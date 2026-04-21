# Section 12 Review — Masking, symmetry, diagonal conventions

**Summary verdict:** All six checks pass. No gaps remain.

---

## Spec (§12) requirements

1. Diagonal of E zeroed at data ingestion (`encode_no_edge`, `src/utils.py:73-74`).
2. Model also zeros the diagonal on its output (`diag_mask`, `transformer_model.py:279`).
3. Symmetry of E enforced after every `sample_discrete_features` call via `triu(diagonal=1) + triu(diagonal=1).T`.
4. Initial noise prior enforced symmetrically (same triu construction).
5. Padding applied via `node_mask` multiplication; upstream asserts symmetry inside `PlaceHolder.mask`.
6. Row predicate `(true != 0).any(-1)` drops diagonal and padding simultaneously; no explicit diagonal mask on predictions is needed because target-side zeroing handles it.

---

## Our implementation vs. spec

### Check 1 — Diagonal of E at target construction

**Upstream:** `encode_no_edge` in `src/utils.py:73-74`:
```python
diag = torch.eye(E.shape[1], dtype=torch.bool).unsqueeze(0).expand(E.shape[0], -1, -1)
E[diag] = 0
```
This zeroes the entire `(bs, n, n, de)` slice at the diagonal, making every `(i, i)` position an all-zero vector.

**Ours:** `GraphData.from_pyg_batch`, `graph_types.py:518`:
```python
E_class[:, diag, diag, :] = 0.0
```
where `diag = torch.arange(n_max, device=adj.device)`. Exactly the same invariant. Added in commit `82bcec26`. **PASS.**

### Check 2 — Diagonal zeroed in model output

**Upstream:** `transformer_model.py:279`
```python
E = (E + E_to_out) * diag_mask
```
where `diag_mask` is the boolean complement of the identity (`~eye`, shape `(bs, n, n, 1)`). The diagonal is zeroed before the final symmetrisation.

**Ours:** `models/digress/transformer_model.py:768-778`:
```python
E_final = (E_out + E_to_out) * diag_mask
E_symmetric = 1 / 2 * (E_final + torch.transpose(E_final, 1, 2))
return GraphData(...).mask_zero_diag()
```
`diag_mask` is constructed identically (`~eye`, `unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)`). After the symmetrisation the output goes through `mask_zero_diag()` which additionally zeroes the diagonal in one combined call. The input hidden state also goes through `mask_zero_diag()` immediately after `mlp_in_E` (`line 754`). Upstream applies `PlaceHolder.mask(node_mask)` at the same point, which does not zero the diagonal; our use of `mask_zero_diag` is *strictly more conservative* (zeros both padding and diagonal on the hidden state). **PASS** (and slightly stronger than upstream inside the model).

### Check 3 — Symmetry in `sample_discrete_features`

**Upstream** (`diffusion_utils.py:263-264`):
```python
E_t = torch.triu(E_t, diagonal=1)
E_t = (E_t + torch.transpose(E_t, 1, 2))
```

**Ours** (`diffusion_sampling.py:65-66`):
```python
E_t = torch.triu(E_t, diagonal=1)
E_t = E_t + torch.transpose(E_t, 1, 2)
```
Identical operation. The diagonal in the resulting `E_t` is structurally zero because `triu(diagonal=1)` leaves position `(i,i)` zero and the transpose addition adds zero to it. **PASS.**

### Check 4 — Symmetry of initial noise prior (`sample_discrete_feature_noise`)

**Upstream** (`diffusion_utils.py:385-390`): uses an `upper_triangular_mask` built via `triu_indices(offset=1)`, zeros the lower triangle and diagonal, then adds the transpose.

**Ours** (`diffusion_sampling.py:226-233`): identical construction:
```python
upper_triangular_mask = torch.zeros_like(ue_one_hot)
indices = torch.triu_indices(row=..., col=..., offset=1)
upper_triangular_mask[:, indices[0], indices[1], :] = 1
ue_one_hot = ue_one_hot * upper_triangular_mask
ue_one_hot = ue_one_hot + torch.transpose(ue_one_hot, 1, 2)
```
An explicit symmetry assertion follows (`raise AssertionError("Edge noise is not symmetric")`); upstream only asserts post-sampling in `sample_p_zs_given_zt`. Our assertion is at least as tight. **PASS.**

### Check 5 — Padding via `node_mask` (`.mask()` / `.mask_zero_diag()`)

**Upstream** `PlaceHolder.mask(node_mask, collapse=False)`:
```python
self.X = self.X * x_mask
self.E = self.E * e_mask1 * e_mask2
assert torch.allclose(self.E, torch.transpose(self.E, 1, 2))
```

**Ours** `GraphData.mask()` (`graph_types.py:144-192`): applies the same outer-product mask on both edge fields and asserts symmetry on each populated field. `GraphData.mask_zero_diag()` additionally zeroes the diagonal by composing a `(~eye)` mask with the outer-product mask. Call sites:

| Site | Method called | Effect |
|------|---------------|--------|
| `from_pyg_batch` (after building `E_class`) | returned as-is without calling mask; diagonal already zero | OK — `mask()` applied downstream in `training_step` |
| `_compute_loss` (training, validation) | `target = target.mask()` | zeroes padded rows before CE row-predicate |
| `CategoricalNoiseProcess._apply_noise` | `.mask()` via `data.replace(...).mask()` | correct |
| `CategoricalNoiseProcess.posterior_sample` / `posterior_sample_marginalised` | `z_t.replace(...).mask()` | correct |
| `sample_discrete_feature_noise` (prior) | `.mask()` on the returned `GraphData` | correct |
| `_GraphTransformer.forward` | `.mask_zero_diag()` on hidden state and output | stronger than upstream's `.mask(node_mask)` on the hidden state |

There is one structural difference: upstream calls `PlaceHolder.mask(node_mask)` immediately after `encode_no_edge` in `to_dense`, so the raw batch tensor arriving at `train_loss` is already masked. Our `from_pyg_batch` does not call `.mask()`; the masking is deferred to `_compute_loss` via `target = target.mask()` (added in commit `f6d99185`). The net result is the same: the target tensor reaching `masked_edge_ce` has padding and diagonal zeroed. **PASS.**

### Check 6 — Row predicate `(true != 0).any(-1)` — combined diagonal + padding exclusion

**Upstream** relies on the compound invariant: padding positions are all-zero (multiplied by `e_mask`) and diagonal positions are all-zero (`encode_no_edge`). Therefore `(true != 0).any(-1)` is `False` exactly at those positions.

**Ours** relies on the same invariant, achieved by:
- `from_pyg_batch` sets `E_class[:, diag, diag, :] = 0.0` (diagonal);
- `target = target.mask()` in `_compute_loss` multiplies by `e_mask1 * e_mask2` (padding).

Both `masked_edge_ce` and `masked_node_ce` use the exact same predicate (`train_loss_discrete.py:118`, `68`):
```python
valid = (flat_true != 0).any(dim=-1)
```
No additional diagonal mask is applied on the prediction side; raw logits at diagonal positions are simply never selected by `flat_logits[valid]`. **PASS.**

---

## `mask()` vs `mask_zero_diag()` — usage audit

`mask()` zeros padded positions only; `mask_zero_diag()` zeros padded positions **and** the diagonal.

Correct usage requires `mask_zero_diag()` wherever the diagonal must be suppressed as part of the contract (i.e. inside the transformer where self-loop features must not leak into attention). `mask()` is correct for all other sites where the diagonal is already zero by construction (after `from_pyg_batch` + explicit zeroing) or doesn't matter (e.g. applying padding before loss computation where the row predicate handles the diagonal independently).

Current usage:
- `_GraphTransformer.forward`: uses `mask_zero_diag()` on both the hidden state entering the transformer and the final output. Correct and conservative.
- All noise-process and sampling paths: use `mask()`. Correct — the diagonal is zero by construction after `sample_discrete_features` or `from_pyg_batch`.
- `_compute_loss` target: uses `mask()`. Correct — the diagonal is already zero from `from_pyg_batch`; `mask()` only needs to zero the padding rows.

There is no site where `mask()` is used but `mask_zero_diag()` is required.

---

## Summary

| Check | Verdict |
|-------|---------|
| Diagonal zeroed at `from_pyg_batch` (mirrors `encode_no_edge`) | PASS |
| Model output diagonal zeroed via `diag_mask` + `mask_zero_diag()` | PASS (stronger) |
| `sample_discrete_features`: `triu(diagonal=1) + T` symmetrisation | PASS |
| `sample_discrete_feature_noise`: triu-index construction + assertion | PASS (assertion added) |
| Padding via `mask()` / `mask_zero_diag()`; symmetry asserted | PASS |
| `(true != 0).any(-1)` predicate drops diagonal + padding jointly | PASS |
| `mask()` vs `mask_zero_diag()` — no misuse found | PASS |
