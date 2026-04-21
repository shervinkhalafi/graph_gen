# Section 9: Training Objective — Parity Review

**Date:** 2026-04-21
**Status:** POST-REFACTOR (commits `f6d99185`, `82bcec26`, `59e9593f` all landed on `main`)
**Verdict: MATCH on all eight checks. No remaining gaps.**

---

## Spec summary (§9)

The training objective is the $x_0$-parameterised, unweighted-in-$t$ cross-entropy

$$
\mathcal{L}(\theta) = \mathbb{E}_{t,\mathbf{z}_0,\mathbf{z}_t}\bigl[
  \mathrm{CE}(X_0, p_\theta(X_0|\mathbf{z}_t))
  + \lambda_E\,\mathrm{CE}(E_0, p_\theta(E_0|\mathbf{z}_t))
  + \lambda_y\,\mathrm{CE}(y_0, p_\theta(y_0|\mathbf{z}_t))
\bigr]
$$

where:
- `CrossEntropyMetric.update` calls `F.cross_entropy(logits, argmax(target), reduction='sum')`, then `compute()` divides by the number of valid rows — effectively `reduction='mean'` over valid positions.
- Masked rows: `(true != 0.).any(-1)` after flattening to `(B·n, d)` / `(B·n·n, d)`. All-zero rows (padding, diagonal) are excluded.
- Diagonal is all-zero because `encode_no_edge` in `to_dense` zeroes it (`src/utils.py:73-74`) before the target reaches the loss.
- `lambda_train = [lambda_E, lambda_y] = [5, 0]` (SBM default). Node CE has implicit weight 1.
- Targets are the masked clean data `dense_data.mask(node_mask)`, computed before `apply_noise`.

---

## Our implementation — relevant entry points

- `DiffusionModule.training_step` (`diffusion_module.py:470-511`) — samples `t`, calls `noise_process.forward_sample`, calls `_compute_loss(pred, batch)`.
- `DiffusionModule._compute_loss` (`diffusion_module.py:513-594`) — per-field loop; for categorical fields dispatches to `masked_node_ce` / `masked_edge_ce` after `target = target.mask()`.
- `masked_node_ce` / `masked_edge_ce` (`train_loss_discrete.py:27-128`) — take raw logits; apply `(true != 0).any(-1)` predicate; call `F.cross_entropy(..., reduction='mean', label_smoothing=0.0)`.
- `TrainLossDiscrete` (`train_loss_discrete.py:171-234`) — thin compat wrapper kept for tests; composes the two helpers with `lambda_E` weighting.

---

## Per-check verdicts

### 1. x0-parameterisation

**Spec:** network predicts $p_\theta(x_0|\mathbf{z}_t)$; targets are the clean graph $X_0, E_0$.

**Ours:** `training_step` calls `_compute_loss(pred, batch)` where `pred` is the model output on `z_t` and `batch` is the clean input `GraphData`. `_read_field(target, field)` pulls `batch.X_class` / `batch.E_class` — unmodified clean tensors. The noise process does not overwrite `batch`.

**Verdict: MATCH.**

---

### 2. CE on raw logits (not pre-softmaxed probs)

**Spec:** `CrossEntropyMetric.update` calls `F.cross_entropy(preds, argmax(target), reduction='sum')` on the raw model output. The `log_softmax` is fused inside `F.cross_entropy`.

**Pre-`f6d99185` state:** `_compute_loss` applied `F.softmax` before the CE helpers; helpers used `log(softmax(logits))` — gradient-equivalent but numerically less stable. `BUG_REPORT.md` flagged this as the sole confirmed parity divergence (alongside `_epsilon_renormalise(1e-7)` on both targets and predictions).

**Post-`f6d99185` state (current):** `masked_node_ce` / `masked_edge_ce` receive raw logits unchanged. `_compute_loss` no longer softmaxes. `F.cross_entropy(..., reduction='mean', label_smoothing=0.0)` is called directly. `TrainLossDiscrete.__call__` documentation explicitly notes "Not softmaxed."

**Verdict: MATCH.** Confirmed by `TestUpstreamParity` regression tests at `atol=1e-6`.

---

### 3. `lambda_E = 5.0` — single application, correct term

**Spec:** `loss_X + lambda_train[0] * loss_E + lambda_train[1] * loss_y`. `lambda_train[0] = 5` multiplies the edge CE once. Node CE has weight 1.

**Ours:** `_DEFAULT_LAMBDA_PER_FIELD = {"X_class": 1.0, "E_class": 5.0, ...}`. The constructor merges any `lambda_per_field` overrides then sets `lambda_per_field["E_class"] = float(lambda_E)` (line 401). Inside `_compute_loss` each field's term is `weight * masked_*_ce(...)` summed once; `E_class` gets weight `lambda_E = 5.0`, `X_class` gets 1.0. No double-application.

`TrainLossDiscrete.__call__` returns `loss_x + self.lambda_E * loss_e`, matching upstream line-for-line.

**Verdict: MATCH.**

---

### 4. Row-drop predicate `(true != 0).any(-1)`

**Spec:** after flattening to `(B·n, d_X)` / `(B·n·n, d_E)`, `mask = (true != 0.).any(dim=-1)` drops all-zero rows, excluding padded node positions and diagonal edge positions.

**Ours (`train_loss_discrete.py:68, 118`):**
```python
valid = (flat_true != 0).any(dim=-1)
flat_logits = flat_logits[valid]
flat_true   = flat_true[valid]
```
Identical predicate, identical indexing order.

**Verdict: MATCH.**

---

### 5. Diagonal exclusion

**Spec:** upstream relies on `encode_no_edge` (`src/utils.py:73-74`) zeroing the diagonal of `E` before it reaches the loss. Diagonal rows are all-zero ⟹ the `(true != 0).any(-1)` predicate drops them automatically. No explicit diagonal index mask in the loss function itself.

**Ours:** commit `82bcec26` added `E_class[:, diag, diag, :] = 0.0` in `GraphData.from_pyg_batch` (`graph_types.py:518`). The comment cites the upstream `encode_no_edge` line. All datamodule paths that produce `GraphData` go through `from_pyg_batch` or equivalent construction that preserves the all-zero diagonal invariant. `_compute_loss` then calls `target.mask()` which multiplies by `e_mask1 * e_mask2` (both node masks), further zeroing any edge position involving a padded node. Diagonal entries carry all-zero `E_class` rows from the data layer onward; the `(true != 0).any(-1)` predicate drops them inside the CE helpers.

The mechanism is equivalent to upstream's: both enforce the diagonal invariant at the data layer, not inside the loss function.

**Verdict: MATCH.**

---

### 6. Target masking: `target.mask()` vs upstream's `dense_data.mask(node_mask)`

**Spec:** upstream applies `dense_data.mask(node_mask)` at `src/diffusion_model_discrete.py:108` before the loss call. This multiplies `X` and `E` by the node mask, zeroing padding rows. The masked tensors are then passed as `true_X=X, true_E=E`.

**Pre-`f6d99185` state:** ours was missing this step. The `(true != 0).any(-1)` predicate could see non-zero padding rows under the old soft-CE path, inflating the denominator.

**Post-`f6d99185` state (current):** `_compute_loss` calls `target = target.mask()` (`diffusion_module.py:554`) before any field is read. `GraphData.mask()` multiplies `X_class` by `x_mask` and `E_class` by `e_mask1 * e_mask2`, zeroing all padding positions. The structure-only `X_class` synthesis in `_read_field` also emits all-zero rows at padding positions (line 136: `return synth * node_ind.unsqueeze(-1)` where `node_ind = data.node_mask.float()`).

**Verdict: MATCH.**

---

### 7. Reduction: `reduction='sum' / n_valid` vs `reduction='mean'`

**Spec:** upstream sums CE with `reduction='sum'` into a running accumulator and divides by accumulated `total_samples` at `compute()`. For a single batch this equals `sum_CE_over_valid_rows / n_valid_rows`. This is the global-mean-over-valid-rows within the batch.

**Ours:** `F.cross_entropy(..., reduction='mean')` divides by the number of valid rows passed (after `valid = (flat_true != 0).any(-1)` indexing). The denominator is exactly `n_valid_rows` — identical to upstream's `total_samples` count for the same batch.

When graphs have uniform size (SBM, n=20), both reductions produce the same value per position regardless of which graph each row came from. For variable-size batches, upstream's global mean differs from a per-graph-then-batch mean, but `reduction='mean'` over the full valid-row set is the same global mean that upstream uses — not the two-stage per-graph average of the old helpers. This is the correct reduction.

**Verdict: MATCH.** (The old audit §5 flagged a benign divergence; that was the pre-`f6d99185` code path with the two-stage per-graph MSE-style reduction. The current helpers use flat-index `reduction='mean'` which reproduces upstream's single-stage global mean.)

---

### 8. `label_smoothing` default 0.0

**Spec:** upstream uses `F.cross_entropy(preds, target, reduction='sum')` with no `label_smoothing` argument — i.e., `label_smoothing=0.0`.

**Ours:** `masked_node_ce` and `masked_edge_ce` have `label_smoothing: float = 0.0` as a keyword-only parameter with that default. No call site overrides it. `TrainLossDiscrete` stores `self.label_smoothing = label_smoothing` (default 0.0) and passes it through.

The `TestUpstreamParity` regression tests pin bit-for-bit equivalence at `atol=1e-6` with this default, enforcing that no accidental smoothing is added.

**Verdict: MATCH.**

---

## Summary verdict table

| Check | Verdict |
|---|---|
| x0-parameterisation | MATCH |
| CE on raw logits (post-`f6d99185`) | MATCH |
| `lambda_E = 5.0`, single application | MATCH |
| Row-drop predicate `(true != 0).any(-1)` | MATCH |
| Diagonal exclusion via data-layer fix (`82bcec26`) | MATCH |
| Target masking (`target.mask()`, post-`f6d99185`) | MATCH |
| Reduction: flat-index `mean` = upstream `sum / n_valid` | MATCH |
| `label_smoothing` default 0.0 | MATCH |

All eight checks pass. The `_epsilon_renormalise(1e-7)` parity divergence documented in `BUG_REPORT.md` is eliminated by `f6d99185`. No new gaps introduced.

---

## Remaining non-gap observations

- The `y`-loss path is not tested by the SBM configuration (`lambda_y = 0`, no `y` conditioning), but `TrainLossDiscrete` includes `loss_y` weighted by `lambda_train[1]`. Our module does not currently iterate over a `y` field; the per-field loop in `_compute_loss` only processes fields declared by `noise_process.fields`. Extending to molecular datasets that use `y` conditioning would require registering `y` as a field. This is a scope gap for future work, not a bug in the current SBM/structure-only regime.
- `node_mask` parameter is accepted by `masked_node_ce` / `masked_edge_ce` but immediately deleted (`del node_mask`). The comment explains this is intentional: the `(true != 0).any(-1)` predicate is the row filter, and the mask is kept in the signature for API stability. This matches upstream's design, where there is no explicit `node_mask` argument to `TrainLossDiscrete.forward` — padding is handled solely by the all-zero row convention.
- The diagonal-exclusion mechanism (data-layer fix vs inline `encode_no_edge`) is functionally equivalent, but ours applies the fix once at dataset construction time (`from_pyg_batch`) rather than at each forward pass. The invariant is established earlier in the pipeline; the loss helpers see it as a precondition rather than enforcing it themselves.
