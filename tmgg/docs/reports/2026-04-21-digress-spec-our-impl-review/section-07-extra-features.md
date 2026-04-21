# Section 7 — Extra Features: Spec vs Implementation Review

**Spec source:** `docs/reports/2026-04-21-digress-upstream-spec.md § 7`
**Upstream reference:** `digress-upstream-readonly/src/diffusion/extra_features.py` @ `780242b`
**Our code:** `src/tmgg/models/digress/extra_features.py`, `src/tmgg/models/digress/transformer_model.py`
**Prior audits incorporated:** `analysis/digress-loss-check/parity-audit/04-extra-features.md` (6 checks), `analysis/digress-loss-check/parity-audit/deep-extra-features.md` (8 checks)

---

## Spec summary (§ 7)

For `extra_features_type='all'`, the umbrella `ExtraFeatures` class produces:

- **Node count `n`:** `node_mask.sum(1) / max_n_nodes`, shape `(B, 1)`, appended to every `y`.
- **Cycles** (`NodeCycleFeatures`): adjacency recovered as `E_t[..., 1:].sum(-1)`. Matrix powers
  A¹–A⁶ computed; per-node counts for k=3,4,5, per-graph counts for k=3,4,5,6. Scaled `/10`,
  clamped at 1. Shapes: `(B, n, 3)` node, `(B, 4)` graph.
- **Spectral** (`EigenFeatures`, mode `'all'`): combinatorial Laplacian L = D − A, symmetrised
  `(L + Lᵀ)/2`, with large diagonal patch for padded nodes (`2·n·I` on masked rows/cols).
  `eigh(L)` → eigenvalues divided by actual node count, eigenvectors masked to valid nodes.
  - `get_eigenvalues_features(k=5)`: threshold `< 1e-5` for connected-component count `(B, 1)`;
    gather k=5 smallest non-zero eigenvalues, padding with `2` if `n_connected + k > n`.
  - `get_eigenvectors_features(k=2)`: LCC indicator from rounded first-eigenvector mode `(B, n, 1)`;
    k=2 Fiedler-class eigenvectors `(B, n, 2)`.
- **Assembly** (`'all'`):
  - `extra_X = cat(x_cycles[3], nonlcc_indicator[1], k_lowest_eigvec[2])` → `+6` node dims.
  - `extra_E = zeros(..., 0)` → `+0` edge dims.
  - `extra_y = hstack(n[1], y_cycles[4], n_components[1], batched_eigenvalues[5])` → `+11` graph dims.
- **Timestep:** `t/T` appended LAST to `y` by the outer model (`compute_extra_data` in upstream;
  `GraphTransformer.forward` with `use_timestep=True` in ours). Not included in the `+11` count.
  Total `y` width fed to the transformer: `d_y_original + 11 + 1 = 12` for non-molecular SBM
  (`d_y_original = 0`).

---

## Our implementation

### `ExtraFeatures` and sub-modules (`extra_features.py`)

The class takes `extra_features_type` and `max_n_nodes` as constructor arguments; the `noisy_data`
dict interface is replaced by explicit `(X, E, y, node_mask)` tensors, consistent with the tmgg
convention. All three feature modes (`'cycles'`, `'eigenvalues'`, `'all'`) are implemented and
the `extra_features_dims()` helper accurately declares `(6, 0, 11)` for `'all'`.

### `GraphTransformer.forward` (`transformer_model.py:943–950`)

```python
if self.extra_features is not None:
    extra_X, extra_E, extra_y = self.extra_features(X, E, y, node_mask)
    X = torch.cat([X, extra_X], dim=-1)
    E = torch.cat([E, extra_E], dim=-1)
    y = torch.cat([y, extra_y], dim=-1)

if self._use_timestep and t is not None:
    y = torch.cat([y, t.unsqueeze(-1)], dim=-1)
```

Extras precede timestep; `adjust_dims` widens `input_dims` by `(+6, +0, +11)` at construction,
and the `use_timestep` branch adds a further `+1` to `y`, so the `_GraphTransformer`'s expected
input width matches.

---

## Per-check verdicts

### 1. Feature types for `'all'` — cycles (3/4/5/6), eigenvalues (top-k), eigenvectors (top-k_ev), component count

**MATCH.** `KNodeCycles.k_cycles` assembles `(k3x, k4x, k5x)` for nodes and `(k3y, k4y, k5y, k6y)`
for the graph, with identical polynomial formulas (same signs, divisors, and power-matrix
dependencies as upstream). `EigenFeatures` mode `'all'` computes both `eigvalsh`/`eigh` branches
with the same k=5 / k=2 defaults.

The adjacency is recovered identically in both cycle and spectral branches:
`E[..., 1:].sum(dim=-1)` treats class 0 as "no edge" and sums remaining channels.

### 2. Clean-vs-noisy input — call sites feed `z_t`, never `x_0`

**MATCH** across all four call sites:
- Training (`diffusion_module.py`): `z_t = noise_process.forward_sample(batch, t_int)` → `model(z_t, t=condition)`.
- Validation: same pattern with `lowest_t=1`.
- Reconstruction pass: `z_1` (one-step noised).
- Sampler reverse chain (`sampler.py:311`): `model(z_t, t=condition)` where `z_t` is the running
  latent updated each step.

The transformer's `forward` receives `data.X_class` / `data.E_class` directly as `X` / `E` before
calling `self.extra_features(X, E, y, node_mask)`. No clean batch fields reach the feature
computation on any path.

### 3. `max_n_nodes` role — normalisation only?

**MATCH.** `max_n_nodes` appears in exactly one place: `n = node_mask.sum(1).float() / self.max_n_nodes`
(our line 185). It is not used for cycle scaling (that uses hard-coded `/10`) and not for
eigenvalue scaling (that uses the per-sample actual node count). Confirmed identical in upstream.

### 4. Laplacian choice — combinatorial `L = D − A`, symmetrised on both sides?

**MATCH.** Our `compute_laplacian(A, normalized=False, symmetrize=True)` produces `(D − A + (D − A)ᵀ)/2`
= `D − A` (since `D − A` is already symmetric for symmetric `A`). Upstream's `compute_laplacian(A, normalize=False)`
always symmetrises internally. The adjacency `A` is masked to valid nodes before Laplacian
construction, so asymmetry from padding cannot arise. The large diagonal patch for padded rows
(`2 · n_max · I · (~mask_i) · (~mask_j)`) is constructed identically to upstream (line 515–517 vs
upstream line 87–89).

### 5. Connected-component threshold `< 1e-5` and padding conventions

**MATCH.** `n_connected = (ev < 1e-5).sum(-1)` with the same assertion `(n_connected > 0).all()`.
Padding uses `2 · ones(bs, to_extend)` in both codebases (our line 581, upstream line 152).
Eigenvector column padding uses `zeros(bs, n, to_extend)` (our line 629, upstream line 179).
The threshold is numerically safe: the Fiedler value for the graphs in our experiments is
multiple orders of magnitude above 1e-5.

### 6. `y` concatenation order — `(n[1], cycles[4], n_components[1], eigenvalues[5])` = 11 dims

**MATCH.** Both codebases produce `hstack((n, y_cycles, n_components, batched_eigenvalues))` as
the `extra_y` tensor. Widths: 1 + 4 + 1 + 5 = 11. For `'all'` mode, the X concatenation order is
`cat(x_cycles[3], nonlcc_indicator[1], k_lowest_eigvec[2])` = 6 node dims. Verified line-by-line
against upstream lines 47–52.

### 7. Timestep placement — `t/T` last in `y`, at every call site

**MATCH.** The timestep is appended last in `y` at every call site (training, validation, sampler).
In our code, `GraphTransformer.forward` performs this at lines 949–950:
`y = cat([y, t.unsqueeze(-1)], dim=-1)` after the structural extras are already concatenated.
This matches upstream's `compute_extra_data` which appends `t` to `extra_y` last, before
`hstack((y_t, extra_y))` in the model's `forward`.

`t` shape: upstream has `(B, 1)` throughout; ours has `(B,)` at the call boundary and
`unsqueeze(-1)` inside the transformer. Numerically identical after concatenation.

### 8. Eigenvector masking — `mask_2d` vs `mask.unsqueeze(1) * mask.unsqueeze(2)`

**MATCH with one structural note.** Upstream (line 101) masks eigenvectors as
`eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)`, which is `(B, n, n)` × `(B, 1, n)` × `(B, n, 1)` —
a 2D node-product mask on the eigenvector matrix. Our code at line 530 uses
`eigvectors * mask_2d(node_mask)` where `mask_2d` expands to `(B, n, n)` via
`node_mask.unsqueeze(1) * node_mask.unsqueeze(2)`. These are the same 2D mask; the difference
is cosmetic.

---

## Summary

| Check | Verdict |
|---|---|
| Feature types for `'all'` (cycles 3/4/5/6, eigenvalues k=5, eigenvectors k=2, component count) | MATCH |
| Clean-vs-noisy: all call sites feed `z_t` (training, validation, sampler, reconstruction) | MATCH |
| `max_n_nodes` used only for node-count normalisation, nowhere else | MATCH |
| Laplacian: combinatorial `D − A`, symmetrised, same diagonal patch for padding | MATCH |
| Connected-component threshold `1e-5`, padding value `2`, both clamped | MATCH |
| `y` layout: `(n, y_cycles, n_components, eigenvalues)` = 11 dims | MATCH |
| `t/T` appended LAST to `y` at every call site | MATCH |
| Eigenvector masking (2D node-product mask) | MATCH (cosmetic diff only) |

**No divergences found.** The implementation is numerically equivalent to the upstream spec on
every axis checked. The interface refactor (explicit tensor args instead of `noisy_data` dict,
`symmetrize` flag on `compute_laplacian`, `.clamp()` vs in-place clip) introduces no
behavioural differences.

### Remaining gaps / not audited here

- The `EigenvectorAugmentation` class in our `extra_features.py` (lines 231–297) has no upstream
  counterpart; it is a tmgg-specific extension and falls outside the spec scope.
- `extra_features_dims` returns `(3, 0, 5)` for mode `'cycles'`. The spec's "eigenvalues" mode
  comment in the docstring says `y+11` but the spec table gives `y+5` for the non-`'all'`
  path — not audited here since the failing-run config uses `'all'`.
- The `DummyExtraFeatures.adjust_dims` method returns `input_dims` unchanged, which is correct,
  but if a caller passes `extra_features=DummyExtraFeatures()` alongside `use_timestep=True`
  the total `y` input width is `d_y + 1`, which differs from the upstream non-molecular path
  where `y_t` is width 0 and `extra_y` is width `0 + 1` (just timestep). This is correct
  behaviour; noted as a configuration concern, not a code bug.
