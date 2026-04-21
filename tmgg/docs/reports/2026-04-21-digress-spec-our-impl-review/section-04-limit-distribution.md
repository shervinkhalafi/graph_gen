# Section 4 — Limit Distribution: spec vs. our implementation

**Summary verdict.** The two shipped variants (`uniform`, `empirical_marginal`) are fully implemented and behaviourally match upstream. π estimation is a single `(K,)` scalar marginal aggregated across all positions and all training graphs on both sides — confirmed by two independent prior audits. π_X and π_E are independent. The absorbing-state variant is not implemented in our code; upstream's `AbsorbingStateTransition` is not instantiated by any shipped config and carries a known typo. No gaps for the shipped SBM configuration.

---

## 1. Spec summary

Spec §4 (`docs/reports/2026-04-21-digress-upstream-spec.md:169-223`) defines three limit variants:

- `uniform` — flat PMF `1/K` per class, both X and E.
- `marginal` — empirical node-type and edge-type distributions from the training set, estimated by `AbstractDataModule.node_types` and `edge_counts` (`abstract_dataset.py:34-72`). `edge_counts` counts ordered pairs (so both directions of each undirected edge, plus both orderings of each absent pair) and normalises to a `(K,)` PMF.
- `absorbing` — `AbsorbingStateTransition` (`noise_schedule.py:190-223`), uninstantiated by any shipped config; carries a typo at line 203 (`u_e` written where `u_y` was intended).

Default config: `transition='marginal'` (`configs/model/discrete.yaml:3`).

---

## 2. Our implementation

**File:** `src/tmgg/diffusion/noise_process.py` — `CategoricalNoiseProcess`, lines 719–1258.

### Variants supported

Constructor (`noise_process.py:747`):
```python
limit_distribution: Literal["uniform", "empirical_marginal"] = "uniform",
```
Uniform sets `_limit_x`, `_limit_e`, `_limit_y` immediately at `__init__` time via `_set_stationary_distribution` (lines 760–777). `empirical_marginal` defers to `initialize_from_data`. Any other value raises `ValueError` at line 779.

Absorbing state: **not implemented** — no class, no variant literal, no code path.

### π estimation (`initialize_from_data`, lines 814–857)

```python
x_counts = torch.zeros(self.x_classes, dtype=torch.float64)
e_counts = torch.zeros(self.e_classes, dtype=torch.float64)
for batch in train_loader:
    x_counts += x_batched[node_mask].sum(dim=0)           # (K_x,) scalar aggregate
    upper_triangle = torch.triu(ones(n, n), diagonal=1).unsqueeze(0)
    valid_edges = node_mask.unsqueeze(1) & node_mask.unsqueeze(2) & upper_triangle
    if valid_edges.any():
        e_counts += e_batched[valid_edges].sum(dim=0)      # (K_e,) scalar aggregate
x_marginals = _normalise_counts_or_uniform(x_counts)
e_marginals = _normalise_counts_or_uniform(e_counts)
```

Shape validation in `_set_stationary_distribution` (line 892–894) enforces `(K,)` and will raise on anything else.

### _limit_x / _limit_e / _limit_y buffers

Registered at `__init__` as `None` (`register_buffer`, lines 756–758) and written by `_set_stationary_distribution` (line 896–898). Accessed exclusively through `_stationary_distribution()` (lines 919–926), which raises `RuntimeError` if any buffer is still `None`.

### π_X and π_E independence

`_categorical_kernel` (lines 209–223) is called separately for X and E limits with no cross-coupling:
- `q_x = _categorical_kernel(1.0 - beta_t, x_limit)` (line 994)
- `q_e = _categorical_kernel(1.0 - beta_t, e_limit)` (line 995)

---

## 3. Behavioural match per check

### Variants: uniform and empirical_marginal

**Match.** Both are implemented. Upstream names them `'uniform'` and `'marginal'`; ours uses `'uniform'` and `'empirical_marginal'`. The semantics are identical.

### Absorbing state

**Gap — by design.** Upstream defines `AbsorbingStateTransition` but no shipped config instantiates it. Our code does not implement this variant. The gap is intentional and has no effect on any configuration we run.

Upstream typo (spec §5, `noise_schedule.py:203`): `self.u_e[:, :, abs_state] = 1` is written twice; `self.u_y` is never set to anything other than zeros. Our code has no equivalent because the variant does not exist here.

### π estimation — aggregation scheme

**Match, confirmed.** Both sides produce a single `(K,)` scalar PMF over all edge/node positions, all graphs in the training set.

Structural difference: upstream uses the PyG `edge_index` (which stores both directions of every undirected edge) plus an explicit non-edge count derived from `count * (count - 1)`. Ours uses the strict upper triangle of the dense `E_class` tensor (`triu(..., diagonal=1)`). Both produce the same marginals up to floating-point rounding: upstream's ordered-pair count equals twice the undirected count, and after normalisation the ratio is identical to the ratio produced by the upper-triangle count.

For SBM with `p_intra=1.0, p_inter=0.0` at `n=20` (two balanced blocks of size 10): intra-block undirected pairs = 2 × C(10, 2) = 90 (all edges), inter-block undirected pairs = 10 × 10 = 100 (all non-edges), total = 190. Thus `pi_E ≈ [100/190, 90/190] ≈ [0.526, 0.474]`. The upper-triangle counting and the ordered-pair counting both yield this ratio.

### π_X and π_E independence

**Match.** Both store separate `(K_x,)` and `(K_e,)` vectors with no coupling. The transition kernels are built from them independently.

### Prior buffer access at sampling time

**Match.** Upstream stores `limit_dist` as a `PlaceHolder(X=x_marginals, E=e_marginals, ...)` and passes it to `sample_discrete_feature_noise`. Ours calls `_stationary_distribution()` inside `sample_prior` (line 870–872). Both read the same pre-computed PMFs without re-traversing the data loader.

---

## 4. Remaining gaps

| Item | Status |
|------|--------|
| `absorbing` variant | Not implemented; upstream's version is also unused and buggy. No action needed for SBM runs. |
| `y_probs` always uniform | Ours always sets `y_probs = uniform(y_classes)` (lines 852–856). Upstream does the same (`u_y = ones / y_classes` in `MarginalUniformTransition.__init__:148-150`). Match, no gap. |
| Upper-triangle vs. ordered-pair counting | Produces identical PMFs after normalisation. Numerical difference is O(1e-7) from float64 accumulation; inconsequential. |
