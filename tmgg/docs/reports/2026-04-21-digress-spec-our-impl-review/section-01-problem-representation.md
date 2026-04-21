# Section 1 review — Problem statement and graph representation

Reviewer: agent, 2026-04-21. Spec: `docs/reports/2026-04-21-digress-upstream-spec.md §1`.

---

## 1. Spec summary

Upstream encodes a graph as a triple `(X, E, y)` with a paired `node_mask`.
`X ∈ {0,1}^{B×n×dX}` is a one-hot node-type float tensor; `E ∈ {0,1}^{B×n×n×dE}`
is a one-hot edge-type tensor, symmetric, with class 0 = "no edge" and the
diagonal set to the all-zero vector. `y ∈ ℝ^{B×dy}` is graph-level; for
SPECTRE datasets it is empty (`dy=0`). The batch is padded to `n_max` and
`node_mask: (B, n_max)` bool tracks real nodes. The container is `PlaceHolder`
with a `.mask(node_mask)` method. Upstream calls `to_dense` → `encode_no_edge`
at a single data-layer boundary (`src/utils.py:53-75`) before any diffusion
or loss code touches the tensors.

---

## 2. Our implementation

### Container: `GraphData`

`src/tmgg/data/datasets/graph_types.py:29-649`.

We use a frozen dataclass with split fields `(X_class, X_feat, E_class, E_feat, y, node_mask)`.
`X_class` is `Tensor | None`; `E_class` is `Tensor | None` (one-hot categorical);
`E_feat` is `Tensor | None` (continuous). `node_mask: (B, n_max)` bool.
`__post_init__` (lines 70–133) validates shape consistency. `.mask()` (lines 144–192)
zeroes padded rows and asserts symmetry on every non-None edge field.
`mask_zero_diag()` (lines 194–224) additionally zeros the diagonal.

Key shape contract: `X_class: (B, n, dX)` float, `E_class: (B, n, n, dE)` float,
`y: (B, dy)`, `node_mask: (B, n)` bool — identical to upstream.

### Dense-batch construction: `GraphData.from_pyg_batch`

`src/tmgg/data/datasets/graph_types.py:466–522`.

Calls `to_dense_adj(edge_index, batch_vec)`, builds node_mask from `bincount`,
symmetrises via `(adj + adj.T).clamp(max=1)`, zeros the diagonal, then builds
`E_class = stack([1-adj, adj], dim=-1)` (two-channel, class 0 = no-edge).
At line 518: `E_class[:, diag, diag, :] = 0.0` — zeroes the all-diagonal
of the target tensor. `y = zeros(bs, 0)` for structure-only datasets.

The corresponding upstream call chain is `to_dense` → `encode_no_edge` at
`digress-upstream-readonly/src/utils.py:53-75`.

### Collation path

`src/tmgg/data/data_modules/multigraph_data_module.py:56–64`.
`_collate_pyg_to_graphdata` calls `Batch.from_data_list` then `GraphData.from_pyg_batch`.
Used by both `MultiGraphDataModule` and `SpectreSBMDataModule`.

### `X_class` handling for structure-only datasets

Structure-only batches (`from_pyg_batch`) leave `X_class=None` — the
spec's "architecture-internal concern" note. At training time,
`_read_field` in `training/lightning_modules/diffusion_module.py:118–137`
synthesises a degenerate `[no-node, node]` one-hot from `node_mask` with
all-zero rows at padded positions, so the `(true != 0).any(-1)` predicate
in `masked_node_ce` still excludes padding correctly.

### No `remove_self_loops` call

Upstream's `to_dense` explicitly calls
`torch_geometric.utils.remove_self_loops` before `to_dense_adj`
(`digress-upstream-readonly/src/utils.py:35-36`). Our `from_pyg_batch`
does not; instead it zeroes the diagonal of `adj` after calling
`to_dense_adj` (line 503: `adj[:, diag, diag] = 0.0`).

### `PlaceHolder.mask` collapse path vs our `mask()`

Upstream's `PlaceHolder.mask(node_mask, collapse=False)` multiplies by
masks and asserts symmetry; `collapse=True` argmaxes to integer labels
and writes −1 at padded positions. Our `GraphData.mask()` mirrors only
the multiplication + symmetry-assert path; integer-label collapse is
handled separately by `collapse_to_indices` (lines 652–691) which writes
−1 at masked positions.

---

## 3. Behavioural match

**E_class encoding: MATCH.**
Both pipelines produce a float one-hot tensor over `(B, n, n, dE)` with
class 0 = no-edge, symmetric upper-and-lower triangle, and an all-zero
diagonal. The invariant is enforced at the same logical boundary (PyG →
dense conversion) in both codebases.

**X_class for SPECTRE (structure-only): DIVERGES (benign).**
Upstream SPECTRE datasets carry `X: ones(n,1)` — a trivial single-class
node tensor (`dX=1`). Our `from_pyg_batch` emits `X_class=None` and
synthesises a two-channel `[no-node, node]` one-hot only inside the
training-loss path. Numerically the loss contribution from the node CE
term is identical (only one valid class), but the tensor shapes differ:
upstream `(B, n, 1)` vs our synthesised `(B, n, 2)`. This is an
intentional design choice (the unified-spec removes the redundant
node-present field); it does not affect the edge-CE path or any
generation logic.

**`node_mask` shape and semantics: MATCH.** Both produce `(B, n_max)` bool,
where `n_max` is the maximum node count in the batch.

**`y` for SPECTRE: MATCH.** Both emit `zeros(B, 0)` for non-molecular datasets.

**`remove_self_loops` vs diagonal zeroing: DIVERGES (benign).**
Upstream removes self-loops from the sparse representation before densification;
we zero the diagonal of the dense adjacency after densification. The result
is the same: no self-loops appear in `E_class` and the loss row predicate
excludes diagonal entries.

**`PlaceHolder.mask` collapse path: DIVERGES (benign).**
The integer-collapse (`collapse=True`) is separated into `collapse_to_indices`
in our codebase. The semantics are identical (argmax + −1 sentinel at masked
positions). No training or loss code calls the upstream `collapse=True` path.

---

## 4. Already fixed on main

**`82bcec26`** — zeroed `E_class[:, diag, diag, :]` in `from_pyg_batch`.
Before this, diagonal entries emitted `[1, 0]` (class-0 one-hot "no-edge"),
which survived the `(true != 0).any(-1)` predicate in `masked_edge_ce` and
inflated the CE denominator by ~5% on `n=20` batches. The fix mirrors
`encode_no_edge`'s `E[diag] = 0` at `digress-upstream-readonly/src/utils.py:73-74`.

**`f6d99185`** — rewrote `masked_node_ce` / `masked_edge_ce` to accept raw
logits and use `F.cross_entropy` with the `(true != 0).any(-1)` row predicate,
matching `digress-upstream-readonly/src/metrics/train_metrics.py:95-102` exactly.
Added explicit `target = target.mask()` in `_compute_loss` to zero padding rows
before the predicates run, matching the upstream `dense_data.mask(node_mask)`
call at `diffusion_model_discrete.py:108`.

**`59e9593f`** — removed dead `mask_distributions` helper; its role in
padding/diagonal handling for the legacy soft-CE path is now covered by the
`(true != 0).any(-1)` predicate.

---

## 5. Remaining gaps

1. **`X_class=None` vs upstream's trivial one-hot (`dX=1`).** The synthesised
   two-channel node tensor in `_read_field` produces node-CE loss over two
   classes instead of one. For any structure-only SBM run, the node CE term
   is effectively zero (every real node gets class-1, every padding node is
   excluded), but the `dX=2` vs `dX=1` difference means `lambda_X × node_CE`
   would differ in theory if a model ever emits non-trivial `X_class` logits.
   In practice the transformer for structure-only SBM does not output
   `X_class`, so this gap is latent rather than active.

2. **No assertion of E symmetry at the `from_pyg_batch` boundary.** Upstream
   asserts `E == E.transpose` inside `PlaceHolder.mask` (called immediately
   after `to_dense` in the training step). Our `GraphData.mask()` does assert
   symmetry, but `from_pyg_batch` itself does not. The symmetrisation via
   `(adj + adj.T).clamp(max=1)` is correct in practice; a post-construction
   assert would harden the invariant.

3. **`max_n_nodes` derivation.** Upstream's `AbstractDatasetInfos.complete_infos`
   computes `max_n_nodes` from train+val. Our `SpectreSBMDataModule` sets
   `num_nodes = max(n_nodes)` over the full dataset (all splits), which is
   equivalent for SPECTRE (the upstream split logic is the same), but the
   derivation is not explicitly restricted to train+val. No functional impact
   for the fixed 200-graph SPECTRE fixture.
