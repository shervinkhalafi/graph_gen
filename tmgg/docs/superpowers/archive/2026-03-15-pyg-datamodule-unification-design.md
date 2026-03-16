# G40: Unify datamodule storage on PyG Data/Batch

**Context:** The datamodule hierarchy has three different internal storage formats (batched tensor, list of tensors, list of GraphData) with three different wrapper datasets (_UnwrapDataset, GraphDataset, _GraphDataDataset). All three yield the same thing externally: dense `GraphData` batches. The user wants to leverage PyG's native `Data`/`Batch` ecosystem as the universal internal storage, converting to dense `GraphData` only at the batch boundary.

**Goal:** Store all graphs internally as `list[torch_geometric.data.Data]`. Use PyG's `Batch.from_data_list()` for collation. Convert to dense `GraphData` in a custom collate function. Eliminate `_UnwrapDataset`, `_GraphDataDataset`, the triple storage format, and `GraphDataModule`'s redundant `val_adjacency_matrices` lists.

---

## Design

### Storage: `list[pyg.data.Data]`

All datamodules store graphs as `list[torch_geometric.data.Data]` after `setup()`. Each `Data` object holds `edge_index` (COO), optional `x` (node features), and `num_nodes`. This is PyG's native representation — no subclassing or reimplementation.

For synthetic graphs (SBM, ER, etc.), `setup()` generates numpy adjacency matrices as today, then converts each to a `Data` object via `torch_geometric.utils.dense_to_sparse()`.

For PyG benchmark datasets (QM9, ENZYMES), the `Data` objects come directly from PyG's dataset classes — no eager `to_dense_adj()` conversion. We keep their native format.

### Collation: PyG Batch → dense GraphData

A single collate function replaces all three wrapper datasets:

```python
def collate_to_graphdata(data_list: list[Data]) -> GraphData:
    """Collate PyG Data objects into a dense GraphData batch."""
    batch = Batch.from_data_list(data_list)
    # to_dense_adj and to_dense_batch handle variable-size graphs
    adj = to_dense_adj(batch.edge_index, batch.batch, max_num_nodes=...)
    x = to_dense_batch(batch.x, batch.batch, max_num_nodes=...)[0]
    node_mask = to_dense_batch(torch.ones(...), batch.batch, ...)[0]
    # Convert to one-hot categorical GraphData
    return GraphData(X=..., E=..., y=..., node_mask=node_mask)
```

This lives on `GraphData` as a classmethod: `GraphData.from_pyg_batch(batch: Batch) -> GraphData`.

### Datamodule simplification

**`MultiGraphDataModule`**: `setup()` generates adjacencies → converts to `list[Data]` via `dense_to_sparse`. Dataloaders use the shared collate function. Delete `_UnwrapDataset`.

**`GraphDataModule`**: The `samples_per_graph` feature (multiple noise realizations per graph) remains, but implemented by repeating `Data` objects in the list rather than via a special `GraphDataset`. Delete `val_adjacency_matrices`, `_create_dataloader`. Most of `GraphDataModule` collapses into `MultiGraphDataModule` with a `samples_per_graph` parameter.

**`SyntheticCategoricalDataModule`**: Same storage (`list[Data]`), but additionally computes marginals from the data at `setup()` time. The marginal computation converts to dense temporarily, computes, discards. Delete `_GraphDataDataset`.

**`SingleGraphDataModule`**: Stores one `Data` per split. Dataloader repeats it N times.

### What stays the same

- `GraphData` as the model-facing batch type — no model changes
- `GraphData.collate()` still exists (for any code that constructs GraphData directly)
- All Hydra YAML configs — `_target_` still points at the same datamodule classes
- External API: `train_dataloader()`, `val_dataloader()`, `test_dataloader()`, `get_reference_graphs()`

---

## Files to modify

| File | Change |
|------|--------|
| `src/tmgg/data/datasets/graph_types.py` | Add `GraphData.from_pyg_batch(batch: Batch) -> GraphData` classmethod. Add `to_pyg(self) -> Data` instance method for roundtrip. |
| `src/tmgg/data/datasets/pyg_datasets.py` | Simplify `PyGDatasetWrapper` to return `Data` objects instead of numpy arrays. Remove eager `to_dense_adj` conversion. |
| `src/tmgg/data/data_modules/multigraph_data_module.py` | Change `_train_data`/`_val_data`/`_test_data` from `Tensor` to `list[Data]`. Delete `_UnwrapDataset`. Use `collate_to_graphdata` in dataloaders. |
| `src/tmgg/data/data_modules/data_module.py` | Collapse most logic into `MultiGraphDataModule`. Keep only PyG dataset loading and `samples_per_graph` as distinct features. Delete `val_adjacency_matrices`, `_create_dataloader`. |
| `src/tmgg/experiments/discrete_diffusion_generative/datamodule.py` | Change `_train_data` from `list[GraphData]` to `list[Data]`. Delete `_GraphDataDataset`. Marginal computation reads from `Data` objects. |
| `src/tmgg/data/data_modules/base_data_module.py` | Update `get_reference_graphs()` to convert from `Data` objects instead of iterating dataloaders. |
| Tests | Update tests that access `_train_data`, `val_adjacency_matrices`, etc. |

## Verification

- `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow and not modal"` — all tests pass
- `uv run tmgg-spectral-arch --cfg job` — config still composes
- `uv run tmgg-spectral-arch trainer.max_steps=10` — training runs
- basedpyright clean on all modified files
- Memory comparison: load a PyG dataset (ENZYMES) before and after to verify no eager densification

## Risks

- **PyG version coupling**: We depend on `torch_geometric.utils.to_dense_adj` and `Batch.from_data_list()` stability. Both are core PyG utilities unlikely to break.
- **Performance**: `dense_to_sparse` at setup time + `to_dense_adj` per batch may be marginally slower than current eager conversion for small fixed-size graphs. For large/variable-size graphs it's a net win.
- **Edge features**: Current code only uses binary adjacency (2-class one-hot edges). The conversion must handle this correctly — `edge_attr` of `None` maps to binary presence/absence.
