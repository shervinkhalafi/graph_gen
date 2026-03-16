# PyG Datamodule Storage Unification — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Unify all datamodule internal storage on `list[torch_geometric.data.Data]`, converting to dense `GraphData` only at the batch boundary via a shared collate function.

**Architecture:** Each datamodule stores graphs as PyG `Data` objects (COO edge_index). A single `collate_to_graphdata()` function replaces three wrapper datasets (`_UnwrapDataset`, `GraphDataset`, `_GraphDataDataset`) by converting a list of `Data` objects to a dense `GraphData` batch via `Batch.from_data_list()` + `to_dense_adj()`. The `GraphData` model-facing interface is unchanged.

**Tech Stack:** PyTorch, PyTorch Geometric (`torch_geometric.data.Data`, `torch_geometric.data.Batch`, `torch_geometric.utils.to_dense_adj`, `torch_geometric.utils.dense_to_sparse`), PyTorch Lightning

**Spec:** `docs/superpowers/specs/2026-03-15-pyg-datamodule-unification-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `src/tmgg/data/datasets/graph_types.py` | Modify | Add `from_pyg_batch()` classmethod, `to_pyg()` instance method |
| `src/tmgg/data/data_modules/multigraph_data_module.py` | Modify | Change storage to `list[Data]`, delete `_UnwrapDataset`, use shared collate |
| `src/tmgg/data/data_modules/data_module.py` | Modify | Delete `val_adjacency_matrices`, simplify to thin subclass |
| `src/tmgg/data/data_modules/single_graph_data_module.py` | Modify | Change storage to `Data` objects |
| `src/tmgg/experiments/discrete_diffusion_generative/datamodule.py` | Modify | Delete `_GraphDataDataset`, use shared collate, marginals from `Data` |
| `src/tmgg/data/datasets/pyg_datasets.py` | Modify | Return `list[Data]` instead of numpy arrays |
| `src/tmgg/data/data_modules/base_data_module.py` | Modify | Update `get_reference_graphs()` to work with new storage |
| `src/tmgg/data/datasets/graph_dataset.py` | Delete or simplify | `GraphDataset` replaced by repeating `Data` objects |
| Tests | Update | Adapt to new storage types and removed classes |

---

## Chunk 1: GraphData ↔ PyG conversion methods

### Task 1: Add `from_pyg_batch()` and `to_pyg()` to GraphData

**Files:**
- Modify: `src/tmgg/data/datasets/graph_types.py`
- Test: `tests/data_modules/test_pyg_conversion.py` (create)

- [ ] **Step 1a: Write tests for from_pyg_batch and to_pyg**

Create `tests/data_modules/test_pyg_conversion.py`:

```python
"""Tests for GraphData ↔ PyG Data conversion.

Test Rationale
--------------
from_pyg_batch and to_pyg bridge the PyG sparse representation (COO
edge_index) and the dense categorical representation (one-hot X, E)
used by all models. Round-trip fidelity is critical: converting
adjacency → Data → Batch → GraphData → adjacency must reproduce
the original.
"""

from __future__ import annotations

import torch
import pytest
from torch_geometric.data import Data, Batch

from tmgg.data.datasets.graph_types import GraphData


def _make_triangle_graph() -> Data:
    """Triangle graph: 3 nodes, 3 undirected edges."""
    edge_index = torch.tensor([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], dtype=torch.long)
    return Data(edge_index=edge_index, num_nodes=3)


def _make_square_graph() -> Data:
    """Square graph: 4 nodes, 4 undirected edges (cycle)."""
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    return Data(edge_index=edge_index, num_nodes=4)


class TestFromPygBatch:
    """Verify from_pyg_batch converts PyG Batch to dense GraphData."""

    def test_single_graph(self) -> None:
        data = _make_triangle_graph()
        batch = Batch.from_data_list([data])
        gd = GraphData.from_pyg_batch(batch)
        assert gd.X.shape == (1, 3, 2)  # (bs, n, dx=2)
        assert gd.E.shape == (1, 3, 3, 2)  # (bs, n, n, de=2)
        assert gd.node_mask.shape == (1, 3)
        assert gd.node_mask.all()

    def test_batch_of_two(self) -> None:
        batch = Batch.from_data_list([_make_triangle_graph(), _make_square_graph()])
        gd = GraphData.from_pyg_batch(batch)
        assert gd.X.shape[0] == 2  # batch size
        assert gd.X.shape[1] == 4  # padded to max nodes (4)
        # Triangle (3 nodes) should have node_mask[0, 3] = False
        assert gd.node_mask[0, :3].all()
        assert not gd.node_mask[0, 3]
        # Square (4 nodes) should have all True
        assert gd.node_mask[1].all()

    def test_adjacency_round_trip(self) -> None:
        """Data → Batch → GraphData → adjacency matches original."""
        data = _make_triangle_graph()
        batch = Batch.from_data_list([data])
        gd = GraphData.from_pyg_batch(batch)
        adj = gd.to_adjacency()  # (1, 3, 3)
        # Triangle has edges 0-1, 1-2, 0-2
        assert adj[0, 0, 1] == 1.0
        assert adj[0, 1, 2] == 1.0
        assert adj[0, 0, 2] == 1.0
        # No self-loops
        assert adj[0, 0, 0] == 0.0

    def test_variable_size_padding(self) -> None:
        """Variable-size batch pads correctly."""
        g2 = Data(edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long), num_nodes=2)
        g5 = Data(
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], dtype=torch.long),
            num_nodes=5,
        )
        batch = Batch.from_data_list([g2, g5])
        gd = GraphData.from_pyg_batch(batch)
        assert gd.X.shape == (2, 5, 2)
        assert gd.node_mask[0, :2].all()
        assert not gd.node_mask[0, 2:].any()


class TestToPyg:
    """Verify to_pyg converts GraphData back to PyG Data."""

    def test_single_unbatched(self) -> None:
        adj = torch.zeros(4, 4)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[2, 3] = adj[3, 2] = 1.0
        gd = GraphData.from_adjacency(adj)
        data = gd.to_pyg()
        assert isinstance(data, Data)
        assert data.num_nodes == 4
        assert data.edge_index.shape[0] == 2

    def test_round_trip(self) -> None:
        """adjacency → GraphData → Data → Batch → GraphData → adjacency."""
        adj = torch.zeros(5, 5)
        adj[0, 1] = adj[1, 0] = 1.0
        adj[1, 2] = adj[2, 1] = 1.0
        gd = GraphData.from_adjacency(adj)
        data = gd.to_pyg()
        batch = Batch.from_data_list([data])
        gd2 = GraphData.from_pyg_batch(batch)
        adj2 = gd2.to_adjacency()
        assert torch.allclose(adj.unsqueeze(0), adj2)
```

- [ ] **Step 1b: Run tests to verify they fail**

```bash
uv run pytest tests/data_modules/test_pyg_conversion.py -x -v
```

- [ ] **Step 1c: Implement from_pyg_batch and to_pyg**

In `graph_types.py`, add after the existing `from_adjacency` classmethod:

```python
@classmethod
def from_pyg_batch(cls, batch: "Batch") -> GraphData:
    """Convert a PyG ``Batch`` of graphs to a dense ``GraphData`` batch.

    Uses ``to_dense_adj`` and constructs one-hot categorical features
    matching the format expected by all TMGG models.

    Parameters
    ----------
    batch
        A ``torch_geometric.data.Batch`` produced by
        ``Batch.from_data_list()``.

    Returns
    -------
    GraphData
        Dense batched representation with padded node/edge features.
    """
    from torch_geometric.utils import to_dense_adj

    bs = int(batch.num_graphs)
    # Dense adjacency: (bs, n_max, n_max)
    adj = to_dense_adj(batch.edge_index, batch.batch)
    n_max = adj.shape[1]

    # Node mask: which positions are real nodes
    node_counts = torch.bincount(batch.batch, minlength=bs)
    arange = torch.arange(n_max, device=adj.device).unsqueeze(0).expand(bs, -1)
    node_mask = arange < node_counts.unsqueeze(1)

    # Convert to one-hot categorical (same logic as from_adjacency)
    # Zero diagonal (no self-loops)
    diag = torch.arange(n_max, device=adj.device)
    adj[:, diag, diag] = 0.0
    # Symmetrise
    adj = (adj + adj.transpose(1, 2)).clamp(max=1.0)

    # Edge features: (bs, n, n, 2) one-hot [no-edge, edge]
    E = torch.stack([1.0 - adj, adj], dim=-1)

    # Node features: (bs, n, 2) one-hot [no-node, node]
    node_indicators = node_mask.float()
    X = torch.stack([1.0 - node_indicators, node_indicators], dim=-1)

    # Global features: empty
    y = torch.zeros(bs, 0, device=adj.device)

    return cls(X=X, E=E, y=y, node_mask=node_mask)

def to_pyg(self) -> "Data":
    """Convert this (unbatched) GraphData to a PyG ``Data`` object.

    Parameters
    ----------
    Returns
    -------
    torch_geometric.data.Data
        Sparse COO representation with ``edge_index`` and ``num_nodes``.

    Raises
    ------
    ValueError
        If the GraphData is batched (has a batch dimension).
    """
    from torch_geometric.data import Data
    from torch_geometric.utils import dense_to_sparse

    adj = self.to_adjacency()
    if adj.ndim == 3:
        if adj.shape[0] != 1:
            raise ValueError(
                f"to_pyg() requires unbatched GraphData or batch size 1, "
                f"got batch size {adj.shape[0]}"
            )
        adj = adj.squeeze(0)

    n = int(self.node_mask.sum().item()) if self.node_mask.ndim == 1 else adj.shape[0]
    adj_trimmed = adj[:n, :n]
    edge_index, _ = dense_to_sparse(adj_trimmed)
    return Data(edge_index=edge_index, num_nodes=n)
```

Add `TYPE_CHECKING` guard for PyG imports at top of file:

```python
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch_geometric.data import Batch, Data
```

- [ ] **Step 1d: Run tests**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/data/datasets/graph_types.py
uv run pytest tests/data_modules/test_pyg_conversion.py -x -v
```

- [ ] **Step 1e: Commit**

```bash
git add src/tmgg/data/datasets/graph_types.py tests/data_modules/test_pyg_conversion.py
git commit -m "feat: add GraphData.from_pyg_batch() and to_pyg() conversion methods

Bridge between PyG sparse (COO edge_index) and TMGG dense (one-hot
categorical) representations. from_pyg_batch converts a PyG Batch to
GraphData; to_pyg converts back for storage.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 2: Collate function + MultiGraphDataModule conversion

### Task 2: Replace _UnwrapDataset with PyG-native storage in MultiGraphDataModule

**Files:**
- Modify: `src/tmgg/data/data_modules/multigraph_data_module.py`
- Modify: `tests/data_modules/test_reference_graphs.py` (update if needed)

- [ ] **Step 2a: Write a helper to convert numpy adjacencies to list[Data]**

In `multigraph_data_module.py`, add a module-level function after imports:

```python
def _adjacencies_to_pyg(adjs: np.ndarray) -> list[Data]:
    """Convert numpy adjacency matrices to PyG Data objects.

    Parameters
    ----------
    adjs
        Adjacency matrices, shape ``(N, n, n)``.

    Returns
    -------
    list[Data]
        One ``Data`` per graph with COO ``edge_index``.
    """
    from torch_geometric.data import Data
    from torch_geometric.utils import dense_to_sparse

    result: list[Data] = []
    for i in range(len(adjs)):
        adj_t = torch.from_numpy(adjs[i]).float()
        edge_index, _ = dense_to_sparse(adj_t)
        result.append(Data(edge_index=edge_index, num_nodes=adj_t.shape[0]))
    return result
```

- [ ] **Step 2b: Define the collate function**

Add after `_adjacencies_to_pyg`:

```python
def _collate_pyg_to_graphdata(data_list: list[Data]) -> GraphData:
    """Collate PyG Data objects into a dense GraphData batch."""
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(data_list)
    return GraphData.from_pyg_batch(batch)
```

- [ ] **Step 2c: Change storage types and setup()**

Change the type annotations (lines 118–120):
```python
self._train_data: list[Data] | None = None
self._val_data: list[Data] | None = None
self._test_data: list[Data] | None = None
```

Add `from torch_geometric.data import Data` to imports (use `TYPE_CHECKING` guard for pyright).

Change `setup()` (lines 323–325) from:
```python
self._train_data = torch.from_numpy(train).float()
self._val_data = torch.from_numpy(val).float()
self._test_data = torch.from_numpy(test).float()
```
To:
```python
self._train_data = _adjacencies_to_pyg(train)
self._val_data = _adjacencies_to_pyg(val)
self._test_data = _adjacencies_to_pyg(test)
```

- [ ] **Step 2d: Update dataloaders**

Change all three dataloaders from `_UnwrapDataset(self._xxx_data)` to a simple list dataset. Use PyTorch's built-in list-indexing:

```python
def train_dataloader(self) -> DataLoader[GraphData]:
    if self._train_data is None:
        raise RuntimeError("DataModule not setup. Call setup() first.")
    return self._make_dataloader(
        self._train_data, shuffle=True, collate_fn=_collate_pyg_to_graphdata
    )
```

Same for `val_dataloader` and `test_dataloader` (with `shuffle=False`).

Note: `_make_dataloader` expects a `Dataset`, but `list[Data]` isn't a `Dataset`. We need a thin wrapper or use `torch.utils.data.default_collate`. Simplest: keep a minimal wrapper:

```python
class _ListDataset(Dataset[Data]):
    """Thin wrapper making a list indexable as a Dataset."""

    def __init__(self, data: list[Data]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Data:
        return self._data[idx]
```

Use `_ListDataset(self._train_data)` instead of `_UnwrapDataset(self._train_data)`.

- [ ] **Step 2e: Delete _UnwrapDataset**

Remove the class definition (lines 38–48).

- [ ] **Step 2f: Update get_reference_graphs() in base_data_module.py**

The current implementation iterates the dataloader and calls `batch.to_adjacency()`. Since the dataloaders still yield `GraphData` batches (the collate function handles conversion), `get_reference_graphs()` doesn't need changes. Verify this by running:

```bash
uv run pytest tests/data_modules/test_reference_graphs.py -x -v
```

- [ ] **Step 2g: Run full test suite**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/data/data_modules/multigraph_data_module.py
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py --ignore=tests/experiment_utils/test_eigenstructure_study.py -m "not slow and not modal" -q
```

- [ ] **Step 2h: Commit**

```bash
git add src/tmgg/data/data_modules/multigraph_data_module.py src/tmgg/data/data_modules/base_data_module.py
git commit -m "refactor(G40): convert MultiGraphDataModule to PyG Data storage

Replace _UnwrapDataset + batched tensor storage with list[Data] +
_collate_pyg_to_graphdata. Graphs stored as COO edge_index internally,
converted to dense GraphData at batch boundary.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 3: GraphDataModule simplification

### Task 3: Simplify GraphDataModule to thin subclass

**Files:**
- Modify: `src/tmgg/data/data_modules/data_module.py`
- Modify: `src/tmgg/data/datasets/graph_dataset.py` (delete or simplify)

- [ ] **Step 3a: Change storage to list[Data]**

Replace `val_adjacency_matrices: list[torch.Tensor] | None` and siblings with `list[Data] | None` storage via the parent class's `_train_data`/`_val_data`/`_test_data`.

In `setup()`:
- For synthetic graphs: call `super().setup(stage)` (parent handles generation → `list[Data]`).
- For PyG datasets: call `_setup_pyg_dataset()` which should now store `list[Data]` directly from `PyGDatasetWrapper`.

- [ ] **Step 3b: Handle samples_per_graph**

`samples_per_graph` repeats each graph N times in the training set. With `list[Data]` storage, this becomes:

```python
if self.samples_per_graph > 1:
    self._train_data = self._train_data * self.samples_per_graph
```

This replaces the `GraphDataset` class entirely — list repetition achieves the same sampling effect.

- [ ] **Step 3c: Delete GraphDataset usage**

Remove `_create_dataloader` method. The parent's dataloaders (using `_ListDataset` + `_collate_pyg_to_graphdata`) handle everything.

- [ ] **Step 3d: Update _setup_pyg_dataset()**

`PyGDatasetWrapper` currently returns numpy arrays. After Task 4 (below) it will return `list[Data]` directly. For now, convert numpy → `list[Data]` using `_adjacencies_to_pyg()` from the parent module.

- [ ] **Step 3e: Run tests**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/data/data_modules/data_module.py
uv run pytest tests/experiment_utils/test_data_module.py tests/experiment_utils/test_dataset_wrappers.py -x -v
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py --ignore=tests/experiment_utils/test_eigenstructure_study.py -m "not slow and not modal" -q
```

- [ ] **Step 3f: Commit**

```bash
git add src/tmgg/data/data_modules/data_module.py src/tmgg/data/datasets/graph_dataset.py
git commit -m "refactor(G40): simplify GraphDataModule to thin subclass

Delete val_adjacency_matrices, _create_dataloader. samples_per_graph
implemented via list repetition. Delegates to parent for PyG storage
and dataloaders.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 4: SyntheticCategoricalDataModule + PyGDatasetWrapper

### Task 4: Convert SyntheticCategoricalDataModule to list[Data] storage

**Files:**
- Modify: `src/tmgg/experiments/discrete_diffusion_generative/datamodule.py`

- [ ] **Step 4a: Change storage and delete _GraphDataDataset**

Replace `_train_data: list[GraphData] | None` with `list[Data] | None` (using parent's type).

In `setup()`, replace:
```python
self._train_data = [GraphData.from_adjacency(train_adj[i]) for i in range(len(train_adj))]
```
With:
```python
self._train_data = _adjacencies_to_pyg(train_np)
```

Import `_adjacencies_to_pyg` from `multigraph_data_module`.

Delete `_GraphDataDataset` class.

- [ ] **Step 4b: Update marginal computation**

The marginal computation currently iterates `list[GraphData]` and reads `g.X` and `g.E`. After the change, `_train_data` is `list[Data]` (COO format). The marginals need dense one-hot features.

Convert to dense temporarily for marginal computation:

```python
def _compute_marginals(self) -> None:
    if self._train_data is None:
        return

    from torch_geometric.data import Batch

    # Convert all training graphs to dense for marginal computation
    batch = Batch.from_data_list(self._train_data)
    dense = GraphData.from_pyg_batch(batch)

    # Node marginals: count per-class frequencies
    node_counts = dense.X[dense.node_mask.bool()].sum(dim=0)
    self._node_marginals = node_counts / node_counts.sum()

    # Edge marginals: upper triangle only, excluding diagonal
    bs, n, _, de = dense.E.shape
    mask_2d = dense.node_mask.unsqueeze(-1) & dense.node_mask.unsqueeze(-2)
    triu_mask = torch.triu(torch.ones(n, n, dtype=torch.bool), diagonal=1)
    full_mask = mask_2d & triu_mask.unsqueeze(0)
    edge_counts = dense.E[full_mask].sum(dim=0)
    self._edge_marginals = edge_counts / edge_counts.sum()
```

- [ ] **Step 4c: Update dataloaders to use parent's pattern**

Use `_ListDataset` + `_collate_pyg_to_graphdata` from the parent, or override to call parent's dataloader methods directly.

- [ ] **Step 4d: Run tests**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/experiments/discrete_diffusion_generative/datamodule.py
uv run pytest tests/experiments/test_categorical_datamodule.py -x -v
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py --ignore=tests/experiment_utils/test_eigenstructure_study.py -m "not slow and not modal" -q
```

- [ ] **Step 4e: Commit**

```bash
git add src/tmgg/experiments/discrete_diffusion_generative/datamodule.py
git commit -m "refactor(G40): convert SyntheticCategoricalDataModule to PyG Data storage

Delete _GraphDataDataset. Store list[Data] internally, compute
marginals by temporarily converting to dense. Dataloaders use
shared collate function.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 5: Update PyGDatasetWrapper to return list[Data]

**Files:**
- Modify: `src/tmgg/data/datasets/pyg_datasets.py`

- [ ] **Step 5a: Change PyGDatasetWrapper to preserve PyG Data objects**

Instead of eagerly converting to dense numpy arrays, keep the original `Data` objects from PyG datasets. Add a `data_list` attribute:

```python
self.data_list: list[Data] = [dataset[i] for i in range(len(dataset))]
```

Keep the existing `adjacencies` and `num_nodes` attributes for backward compatibility (some code may still use them), but make them lazy or computed from `data_list`.

- [ ] **Step 5b: Update GraphDataModule._setup_pyg_dataset() to use data_list**

In `data_module.py`, change `_setup_pyg_dataset()` to read `wrapper.data_list` directly instead of converting `wrapper.adjacencies` through numpy.

- [ ] **Step 5c: Run tests**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/data/datasets/pyg_datasets.py src/tmgg/data/data_modules/data_module.py
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py --ignore=tests/experiment_utils/test_eigenstructure_study.py -m "not slow and not modal" -q
```

- [ ] **Step 5d: Commit**

```bash
git add src/tmgg/data/datasets/pyg_datasets.py src/tmgg/data/data_modules/data_module.py
git commit -m "refactor(G40): PyGDatasetWrapper preserves native Data objects

Keep PyG Data list alongside existing adjacency arrays. GraphDataModule
reads from data_list directly, avoiding eager densification.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Chunk 5: SingleGraphDataModule + cleanup

### Task 6: Convert SingleGraphDataModule to Data storage

**Files:**
- Modify: `src/tmgg/data/data_modules/single_graph_data_module.py`

- [ ] **Step 6a: Change storage from numpy to Data**

Replace `train_graph: npt.NDArray | None` with `train_data: Data | None`. In `setup()`, convert the generated numpy adjacency to a `Data` object.

The `SingleGraphDataset` can be simplified to repeat a single `Data` object N times.

- [ ] **Step 6b: Update dataloaders**

Use `_ListDataset([self.train_data] * self.num_train_samples)` with `_collate_pyg_to_graphdata`.

- [ ] **Step 6c: Run tests**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/data/data_modules/single_graph_data_module.py
uv run pytest tests/test_single_graph_datasets.py -x -v
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py --ignore=tests/experiment_utils/test_eigenstructure_study.py -m "not slow and not modal" -q
```

- [ ] **Step 6d: Commit**

```bash
git add src/tmgg/data/data_modules/single_graph_data_module.py
git commit -m "refactor(G40): convert SingleGraphDataModule to Data storage

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

### Task 7: Final cleanup and mark G40 FIXED

- [ ] **Step 7a: Delete unused GraphDataset if no longer needed**

Check if `GraphDataset` in `graph_dataset.py` is still imported anywhere. If not, delete it.

- [ ] **Step 7b: Run full test suite**

```bash
uv run basedpyright --project pyproject.toml src/tmgg/data/
uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py --ignore=tests/experiment_utils/test_eigenstructure_study.py -m "not slow and not modal" -v
```

- [ ] **Step 7c: Mark G40 FIXED in SUMMARY.md**

- [ ] **Step 7d: Commit**

```bash
git add -u
git commit -m "refactor(G40): complete PyG datamodule unification

All datamodules now store graphs as list[torch_geometric.data.Data].
Dense GraphData conversion happens at the batch boundary via a shared
collate function. Eliminated _UnwrapDataset, _GraphDataDataset, and
triple storage format.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>"
```

---

## Verification

After all tasks:

1. `uv run pytest tests/ -x --ignore=tests/modal/test_eigenstructure_modal.py -m "not slow and not modal" -v` — all tests pass
2. `uv run basedpyright --project pyproject.toml src/tmgg/data/` — 0 errors
3. `uv run tach check` — module boundaries valid
4. `uv run tmgg-spectral-arch --cfg job` — config composes
5. Verify no remaining references to `_UnwrapDataset`, `_GraphDataDataset`, or `val_adjacency_matrices` in `src/`
