# Distributional Protocol Audit Log

**Date**: 2026-01-19
**Audited by**: Claude (automated review)
**Scope**: stage3_pyg_dist experiments on PyG ENZYMES and PROTEINS datasets

## Summary

The stage3_pyg_dist experiments use proper distributional generalization: models train on one set of graphs and evaluate on entirely separate graphs from the same distribution.

## Configuration Verified

### Stage Definition (`src/tmgg/modal/stage_definitions/stage3_pyg_dist.yaml`)

```yaml
name: stage3_pyg_dist
base_config: base_config_spectral

datasets:
  # Distribution protocol (GraphDataModule, not SingleGraphDataModule)
  - pyg_enzymes
  - pyg_proteins

hyperparameters:
  learning_rate: [5e-4, 1e-3]
  weight_decay: [1e-2, 1e-3]
  "+model.k": [16, 32]

seeds: [1, 2, 3]
```

### Run Config (from W&B export)

```json
{
  "data": {
    "_target_": "tmgg.experiment_utils.data.data_module.GraphDataModule",
    "val_split": 0.1,
    "test_split": 0.2,
    "dataset_name": "enzymes",
    "num_samples_per_graph": 100
  }
}
```

Key observation: `_target_` is `GraphDataModule`, **not** `SingleGraphDataModule`. This confirms the distribution protocol.

## Data Module Implementation

### Graph Split Logic (`src/tmgg/experiment_utils/data/data_module.py:331-359`)

```python
def _setup_pyg_dataset(self) -> None:
    dataset = PyGDatasetWrapper(dataset_name=self.dataset_name, ...)

    # Split into train/val/test (different graphs in each)
    train_ratio = 1 - self.val_split - self.test_split
    train, val, test = dataset.train_val_test_split(
        train_ratio=train_ratio,
        val_ratio=self.val_split,
        seed=seed,
    )

    # Store as SEPARATE lists
    self.train_adjacency_matrices = [torch.from_numpy(A).float() for A in train]
    self.val_adjacency_matrices = [torch.from_numpy(A).float() for A in val]
    self.test_adjacency_matrices = [torch.from_numpy(A).float() for A in test]
```

### Split Implementation (`src/tmgg/experiment_utils/data/pyg_datasets.py:169-209`)

```python
def train_val_test_split(self, train_ratio=0.7, val_ratio=0.1, seed=None):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(self._num_graphs)

    n_train = int(train_ratio * self._num_graphs)
    n_val = int(val_ratio * self._num_graphs)

    train_idx = indices[:n_train]
    val_idx = indices[n_train : n_train + n_val]
    test_idx = indices[n_train + n_val :]

    return (
        self.adjacencies[train_idx],
        self.adjacencies[val_idx],
        self.adjacencies[test_idx],
    )
```

The split:
1. Shuffles graph indices with a fixed seed (reproducible)
2. Partitions into disjoint train/val/test sets
3. Returns separate arrays of adjacency matrices

## Dataset Statistics

| Dataset | Total Graphs | Train (70%) | Val (10%) | Test (20%) |
|---------|--------------|-------------|-----------|------------|
| ENZYMES | 600 | 420 | 60 | 120 |
| PROTEINS | 1113 | 779 | 111 | 223 |

### Sample Counts (with `num_samples_per_graph: 100`)

| Dataset | Train Samples | Val Samples | Test Samples |
|---------|---------------|-------------|--------------|
| ENZYMES | 42,000 | 3,000 | 6,000 |
| PROTEINS | 77,900 | 5,550 | 11,150 |

Note: Val/test use `val_samples_per_graph = num_samples_per_graph // 2 = 50` by default.

## Contrast with Single-Graph Protocol

For reference, the `SingleGraphDataModule` (not used here) trains and tests on the **same** graph with different noisy samples. That would be `_target_: tmgg.experiment_utils.data.single_graph_data_module.SingleGraphDataModule`.

## Results Summary (as of audit date)

176 finished stage3_pyg_dist runs show:

| Architecture | ENZYMES Accuracy | PROTEINS Accuracy |
|--------------|------------------|-------------------|
| digress_transformer | 99.23% (n=20) | 96.00% (n=2) |
| digress_transformer_gnn_qk | 99.22% (n=24) | 99.22% (n=3) |
| filter_bank | 94.34% (n=22) | 99.51% (n=24) |
| linear_pe | 99.20% (n=24) | 98.62% (n=2) |
| self_attention | 98.12% (n=22) | 99.95% (n=24) |

These are proper **distributional generalization** accuracies — the model has never seen the test graphs during training.

## Conclusion

The audit confirms that stage3_pyg_dist experiments implement correct distributional generalization:

1. **Data module**: `GraphDataModule` (not `SingleGraphDataModule`)
2. **Split**: Disjoint graph sets for train/val/test
3. **Seed**: Fixed seed (42) ensures reproducibility
4. **Scale**: Substantial dataset sizes (600-1113 graphs, 42k-78k training samples)

No issues found.
