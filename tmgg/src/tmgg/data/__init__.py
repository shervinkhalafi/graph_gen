"""Graph data modules and datasets."""

from __future__ import annotations

from .data_modules import (
    BaseGraphDataModule,
    GraphDataModule,
    MultiGraphDataModule,
    SingleGraphDataModule,
    SyntheticCategoricalDataModule,
)
from .datasets import (
    GraphData,
    PyGDatasetWrapper,
    SyntheticGraphDataset,
    collapse_to_indices,
    generate_block_sizes,
    generate_sbm_adjacency,
    generate_sbm_batch,
)

__all__ = [
    # Graph types
    "GraphData",
    "collapse_to_indices",
    # Data Modules
    "BaseGraphDataModule",
    "MultiGraphDataModule",
    "GraphDataModule",
    "SingleGraphDataModule",
    "SyntheticCategoricalDataModule",
    # Datasets
    "SyntheticGraphDataset",
    "PyGDatasetWrapper",
    # SBM functions
    "generate_block_sizes",
    "generate_sbm_adjacency",
    "generate_sbm_batch",
]
