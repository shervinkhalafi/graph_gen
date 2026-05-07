"""Graph dataset definitions, types, and generation utilities."""

from __future__ import annotations

from .graph_types import GraphData, collapse_to_indices
from .pyg_datasets import PyGDatasetWrapper
from .sbm import generate_block_sizes, generate_sbm_adjacency, generate_sbm_batch
from .synthetic_graphs import SyntheticGraphDataset

__all__ = [
    "GraphData",
    "collapse_to_indices",
    "SyntheticGraphDataset",
    "PyGDatasetWrapper",
    "generate_block_sizes",
    "generate_sbm_adjacency",
    "generate_sbm_batch",
]
