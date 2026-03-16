"""Graph data modules, datasets, noise, and generation utilities."""

from __future__ import annotations

from .data_modules import (
    BaseGraphDataModule,
    GraphDataModule,
    MultiGraphDataModule,
    SingleGraphDataModule,
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
from .noising import (
    DigressNoiseGenerator,
    EdgeFlipNoiseGenerator,
    GaussianNoiseGenerator,
    LogitNoiseGenerator,
    NoiseGenerator,
    RotationNoiseGenerator,
    SizeDistribution,
    add_digress_noise,
    add_edge_flip_noise,
    add_gaussian_noise,
    add_logit_noise,
    add_rotation_noise,
    create_noise_generator,
    random_skew_symmetric_matrix,
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
    # Datasets
    "SyntheticGraphDataset",
    "PyGDatasetWrapper",
    # Noise functions
    "add_digress_noise",
    "add_edge_flip_noise",
    "add_gaussian_noise",
    "add_logit_noise",
    "add_rotation_noise",
    "random_skew_symmetric_matrix",
    # Noise generators
    "NoiseGenerator",
    "DigressNoiseGenerator",
    "GaussianNoiseGenerator",
    "EdgeFlipNoiseGenerator",
    "LogitNoiseGenerator",
    "RotationNoiseGenerator",
    "create_noise_generator",
    # Size distribution
    "SizeDistribution",
    # SBM functions
    "generate_block_sizes",
    "generate_sbm_adjacency",
    "generate_sbm_batch",
]
