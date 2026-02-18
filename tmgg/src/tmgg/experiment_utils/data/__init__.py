"""Public API for data utilities."""

from .base_data_module import BaseGraphDataModule
from .data_module import GraphDataModule
from .dataset import AdjacencyMatrixDataset, GraphDataset, PermutedAdjacencyDataset
from .dataset_wrappers import (
    ANUDatasetWrapper,
    ClassicalGraphsWrapper,
    NXGraphWrapperWrapper,
    create_dataset_wrapper,
)
from .eigendecomposition import (
    compute_eigendecomposition,
    compute_spectral_distance,
    compute_top_k_eigendecomposition,
)
from .multigraph_data_module import MultiGraphDataModule
from .noise import (
    add_digress_noise,
    add_edge_flip_noise,
    add_gaussian_noise,
    add_logit_noise,
    add_rotation_noise,
    random_skew_symmetric_matrix,
)
from .noise_generators import (
    DigressNoiseGenerator,
    EdgeFlipNoiseGenerator,
    GaussianNoiseGenerator,
    LogitNoiseGenerator,
    NoiseGenerator,
    RotationNoiseGenerator,
    create_noise_generator,
)
from .sbm import generate_block_sizes, generate_sbm_adjacency, generate_sbm_batch
from .single_graph_data_module import SingleGraphDataModule

__all__ = [
    # Data Modules
    "BaseGraphDataModule",
    "MultiGraphDataModule",
    "GraphDataModule",
    "SingleGraphDataModule",
    # Datasets
    "GraphDataset",
    "AdjacencyMatrixDataset",
    "PermutedAdjacencyDataset",
    # Dataset Wrappers
    "ANUDatasetWrapper",
    "ClassicalGraphsWrapper",
    "NXGraphWrapperWrapper",
    "create_dataset_wrapper",
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
    # SBM functions
    "generate_block_sizes",
    "generate_sbm_adjacency",
    "generate_sbm_batch",
    # Eigendecomposition functions
    "compute_eigendecomposition",
    "compute_top_k_eigendecomposition",
    "compute_spectral_distance",
]
