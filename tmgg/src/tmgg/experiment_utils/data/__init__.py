"""Public API for data utilities."""

from .data_module import GraphDataModule
from .dataset import AdjacencyMatrixDataset, GraphDataset, PermutedAdjacencyDataset
from .dataset_wrappers import (
    ANUDatasetWrapper,
    ClassicalGraphsWrapper,
    NXGraphWrapperWrapper,
)
from .noise import (
    add_digress_noise,
    add_gaussian_noise,
    add_rotation_noise,
    random_skew_symmetric_matrix,
)
from .noise_generators import (
    NoiseGenerator,
    GaussianNoiseGenerator,
    DigressNoiseGenerator,
    RotationNoiseGenerator,
    create_noise_generator,
)
from .sbm import generate_block_sizes, generate_sbm_adjacency
from .eigendecomposition import (
    compute_eigendecomposition,
    compute_top_k_eigendecomposition,
    compute_spectral_distance,
)

__all__ = [
    # Data Module
    "GraphDataModule",
    # Datasets
    "GraphDataset",
    "AdjacencyMatrixDataset",
    "PermutedAdjacencyDataset",
    # Dataset Wrappers
    "ANUDatasetWrapper",
    "ClassicalGraphsWrapper",
    "NXGraphWrapperWrapper",
    # Noise functions
    "add_digress_noise",
    "add_gaussian_noise",
    "add_rotation_noise",
    "random_skew_symmetric_matrix",
    # Noise generators
    "NoiseGenerator",
    "GaussianNoiseGenerator",
    "DigressNoiseGenerator",
    "RotationNoiseGenerator",
    "create_noise_generator",
    # SBM functions
    "generate_block_sizes",
    "generate_sbm_adjacency",
    # Eigendecomposition functions
    "compute_eigendecomposition",
    "compute_top_k_eigendecomposition",
    "compute_spectral_distance",
]
