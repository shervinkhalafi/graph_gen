"""Data generation and manipulation utilities for graph denoising experiments."""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.linalg import expm
from typing import List, Tuple, Optional, Union
import random


class GraphDataset(Dataset):
    """
    Unified dataset for graph adjacency matrices with permutation support.

    This dataset can handle both single and multiple adjacency matrices,
    and optionally applies random permutations to increase data diversity.
    """

    def __init__(
        self,
        adjacency_matrices: Union[
            np.ndarray, torch.Tensor, List[Union[np.ndarray, torch.Tensor]]
        ],
        num_samples: int,
        apply_permutation: bool = True,
        return_original_idx: bool = False,
    ):
        """
        Initialize graph dataset.

        Args:
            adjacency_matrices: Single adjacency matrix or list of matrices
            num_samples: Number of samples to generate
            apply_permutation: Whether to apply random permutations
            return_original_idx: Whether to return the index of the original matrix
        """
        # Convert to list if single matrix provided
        if not isinstance(adjacency_matrices, list):
            adjacency_matrices = [adjacency_matrices]

        # Convert all matrices to torch tensors
        self.adjacency_matrices = []
        for mat in adjacency_matrices:
            if isinstance(mat, np.ndarray):
                mat = torch.tensor(mat, dtype=torch.float32)
            elif not isinstance(mat, torch.Tensor):
                raise ValueError(f"Unsupported matrix type: {type(mat)}")
            self.adjacency_matrices.append(mat)

        self.num_samples = num_samples
        self.apply_permutation = apply_permutation
        self.return_original_idx = return_original_idx
        self.num_matrices = len(self.adjacency_matrices)

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
        """
        Get a sample from the dataset.

        Returns:
            If return_original_idx is False: permuted adjacency matrix
            If return_original_idx is True: (permuted adjacency matrix, original matrix index)
        """
        # Choose a matrix (round-robin if fewer samples than matrices)
        if self.num_matrices == 1:
            matrix_idx = 0
        else:
            matrix_idx = torch.randint(0, self.num_matrices, (1,)).item()

        adjacency_matrix = self.adjacency_matrices[matrix_idx]

        # Apply permutation if requested
        if self.apply_permutation:
            permuted_indices = torch.randperm(adjacency_matrix.size(0))
            adjacency_matrix = adjacency_matrix[permuted_indices, :][
                :, permuted_indices
            ]

        if self.return_original_idx:
            return adjacency_matrix, matrix_idx
        else:
            return adjacency_matrix


# Backward compatibility aliases
AdjacencyMatrixDataset = GraphDataset
PermutedAdjacencyDataset = GraphDataset

