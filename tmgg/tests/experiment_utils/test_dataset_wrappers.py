"""Tests for GraphDataset.

GraphDataset wraps adjacency matrices into a PyTorch Dataset that produces
GraphData instances with optional permutation augmentation.

Testing Strategy:
- Verify GraphDataset produces correct output types and shapes
- Test permutation behavior and round-robin matrix selection
- Test numpy/tensor input acceptance
"""

from __future__ import annotations

import numpy as np
import torch

from tmgg.data.datasets.graph_dataset import GraphDataset
from tmgg.data.datasets.graph_types import GraphData


class TestGraphDataset:
    """Tests for GraphDataset class."""

    def test_single_matrix(self) -> None:
        """Single adjacency matrix produces GraphData items of correct shape."""
        A = torch.eye(5, dtype=torch.float32)
        ds = GraphDataset(A, num_samples=10)
        assert len(ds) == 10
        item = ds[0]
        assert isinstance(item, GraphData)
        assert item.to_adjacency().shape == (5, 5)

    def test_multiple_matrices(self) -> None:
        """Multiple matrices are sampled from without error."""
        A1 = torch.eye(5, dtype=torch.float32)
        A2 = torch.ones(5, 5, dtype=torch.float32)
        ds = GraphDataset([A1, A2], num_samples=20)
        assert len(ds) == 20

    def test_numpy_input(self) -> None:
        """Numpy arrays are accepted and converted transparently."""
        A = np.eye(4, dtype=np.float32)
        ds = GraphDataset(A, num_samples=5)
        item = ds[0]
        assert isinstance(item, GraphData)

    def test_return_original_idx(self) -> None:
        """With return_original_idx=True, dataset yields (GraphData, int)."""
        A1 = torch.eye(3, dtype=torch.float32)
        A2 = torch.eye(3, dtype=torch.float32) * 2
        ds = GraphDataset([A1, A2], num_samples=10, return_original_idx=True)
        result = ds[0]
        assert isinstance(result, tuple)
        graph_data, idx = result
        assert isinstance(graph_data, GraphData)
        assert isinstance(idx, int)
        assert idx in (0, 1)

    def test_no_permutation(self) -> None:
        """With apply_permutation=False, output matches input exactly."""
        A = torch.tensor([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=torch.float32)
        ds = GraphDataset(A, num_samples=5, apply_permutation=False)
        result = ds[0]
        assert isinstance(result, GraphData)
        assert torch.allclose(result.to_adjacency(), A)
