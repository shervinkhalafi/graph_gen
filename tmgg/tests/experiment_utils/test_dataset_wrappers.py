"""Tests for dataset wrappers module.

This module tests the GraphCollection class and create_dataset_wrapper factory
function, which provide a unified interface for various graph dataset types.

Testing Strategy:
- Use mock datasets to avoid external dependencies
- Test padding behavior with variable-sized graphs
- Test type conversion from numpy/tensor to float tensors
- Test factory function with valid and invalid types

Key Invariants:
- get_adjacency_matrices returns list of torch.Tensor with dtype float
- All returned matrices have the same size (padded to max)
- Empty dataset returns empty list
- Unknown dataset type raises ValueError
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch


class MockDataset:
    """Mock dataset for testing GraphCollection behavior."""

    def __init__(self, graphs: list[tuple[Any, np.ndarray | torch.Tensor]]):
        """Initialize with list of (features, adjacency) tuples."""
        self.graphs = graphs

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, idx: int) -> tuple[Any, np.ndarray | torch.Tensor]:
        return self.graphs[idx]


class TestGraphCollection:
    """Tests for GraphCollection class."""

    def test_empty_dataset_returns_empty_list(self) -> None:
        """Empty dataset should return empty list without error.

        Rationale: Edge case handling - the code should handle empty datasets
        gracefully without crashing or returning invalid data.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import GraphCollection

        empty_dataset = MockDataset([])
        wrapper = GraphCollection(empty_dataset)
        result = wrapper.get_adjacency_matrices()

        assert result == []
        assert isinstance(result, list)

    def test_single_graph_no_padding(self) -> None:
        """Single graph shouldn't require padding.

        Rationale: When there's only one graph, no padding is needed since
        max_size equals the graph size.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import GraphCollection

        A = np.eye(5)
        dataset = MockDataset([(None, A)])
        wrapper = GraphCollection(dataset)
        result = wrapper.get_adjacency_matrices()

        assert len(result) == 1
        assert result[0].shape == (5, 5)
        assert result[0].dtype == torch.float

    def test_padding_to_max_size(self) -> None:
        """Variable-sized graphs should be padded to max size.

        Rationale: The wrapper must produce uniformly-sized matrices for
        batch processing. Smaller graphs should be padded with zeros.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import GraphCollection

        A1 = np.eye(3)  # 3x3
        A2 = np.eye(5)  # 5x5
        A3 = np.eye(4)  # 4x4
        dataset = MockDataset([(None, A1), (None, A2), (None, A3)])
        wrapper = GraphCollection(dataset)
        result = wrapper.get_adjacency_matrices()

        assert len(result) == 3
        # All should be padded to 5x5 (max size)
        for mat in result:
            assert mat.shape == (5, 5)
            assert mat.dtype == torch.float

        # Check padding is zeros - A1 was 3x3, so positions [3:5, :] and [:, 3:5] should be 0
        assert torch.all(result[0][3:, :] == 0)
        assert torch.all(result[0][:, 3:] == 0)

        # A2 was already 5x5, should be unchanged (identity matrix)
        assert torch.allclose(result[1], torch.eye(5))

    def test_dtype_conversion_numpy(self) -> None:
        """Numpy arrays should be converted to torch float.

        Rationale: The wrapper must ensure consistent dtype for downstream
        processing regardless of input dtype.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import GraphCollection

        # Test various numpy dtypes
        A_int = np.eye(4, dtype=np.int32)
        A_float64 = np.eye(4, dtype=np.float64)
        A_bool = np.eye(4, dtype=bool)

        dataset = MockDataset(
            [
                (None, A_int),
                (None, A_float64),
                (None, A_bool),
            ]
        )
        wrapper = GraphCollection(dataset)
        result = wrapper.get_adjacency_matrices()

        for mat in result:
            assert mat.dtype == torch.float
            assert isinstance(mat, torch.Tensor)

    def test_dtype_conversion_tensor(self) -> None:
        """Torch tensors should be converted to float.

        Rationale: Input tensors may have various dtypes; the wrapper
        ensures they are all float for consistent processing.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import GraphCollection

        A_int = torch.eye(4, dtype=torch.int32)
        A_float64 = torch.eye(4, dtype=torch.float64)
        A_float32 = torch.eye(4, dtype=torch.float32)

        dataset = MockDataset(
            [
                (None, A_int),
                (None, A_float64),
                (None, A_float32),
            ]
        )
        wrapper = GraphCollection(dataset)
        result = wrapper.get_adjacency_matrices()

        for mat in result:
            assert mat.dtype == torch.float
            assert isinstance(mat, torch.Tensor)

    def test_unsupported_type_raises(self) -> None:
        """Unsupported adjacency matrix type should raise TypeError on init.

        Rationale: Early validation catches type errors at construction time
        rather than later during iteration, making debugging easier.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import GraphCollection

        # Create a mock dataset that returns an unsupported type (string)
        class BadDataset:
            def __len__(self) -> int:
                return 1

            def __getitem__(self, idx: int) -> tuple[None, str]:
                return (None, "not_an_array")  # String is not array/tensor

        # Should raise on construction (early validation)
        with pytest.raises(TypeError, match="Unsupported adjacency matrix type"):
            GraphCollection(BadDataset())

    def test_preserves_graph_structure(self) -> None:
        """Padding should preserve original graph structure.

        Rationale: The original adjacency values should remain unchanged;
        only padding areas should be zeros.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import GraphCollection

        # Create a specific graph structure (path graph P3)
        A = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=np.float32,
        )

        # Add another larger graph to force padding
        A_large = np.eye(5)

        dataset = MockDataset([(None, A), (None, A_large)])
        wrapper = GraphCollection(dataset)
        result = wrapper.get_adjacency_matrices()

        # Check the padded P3 graph - original 3x3 portion should be unchanged
        expected = torch.tensor(
            [
                [0, 1, 0, 0, 0],
                [1, 0, 1, 0, 0],
                [0, 1, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            dtype=torch.float,
        )
        assert torch.allclose(result[0], expected)


class TestCreateDatasetWrapper:
    """Tests for create_dataset_wrapper factory function."""

    def test_invalid_type_raises(self) -> None:
        """Invalid dataset_type should raise ValueError.

        Rationale: Clear error message with available options helps users
        fix typos or discover supported datasets.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import create_dataset_wrapper

        with pytest.raises(ValueError, match="Unknown dataset type"):
            create_dataset_wrapper("nonexistent_type")

    def test_error_lists_available_types(self) -> None:
        """Error message should list available dataset types.

        Rationale: Users should know what options are available without
        reading documentation.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import create_dataset_wrapper

        with pytest.raises(ValueError) as excinfo:
            create_dataset_wrapper("invalid")

        error_msg = str(excinfo.value)
        assert "anu" in error_msg
        assert "classical" in error_msg
        assert "nx" in error_msg

    @patch("tmgg.experiment_utils.data.dataset_wrappers.ANUDataset")
    def test_anu_wrapper_creation(self, mock_anu: MagicMock) -> None:
        """create_dataset_wrapper('anu') should create ANUDataset wrapper.

        Rationale: Factory should correctly instantiate the underlying dataset
        with provided kwargs.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import (
            GraphCollection,
            create_dataset_wrapper,
        )

        mock_anu.return_value = MagicMock()

        wrapper = create_dataset_wrapper("anu", graph_type="grid", k=5)

        mock_anu.assert_called_once_with(graph_type="grid", k=5)
        assert isinstance(wrapper, GraphCollection)

    @patch("tmgg.experiment_utils.data.dataset_wrappers.CassicalGraphs")
    def test_classical_wrapper_creation(self, mock_classical: MagicMock) -> None:
        """create_dataset_wrapper('classical') should create ClassicalGraphs wrapper.

        Rationale: Factory should correctly instantiate classical graphs dataset.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import (
            GraphCollection,
            create_dataset_wrapper,
        )

        mock_classical.return_value = MagicMock()

        wrapper = create_dataset_wrapper("classical", k=10)

        mock_classical.assert_called_once_with(k=10)
        assert isinstance(wrapper, GraphCollection)

    @patch("tmgg.experiment_utils.data.dataset_wrappers.NXGraphWrapper")
    def test_nx_wrapper_creation(self, mock_nx: MagicMock) -> None:
        """create_dataset_wrapper('nx') should create NXGraphWrapper wrapper.

        Rationale: Factory should correctly instantiate NetworkX graph wrapper.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import (
            GraphCollection,
            create_dataset_wrapper,
        )

        mock_nx.return_value = MagicMock()

        wrapper = create_dataset_wrapper("nx", n_nodes=20)

        mock_nx.assert_called_once_with(n_nodes=20)
        assert isinstance(wrapper, GraphCollection)


class TestLegacyWrappers:
    """Tests for legacy wrapper classes (backward compatibility)."""

    @patch("tmgg.experiment_utils.data.dataset_wrappers.ANUDataset")
    def test_anu_dataset_wrapper(self, mock_anu: MagicMock) -> None:
        """ANUDatasetWrapper should create wrapper for ANUDataset.

        Rationale: Legacy class should still work for backward compatibility.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import (
            ANUDatasetWrapper,
            GraphCollection,
        )

        mock_anu.return_value = MagicMock()

        wrapper = ANUDatasetWrapper(graph_type="ladder")

        mock_anu.assert_called_once_with(graph_type="ladder")
        assert isinstance(wrapper, GraphCollection)

    @patch("tmgg.experiment_utils.data.dataset_wrappers.CassicalGraphs")
    def test_classical_graphs_wrapper(self, mock_classical: MagicMock) -> None:
        """ClassicalGraphsWrapper should create wrapper for CassicalGraphs.

        Rationale: Legacy class should still work for backward compatibility.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import (
            ClassicalGraphsWrapper,
            GraphCollection,
        )

        mock_classical.return_value = MagicMock()

        wrapper = ClassicalGraphsWrapper()

        mock_classical.assert_called_once_with()
        assert isinstance(wrapper, GraphCollection)

    @patch("tmgg.experiment_utils.data.dataset_wrappers.NXGraphWrapper")
    def test_nx_graph_wrapper_wrapper(self, mock_nx: MagicMock) -> None:
        """NXGraphWrapperWrapper should create wrapper for NXGraphWrapper.

        Rationale: Legacy class should still work for backward compatibility.
        """
        from tmgg.experiment_utils.data.dataset_wrappers import (
            GraphCollection,
            NXGraphWrapperWrapper,
        )

        mock_nx.return_value = MagicMock()

        wrapper = NXGraphWrapperWrapper()

        mock_nx.assert_called_once_with()
        assert isinstance(wrapper, GraphCollection)
