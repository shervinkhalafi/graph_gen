"""
Wrappers for various graph datasets to make them compatible with the
experimental data modules, which expect a list of uniformly-sized adjacency matrices.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, overload

import numpy as np
import torch

from .ggg_data.dense.anu_graphs.anudataset import ANUDataset
from .ggg_data.dense.classical import CassicalGraphs
from .ggg_data.dense.nx_graphs import NXGraphWrapper


class SizedDatasetProtocol(Protocol):
    """Protocol for datasets that can be iterated and have a length."""

    def __len__(self) -> int: ...
    # Parameter name intentionally generic to match various dataset implementations
    def __getitem__(self, __idx: int, /) -> Any: ...  # pyright: ignore[reportExplicitAny]


class GraphCollection:
    """A wrapper to provide a unified interface for various graph dataset types.

    Validates dataset structure on construction to fail fast if the dataset
    returns unsupported adjacency matrix types.
    """

    def __init__(self, dataset: SizedDatasetProtocol) -> None:
        self.dataset = dataset
        # Validate first item immediately if dataset is non-empty
        if len(dataset) > 0:
            first_item = dataset[0]
            adj = first_item[1] if isinstance(first_item, tuple) else first_item
            if not isinstance(adj, np.ndarray | torch.Tensor):
                raise TypeError(
                    f"Unsupported adjacency matrix type: {type(adj).__name__}. "
                    f"Expected np.ndarray or torch.Tensor."
                )

    def get_adjacency_matrices(self) -> list[torch.Tensor]:
        """
        Extracts all adjacency matrices from the wrapped dataset and pads them
        to a uniform size.
        """
        adj_matrices = []
        if len(self.dataset) == 0:
            return adj_matrices

        for i in range(len(self.dataset)):
            # The datasets typically return (X, A) where A is the adjacency matrix
            _, A = self.dataset[i]

            if isinstance(A, np.ndarray):
                A_tensor = torch.from_numpy(A).float()
            elif isinstance(A, torch.Tensor):
                A_tensor = A.float()
            else:
                raise TypeError(f"Unsupported adjacency matrix type: {type(A)}")
            adj_matrices.append(A_tensor)

        # Pad matrices to the same size if they are not already
        max_size = max(m.shape[0] for m in adj_matrices)

        padded_matrices = [
            torch.nn.functional.pad(
                m, (0, max_size - m.shape[0], 0, max_size - m.shape[0]), "constant", 0
            )
            if m.shape[0] < max_size
            else m
            for m in adj_matrices
        ]
        # Verify all matrices are float dtype (should always be true after .float() calls)
        invalid_dtypes = [
            (i, m.dtype)
            for i, m in enumerate(padded_matrices)
            if m.dtype != torch.float
        ]
        if invalid_dtypes:
            raise TypeError(
                f"Internal error: expected all matrices to have dtype torch.float, "
                f"but found {invalid_dtypes}"
            )
        return padded_matrices


@overload
def create_dataset_wrapper(
    dataset_type: Literal["anu"], **kwargs: Any
) -> GraphCollection: ...


@overload
def create_dataset_wrapper(
    dataset_type: Literal["classical"], **kwargs: Any
) -> GraphCollection: ...


@overload
def create_dataset_wrapper(
    dataset_type: Literal["nx"], **kwargs: Any
) -> GraphCollection: ...


@overload
def create_dataset_wrapper(dataset_type: str, **kwargs: Any) -> GraphCollection: ...


def create_dataset_wrapper(dataset_type: str, **kwargs: Any) -> GraphCollection:
    """
    Factory function to create dataset wrappers.

    Args:
        dataset_type: Type of dataset ("anu", "classical", "nx")
        **kwargs: Parameters passed to the dataset constructor

    Returns:
        GraphCollection wrapper around the specified dataset

    Raises:
        ValueError: If dataset_type is not recognized
    """
    dataset_classes: dict[
        str, type[ANUDataset] | type[CassicalGraphs] | type[NXGraphWrapper]
    ] = {
        "anu": ANUDataset,
        "classical": CassicalGraphs,
        "nx": NXGraphWrapper,
    }

    if dataset_type not in dataset_classes:
        raise ValueError(
            f"Unknown dataset type: {dataset_type}. Available: {list(dataset_classes.keys())}"
        )

    dataset = dataset_classes[dataset_type](**kwargs)
    return GraphCollection(dataset)


# Keep legacy wrapper classes for backward compatibility
class ANUDatasetWrapper(GraphCollection):
    """Wrapper for ANUDataset to make it compatible with experiment data modules."""

    def __init__(self, **kwargs):
        dataset = ANUDataset(**kwargs)
        super().__init__(dataset)


class ClassicalGraphsWrapper(GraphCollection):
    """Wrapper for CassicalGraphs to make it compatible with experiment data modules."""

    def __init__(self, **kwargs):
        dataset = CassicalGraphs(**kwargs)
        super().__init__(dataset)


class NXGraphWrapperWrapper(GraphCollection):
    """Wrapper for NXGraphWrapper to make it compatible with experiment data modules."""

    def __init__(self, **kwargs):
        dataset = NXGraphWrapper(**kwargs)
        super().__init__(dataset)
