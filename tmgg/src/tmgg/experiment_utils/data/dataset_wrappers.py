"""
Wrappers for various graph datasets to make them compatible with the
experimental data modules, which expect a list of uniformly-sized adjacency matrices.
"""

from typing import List
import torch
import numpy as np
from torch._C import dtype

from .ggg_data.dense.anu_graphs.anudataset import ANUDataset
from .ggg_data.dense.classical import CassicalGraphs
from .ggg_data.dense.nx_graphs import NXGraphWrapper


class GraphCollection:
    """A wrapper to provide a unified interface for various graph dataset types."""

    def __init__(self, dataset):
        self.dataset = dataset

    def get_adjacency_matrices(self) -> List[torch.Tensor]:
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
        dtypes = [x.dtype for x in padded_matrices]
        assert all(x == torch.float for x in dtypes), f"padded_matrices.dtypes={dtypes}"
        return padded_matrices


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
