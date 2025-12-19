"""PyTorch Geometric dataset wrappers for graph denoising experiments.

This module provides wrappers around PyTorch Geometric datasets (QM9, ENZYMES,
PROTEINS) that extract adjacency matrices and handle variable graph sizes
through padding.
"""

from pathlib import Path

import numpy as np
import torch


class PyGDatasetWrapper:
    """Wrapper for PyTorch Geometric datasets that extracts adjacency matrices.

    Handles variable-size graphs by padding to the maximum size in the dataset.
    Node features are ignored as per experiment design (adjacency-only).

    Parameters
    ----------
    dataset_name : str
        Name of the dataset: "qm9", "enzymes", or "proteins".
    root : str or Path, optional
        Root directory for downloading/storing datasets.
        Default is "~/.pyg_data".
    max_graphs : int, optional
        Maximum number of graphs to load. Default is None (all graphs).

    Attributes
    ----------
    adjacencies : np.ndarray
        Padded adjacency matrices of shape (num_graphs, max_n, max_n).
    num_nodes : np.ndarray
        Actual number of nodes per graph (before padding).
    max_n : int
        Maximum number of nodes across all graphs.
    """

    VALID_DATASETS = {"qm9", "enzymes", "proteins"}

    def __init__(
        self,
        dataset_name: str,
        root: str | None = None,
        max_graphs: int | None = None,
    ):
        if dataset_name.lower() not in self.VALID_DATASETS:
            raise ValueError(
                f"dataset_name must be one of {self.VALID_DATASETS}, "
                f"got '{dataset_name}'"
            )

        self.dataset_name = dataset_name.lower()
        self.root = Path(root) if root else Path.home() / ".pyg_data"
        self.max_graphs = max_graphs

        # Load dataset
        self._load_dataset()

    def _load_dataset(self) -> None:
        """Load PyG dataset and extract adjacency matrices."""
        try:
            from torch_geometric.datasets import QM9, TUDataset
            from torch_geometric.utils import to_dense_adj
        except ImportError as e:
            raise ImportError(
                "PyTorch Geometric is required for benchmark datasets. "
                "Install with: pip install torch-geometric"
            ) from e

        # Load appropriate dataset
        if self.dataset_name == "qm9":
            dataset = QM9(root=str(self.root / "QM9"))
        elif self.dataset_name == "enzymes":
            dataset = TUDataset(root=str(self.root), name="ENZYMES")
        elif self.dataset_name == "proteins":
            dataset = TUDataset(root=str(self.root), name="PROTEINS")
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        # Limit number of graphs if specified
        if self.max_graphs is not None:
            dataset = dataset[: self.max_graphs]

        # Extract adjacency matrices
        adjacencies = []
        num_nodes = []

        for i in range(len(dataset)):
            data = dataset[i]  # pyright: ignore[reportArgumentType]
            # Get number of nodes - use getattr for PyG Data dynamic attributes
            n = getattr(data, "num_nodes")  # noqa: B009
            num_nodes.append(n)

            # Convert edge_index to dense adjacency
            edge_index = getattr(data, "edge_index")  # noqa: B009
            A = to_dense_adj(edge_index, max_num_nodes=n).squeeze(0)
            adjacencies.append(A.numpy().astype(np.float32))

        self.num_nodes = np.array(num_nodes)
        self.max_n = int(self.num_nodes.max())

        # Pad all adjacencies to max size
        padded = []
        for A in adjacencies:
            n = A.shape[0]
            if n < self.max_n:
                # Pad with zeros
                padded_A = np.zeros((self.max_n, self.max_n), dtype=np.float32)
                padded_A[:n, :n] = A
                padded.append(padded_A)
            else:
                padded.append(A)

        self.adjacencies = np.stack(padded, axis=0)
        self._num_graphs = len(self.adjacencies)

    def __len__(self) -> int:
        return self._num_graphs

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.adjacencies[idx]

    def get_adjacency_matrices(self) -> np.ndarray:
        """Return all padded adjacency matrices.

        Returns
        -------
        np.ndarray
            Adjacency matrices of shape (num_graphs, max_n, max_n).
        """
        return self.adjacencies

    def get_node_counts(self) -> np.ndarray:
        """Return actual node counts for each graph.

        Returns
        -------
        np.ndarray
            Number of nodes per graph (before padding).
        """
        return self.num_nodes

    def to_torch(self) -> torch.Tensor:
        """Convert adjacency matrices to PyTorch tensor.

        Returns
        -------
        torch.Tensor
            Adjacency matrices as float32 tensor.
        """
        return torch.from_numpy(self.adjacencies).float()

    def get_masks(self) -> torch.Tensor:
        """Get node masks indicating valid (non-padded) regions.

        Returns
        -------
        torch.Tensor
            Boolean masks of shape (num_graphs, max_n) where True indicates
            a valid node.
        """
        masks = torch.zeros(self._num_graphs, self.max_n, dtype=torch.bool)
        for i, n in enumerate(self.num_nodes):
            masks[i, :n] = True
        return masks

    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        seed: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split dataset into train/validation/test sets.

        Parameters
        ----------
        train_ratio : float
            Fraction of data for training. Default 0.7.
        val_ratio : float
            Fraction of data for validation. Default 0.1.
        seed : int, optional
            Random seed for shuffling.

        Returns
        -------
        train : np.ndarray
            Training adjacency matrices.
        val : np.ndarray
            Validation adjacency matrices.
        test : np.ndarray
            Test adjacency matrices.
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self._num_graphs)

        n_train = int(train_ratio * self._num_graphs)
        n_val = int(val_ratio * self._num_graphs)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        return (
            self.adjacencies[train_idx],
            self.adjacencies[val_idx],
            self.adjacencies[test_idx],
        )


def load_pyg_dataset(
    name: str,
    root: str | None = None,
    max_graphs: int | None = None,
) -> PyGDatasetWrapper:
    """Convenience function to load a PyG dataset.

    Parameters
    ----------
    name : str
        Dataset name: "qm9", "enzymes", or "proteins".
    root : str, optional
        Root directory for datasets.
    max_graphs : int, optional
        Maximum number of graphs to load.

    Returns
    -------
    PyGDatasetWrapper
        Loaded dataset wrapper.

    Examples
    --------
    >>> dataset = load_pyg_dataset("enzymes", max_graphs=100)
    >>> print(f"Loaded {len(dataset)} graphs with max {dataset.max_n} nodes")
    """
    return PyGDatasetWrapper(name, root=root, max_graphs=max_graphs)
