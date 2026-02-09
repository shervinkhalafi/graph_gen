"""SyntheticCategoricalDataModule for discrete diffusion experiments.

Generates synthetic graphs as adjacency matrices, converts them to the
categorical (X, E, y, node_mask) representation used by discrete diffusion
models, and provides train/val/test DataLoaders with proper collation.
"""

from __future__ import annotations

from typing import Any, override

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tmgg.experiment_utils.data.base_data_module import BaseGraphDataModule
from tmgg.experiment_utils.data.conversions import (
    adjacency_to_categorical,
    collate_categorical,
)
from tmgg.experiment_utils.data.sbm import generate_sbm_adjacency
from tmgg.experiment_utils.data.synthetic_graphs import SyntheticGraphDataset

# Type alias for the per-graph tuple consumed by collate_categorical.
_CategoricalSample = tuple[Tensor, Tensor, Tensor, int]


class _CategoricalTupleDataset(Dataset[_CategoricalSample]):
    """Thin Dataset wrapping a list of (X, E, y, num_nodes) tuples.

    Each element matches the signature expected by ``collate_categorical``.
    """

    _data: list[_CategoricalSample]

    def __init__(self, data: list[_CategoricalSample]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    @override
    def __getitem__(self, idx: int) -> _CategoricalSample:
        return self._data[idx]


class SyntheticCategoricalDataModule(BaseGraphDataModule):
    """Data module that generates synthetic graphs in categorical format.

    Generates graphs as adjacency matrices using the existing SBM or
    SyntheticGraphDataset generators, converts each to the one-hot
    categorical representation ``(X, E, y, node_mask)`` via
    ``adjacency_to_categorical``, and splits into train/val/test sets.

    The categorical representation uses ``dx=2`` (node present / absent)
    and ``de=2`` (edge present / absent), matching the DiGress convention
    for synthetic binary graphs.

    Parameters
    ----------
    dataset_type : str
        Graph type to generate. ``"sbm"`` uses the SBM generator directly;
        all other types (``"er"``, ``"regular"``, ``"tree"``, ``"ws"``,
        ``"rg"``, ``"lfr"``, etc.) delegate to ``SyntheticGraphDataset``.
    num_nodes : int
        Number of nodes per graph.
    num_graphs : int
        Total number of graphs to generate across all splits.
    train_ratio : float
        Fraction of graphs allocated to the training split.
    val_ratio : float
        Fraction allocated to validation. The remainder goes to test.
    batch_size : int
        Batch size for all DataLoaders.
    num_workers : int
        Number of DataLoader worker processes.
    seed : int
        Random seed for graph generation and splitting.
    dataset_config : dict or None
        Extra keyword arguments forwarded to the graph generator.
        For SBM: ``num_blocks``, ``p_in``, ``p_out``.
        For other types: see ``SyntheticGraphDataset`` documentation.
    """

    dataset_type: str
    num_nodes: int
    num_graphs: int
    train_ratio: float
    val_ratio: float
    seed: int
    dataset_config: dict[str, Any]  # pyright: ignore[reportExplicitAny]

    def __init__(
        self,
        dataset_type: str = "sbm",
        num_nodes: int = 50,
        num_graphs: int = 1000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
        seed: int = 42,
        dataset_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> None:
        super().__init__(batch_size=batch_size, num_workers=num_workers)
        self.dataset_type = dataset_type
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.seed = seed
        self.dataset_config = dataset_config or {}

        # Populated by setup()
        self._train_data: list[_CategoricalSample] | None = None
        self._val_data: list[_CategoricalSample] | None = None
        self._test_data: list[_CategoricalSample] | None = None
        self._node_marginals: Tensor | None = None
        self._edge_marginals: Tensor | None = None

    # ------------------------------------------------------------------
    # Graph generation (mirrors GraphDistributionDataModule)
    # ------------------------------------------------------------------

    def _generate_adjacencies(self) -> np.ndarray:
        """Generate adjacency matrices according to ``dataset_type``.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_graphs, num_nodes, num_nodes)``.
        """
        if self.dataset_type == "sbm":
            return self._generate_sbm_graphs()

        dataset = SyntheticGraphDataset(
            graph_type=self.dataset_type,
            n=self.num_nodes,
            num_graphs=self.num_graphs,
            seed=self.seed,
            **self.dataset_config,
        )
        return dataset.get_adjacency_matrices()

    def _generate_sbm_graphs(self) -> np.ndarray:
        """Generate stochastic block model graphs.

        Reuses the same parameterisation as
        ``GraphDistributionDataModule._generate_sbm_graphs``.
        """
        rng = np.random.default_rng(self.seed)

        num_blocks: int = self.dataset_config.get("num_blocks", 2)
        p_in: float = self.dataset_config.get("p_in", 0.7)
        p_out: float = self.dataset_config.get("p_out", 0.1)

        block_size = self.num_nodes // num_blocks
        remainder = self.num_nodes % num_blocks
        block_sizes = [block_size] * num_blocks
        for i in range(remainder):
            block_sizes[i] += 1

        adjacencies: list[np.ndarray] = []
        for _ in range(self.num_graphs):
            adj = generate_sbm_adjacency(
                block_sizes=block_sizes, p=p_in, q=p_out, rng=rng
            )
            adj = np.triu(adj, k=1)
            adj = adj + adj.T
            np.fill_diagonal(adj, 0)
            adjacencies.append(adj.astype(np.float32))

        return np.stack(adjacencies, axis=0)

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _adjacency_to_tuple(
        adj: Tensor,
    ) -> _CategoricalSample:
        """Convert a single adjacency matrix to a collate-compatible tuple.

        Parameters
        ----------
        adj : Tensor
            A single adjacency matrix of shape ``(n, n)``.

        Returns
        -------
        _CategoricalSample
            ``(X, E, y, n)`` matching the ``collate_categorical`` interface.
        """
        X, E, y, _node_mask = adjacency_to_categorical(adj)
        # X: (n, dx), E: (n, n, de), y: (dy,), _node_mask: (n,)
        n: int = int(adj.shape[0])
        return (X, E, y, n)

    # ------------------------------------------------------------------
    # Setup and DataLoaders
    # ------------------------------------------------------------------

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate graphs, convert to categorical, and split.

        Idempotent: calling setup multiple times is safe.
        """
        if self._train_data is not None:
            return

        # 1. Generate adjacency matrices
        adjacencies_np = self._generate_adjacencies()

        # 2. Shuffle and split indices
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(adjacencies_np))

        n_train = int(self.train_ratio * len(adjacencies_np))
        n_val = int(self.val_ratio * len(adjacencies_np))

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        # 3. Convert each adjacency matrix to categorical tuple
        all_adj = torch.from_numpy(adjacencies_np).float()  # pyright: ignore[reportUnknownMemberType]

        self._train_data = [self._adjacency_to_tuple(all_adj[i]) for i in train_idx]
        self._val_data = [self._adjacency_to_tuple(all_adj[i]) for i in val_idx]
        self._test_data = [self._adjacency_to_tuple(all_adj[i]) for i in test_idx]

        # 4. Compute marginals from training data only
        self._compute_marginals()

    def _compute_marginals(self) -> None:
        """Compute empirical node and edge type distributions from training data.

        Marginals are normalised histograms over the one-hot categories,
        restricted to real (unmasked) positions.
        """
        assert self._train_data is not None, "setup() must be called first"

        # dx=2 (no-node, node), de=2 (no-edge, edge) for synthetic graphs
        dx: int = int(self._train_data[0][0].shape[-1])
        de: int = int(self._train_data[0][1].shape[-1])

        node_counts = torch.zeros(dx)
        edge_counts = torch.zeros(de)

        for X_i, E_i, _y_i, n_i in self._train_data:
            # X_i: (n, dx), E_i: (n, n, de) -- all nodes are real
            node_counts += X_i[:n_i].sum(dim=0)

            # Upper triangle only to avoid double-counting symmetric edges,
            # and exclude diagonal (no self-loops).
            triu_idx = torch.triu_indices(n_i, n_i, offset=1)  # pyright: ignore[reportUnknownMemberType]
            upper_E = E_i[triu_idx[0], triu_idx[1]]  # (num_edges, de)
            edge_counts += upper_E.sum(dim=0)

        # Normalise to probability distributions
        node_total = node_counts.sum()
        edge_total = edge_counts.sum()

        if node_total > 0:
            self._node_marginals = node_counts / node_total
        else:
            self._node_marginals = torch.ones(dx) / dx

        if edge_total > 0:
            self._edge_marginals = edge_counts / edge_total
        else:
            self._edge_marginals = torch.ones(de) / de

    @property
    def node_marginals(self) -> Tensor:
        """Empirical node type distribution from the training split.

        Returns
        -------
        Tensor
            Shape ``(dx,)`` summing to 1.
        """
        if self._node_marginals is None:
            raise RuntimeError("Marginals not available. Call setup() first.")
        return self._node_marginals

    @property
    def edge_marginals(self) -> Tensor:
        """Empirical edge type distribution from the training split.

        Returns
        -------
        Tensor
            Shape ``(de,)`` summing to 1.
        """
        if self._edge_marginals is None:
            raise RuntimeError("Marginals not available. Call setup() first.")
        return self._edge_marginals

    @override
    def train_dataloader(self) -> DataLoader[_CategoricalSample]:
        """Return the training DataLoader."""
        if self._train_data is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return DataLoader(
            _CategoricalTupleDataset(self._train_data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_categorical,
            pin_memory=True,
        )

    @override
    def val_dataloader(self) -> DataLoader[_CategoricalSample]:
        """Return the validation DataLoader."""
        if self._val_data is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return DataLoader(
            _CategoricalTupleDataset(self._val_data),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_categorical,
            pin_memory=True,
        )

    @override
    def test_dataloader(self) -> DataLoader[_CategoricalSample]:
        """Return the test DataLoader."""
        if self._test_data is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return DataLoader(
            _CategoricalTupleDataset(self._test_data),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_categorical,
            pin_memory=True,
        )

    @override
    def get_dataset_info(self) -> dict[str, int]:
        """Return metadata about the categorical dataset.

        Returns
        -------
        dict
            Keys: ``num_graphs``, ``num_nodes``, ``num_node_classes``,
            ``num_edge_classes``.
        """
        return {
            "num_graphs": self.num_graphs,
            "num_nodes": self.num_nodes,
            "num_node_classes": 2,
            "num_edge_classes": 2,
        }
