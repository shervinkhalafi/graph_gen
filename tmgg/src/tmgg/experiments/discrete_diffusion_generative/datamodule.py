"""SyntheticCategoricalDataModule for discrete diffusion experiments.

Generates synthetic graphs as adjacency matrices, converts them to the
categorical ``GraphData`` representation used by discrete diffusion models,
and provides train/val/test DataLoaders with proper collation.

Graph generation and index splitting are handled by the
``MultiGraphDataModule`` superclass.
"""

from __future__ import annotations

from typing import Any, override

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from tmgg.data.data_modules.multigraph_data_module import (
    MultiGraphDataModule,
)
from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.noising.size_distribution import SizeDistribution


class _GraphDataDataset(Dataset[GraphData]):
    """Thin Dataset wrapping a list of unbatched ``GraphData`` instances."""

    _data: list[GraphData]

    def __init__(self, data: list[GraphData]) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    @override
    def __getitem__(self, idx: int) -> GraphData:
        return self._data[idx]


class SyntheticCategoricalDataModule(MultiGraphDataModule):
    """Data module that generates synthetic graphs in categorical format.

    Generates graphs as adjacency matrices using the shared
    ``_generate_adjacencies()`` from ``MultiGraphDataModule``, converts
    each to a ``GraphData`` instance via ``GraphData.from_adjacency()``,
    and splits into train/val/test sets.

    The categorical representation uses ``dx=2`` (node present / absent)
    and ``de=2`` (edge present / absent), matching the DiGress convention
    for synthetic binary graphs.

    Parameters
    ----------
    graph_type
        Graph type to generate. ``"sbm"`` uses the SBM generator directly;
        all other types delegate to ``SyntheticGraphDataset``.
    num_nodes
        Number of nodes per graph.
    num_graphs
        Total number of graphs to generate across all splits.
    train_ratio
        Fraction of graphs allocated to the training split.
    val_ratio
        Fraction allocated to validation. The remainder goes to test.
    batch_size
        Batch size for all DataLoaders.
    num_workers
        Number of DataLoader worker processes.
    seed
        Random seed for graph generation and splitting.
    graph_config
        Extra keyword arguments forwarded to the graph generator.
    """

    train_ratio: float
    val_ratio: float

    def __init__(
        self,
        graph_type: str = "sbm",
        num_nodes: int = 50,
        num_graphs: int = 1000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        seed: int = 42,
        graph_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> None:
        super().__init__(
            graph_type=graph_type,
            num_nodes=num_nodes,
            num_graphs=num_graphs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            graph_config=graph_config,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )

        # Populated by setup()
        self._train_data: list[GraphData] | None = None
        self._val_data: list[GraphData] | None = None
        self._test_data: list[GraphData] | None = None
        self._node_marginals: Tensor | None = None
        self._edge_marginals: Tensor | None = None

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

        # 1. Generate and split adjacency matrices (from MultiGraphDataModule)
        train_np, val_np, test_np = self._generate_and_split()

        # 2. Convert each adjacency matrix to GraphData
        train_adj = torch.from_numpy(train_np).float()  # pyright: ignore[reportUnknownMemberType]
        val_adj = torch.from_numpy(val_np).float()  # pyright: ignore[reportUnknownMemberType]
        test_adj = torch.from_numpy(test_np).float()  # pyright: ignore[reportUnknownMemberType]

        self._train_data = [
            GraphData.from_adjacency(train_adj[i]) for i in range(len(train_adj))
        ]
        self._val_data = [
            GraphData.from_adjacency(val_adj[i]) for i in range(len(val_adj))
        ]
        self._test_data = [
            GraphData.from_adjacency(test_adj[i]) for i in range(len(test_adj))
        ]

        # 3. Compute marginals from training data only
        self._compute_marginals()

    def _compute_marginals(self) -> None:
        """Compute empirical node and edge type distributions from training data.

        Marginals are normalised histograms over the one-hot categories,
        restricted to real (unmasked) positions.
        """
        assert self._train_data is not None, "setup() must be called first"

        dx: int = int(self._train_data[0].X.shape[-1])
        de: int = int(self._train_data[0].E.shape[-1])

        node_counts = torch.zeros(dx)
        edge_counts = torch.zeros(de)

        for g in self._train_data:
            n = g.node_mask.shape[0]
            node_counts += g.X.sum(dim=0)

            # Upper triangle only to avoid double-counting symmetric edges,
            # and exclude diagonal (no self-loops).
            triu_idx = torch.triu_indices(n, n, offset=1)  # pyright: ignore[reportUnknownMemberType]
            upper_E = g.E[triu_idx[0], triu_idx[1]]  # (num_edges, de)
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
    def train_dataloader(self) -> DataLoader[GraphData]:
        """Return the training DataLoader."""
        if self._train_data is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._make_dataloader(
            _GraphDataDataset(self._train_data),
            shuffle=True,
            collate_fn=GraphData.collate,
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphData]:
        """Return the validation DataLoader."""
        if self._val_data is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._make_dataloader(
            _GraphDataDataset(self._val_data),
            shuffle=False,
            collate_fn=GraphData.collate,
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphData]:
        """Return the test DataLoader."""
        if self._test_data is None:
            raise RuntimeError("DataModule not set up. Call setup() first.")
        return self._make_dataloader(
            _GraphDataDataset(self._test_data),
            shuffle=False,
            collate_fn=GraphData.collate,
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

    @override
    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
        """Return the graph size distribution for a split or the whole dataset.

        After ``setup()``, computes the empirical distribution from the
        stored per-graph node counts. Before ``setup()`` (or if the
        requested split is empty), falls back to the degenerate
        distribution at ``num_nodes``.

        Parameters
        ----------
        split
            ``"train"``, ``"val"``, ``"test"``, or ``None`` for all
            splits combined.

        Returns
        -------
        SizeDistribution
        """
        split_map: dict[str | None, list[GraphData] | None] = {
            "train": self._train_data,
            "val": self._val_data,
            "test": self._test_data,
        }

        if split is None:
            all_data = (
                (self._train_data or [])
                + (self._val_data or [])
                + (self._test_data or [])
            )
        elif split in split_map:
            all_data = split_map[split] or []
        else:
            raise ValueError(
                f"Unknown split {split!r}. Expected 'train', 'val', 'test', or None."
            )

        if not all_data:
            return SizeDistribution.fixed(self.num_nodes)

        node_counts = [g.node_mask.shape[0] for g in all_data]
        return SizeDistribution.from_node_counts(node_counts)

    def sample_n_nodes(self, batch_size: int) -> Tensor:
        """Sample graph sizes from the training split's empirical distribution.

        For fixed-size synthetic data this returns ``num_nodes`` repeated
        ``batch_size`` times (degenerate distribution). When variable-size
        datasets are wired in, the distribution reflects actual per-graph
        sizes from the training split.

        Parameters
        ----------
        batch_size
            Number of node counts to sample.

        Returns
        -------
        Tensor
            Integer tensor of shape ``(batch_size,)`` with sampled sizes.
        """
        dist = self.get_size_distribution("train")
        return dist.sample(batch_size)
