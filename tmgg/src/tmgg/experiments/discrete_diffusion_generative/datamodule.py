"""SyntheticCategoricalDataModule for discrete diffusion experiments.

Generates synthetic graphs as adjacency matrices, stores them as PyG
``Data`` objects (matching the parent ``MultiGraphDataModule``), and
provides train/val/test DataLoaders that collate into dense ``GraphData``
batches. Marginal distributions over node and edge types are computed
from the training split at ``setup()`` time.

Graph generation and index splitting are handled by the
``MultiGraphDataModule`` superclass.
"""

# pyright: reportUnknownMemberType=false
# pyright: reportExplicitAny=false
# torch.from_numpy().float() and config dict Any parameters have
# incomplete type stubs.

from __future__ import annotations

from typing import Any, override

import torch
from torch import Tensor
from torch_geometric.data import Data

from tmgg.data.data_modules.multigraph_data_module import (
    MultiGraphDataModule,
)
from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.noising.size_distribution import SizeDistribution


class SyntheticCategoricalDataModule(MultiGraphDataModule):
    """Data module that generates synthetic graphs in categorical format.

    Generates graphs as adjacency matrices using the shared
    ``_generate_adjacencies()`` from ``MultiGraphDataModule``, converts
    each to a PyG ``Data`` object, and splits into train/val/test sets.
    DataLoaders collate into dense ``GraphData`` batches via the parent's
    ``_collate_pyg_to_graphdata`` collation.

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
        graph_config: dict[str, Any] | None = None,
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
        self._node_marginals: Tensor | None = None
        self._edge_marginals: Tensor | None = None

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate graphs, convert to PyG Data, split, and compute marginals.

        Delegates graph generation and storage to the parent, which
        populates ``_train_data`` / ``_val_data`` / ``_test_data`` as
        ``list[Data]``. Then computes marginals from training data.

        Idempotent: calling setup multiple times is safe.
        """
        if self._train_data is not None:
            return

        # Parent generates adjacencies and converts to list[Data]
        super().setup(stage)

        # Compute marginals from training data
        self._compute_marginals()

    def _compute_marginals(self) -> None:
        """Compute empirical node and edge type distributions from training data.

        Marginals are normalised histograms over the one-hot categories,
        restricted to real (unmasked) positions. Converts to dense
        representation temporarily for computation.
        """
        assert self._train_data is not None, "setup() must be called first"

        from typing import cast

        from torch_geometric.data import Batch
        from torch_geometric.data.data import BaseData

        # Convert to dense GraphData for marginal computation
        batch = Batch.from_data_list(cast(list[BaseData], self._train_data))
        dense = GraphData.from_pyg_batch(batch)

        bs, n_max = dense.node_mask.shape
        dx = int(dense.X.shape[-1])
        de = int(dense.E.shape[-1])

        # Node marginals: count per-class over real nodes
        real_nodes = dense.node_mask.bool()  # (bs, n_max)
        node_counts = dense.X[real_nodes].sum(dim=0)  # (dx,)

        # Edge marginals: upper triangle of real positions only
        triu = torch.triu_indices(n_max, n_max, offset=1)
        mask_2d = dense.node_mask.unsqueeze(-1) & dense.node_mask.unsqueeze(
            -2
        )  # (bs, n, n)
        upper_mask = mask_2d[:, triu[0], triu[1]]  # (bs, num_triu)
        upper_E = dense.E[:, triu[0], triu[1]]  # (bs, num_triu, de)
        edge_counts = upper_E[upper_mask].sum(dim=0)  # (de,)

        # Normalise
        node_total = node_counts.sum()
        edge_total = edge_counts.sum()
        self._node_marginals = (
            node_counts / node_total if node_total > 0 else torch.ones(dx) / dx
        )
        self._edge_marginals = (
            edge_counts / edge_total if edge_total > 0 else torch.ones(de) / de
        )

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

    # ------------------------------------------------------------------
    # Dataset info and size distribution
    # ------------------------------------------------------------------

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
        split_map: dict[str | None, list[Data] | None] = {
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

        node_counts = [int(d.num_nodes) for d in all_data]  # type: ignore[arg-type]
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
