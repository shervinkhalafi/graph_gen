"""Abstract base for all graph data modules in TMGG.

Defines the minimal contract that every graph data module must satisfy,
independent of the graph representation (adjacency matrices vs.
categorical one-hot features) or the generation protocol (multi-graph
vs. single-graph). Concrete generation and splitting logic lives in
``MultiGraphDataModule`` and the leaf classes.
"""

# pyright: reportExplicitAny=false
# DataLoader/Dataset generic parameters and config dicts require Any
# until PyTorch provides complete generic stubs.

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, override

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset

from tmgg.utils.noising.size_distribution import SizeDistribution


class BaseGraphDataModule(pl.LightningDataModule, abc.ABC):
    """Abstract base class for all graph data modules.

    Provides the shared DataLoader configuration (batch size, workers,
    pin_memory) and a concrete ``_make_dataloader`` factory. Subclasses
    implement generation, splitting, setup, and dataloaders.

    Parameters
    ----------
    batch_size
        Batch size for all dataloaders.
    num_workers
        Number of dataloader worker processes.
    pin_memory
        Whether to pin memory in DataLoaders for faster GPU transfer.
    prefetch_factor
        Per-worker batches to prefetch ahead. PyTorch default is 2;
        4 hides typical step latency without much memory overhead. Has
        no effect when ``num_workers == 0`` (PyTorch ignores it). See
        ``analysis/digress-loss-check/vignac-repro-health-check/
        speedup-options.md §A``.
    seed
        Random seed for graph generation and splitting.
    """

    batch_size: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    seed: int
    graph_type: str
    num_nodes: int

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 4,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.seed = seed

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    def _make_dataloader(
        self,
        dataset: Dataset[Any],
        shuffle: bool,
        collate_fn: Callable[..., Any] | None = None,
    ) -> DataLoader[Any]:
        """Create a DataLoader with the module's shared configuration.

        Parameters
        ----------
        dataset
            The dataset to wrap.
        shuffle
            Whether to shuffle each epoch.
        collate_fn
            Custom collation function. ``None`` uses the default
            torch collation.

        Returns
        -------
        DataLoader
            Configured with ``self.batch_size``, ``self.num_workers``,
            ``self.pin_memory``, and ``persistent_workers`` when workers
            are present. ``prefetch_factor`` only forwarded when workers
            are spawned (PyTorch raises if you pass it with
            ``num_workers == 0``).
        """
        kwargs: dict[str, Any] = dict(
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )
        if self.num_workers > 0:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return DataLoader(dataset, **kwargs)

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @override
    @abc.abstractmethod
    def setup(self, stage: str | None = None) -> None:
        """Prepare datasets for the given stage.

        Parameters
        ----------
        stage : str or None
            One of ``"fit"``, ``"validate"``, ``"test"``, ``"predict"``,
            or None (all stages).
        """
        ...

    @override
    @abc.abstractmethod
    def train_dataloader(self) -> DataLoader[Any]:
        """Return the training dataloader."""
        ...

    @override
    @abc.abstractmethod
    def val_dataloader(self) -> DataLoader[Any]:
        """Return the validation dataloader."""
        ...

    @override
    @abc.abstractmethod
    def test_dataloader(self) -> DataLoader[Any]:
        """Return the test dataloader."""
        ...

    def train_dataloader_raw_pyg(self) -> DataLoader[Any]:
        """Return a training dataloader yielding *raw* PyG ``Batch`` objects.

        Sits one layer below ``train_dataloader``: bypasses the dense
        ``GraphData`` collator so consumers that need the sparse PyG
        representation (notably the upstream-parity edge / node count
        helpers in :mod:`tmgg.data.utils.edge_counts`) can iterate the
        same training data without going through densification.

        Subclasses that produce PyG ``Data`` lists must override this to
        wire their stored split through the raw collator. The base
        method raises so the omission is loud rather than silent (any
        datamodule whose training data is *not* PyG-backed must declare
        that explicitly).
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not expose a raw PyG training "
            "dataloader; override train_dataloader_raw_pyg() if the "
            "datamodule's training data is PyG-backed."
        )

    # ------------------------------------------------------------------
    # Concrete utility methods
    # ------------------------------------------------------------------

    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
        """Return the graph-size distribution for a dataset split.

        The base contract assumes fixed-size graphs and therefore returns
        a degenerate distribution at ``self.num_nodes`` regardless of
        *split*. Variable-size datamodules should override this method.

        Parameters
        ----------
        split
            Split selector. Accepted for interface compatibility with
            variable-size subclasses and ignored by the fixed-size base
            implementation.

        Returns
        -------
        SizeDistribution
            Degenerate distribution at ``self.num_nodes``.
        """
        _ = split
        return SizeDistribution.fixed(self.num_nodes)

    def get_reference_graphs(self, stage: str, max_graphs: int) -> list[Any]:
        """Extract up to *max_graphs* from a dataset split as NetworkX graphs.

        Iterates the appropriate dataloader, converts ``GraphData`` batches
        to adjacency matrices, and builds NetworkX graphs respecting
        ``node_mask`` for variable-size graphs. Subclasses may override for
        efficiency (e.g., reading directly from internal tensors).

        Parameters
        ----------
        stage : str
            ``"val"`` or ``"test"``.
        max_graphs : int
            Maximum number of graphs to return.

        Returns
        -------
        list[nx.Graph]
            Up to *max_graphs* NetworkX graphs from the requested split.

        Raises
        ------
        ValueError
            If *stage* is not ``"val"`` or ``"test"``.
        """
        import networkx as nx

        if stage == "val":
            loader = self.val_dataloader()
        elif stage == "test":
            loader = self.test_dataloader()
        else:
            raise ValueError(f"stage must be 'val' or 'test', got {stage!r}")

        graphs: list[Any] = []
        for batch in loader:
            adj = batch.binarised_adjacency()  # (B, N, N)
            bs = adj.shape[0]
            for i in range(bs):
                if len(graphs) >= max_graphs:
                    return graphs
                n = int(batch.node_mask[i].sum().item())
                A_np = adj[i, :n, :n].cpu().numpy()
                graphs.append(nx.from_numpy_array(A_np))
        return graphs
