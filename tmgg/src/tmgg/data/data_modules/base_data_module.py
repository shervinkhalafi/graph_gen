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

    @abc.abstractmethod
    def train_dataloader_raw_pyg(self) -> DataLoader[Any]:
        """Return a training dataloader yielding *raw* PyG ``Batch`` objects.

        Sits one layer below ``train_dataloader``: bypasses the dense
        ``GraphData`` collator so consumers that need the sparse PyG
        representation (notably the upstream-parity edge / node count
        helpers in :mod:`tmgg.data.utils.edge_counts` and
        :meth:`CategoricalNoiseProcess.initialize_from_data`) can
        iterate the same training data without going through
        densification.

        Marked ``@abc.abstractmethod`` so a subclass that omits the
        override fails at instantiation time (``TypeError: Can't
        instantiate abstract class ... with abstract method
        train_dataloader_raw_pyg``) rather than at config-preflight
        time on a Modal worker. Datamodules whose training data is not
        PyG-backed should still implement this and raise
        ``NotImplementedError`` from the body, so the failure mode is
        explicit at the leaf class rather than inherited from the base.
        """
        ...

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
        """Extract up to *max_graphs* from a dataset split as ``GraphData``.

        Iterates the appropriate dataloader and slices each batched
        ``GraphData`` along the leading dim, returning a flat list of
        per-graph (single-instance) ``GraphData`` with a 1-D
        ``node_mask`` and the optional split fields re-indexed to drop
        the batch axis. Padding is preserved so downstream consumers can
        decide whether to trim (graph evaluators) or to keep padded
        tensors as-is (molecular SMILES decode reads ``node_mask``).

        Per spec ``docs/specs/2026-05-01-graphdata-eval-pipeline-minispec.md``
        the return type is ``list[GraphData]`` rather than
        ``list[nx.Graph]``: GraphData is the universal transport format
        for the whole evaluation pipeline. Callers that need NetworkX
        graphs convert at the consumption site via
        :meth:`GraphData.to_networkx` (cheap, local, lossless).

        Parameters
        ----------
        stage : str
            ``"val"`` or ``"test"``.
        max_graphs : int
            Maximum number of graphs to return.

        Returns
        -------
        list[GraphData]
            Up to *max_graphs* per-graph ``GraphData`` from the
            requested split.

        Raises
        ------
        ValueError
            If *stage* is not ``"val"`` or ``"test"``.
        """
        from tmgg.data.datasets.graph_types import GraphData

        if stage == "val":
            loader = self.val_dataloader()
        elif stage == "test":
            loader = self.test_dataloader()
        else:
            raise ValueError(f"stage must be 'val' or 'test', got {stage!r}")

        graphs: list[Any] = []
        for batch in loader:
            bs = int(batch.node_mask.shape[0])
            for i in range(bs):
                if len(graphs) >= max_graphs:
                    return graphs
                graphs.append(
                    GraphData(
                        node_mask=batch.node_mask[i],
                        X_class=batch.X_class[i] if batch.X_class is not None else None,
                        X_feat=batch.X_feat[i] if batch.X_feat is not None else None,
                        E_class=batch.E_class[i] if batch.E_class is not None else None,
                        E_feat=batch.E_feat[i] if batch.E_feat is not None else None,
                        y=batch.y[i] if batch.y is not None else batch.y,
                    )
                )
        return graphs
