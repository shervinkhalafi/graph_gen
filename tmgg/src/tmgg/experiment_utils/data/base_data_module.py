"""Abstract base for all graph data modules in TMGG.

Defines the minimal contract that every graph data module must satisfy,
independent of the graph representation (adjacency matrices vs.
categorical one-hot features) or the generation protocol (multi-graph
vs. single-graph). Concrete generation and splitting logic lives in
``MultiGraphDataModule`` and the leaf classes.
"""

from __future__ import annotations

import abc
from collections.abc import Callable
from typing import Any, override

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


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
    seed
        Random seed for graph generation and splitting.
    """

    batch_size: int
    num_workers: int
    pin_memory: bool
    seed: int

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.seed = seed

    # ------------------------------------------------------------------
    # DataLoader factory
    # ------------------------------------------------------------------

    def _make_dataloader(
        self,
        dataset: Dataset[Any],  # pyright: ignore[reportExplicitAny]
        shuffle: bool,
        collate_fn: Callable[..., Any] | None = None,  # pyright: ignore[reportExplicitAny]
    ) -> DataLoader[Any]:  # pyright: ignore[reportExplicitAny]
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
            are present.
        """
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_fn,
        )

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
    def train_dataloader(self) -> DataLoader[Any]:  # pyright: ignore[reportExplicitAny]
        """Return the training dataloader."""
        ...

    @override
    @abc.abstractmethod
    def val_dataloader(self) -> DataLoader[Any]:  # pyright: ignore[reportExplicitAny]
        """Return the validation dataloader."""
        ...

    @override
    @abc.abstractmethod
    def test_dataloader(self) -> DataLoader[Any]:  # pyright: ignore[reportExplicitAny]
        """Return the test dataloader."""
        ...

    @abc.abstractmethod
    def get_dataset_info(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Return metadata about the dataset.

        The returned dict should contain at minimum:

        - ``"num_graphs"``: total number of graphs across all splits.
        - ``"num_nodes"``: node count (int for fixed-size, tuple for
          min/max range).

        Subclasses add representation-specific keys (e.g.,
        ``"num_node_classes"`` for categorical data modules).
        """
        ...
