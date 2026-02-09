"""Abstract base for all graph data modules in TMGG.

Defines the contract that every graph data module must satisfy,
independent of the graph representation (adjacency matrices vs.
categorical one-hot features). Concrete subclasses fall into two
families:

- ``AdjacencyDataModule``: returns Tensor batches of adjacency matrices.
- ``CategoricalDataModule``: returns CategoricalBatch objects for
  discrete diffusion.
"""

from __future__ import annotations

import abc
from typing import Any, override

import pytorch_lightning as pl
from torch.utils.data import DataLoader


class BaseGraphDataModule(pl.LightningDataModule, abc.ABC):
    """Abstract base class for graph data modules.

    Provides the minimal interface shared by all graph data modules:
    setup, train/val/test dataloaders, and dataset metadata. Subclass
    families extend this with representation-specific attributes (e.g.,
    noise_levels for adjacency modules, class counts for categorical
    modules).

    Parameters
    ----------
    batch_size : int
        Batch size for all dataloaders.
    num_workers : int
        Number of dataloader worker processes.
    """

    batch_size: int
    num_workers: int

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

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
    def get_dataset_info(self) -> dict[str, Any]:
        """Return metadata about the dataset.

        The returned dict should contain at minimum:

        - ``"num_graphs"``: total number of graphs across all splits.
        - ``"num_nodes"``: node count (int for fixed-size, tuple for
          min/max range).

        Subclasses add representation-specific keys (e.g.,
        ``"num_node_classes"`` for categorical data modules).
        """
        ...
