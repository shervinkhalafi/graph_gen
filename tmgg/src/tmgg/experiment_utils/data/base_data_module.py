"""Abstract base for all graph data modules in TMGG.

Defines the contract that every graph data module must satisfy,
independent of the graph representation (adjacency matrices vs.
categorical one-hot features). Includes shared graph generation
and splitting logic that both generative datamodule families reuse.
"""

from __future__ import annotations

import abc
from typing import Any, override

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from tmgg.experiment_utils.data.sbm import (
    generate_sbm_batch,  # pyright: ignore[reportAttributeAccessIssue]  # runtime-verified; basedpyright module resolution quirk
)
from tmgg.experiment_utils.data.synthetic_graphs import SyntheticGraphDataset


class BaseGraphDataModule(pl.LightningDataModule, abc.ABC):
    """Abstract base class for graph data modules.

    Provides the minimal interface shared by all graph data modules:
    setup, train/val/test dataloaders, and dataset metadata. Also
    provides concrete graph generation and index-splitting utilities
    that generative datamodules reuse.

    Parameters
    ----------
    batch_size
        Batch size for all dataloaders.
    num_workers
        Number of dataloader worker processes.
    dataset_type
        Graph type to generate (``"sbm"``, ``"er"``, ``"tree"``, etc.).
        Subclasses that handle generation themselves may leave this as
        the default.
    num_nodes
        Number of nodes per graph.
    num_graphs
        Total number of graphs to generate across all splits.
    seed
        Random seed for graph generation and splitting.
    dataset_config
        Extra keyword arguments forwarded to the graph generator.
    """

    batch_size: int
    num_workers: int
    dataset_type: str
    num_nodes: int
    num_graphs: int
    seed: int
    dataset_config: dict[str, Any]  # pyright: ignore[reportExplicitAny]

    def __init__(
        self,
        batch_size: int = 32,
        num_workers: int = 0,
        dataset_type: str = "sbm",
        num_nodes: int = 50,
        num_graphs: int = 1000,
        seed: int = 42,
        dataset_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_type = dataset_type
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.seed = seed
        self.dataset_config = dataset_config or {}

    # ------------------------------------------------------------------
    # Graph generation (shared by both generative datamodule families)
    # ------------------------------------------------------------------

    def _generate_adjacencies(self) -> np.ndarray:
        """Generate adjacency matrices according to ``dataset_type``.

        For ``"sbm"`` graphs, delegates to ``generate_sbm_batch``.
        For all other types, delegates to ``SyntheticGraphDataset``.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_graphs, num_nodes, num_nodes)``.
        """
        if self.dataset_type == "sbm":
            return generate_sbm_batch(
                num_graphs=self.num_graphs,
                num_nodes=self.num_nodes,
                num_blocks=self.dataset_config.get("num_blocks", 2),
                p_intra=self.dataset_config.get("p_intra", 0.7),
                p_inter=self.dataset_config.get("p_inter", 0.1),
                seed=self.seed,
            )

        dataset = SyntheticGraphDataset(
            graph_type=self.dataset_type,
            n=self.num_nodes,
            num_graphs=self.num_graphs,
            seed=self.seed,
            **self.dataset_config,
        )
        return dataset.get_adjacency_matrices()

    # ------------------------------------------------------------------
    # Splitting utility
    # ------------------------------------------------------------------

    @staticmethod
    def _split_indices(
        n: int,
        train_ratio: float,
        val_ratio: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split ``n`` indices into train/val/test by ratio.

        Parameters
        ----------
        n
            Total number of samples.
        train_ratio
            Fraction for training.
        val_ratio
            Fraction for validation. Remainder goes to test.
        seed
            Random seed for the permutation.

        Returns
        -------
        tuple of np.ndarray
            ``(train_idx, val_idx, test_idx)`` index arrays.
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(n)

        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)

        return (
            indices[:n_train],
            indices[n_train : n_train + n_val],
            indices[n_train + n_val :],
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
