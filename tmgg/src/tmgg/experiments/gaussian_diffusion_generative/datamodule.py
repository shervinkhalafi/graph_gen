"""DataModule for gaussian diffusion graph generation experiments.

Provides train/val/test splits of graph collections as adjacency matrix
tensors. Graph generation and index splitting are handled by the
``BaseGraphDataModule`` superclass.
"""

from __future__ import annotations

from typing import Any, override

import torch
from torch.utils.data import DataLoader

from tmgg.experiment_utils.data.base_data_module import BaseGraphDataModule


class GraphDistributionDataModule(BaseGraphDataModule):
    """DataModule for graph distribution learning (continuous diffusion).

    Generates synthetic graphs, splits into train/val/test, and serves
    adjacency matrices as float tensors via DataLoaders.

    Parameters
    ----------
    dataset_type
        Type of graph distribution (``"sbm"``, ``"er"``, ``"tree"``, etc.).
    num_nodes
        Number of nodes per graph.
    num_graphs
        Total number of graphs to generate.
    train_ratio
        Fraction of graphs for training.
    val_ratio
        Fraction of graphs for validation. Remainder goes to test.
    batch_size
        Batch size for dataloaders.
    num_workers
        Number of dataloader workers.
    seed
        Random seed for reproducibility.
    dataset_config
        Additional configuration for the graph generator.
    """

    train_ratio: float
    val_ratio: float

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
        **kwargs: Any,
    ):
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_type=dataset_type,
            num_nodes=num_nodes,
            num_graphs=num_graphs,
            seed=seed,
            dataset_config=dataset_config,
        )
        self.save_hyperparameters()
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio

        # Populated by setup()
        self._train_data: torch.Tensor | None = None
        self._val_data: torch.Tensor | None = None
        self._test_data: torch.Tensor | None = None

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate graphs, split, and convert to tensors.

        Parameters
        ----------
        stage
            Lightning stage (``"fit"``, ``"test"``, etc.) or None.
        """
        if self._train_data is not None:
            return  # Already setup

        adjacencies = self._generate_adjacencies()
        train_idx, val_idx, test_idx = self._split_indices(
            len(adjacencies), self.train_ratio, self.val_ratio, self.seed
        )

        self._train_data = torch.from_numpy(adjacencies[train_idx]).float()
        self._val_data = torch.from_numpy(adjacencies[val_idx]).float()
        self._test_data = torch.from_numpy(adjacencies[test_idx]).float()

    @override
    def train_dataloader(self) -> DataLoader[torch.Tensor]:
        """Create training dataloader."""
        if self._train_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return DataLoader(
            _UnwrapDataset(self._train_data),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @override
    def val_dataloader(self) -> DataLoader[torch.Tensor]:
        """Create validation dataloader."""
        if self._val_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return DataLoader(
            _UnwrapDataset(self._val_data),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @override
    def test_dataloader(self) -> DataLoader[torch.Tensor]:
        """Create test dataloader."""
        if self._test_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return DataLoader(
            _UnwrapDataset(self._test_data),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    @override
    def get_dataset_info(self) -> dict[str, Any]:
        """Return metadata about the dataset.

        Returns
        -------
        dict
            Keys: ``num_graphs``, ``num_nodes``, ``dataset_type``.
        """
        return {
            "num_graphs": self.num_graphs,
            "num_nodes": self.num_nodes,
            "dataset_type": self.dataset_type,
        }

    def get_sample_graph(self) -> torch.Tensor:
        """Get a sample graph for visualization."""
        if self._train_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self._train_data[0]


class _UnwrapDataset(torch.utils.data.Dataset[torch.Tensor]):
    """Dataset wrapper that returns tensors directly instead of tuples."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
