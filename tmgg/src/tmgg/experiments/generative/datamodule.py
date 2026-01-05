"""DataModule for graph generation experiments.

Provides train/val/test splits of graph collections for generative modeling.
Reuses existing synthetic graph generators from the codebase.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from tmgg.experiment_utils.data.sbm import generate_sbm_adjacency
from tmgg.experiment_utils.data.synthetic_graphs import SyntheticGraphDataset


class GraphDistributionDataModule(pl.LightningDataModule):
    """DataModule for graph distribution learning.

    Supports various synthetic graph types and train/val/test splits.

    Parameters
    ----------
    dataset_type
        Type of graph distribution. Options:
        - "sbm": Stochastic block model
        - "regular": d-regular graphs
        - "tree": Random trees
        - "erdos_renyi" / "er": Erdos-Renyi random graphs
        - "watts_strogatz" / "ws": Small-world graphs
        - "random_geometric" / "rg": Geometric proximity graphs
        - "lfr": LFR benchmark graphs
    num_nodes
        Number of nodes per graph.
    num_graphs
        Total number of graphs to generate.
    train_ratio
        Fraction of graphs for training.
    val_ratio
        Fraction of graphs for validation.
    batch_size
        Batch size for dataloaders.
    num_workers
        Number of dataloader workers.
    seed
        Random seed for reproducibility.
    dataset_config
        Additional configuration for the graph generator.
    noise_levels
        Noise levels (for compatibility with denoising modules).
    """

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
        dataset_config: dict[str, Any] | None = None,
        noise_levels: list[float] | None = None,
        **kwargs: Any,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.dataset_type = dataset_type
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed
        self.dataset_config = dataset_config or {}

        # Noise levels for compatibility with DenoisingLightningModule
        self._noise_levels = noise_levels or [0.1, 0.3, 0.5]

        # Will be populated in setup()
        self._train_data: torch.Tensor | None = None
        self._val_data: torch.Tensor | None = None
        self._test_data: torch.Tensor | None = None

    @property
    def noise_levels(self) -> list[float]:
        """Noise levels for training/evaluation."""
        return self._noise_levels

    def setup(self, stage: str | None = None) -> None:
        """Generate and split the dataset.

        Parameters
        ----------
        stage
            Either 'fit', 'validate', 'test', or 'predict'.
        """
        if self._train_data is not None:
            return  # Already setup

        # Generate graphs based on dataset type
        if self.dataset_type == "sbm":
            adjacencies = self._generate_sbm_graphs()
        else:
            # Use SyntheticGraphDataset for other types
            dataset = SyntheticGraphDataset(
                graph_type=self.dataset_type,
                n=self.num_nodes,
                num_graphs=self.num_graphs,
                seed=self.seed,
                **self.dataset_config,
            )
            adjacencies = dataset.get_adjacency_matrices()

        # Split into train/val/test
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(adjacencies))

        n_train = int(self.train_ratio * len(adjacencies))
        n_val = int(self.val_ratio * len(adjacencies))

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        # Convert to tensors
        self._train_data = torch.from_numpy(adjacencies[train_idx]).float()
        self._val_data = torch.from_numpy(adjacencies[val_idx]).float()
        self._test_data = torch.from_numpy(adjacencies[test_idx]).float()

    def _generate_sbm_graphs(self) -> np.ndarray:
        """Generate stochastic block model graphs.

        Returns
        -------
        np.ndarray
            Adjacency matrices of shape (num_graphs, num_nodes, num_nodes).
        """
        rng = np.random.default_rng(self.seed)
        adjacencies = []

        # SBM parameters from config
        num_blocks = self.dataset_config.get("num_blocks", 2)
        p_in = self.dataset_config.get("p_in", 0.7)
        p_out = self.dataset_config.get("p_out", 0.1)

        # Equal block sizes
        block_size = self.num_nodes // num_blocks
        remainder = self.num_nodes % num_blocks
        block_sizes = [block_size] * num_blocks
        # Distribute remainder
        for i in range(remainder):
            block_sizes[i] += 1

        for _ in range(self.num_graphs):
            A = generate_sbm_adjacency(
                block_sizes=block_sizes,
                p=p_in,
                q=p_out,
                rng=rng,
            )
            # Ensure symmetric (upper triangular already copied to lower)
            A = np.triu(A, k=1)
            A = A + A.T
            # Zero diagonal
            np.fill_diagonal(A, 0)
            adjacencies.append(A.astype(np.float32))

        return np.stack(adjacencies, axis=0)

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

    def get_sample_graph(self) -> torch.Tensor:
        """Get a sample graph for visualization."""
        if self._train_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self._train_data[0]


class _UnwrapDataset(torch.utils.data.Dataset):
    """Dataset wrapper that returns tensors directly instead of tuples."""

    def __init__(self, data: torch.Tensor):
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]
