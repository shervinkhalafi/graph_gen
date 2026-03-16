"""PyTorch Lightning data module for denoising graph experiments.

Handles multiple data sources: SBM and synthetic graph generation (via
inherited ``_generate_and_split()``), and PyG benchmark datasets. All
sources produce ``list[torch.Tensor]`` adjacency matrices split into
train/val/test.
"""

# pyright: reportExplicitAny=false
# DataLoader/Dataset generic parameters and config dicts require Any
# until PyTorch provides complete generic stubs.

from __future__ import annotations

import random
from typing import Any, Protocol, override

import numpy as np
import torch
from torch.utils.data import DataLoader

from tmgg.data._split import split_indices
from tmgg.data.datasets.graph_dataset import GraphDataset
from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.datasets.pyg_datasets import PyGDatasetWrapper

from .multigraph_data_module import MultiGraphDataModule

# PyG benchmark datasets
PYG_DATASETS = {"qm9", "enzymes", "proteins"}


class SampledDatasetProtocol(Protocol):
    """Protocol for datasets that expose num_samples attribute."""

    num_samples: int

    def __getitem__(self, idx: int) -> torch.Tensor: ...

    def __len__(self) -> int: ...


class GraphDataModule(MultiGraphDataModule):
    """Data module for denoising experiments with multiple graph sources.

    Supports SBM, all synthetic graph types (ER, regular, tree, LFR,
    ring_of_cliques, lollipop, circular_ladder, star, square_grid,
    triangle_grid), and PyG benchmark datasets (QM9, ENZYMES, PROTEINS).

    SBM and synthetic graph generation delegates to the inherited
    ``_generate_and_split()`` from ``MultiGraphDataModule``. PyG
    datasets are loaded and split via their own method.
    """

    samples_per_graph: int
    val_samples_per_graph: int
    train_adjacency_matrices: list[torch.Tensor] | None
    val_adjacency_matrices: list[torch.Tensor] | None
    test_adjacency_matrices: list[torch.Tensor] | None
    _rng: random.Random

    def __init__(
        self,
        graph_type: str,
        graph_config: dict[str, Any],
        samples_per_graph: int = 1000,
        val_samples_per_graph: int | None = None,
        batch_size: int = 100,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_ratio: float = 0.6,
        val_ratio: float = 0.2,
        noise_levels: list[float] | None = None,
        noise_type: str = "digress",
        seed: int = 42,
    ):
        """Initialize the GraphDataModule.

        Parameters
        ----------
        graph_type
            Type of graph to use (e.g., ``"sbm"``, ``"erdos_renyi"``).
        graph_config
            Dictionary of parameters for the chosen graph type.
        samples_per_graph
            Number of samples (permutations) per graph for training.
        val_samples_per_graph
            Number of samples per graph for validation/test.
            Defaults to ``samples_per_graph // 2`` if not specified.
        batch_size
            Batch size for data loaders.
        num_workers
            Number of worker processes for data loading.
        pin_memory
            Whether to pin memory for faster GPU transfer.
        train_ratio
            Fraction of data to use for training.
        val_ratio
            Fraction of data to use for validation. Remainder goes to test.
        noise_levels
            Accepted for Hydra compatibility (interpolated by task configs)
            but not stored — the training module owns noise configuration.
        noise_type
            Accepted for Hydra compatibility (interpolated by task configs)
            but not stored — the training module owns noise configuration.
        seed
            Random seed for reproducible splitting and generation.
        """
        super().__init__(
            graph_type=graph_type,
            num_nodes=graph_config.get("num_nodes", 50),
            num_graphs=graph_config.get("num_graphs", 1000),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            graph_config=graph_config,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
        self.save_hyperparameters()

        self.samples_per_graph = samples_per_graph
        self.val_samples_per_graph = (
            val_samples_per_graph
            if val_samples_per_graph is not None
            else samples_per_graph // 2
        )
        # Dedicated RNG for wrapper shuffling and sample selection
        self._rng = random.Random(self.seed)

        self.train_adjacency_matrices = None
        self.val_adjacency_matrices = None
        self.test_adjacency_matrices = None

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate or load graphs and split into train/val/test.

        PyG datasets have their own loading logic. SBM and all synthetic
        graph types delegate to ``_generate_and_split()`` from
        ``MultiGraphDataModule``.
        """
        if self.train_adjacency_matrices is not None:
            return

        if self.graph_type.lower() in PYG_DATASETS:
            self._setup_pyg_dataset()
        else:
            # SBM + all synthetic types → inherited _generate_and_split()
            train_np, val_np, test_np = self._generate_and_split()
            self.train_adjacency_matrices = [
                torch.from_numpy(train_np[i]).float() for i in range(len(train_np))
            ]
            self.val_adjacency_matrices = [
                torch.from_numpy(val_np[i]).float() for i in range(len(val_np))
            ]
            self.test_adjacency_matrices = [
                torch.from_numpy(test_np[i]).float() for i in range(len(test_np))
            ]

    def _setup_pyg_dataset(self) -> None:
        """Load and split graphs from a PyTorch Geometric dataset."""
        if self.train_adjacency_matrices is not None:
            return

        root: str | None = self.graph_config.get("root", None)
        max_graphs: int | None = self.graph_config.get("max_graphs", None)
        seed: int = self.graph_config.get("seed", 42)

        dataset = PyGDatasetWrapper(
            dataset_name=self.graph_type,
            root=root,
            max_graphs=max_graphs,
        )

        adjacencies = dataset.adjacencies
        train_idx, val_idx, test_idx = split_indices(
            len(adjacencies), self.train_ratio, self.val_ratio, seed
        )

        self.train_adjacency_matrices = [
            torch.from_numpy(A).float() for A in adjacencies[train_idx]
        ]
        self.val_adjacency_matrices = [
            torch.from_numpy(A).float() for A in adjacencies[val_idx]
        ]
        self.test_adjacency_matrices = [
            torch.from_numpy(A).float() for A in adjacencies[test_idx]
        ]

    def _create_dataloader(
        self,
        adjacency_matrices: list[torch.Tensor],
        samples_per_graph: int,
        shuffle: bool,
    ) -> DataLoader[GraphData]:
        """Create a data loader with common parameters."""
        matrices: list[np.ndarray | torch.Tensor] = list(adjacency_matrices)
        dataset = GraphDataset(matrices, samples_per_graph)
        return self._make_dataloader(
            dataset, shuffle=shuffle, collate_fn=GraphData.collate
        )

    @override
    def train_dataloader(self) -> DataLoader[GraphData]:
        """Create training data loader."""
        if self.train_adjacency_matrices is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._create_dataloader(
            self.train_adjacency_matrices, self.samples_per_graph, shuffle=True
        )

    @override
    def val_dataloader(self) -> DataLoader[GraphData]:
        """Create validation data loader."""
        if self.val_adjacency_matrices is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._create_dataloader(
            self.val_adjacency_matrices, self.val_samples_per_graph, shuffle=False
        )

    @override
    def test_dataloader(self) -> DataLoader[GraphData]:
        """Create test data loader."""
        if self.test_adjacency_matrices is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._create_dataloader(
            self.test_adjacency_matrices, self.val_samples_per_graph, shuffle=False
        )

    def get_sample_adjacency_matrix(self, stage: str = "train") -> torch.Tensor:
        """Get a sample adjacency matrix for visualization."""
        matrices: list[torch.Tensor] | None = None
        if stage == "train":
            matrices = self.train_adjacency_matrices
        elif stage == "val":
            matrices = self.val_adjacency_matrices
        elif stage == "test":
            matrices = self.test_adjacency_matrices

        if not matrices:
            raise RuntimeError(
                f"No data available for stage '{stage}'. "
                "Please ensure setup() has been called and the dataset is not empty."
            )

        return self._rng.choice(matrices)

    @override
    def get_dataset_info(self) -> dict[str, Any]:
        """Return metadata about the dataset."""
        info: dict[str, Any] = {
            "graph_type": self.graph_type,
            "samples_per_graph": self.samples_per_graph,
        }
        num_nodes = self.graph_config.get("num_nodes")
        if num_nodes is not None:
            info["num_nodes"] = num_nodes
        return info
