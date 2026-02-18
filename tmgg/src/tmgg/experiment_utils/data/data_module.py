"""PyTorch Lightning data module for denoising graph experiments.

Handles multiple data sources: SBM and synthetic graph generation (via
inherited ``_generate_and_split()``), external wrapper datasets (ANU,
Classical, NX), and PyG benchmark datasets. All sources produce
``list[torch.Tensor]`` adjacency matrices split into train/val/test.
"""

from __future__ import annotations

import random
from typing import Any, Protocol, override

import numpy as np
import torch
from torch.utils.data import DataLoader

from tmgg.experiment_utils.data.dataset import GraphDataset
from tmgg.experiment_utils.data.dataset_wrappers import (
    ANUDatasetWrapper,
    ClassicalGraphsWrapper,
    NXGraphWrapperWrapper,
)
from tmgg.experiment_utils.data.multigraph_data_module import MultiGraphDataModule
from tmgg.experiment_utils.data.pyg_datasets import PyGDatasetWrapper

# Wrapper mapping for external graph datasets
DATASET_WRAPPERS = {
    "anu": ANUDatasetWrapper,
    "classical": ClassicalGraphsWrapper,
    "nx": NXGraphWrapperWrapper,
}

# PyG benchmark datasets
PYG_DATASETS = {"qm9", "enzymes", "proteins"}


class SampledDatasetProtocol(Protocol):
    """Protocol for datasets that expose num_samples attribute."""

    num_samples: int

    def __getitem__(self, idx: int) -> torch.Tensor: ...

    def __len__(self) -> int: ...


class GraphDataModule(MultiGraphDataModule):
    """Data module for denoising experiments with multiple graph sources.

    Supports SBM, synthetic graph types (ER, regular, tree, LFR,
    ring_of_cliques), external wrapper datasets (ANU, Classical, NX),
    and PyG benchmark datasets (QM9, ENZYMES, PROTEINS).

    SBM and synthetic graph generation delegates to the inherited
    ``_generate_and_split()`` from ``MultiGraphDataModule``. Wrapper and
    PyG datasets are loaded and split via their own methods.
    """

    samples_per_graph: int
    val_samples_per_graph: int
    noise_levels: list[float]
    noise_type: str
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
            Type of graph to use (e.g., ``"sbm"``, ``"classical"``).
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
            List of noise levels for evaluation.
        noise_type
            Type of noise to apply (``"digress"``, ``"gaussian"``, etc.).
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
        self.noise_levels = noise_levels or [0.005, 0.02, 0.05, 0.1, 0.25, 0.4, 0.5]
        self.noise_type = noise_type

        # Dedicated RNG for wrapper shuffling and sample selection
        self._rng = random.Random(self.seed)

        self.train_adjacency_matrices = None
        self.val_adjacency_matrices = None
        self.test_adjacency_matrices = None

    @override
    def prepare_data(self) -> None:
        """Download external data if needed. Called once per node.

        Only relevant for wrapper datasets (ANU, Classical, NX) whose
        instantiation may trigger file downloads. SBM and synthetic
        graph generation is handled entirely in ``setup()``.
        """
        if self.graph_type in DATASET_WRAPPERS:
            wrapper_cls = DATASET_WRAPPERS[self.graph_type]
            _ = wrapper_cls(**self.graph_config)

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate or load graphs and split into train/val/test.

        Wrapper and PyG datasets have their own loading logic. SBM and
        all synthetic graph types delegate to the inherited
        ``_generate_and_split()`` from ``MultiGraphDataModule``.
        """
        if self.train_adjacency_matrices is not None:
            return

        if self.graph_type in DATASET_WRAPPERS:
            self._setup_from_wrapper()
        elif self.graph_type.lower() in PYG_DATASETS:
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

    def _setup_from_wrapper(self) -> None:
        """Load and split graphs from an external wrapper dataset."""
        if self.train_adjacency_matrices is not None:
            return

        wrapper_cls = DATASET_WRAPPERS[self.graph_type]
        dataset_wrapper = wrapper_cls(**self.graph_config)
        all_matrices = dataset_wrapper.get_adjacency_matrices()
        invalid_dtypes = [
            (i, m.dtype) for i, m in enumerate(all_matrices) if m.dtype != torch.float
        ]
        if invalid_dtypes:
            raise TypeError(
                f"Dataset wrapper returned matrices with invalid dtypes: {invalid_dtypes}. "
                f"Expected torch.float."
            )

        self._rng.shuffle(all_matrices)

        num_graphs = len(all_matrices)
        test_ratio = 1.0 - self.train_ratio - self.val_ratio
        num_test = int(test_ratio * num_graphs)
        num_val = int(self.val_ratio * num_graphs)

        self.test_adjacency_matrices = all_matrices[:num_test]
        self.val_adjacency_matrices = all_matrices[num_test : num_test + num_val]
        self.train_adjacency_matrices = all_matrices[num_test + num_val :]

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

        train, val, test = dataset.train_val_test_split(
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=seed,
        )

        self.train_adjacency_matrices = [torch.from_numpy(A).float() for A in train]
        self.val_adjacency_matrices = [torch.from_numpy(A).float() for A in val]
        self.test_adjacency_matrices = [torch.from_numpy(A).float() for A in test]

    def _create_dataloader(
        self,
        adjacency_matrices: list[torch.Tensor],
        samples_per_graph: int,
        shuffle: bool,
    ) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create a data loader with common parameters."""
        matrices: list[np.ndarray | torch.Tensor] = list(adjacency_matrices)
        dataset = GraphDataset(matrices, samples_per_graph)
        return self._make_dataloader(dataset, shuffle=shuffle)

    @override
    def train_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create training data loader."""
        if self.train_adjacency_matrices is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._create_dataloader(
            self.train_adjacency_matrices, self.samples_per_graph, shuffle=True
        )

    @override
    def val_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
        """Create validation data loader."""
        if self.val_adjacency_matrices is None:
            raise RuntimeError("Call setup() before accessing dataloaders")
        return self._create_dataloader(
            self.val_adjacency_matrices, self.val_samples_per_graph, shuffle=False
        )

    @override
    def test_dataloader(self) -> DataLoader[tuple[torch.Tensor, torch.Tensor]]:
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
    def get_dataset_info(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Return metadata about the dataset."""
        info: dict[str, Any] = {  # pyright: ignore[reportExplicitAny]
            "graph_type": self.graph_type,
            "samples_per_graph": self.samples_per_graph,
        }
        num_nodes = self.graph_config.get("num_nodes")
        if num_nodes is not None:
            info["num_nodes"] = num_nodes
        return info
