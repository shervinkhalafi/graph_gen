"""PyTorch Lightning data module for graph data."""

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from typing import List, Optional, Dict, Any
import random
import numpy as np
import networkx as nx

from tmgg.experiment_utils.data.dataset import GraphDataset
from tmgg.experiment_utils.data.sbm import generate_sbm_adjacency, generate_block_sizes
from tmgg.experiment_utils.data.dataset_wrappers import (
    ANUDatasetWrapper,
    ClassicalGraphsWrapper,
    NXGraphWrapperWrapper,
)

DATASET_WRAPPERS = {
    "anu": ANUDatasetWrapper,
    "classical": ClassicalGraphsWrapper,
    "nx": NXGraphWrapperWrapper,
}


class GraphDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for various graph datasets."""

    def __init__(self,
                 dataset_name: str,
                 dataset_config: Dict[str, Any],
                 num_samples_per_graph: int = 1000,
                 batch_size: int = 100,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 val_split: float = 0.2,
                 test_split: float = 0.2,
                 **kwargs):
        """
        Initialize the GraphDataModule.

        Args:
            dataset_name: Name of the dataset to use (e.g., "sbm", "classical").
            dataset_config: Dictionary of parameters for the chosen dataset.
            num_samples_per_graph: Number of samples (permutations) per graph.
            batch_size: Batch size for data loaders.
            num_workers: Number of worker processes for data loading.
            pin_memory: Whether to pin memory for faster GPU transfer.
            val_split: Fraction of data to use for validation (for non-SBM datasets).
            test_split: Fraction of data to use for testing (for non-SBM datasets).
        """
        super().__init__()
        self.save_hyperparameters()

        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.num_samples_per_graph = num_samples_per_graph
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
        self.test_split = test_split

        self.train_adjacency_matrices = None
        self.val_adjacency_matrices = None
        self.test_adjacency_matrices = None
        self.all_partitions = None  # For SBM

    def prepare_data(self) -> None:
        """Download or prepare data. Called only once per node."""
        if self.dataset_name == "sbm":
            sbm_params = self.dataset_config
            self.all_partitions = generate_block_sizes(
                sbm_params['num_nodes'],
                min_blocks=sbm_params.get('min_blocks', 2),
                max_blocks=sbm_params.get('max_blocks', 4),
                min_size=sbm_params.get('min_block_size', 2),
                max_size=sbm_params.get('max_block_size', 15)
            )
        elif self.dataset_name in DATASET_WRAPPERS:
            # Instantiating the wrapper might trigger downloads
            _ = DATASET_WRAPPERS[self.dataset_name](**self.dataset_config)

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up datasets for each stage."""
        if self.dataset_name == "sbm":
            self._setup_sbm(stage)
        elif self.dataset_name in DATASET_WRAPPERS:
            self._setup_from_wrapper()
        else:
            raise ValueError(f"Unknown dataset name: {self.dataset_name}")

    def _setup_sbm(self, stage: Optional[str] = None):
        sbm_params = self.dataset_config
        p_intra = sbm_params.get('p_intra', 1.0)
        q_inter = sbm_params.get('q_inter', 0.0)
        num_train_partitions = sbm_params.get('num_train_partitions', 10)
        num_test_partitions = sbm_params.get('num_test_partitions', 10)

        total_needed = num_train_partitions + num_test_partitions
        if len(self.all_partitions) < total_needed:
            raise ValueError(f"Not enough valid SBM partitions ({len(self.all_partitions)}) for "
                             f"requested train ({num_train_partitions}) and test ({num_test_partitions}) partitions.")

        if stage == "fit" or stage is None:
            if self.train_adjacency_matrices is None:
                train_partitions = random.sample(self.all_partitions, num_train_partitions)
                self.train_partitions = train_partitions  # Save for test set exclusion
                remaining_partitions = [p for p in self.all_partitions if p not in train_partitions]

                num_val_partitions = min(5, len(remaining_partitions) // 2)
                val_partitions = random.sample(remaining_partitions, num_val_partitions)

                self.train_adjacency_matrices = [
                    torch.tensor(generate_sbm_adjacency(p, p_intra, q_inter), dtype=torch.float32)
                    for p in train_partitions
                ]
                self.val_adjacency_matrices = [
                    torch.tensor(generate_sbm_adjacency(p, p_intra, q_inter), dtype=torch.float32)
                    for p in val_partitions
                ]

        if stage == "test" or stage is None:
            if self.test_adjacency_matrices is None:
                train_partitions = getattr(self, 'train_partitions', [])
                test_partitions = random.sample(
                    [p for p in self.all_partitions if p not in train_partitions],
                    num_test_partitions
                )
                self.test_adjacency_matrices = [
                    torch.tensor(generate_sbm_adjacency(p, p_intra, q_inter), dtype=torch.float32)
                    for p in test_partitions
                ]

    def _setup_from_wrapper(self):
        if self.train_adjacency_matrices is not None:
            return

        wrapper_cls = DATASET_WRAPPERS[self.dataset_name]
        dataset = wrapper_cls(**self.dataset_config)

        all_matrices = []
        for i in range(len(dataset)):
            graph_repr = dataset[i]
            if isinstance(graph_repr, tuple):
                graph_repr = graph_repr[1]

            if isinstance(graph_repr, np.ndarray):
                adj = torch.from_numpy(graph_repr).float()
            elif isinstance(graph_repr, torch.Tensor):
                adj = graph_repr.float()
            elif isinstance(graph_repr, nx.Graph):
                adj = torch.from_numpy(nx.to_numpy_array(graph_repr)).float()
            else:
                raise TypeError(f"Unsupported graph type from wrapper: {type(graph_repr)}")
            all_matrices.append(adj)

        random.shuffle(all_matrices)

        num_graphs = len(all_matrices)
        num_test = int(self.test_split * num_graphs)
        num_val = int(self.val_split * num_graphs)

        self.test_adjacency_matrices = all_matrices[:num_test]
        self.val_adjacency_matrices = all_matrices[num_test: num_test + num_val]
        self.train_adjacency_matrices = all_matrices[num_test + num_val:]

    def train_dataloader(self) -> DataLoader:
        """Create training data loader."""
        dataset = GraphDataset(
            self.train_adjacency_matrices,
            self.num_samples_per_graph
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def val_dataloader(self) -> DataLoader:
        """Create validation data loader."""
        dataset = GraphDataset(
            self.val_adjacency_matrices,
            self.num_samples_per_graph // 2
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def test_dataloader(self) -> DataLoader:
        """Create test data loader."""
        dataset = GraphDataset(
            self.test_adjacency_matrices,
            self.num_samples_per_graph // 2
        )
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.num_workers > 0
        )

    def get_sample_adjacency_matrix(self, stage: str = "train") -> torch.Tensor:
        """Get a sample adjacency matrix for visualization."""
        matrices = None
        if stage == "train":
            matrices = self.train_adjacency_matrices
        elif stage == "val":
            matrices = self.val_adjacency_matrices
        elif stage == "test":
            matrices = self.test_adjacency_matrices

        if not matrices:
            raise RuntimeError(f"No data available for stage '{stage}'. "
                             "Please ensure setup() has been called and the dataset is not empty.")

        return random.choice(matrices)
