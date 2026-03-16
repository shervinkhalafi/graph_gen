"""PyTorch Lightning data module for denoising graph experiments.

Thin subclass of ``MultiGraphDataModule`` that adds PyG benchmark dataset
support and ``samples_per_graph`` list repetition. All storage and
dataloader creation is delegated to the parent; this class only extends
``setup()`` for PyG datasets and repeats the ``list[Data]`` splits.
"""

# pyright: reportExplicitAny=false
# DataLoader/Dataset generic parameters and config dicts require Any
# until PyTorch provides complete generic stubs.

from __future__ import annotations

import random
from typing import Any, override

import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from tmgg.data._split import split_indices
from tmgg.data.datasets.pyg_datasets import PyGDatasetWrapper

from .multigraph_data_module import MultiGraphDataModule

# PyG benchmark datasets
PYG_DATASETS = {"qm9", "enzymes", "proteins"}


class GraphDataModule(MultiGraphDataModule):
    """Data module for denoising experiments with multiple graph sources.

    Supports SBM, all synthetic graph types (ER, regular, tree, LFR,
    ring_of_cliques, lollipop, circular_ladder, star, square_grid,
    triangle_grid), and PyG benchmark datasets (QM9, ENZYMES, PROTEINS).

    SBM and synthetic graph generation delegates to the parent's
    ``setup()`` (which calls ``_generate_and_split()``). PyG datasets
    are loaded via ``_setup_pyg_dataset()``. In both cases the result
    is ``list[Data]`` in ``_train_data``/``_val_data``/``_test_data``,
    optionally repeated by ``samples_per_graph``.
    """

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
            Number of samples (repetitions) per graph for training.
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
        # Dedicated RNG for sample selection in get_sample_adjacency_matrix
        self._rng = random.Random(self.seed)

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate or load graphs and split into train/val/test.

        PyG datasets have their own loading logic. SBM and all synthetic
        graph types delegate to the parent's ``setup()`` (which calls
        ``_generate_and_split()``). After population, lists are repeated
        by ``samples_per_graph`` / ``val_samples_per_graph``.
        """
        if self._train_data is not None:
            return

        if self.graph_type.lower() in PYG_DATASETS:
            self._setup_pyg_dataset()
        else:
            super().setup(stage)

        # Repeat graphs for samples_per_graph
        if self._train_data is not None and self.samples_per_graph > 1:
            self._train_data = self._train_data * self.samples_per_graph
        if self._val_data is not None and self.val_samples_per_graph > 1:
            self._val_data = self._val_data * self.val_samples_per_graph
        if self._test_data is not None and self.val_samples_per_graph > 1:
            self._test_data = self._test_data * self.val_samples_per_graph

    def _setup_pyg_dataset(self) -> None:
        """Load and split graphs from a PyTorch Geometric dataset."""
        root: str | None = self.graph_config.get("root", None)
        max_graphs: int | None = self.graph_config.get("max_graphs", None)
        seed: int = self.graph_config.get("seed", 42)

        wrapper = PyGDatasetWrapper(
            dataset_name=self.graph_type,
            root=root,
            max_graphs=max_graphs,
        )

        train_idx, val_idx, test_idx = split_indices(
            len(wrapper), self.train_ratio, self.val_ratio, seed
        )

        self._train_data = [wrapper.data_list[i] for i in train_idx]
        self._val_data = [wrapper.data_list[i] for i in val_idx]
        self._test_data = [wrapper.data_list[i] for i in test_idx]

    def get_sample_adjacency_matrix(self, stage: str = "train") -> torch.Tensor:
        """Get a sample adjacency matrix for visualization.

        Parameters
        ----------
        stage
            One of ``"train"``, ``"val"``, ``"test"``.

        Returns
        -------
        torch.Tensor
            Dense adjacency matrix of shape ``(num_nodes, num_nodes)``.
        """
        data_list: list[Data] | None = None
        if stage == "train":
            data_list = self._train_data
        elif stage == "val":
            data_list = self._val_data
        elif stage == "test":
            data_list = self._test_data

        if not data_list:
            raise RuntimeError(
                f"No data available for stage '{stage}'. "
                "Please ensure setup() has been called and the dataset is not empty."
            )

        sample = self._rng.choice(data_list)
        if sample.edge_index is None:
            raise RuntimeError("Sample graph has no edge_index.")
        adj = to_dense_adj(sample.edge_index, max_num_nodes=sample.num_nodes)
        return adj.squeeze(0)

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
