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

from typing import Any, override

from .graph_generation import load_pyg_dataset_split, uses_pyg_dataset_split
from .multigraph_data_module import MultiGraphDataModule


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
        seed: int = 42,
        num_nodes_max_static: int | None = None,
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
        seed
            Random seed for reproducible splitting and generation.
        num_nodes_max_static
            Safe upper bound on the per-graph node count across the
            dataset. Exposed to model configs via Hydra interpolation
            (e.g. ``${data.num_nodes_max_static}``) so dimensioning
            decisions like ``ExtraFeatures(max_n_nodes=...)`` get a
            single source of truth instead of hardcoding numbers that
            silently drift with the data preset (parity audit D-11
            / #42). When ``None``, falls back to
            ``graph_config["num_nodes"]``.
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
        self.num_nodes_max_static = (
            num_nodes_max_static
            if num_nodes_max_static is not None
            else int(graph_config.get("num_nodes", 50))
        )
        self.save_hyperparameters()

        self.samples_per_graph = samples_per_graph
        self.val_samples_per_graph = (
            val_samples_per_graph
            if val_samples_per_graph is not None
            else samples_per_graph // 2
        )

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

        if uses_pyg_dataset_split(self.graph_type):
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
        self._train_data, self._val_data, self._test_data = load_pyg_dataset_split(
            graph_type=self.graph_type,
            graph_config=self.graph_config,
            train_ratio=self.train_ratio,
            val_ratio=self.val_ratio,
            seed=self.seed,
        )
