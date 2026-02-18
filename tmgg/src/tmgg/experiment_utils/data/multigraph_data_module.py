"""Base for datamodules that generate multiple graphs and split by ratio.

Sits between ``BaseGraphDataModule`` (shared config + DataLoader factory) and
specialized leaf classes (denoising ``GraphDataModule``, discrete generative
``SyntheticCategoricalDataModule``). Provides graph generation, index-based
splitting, SBM partition-aware splitting, and a default setup that converts
adjacency arrays to float tensors served via DataLoaders.

Usable directly (e.g. as the gaussian diffusion generative datamodule) or
subclassed when a different data representation is needed.
"""

from __future__ import annotations

from typing import Any, override

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from tmgg.experiment_utils.data.base_data_module import BaseGraphDataModule
from tmgg.experiment_utils.data.sbm import (
    generate_block_sizes,
    generate_sbm_adjacency,
    generate_sbm_batch,  # pyright: ignore[reportAttributeAccessIssue]  # runtime-verified
)
from tmgg.experiment_utils.data.size_distribution import SizeDistribution
from tmgg.experiment_utils.data.synthetic_graphs import SyntheticGraphDataset


class _UnwrapDataset(Dataset[torch.Tensor]):
    """Dataset wrapper that yields individual tensors from a batched tensor."""

    def __init__(self, data: torch.Tensor) -> None:
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.data[idx]


class MultiGraphDataModule(BaseGraphDataModule):
    """Datamodule that generates N graphs and splits by ratio.

    Extends ``BaseGraphDataModule`` with graph generation parameters and
    concrete methods for adjacency generation, train/val/test splitting,
    and SBM partition-aware generation.

    The default ``setup()`` generates adjacency matrices, converts them to
    float tensors, and serves them via DataLoaders. Subclasses that need a
    different representation (e.g. categorical one-hot) override ``setup()``
    and the dataloaders.

    Parameters
    ----------
    graph_type
        Graph type to generate (``"sbm"``, ``"er"``, ``"tree"``, etc.).
    num_nodes
        Number of nodes per graph.
    num_graphs
        Total number of graphs to generate across all splits.
    train_ratio
        Fraction of graphs for training.
    val_ratio
        Fraction of graphs for validation. Remainder goes to test.
    graph_config
        Extra keyword arguments forwarded to the graph generator.
    batch_size, num_workers, pin_memory, seed
        Passed to ``BaseGraphDataModule``.
    """

    graph_type: str
    num_nodes: int
    num_graphs: int
    train_ratio: float
    val_ratio: float
    graph_config: dict[str, Any]  # pyright: ignore[reportExplicitAny]

    def __init__(
        self,
        graph_type: str = "sbm",
        num_nodes: int = 50,
        num_graphs: int = 1000,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        graph_config: dict[str, Any] | None = None,  # pyright: ignore[reportExplicitAny]
        *,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        seed: int = 42,
    ) -> None:
        super().__init__(
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            seed=seed,
        )
        self.graph_type = graph_type
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.graph_config = graph_config or {}

        self.save_hyperparameters()

        # Populated by setup()
        self._train_data: torch.Tensor | None = None
        self._val_data: torch.Tensor | None = None
        self._test_data: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Graph generation
    # ------------------------------------------------------------------

    def _generate_adjacencies(self) -> np.ndarray:
        """Generate adjacency matrices according to ``graph_type``.

        For ``"sbm"`` graphs, delegates to ``generate_sbm_batch``.
        For all other types, delegates to ``SyntheticGraphDataset``.

        Returns
        -------
        np.ndarray
            Array of shape ``(num_graphs, num_nodes, num_nodes)``.
        """
        if self.graph_type == "sbm":
            return generate_sbm_batch(
                num_graphs=self.num_graphs,
                num_nodes=self.num_nodes,
                num_blocks=self.graph_config.get("num_blocks", 2),
                p_intra=self.graph_config.get("p_intra", 0.7),
                p_inter=self.graph_config.get("p_inter", 0.1),
                seed=self.seed,
            )

        # Filter keys already passed explicitly to avoid duplicate kwargs
        extra = {
            k: v
            for k, v in self.graph_config.items()
            if k not in {"num_nodes", "num_graphs", "seed"}
        }
        dataset = SyntheticGraphDataset(
            graph_type=self.graph_type,
            n=self.num_nodes,
            num_graphs=self.num_graphs,
            seed=self.seed,
            **extra,
        )
        return dataset.get_adjacency_matrices()

    # ------------------------------------------------------------------
    # Splitting utilities
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

    def _generate_and_split(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate graphs and split into (train, val, test) adjacency arrays.

        For SBM with ``partition_mode="enumerated"`` in ``graph_config``:
        enumerates valid block-size partitions, assigns different partitions
        per split, generates one graph per partition. For everything else:
        generates all graphs, splits by index ratio.

        Returns
        -------
        tuple of np.ndarray
            ``(train, val, test)`` arrays, each of shape
            ``(n_split, num_nodes, num_nodes)``.
        """
        partition_mode = self.graph_config.get("partition_mode")

        # Auto-detect partition mode for SBM when not explicitly set.
        # Presence of num_train_partitions triggers enumerated mode (each
        # partition → one graph, different partitions per split).
        if (
            self.graph_type == "sbm"
            and partition_mode is None
            and "num_train_partitions" in self.graph_config
        ):
            partition_mode = "enumerated"

        if self.graph_type == "sbm" and partition_mode in ("enumerated", "fixed"):
            return self._generate_sbm_partitioned()

        adjacencies = self._generate_adjacencies()
        train_idx, val_idx, test_idx = self._split_indices(
            len(adjacencies), self.train_ratio, self.val_ratio, self.seed
        )
        return adjacencies[train_idx], adjacencies[val_idx], adjacencies[test_idx]

    def _generate_sbm_partitioned(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate SBM graphs with partition-aware splitting.

        In ``"fixed"`` mode, uses explicit ``block_sizes`` from config.
        All splits use the same partition but different random draws.

        In ``"enumerated"`` mode, enumerates all valid partitions via
        ``generate_block_sizes()``, samples ``num_train_partitions`` for
        training and ``num_test_partitions`` for testing (non-overlapping).
        Validation gets a subset of the remaining partitions. Each partition
        produces one graph.

        Returns
        -------
        tuple of np.ndarray
            ``(train, val, test)`` arrays of shape
            ``(n_split, num_nodes, num_nodes)``.
        """
        rng = np.random.default_rng(self.seed)
        cfg = self.graph_config
        p_intra: float = cfg.get("p_intra", 0.7)
        p_inter: float = cfg.get("p_inter", 0.1)
        partition_mode = cfg.get("partition_mode", "equal")

        if partition_mode == "fixed":
            block_sizes = cfg["block_sizes"]
            # Same partition, different random draws per split
            train = np.array(
                [generate_sbm_adjacency(block_sizes, p_intra, p_inter, rng=rng)]
            )
            val = np.array(
                [generate_sbm_adjacency(block_sizes, p_intra, p_inter, rng=rng)]
            )
            test = np.array(
                [generate_sbm_adjacency(block_sizes, p_intra, p_inter, rng=rng)]
            )
            return train, val, test

        # Enumerated mode: generate all valid partitions, sample per split
        all_partitions = generate_block_sizes(
            cfg.get("num_nodes", self.num_nodes),
            min_blocks=cfg.get("min_blocks", 2),
            max_blocks=cfg.get("max_blocks", 4),
            min_size=cfg.get("min_block_size", 2),
            max_size=cfg.get("max_block_size", 15),
        )

        num_train = cfg.get("num_train_partitions", 10)
        num_test = cfg.get("num_test_partitions", 10)
        total_needed = num_train + num_test

        if len(all_partitions) < total_needed:
            raise ValueError(
                f"Not enough valid SBM partitions ({len(all_partitions)}) for "
                f"requested train ({num_train}) and test ({num_test}) partitions."
            )

        # Use python random.Random for partition sampling (needs .sample on list)
        import random

        py_rng = random.Random(self.seed)
        train_partitions = py_rng.sample(all_partitions, num_train)
        remaining = [p for p in all_partitions if p not in train_partitions]
        num_val = min(5, max(1, len(remaining) // 2))
        val_partitions = py_rng.sample(remaining, num_val)
        test_remaining = [p for p in remaining if p not in val_partitions]
        test_partitions = py_rng.sample(test_remaining, num_test)

        def _gen_from_partitions(
            partitions: list[list[int]],
        ) -> np.ndarray:
            return np.array(
                [
                    generate_sbm_adjacency(p, p_intra, p_inter, rng=rng)
                    for p in partitions
                ]
            )

        return (
            _gen_from_partitions(train_partitions),
            _gen_from_partitions(val_partitions),
            _gen_from_partitions(test_partitions),
        )

    # ------------------------------------------------------------------
    # Size distribution
    # ------------------------------------------------------------------

    def get_size_distribution(self, split: str | None = None) -> SizeDistribution:
        """Return the graph size distribution for a split or the whole dataset.

        Parameters
        ----------
        split
            ``"train"``, ``"val"``, ``"test"``, or ``None`` for the
            whole dataset (pre-split).

        Returns
        -------
        SizeDistribution
            Default implementation returns a degenerate distribution
            (all graphs have ``num_nodes`` nodes). Subclasses handling
            variable-size data should override this method.
        """
        return SizeDistribution.fixed(self.num_nodes)

    # ------------------------------------------------------------------
    # Default setup and DataLoaders (tensor-serving)
    # ------------------------------------------------------------------

    @override
    def setup(self, stage: str | None = None) -> None:
        """Generate graphs, split, and convert to float tensors.

        Idempotent: calling setup multiple times is safe.

        Parameters
        ----------
        stage
            Lightning stage (``"fit"``, ``"test"``, etc.) or None.
        """
        if self._train_data is not None:
            return

        train, val, test = self._generate_and_split()

        self._train_data = torch.from_numpy(train).float()
        self._val_data = torch.from_numpy(val).float()
        self._test_data = torch.from_numpy(test).float()

    @override
    def train_dataloader(self) -> DataLoader[torch.Tensor]:
        """Create training dataloader."""
        if self._train_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self._make_dataloader(_UnwrapDataset(self._train_data), shuffle=True)

    @override
    def val_dataloader(self) -> DataLoader[torch.Tensor]:
        """Create validation dataloader."""
        if self._val_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self._make_dataloader(_UnwrapDataset(self._val_data), shuffle=False)

    @override
    def test_dataloader(self) -> DataLoader[torch.Tensor]:
        """Create test dataloader."""
        if self._test_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self._make_dataloader(_UnwrapDataset(self._test_data), shuffle=False)

    @override
    def get_dataset_info(self) -> dict[str, Any]:  # pyright: ignore[reportExplicitAny]
        """Return metadata about the dataset.

        Returns
        -------
        dict
            Keys: ``num_graphs``, ``num_nodes``, ``graph_type``.
        """
        return {
            "num_graphs": self.num_graphs,
            "num_nodes": self.num_nodes,
            "graph_type": self.graph_type,
        }

    def get_sample_graph(self) -> torch.Tensor:
        """Get a sample graph for visualization."""
        if self._train_data is None:
            raise RuntimeError("DataModule not setup. Call setup() first.")
        return self._train_data[0]
