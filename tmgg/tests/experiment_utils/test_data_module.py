"""Tests for GraphDataModule.

This module tests the GraphDataModule class which handles loading and
preparation of graph datasets for training.

Testing Strategy:
- Test initialization with different dataset types (SBM, wrappers, synthetic)
- Test prepare_data/setup lifecycle
- Test dataloader creation and properties
- Use real small datasets where practical; mock expensive operations

Key Invariants:
- prepare_data must be called before setup
- setup must be called before accessing dataloaders
- Train/val/test splits should be disjoint
- Dataloader batch sizes should match configuration
"""

from __future__ import annotations

import pytest
import torch

from tmgg.experiment_utils.data.data_module import (
    GraphDataModule,
)


class TestGraphDataModuleInit:
    """Tests for GraphDataModule initialization."""

    def test_default_parameters(self) -> None:
        """Should initialize with sensible defaults.

        Rationale: Default parameters should be reasonable for most use cases
        without requiring explicit configuration.
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 10, "block_sizes": [5, 5]},
        )

        assert dm.num_samples_per_graph == 1000
        assert dm.batch_size == 100
        assert dm.num_workers == 4
        assert dm.pin_memory is True
        assert dm.val_split == 0.2
        assert dm.test_split == 0.2

    def test_custom_parameters(self) -> None:
        """Should accept custom parameters.

        Rationale: Users should be able to override all configurable parameters.
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 10, "block_sizes": [5, 5]},
            num_samples_per_graph=500,
            batch_size=50,
            num_workers=2,
            pin_memory=False,
            val_split=0.1,
            test_split=0.1,
        )

        assert dm.num_samples_per_graph == 500
        assert dm.batch_size == 50
        assert dm.num_workers == 2
        assert dm.pin_memory is False
        assert dm.val_split == 0.1
        assert dm.test_split == 0.1

    def test_default_noise_levels(self) -> None:
        """Should have default noise levels when not specified.

        Rationale: Noise levels are needed for evaluation; reasonable defaults
        should be provided.
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 10, "block_sizes": [5, 5]},
        )

        assert dm.noise_levels is not None
        assert len(dm.noise_levels) > 0
        assert all(0 < n <= 1 for n in dm.noise_levels)

    def test_custom_noise_levels(self) -> None:
        """Should accept custom noise levels."""
        custom_levels = [0.1, 0.2, 0.3]
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 10, "block_sizes": [5, 5]},
            noise_levels=custom_levels,
        )

        assert dm.noise_levels == custom_levels


class TestGraphDataModuleSBM:
    """Tests for SBM dataset setup."""

    def test_sbm_setup_creates_splits(self) -> None:
        """SBM mode should create train/val/test matrices.

        Rationale: After setup(), all three splits should be populated
        with adjacency matrices.
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={
                "num_nodes": 10,
                "block_sizes": [5, 5],
                "p_intra": 0.8,
                "q_inter": 0.1,
            },
        )
        dm.prepare_data()
        dm.setup()

        assert dm.train_adjacency_matrices is not None
        assert dm.val_adjacency_matrices is not None
        assert dm.test_adjacency_matrices is not None
        assert len(dm.train_adjacency_matrices) > 0
        assert len(dm.val_adjacency_matrices) > 0
        assert len(dm.test_adjacency_matrices) > 0

    def test_sbm_fixed_block_sizes(self) -> None:
        """Fixed block_sizes mode should use specified sizes.

        Rationale: When block_sizes is explicitly provided, the SBM
        should use that exact partition structure.
        """
        block_sizes = [3, 4, 3]
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={
                "num_nodes": 10,
                "block_sizes": block_sizes,
                "p_intra": 1.0,
                "q_inter": 0.0,
            },
        )
        dm.prepare_data()
        dm.setup()

        # Verify the adjacency matrix has correct size
        assert dm.train_adjacency_matrices is not None
        A = dm.train_adjacency_matrices[0]
        assert A.shape == (10, 10)

    def test_sbm_random_partitions(self) -> None:
        """Random partitions mode should generate valid splits.

        Rationale: When block_sizes is not provided, the module should
        generate random partitions based on min/max constraints.
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={
                "num_nodes": 20,
                "num_train_partitions": 5,
                "num_test_partitions": 3,
                "min_blocks": 2,
                "max_blocks": 4,
                "min_block_size": 3,
                "max_block_size": 8,
                "p_intra": 0.8,
                "q_inter": 0.1,
            },
        )
        dm.prepare_data()
        dm.setup()

        # Should have different partitions for train/test
        assert dm.train_adjacency_matrices is not None
        assert dm.test_adjacency_matrices is not None
        assert len(dm.train_adjacency_matrices) == 5
        assert len(dm.test_adjacency_matrices) == 3

    def test_sbm_prepare_before_setup_required(self) -> None:
        """setup() should fail if prepare_data() wasn't called.

        Rationale: The two-stage initialization requires prepare_data
        to run first to generate partitions.
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 10, "block_sizes": [5, 5]},
        )

        with pytest.raises(RuntimeError, match="prepare_data.*must be called"):
            dm.setup()


class TestGraphDataModuleSynthetic:
    """Tests for synthetic graph dataset setup."""

    def test_synthetic_er_setup(self) -> None:
        """Should setup Erdos-Renyi graph dataset.

        Rationale: ER graphs should be generated with the specified
        probability parameter and split correctly.
        """
        dm = GraphDataModule(
            dataset_name="er",
            dataset_config={
                "num_nodes": 15,
                "num_graphs": 30,
                "p": 0.2,
                "seed": 42,
            },
            val_split=0.2,
            test_split=0.2,
        )
        dm.prepare_data()
        dm.setup()

        assert dm.train_adjacency_matrices is not None
        assert dm.val_adjacency_matrices is not None
        assert dm.test_adjacency_matrices is not None
        # 60% train, 20% val, 20% test of 30 graphs
        total = (
            len(dm.train_adjacency_matrices)
            + len(dm.val_adjacency_matrices)
            + len(dm.test_adjacency_matrices)
        )
        assert total == 30

        # Check matrix properties
        A = dm.train_adjacency_matrices[0]
        assert A.shape == (15, 15)
        assert torch.allclose(A, A.T)  # Symmetric
        assert A.diag().sum() == 0  # No self-loops

    def test_synthetic_regular_setup(self) -> None:
        """Should setup d-regular graph dataset.

        Rationale: Regular graphs should have consistent degree structure.
        """
        dm = GraphDataModule(
            dataset_name="regular",
            dataset_config={
                "num_nodes": 12,
                "num_graphs": 20,
                "d": 3,
                "seed": 42,
            },
        )
        dm.prepare_data()
        dm.setup()

        assert dm.train_adjacency_matrices is not None

        # Check regular graph property (all degrees = d)
        A = dm.train_adjacency_matrices[0]
        degrees = A.sum(dim=1)
        assert torch.allclose(degrees, torch.full_like(degrees, 3.0))


class TestGraphDataModuleDataLoaders:
    """Tests for dataloader creation."""

    def test_train_dataloader_batch_size(self) -> None:
        """Train dataloader should use configured batch_size.

        Rationale: The batch size should match the configuration for
        proper training throughput.
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 8, "block_sizes": [4, 4]},
            batch_size=16,
            num_samples_per_graph=32,
            num_workers=0,  # Avoid multiprocessing in tests
        )
        dm.prepare_data()
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        # Batch is a tensor of shape (batch_size, n, n)
        assert isinstance(batch, torch.Tensor)
        assert batch.shape[0] == 16
        assert batch.shape[1] == batch.shape[2] == 8

    def test_val_dataloader_default_samples(self) -> None:
        """Val/test dataloaders default to half of num_samples_per_graph.

        Rationale: When val_samples_per_graph is not specified, it defaults
        to num_samples_per_graph // 2.
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 8, "block_sizes": [4, 4]},
            num_samples_per_graph=100,
            batch_size=10,
            num_workers=0,
        )
        dm.prepare_data()
        dm.setup()

        # Train should have 100 samples
        train_loader = dm.train_dataloader()
        train_dataset = train_loader.dataset

        # Val should have 50 samples (default: half)
        val_loader = dm.val_dataloader()
        val_dataset = val_loader.dataset

        # GraphDataset stores total samples as num_samples
        assert getattr(train_dataset, "num_samples", None) == 100
        assert getattr(val_dataset, "num_samples", None) == 50

    def test_val_samples_per_graph_explicit(self) -> None:
        """val_samples_per_graph can be set explicitly.

        Rationale: Users should be able to override the default half-samples
        behavior for validation/test dataloaders.
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 8, "block_sizes": [4, 4]},
            num_samples_per_graph=100,
            val_samples_per_graph=75,  # Explicit value
            batch_size=10,
            num_workers=0,
        )
        dm.prepare_data()
        dm.setup()

        val_loader = dm.val_dataloader()
        val_dataset = val_loader.dataset

        assert getattr(val_dataset, "num_samples", None) == 75

    def test_dataloader_without_setup_raises(self) -> None:
        """Calling dataloader before setup should raise RuntimeError.

        Rationale: Dataloaders require adjacency matrices to be loaded,
        which happens in setup().
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 8, "block_sizes": [4, 4]},
        )

        with pytest.raises(RuntimeError, match="Call setup.*before"):
            dm.train_dataloader()

        with pytest.raises(RuntimeError, match="Call setup.*before"):
            dm.val_dataloader()

        with pytest.raises(RuntimeError, match="Call setup.*before"):
            dm.test_dataloader()

    def test_dataloader_persistent_workers(self) -> None:
        """Dataloaders should have persistent_workers when num_workers > 0.

        Rationale: Persistent workers improve performance by keeping
        worker processes alive between epochs.
        """
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 8, "block_sizes": [4, 4]},
            num_workers=2,
            num_samples_per_graph=20,
        )
        dm.prepare_data()
        dm.setup()

        loader = dm.train_dataloader()
        assert loader.persistent_workers is True


class TestGraphDataModuleUnknownDataset:
    """Tests for error handling with unknown datasets."""

    def test_unknown_dataset_raises(self) -> None:
        """Unknown dataset name should raise ValueError.

        Rationale: Clear error message helps users identify typos
        or unsupported datasets.
        """
        dm = GraphDataModule(
            dataset_name="nonexistent_dataset",
            dataset_config={},
        )
        dm.prepare_data()  # This might not fail yet

        with pytest.raises(ValueError, match="Unknown dataset name"):
            dm.setup()


class TestGraphDataModuleSampleAdjacency:
    """Tests for get_sample_adjacency_matrix method."""

    def test_get_sample_train(self) -> None:
        """Should return a sample from train split."""
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 8, "block_sizes": [4, 4]},
        )
        dm.prepare_data()
        dm.setup()

        sample = dm.get_sample_adjacency_matrix("train")

        assert isinstance(sample, torch.Tensor)
        assert sample.shape == (8, 8)

    def test_get_sample_without_setup_raises(self) -> None:
        """Should raise RuntimeError if setup not called."""
        dm = GraphDataModule(
            dataset_name="sbm",
            dataset_config={"num_nodes": 8, "block_sizes": [4, 4]},
        )

        with pytest.raises(RuntimeError):
            dm.get_sample_adjacency_matrix("train")
