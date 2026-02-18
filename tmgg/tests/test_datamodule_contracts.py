"""Regression tests for all 4 datamodule classes.

These tests verify the full lifecycle (init → setup → dataloader → batch) for
each datamodule, focusing on the external contracts that consumers rely on. They
serve as a safety net for the BaseGraphDataModule unification refactoring: any
change that breaks the batch format, split sizes, or metadata contracts should
cause a failure here.

The existing test files (test_data_module.py, test_single_graph_datasets.py,
test_categorical_datamodule.py, test_generative_integration.py) cover each
module in depth. This file focuses on cross-cutting contract verification.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tmgg.experiment_utils.data.data_module import GraphDataModule
from tmgg.experiment_utils.data.multigraph_data_module import MultiGraphDataModule
from tmgg.experiment_utils.data.single_graph_data_module import SingleGraphDataModule
from tmgg.experiments.discrete_diffusion_generative.datamodule import (
    SyntheticCategoricalDataModule,
)

# ---------------------------------------------------------------------------
# GraphDataModule lifecycle
# ---------------------------------------------------------------------------


class TestGraphDataModuleContract:
    """Full lifecycle for GraphDataModule (denoising, multi-source)."""

    def test_sbm_fixed_lifecycle(self) -> None:
        """SBM with fixed block_sizes: init → prepare → setup → batch.

        Rationale: Fixed-partition SBM is the most common denoising config.
        The batch is a plain Tensor of shape (bs, n, n).
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 12,
                "block_sizes": [6, 6],
                "p_intra": 0.8,
                "p_inter": 0.1,
            },
            samples_per_graph=16,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.prepare_data()
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (4, 12, 12)
        assert batch.dtype == torch.float32

        val_batch = next(iter(dm.val_dataloader()))
        assert isinstance(val_batch, torch.Tensor)
        assert val_batch.shape[1:] == (12, 12)

        test_batch = next(iter(dm.test_dataloader()))
        assert isinstance(test_batch, torch.Tensor)
        assert test_batch.shape[1:] == (12, 12)

    def test_sbm_enumerated_lifecycle(self) -> None:
        """SBM with enumerated partitions: different partitions per split.

        Rationale: Enumerated partitions are the denoising SBM variant that
        tests generalization to unseen community structures. Train and test
        should receive different adjacency matrices.
        """
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={
                "num_nodes": 20,
                "partition_mode": "enumerated",
                "num_train_partitions": 3,
                "num_test_partitions": 2,
                "min_blocks": 2,
                "max_blocks": 4,
                "min_block_size": 3,
                "max_block_size": 8,
                "p_intra": 0.8,
                "p_inter": 0.1,
            },
            samples_per_graph=8,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.prepare_data()
        dm.setup()

        assert dm.train_adjacency_matrices is not None  # pyright: ignore[reportAttributeAccessIssue]
        assert dm.test_adjacency_matrices is not None  # pyright: ignore[reportAttributeAccessIssue]
        assert len(dm.train_adjacency_matrices) == 3  # pyright: ignore[reportAttributeAccessIssue]
        assert len(dm.test_adjacency_matrices) == 2  # pyright: ignore[reportAttributeAccessIssue]

        batch = next(iter(dm.train_dataloader()))
        assert batch.shape == (4, 20, 20)

    def test_er_lifecycle(self) -> None:
        """Erdos-Renyi: synthetic graph generation through SyntheticGraphDataset."""
        dm = GraphDataModule(
            graph_type="er",
            graph_config={"num_nodes": 10, "num_graphs": 20, "p": 0.3, "seed": 42},
            batch_size=4,
            num_workers=0,
            samples_per_graph=8,
        )
        dm.prepare_data()
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, torch.Tensor)
        assert batch.shape[1:] == (10, 10)

    def test_noise_levels_accessible(self) -> None:
        """Denoising LightningModules access datamodule.noise_levels."""
        dm = GraphDataModule(
            graph_type="sbm",
            graph_config={"num_nodes": 8, "block_sizes": [4, 4]},
            noise_levels=[0.01, 0.1, 0.5],
        )
        assert dm.noise_levels == [0.01, 0.1, 0.5]  # pyright: ignore[reportAttributeAccessIssue]


# ---------------------------------------------------------------------------
# SingleGraphDataModule lifecycle
# ---------------------------------------------------------------------------


class TestSingleGraphDataModuleContract:
    """Full lifecycle for SingleGraphDataModule (single-graph denoising)."""

    def test_sbm_same_graph_lifecycle(self) -> None:
        """SBM with same_graph_all_splits=True: all splits use identical graph.

        Rationale: Stage 1 protocol — model sees only noise variation.
        """
        dm = SingleGraphDataModule(
            graph_type="sbm",
            num_nodes=20,
            graph_config={"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 2},
            same_graph_all_splits=True,
            num_train_samples=16,
            num_val_samples=8,
            num_test_samples=8,
            batch_size=4,
            num_workers=0,
        )
        dm.setup()

        # All splits should return the same graph
        assert np.array_equal(dm.get_train_graph(), dm.get_val_graph())
        assert np.array_equal(dm.get_train_graph(), dm.get_test_graph())

        batch = next(iter(dm.train_dataloader()))
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (4, 20, 20)
        assert batch.dtype == torch.float32

    def test_er_different_graphs_lifecycle(self) -> None:
        """ER with same_graph_all_splits=False: different graphs per split.

        Rationale: Stage 2+ protocol — test generalization to new structures.
        """
        dm = SingleGraphDataModule(
            graph_type="erdos_renyi",
            num_nodes=15,
            graph_config={"p": 0.2},
            same_graph_all_splits=False,
            num_train_samples=8,
            num_val_samples=4,
            num_test_samples=4,
            batch_size=2,
            num_workers=0,
        )
        dm.setup()

        assert not np.array_equal(dm.get_train_graph(), dm.get_val_graph())

        batch = next(iter(dm.train_dataloader()))
        assert batch.shape == (2, 15, 15)

    def test_noise_levels_accessible(self) -> None:
        """Denoising LightningModules access datamodule.noise_levels."""
        dm = SingleGraphDataModule(
            graph_type="sbm",
            num_nodes=10,
            graph_config={"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 2},
            noise_levels=[0.01, 0.05, 0.1],
        )
        assert dm.noise_levels == [0.01, 0.05, 0.1]

    def test_graph_properties(self) -> None:
        """Generated graphs should be symmetric, binary, with zero diagonal."""
        dm = SingleGraphDataModule(
            graph_type="sbm",
            num_nodes=20,
            graph_config={"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3},
            same_graph_all_splits=True,
        )
        dm.setup()

        A = dm.get_train_graph()
        assert A.shape == (20, 20)
        assert np.allclose(A, A.T), "Should be symmetric"
        assert set(np.unique(A)).issubset({0.0, 1.0}), "Should be binary"
        assert np.allclose(np.diag(A), 0), "No self-loops"


# ---------------------------------------------------------------------------
# MultiGraphDataModule lifecycle
# ---------------------------------------------------------------------------


class TestMultiGraphDataModuleContract:
    """Full lifecycle for MultiGraphDataModule (gaussian generative)."""

    def test_sbm_lifecycle(self) -> None:
        """SBM: init → setup → all 3 dataloaders → batch shape.

        Rationale: The gaussian diffusion generative module serves plain
        Tensor batches of shape (bs, n, n), same as denoising.
        """
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=30,
            train_ratio=0.8,
            val_ratio=0.1,
            batch_size=4,
            num_workers=0,
            seed=42,
            graph_config={"num_blocks": 2, "p_intra": 0.7, "p_inter": 0.1},
        )
        dm.setup()

        # Train dataloader
        train_batch = next(iter(dm.train_dataloader()))
        assert isinstance(train_batch, torch.Tensor)
        assert train_batch.shape == (4, 16, 16)
        assert train_batch.dtype == torch.float32

        # Val dataloader
        val_batch = next(iter(dm.val_dataloader()))
        assert isinstance(val_batch, torch.Tensor)
        assert val_batch.shape[1:] == (16, 16)

        # Test dataloader
        test_batch = next(iter(dm.test_dataloader()))
        assert isinstance(test_batch, torch.Tensor)
        assert test_batch.shape[1:] == (16, 16)

    def test_er_lifecycle(self) -> None:
        """ER graphs through the generative pipeline."""
        dm = MultiGraphDataModule(
            graph_type="er",
            num_nodes=12,
            num_graphs=20,
            batch_size=4,
            num_workers=0,
            seed=42,
            graph_config={"p": 0.3},
        )
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        assert batch.shape == (4, 12, 12)

    def test_split_sizes(self) -> None:
        """Train/val/test split sizes should match ratios."""
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=10,
            num_graphs=100,
            train_ratio=0.8,
            val_ratio=0.1,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.setup()

        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._val_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._test_data is not None  # pyright: ignore[reportPrivateUsage]
        assert dm._train_data.shape[0] == 80  # pyright: ignore[reportPrivateUsage]
        assert dm._val_data.shape[0] == 10  # pyright: ignore[reportPrivateUsage]
        assert dm._test_data.shape[0] == 10  # pyright: ignore[reportPrivateUsage]

    def test_graph_validity(self) -> None:
        """Generated graphs should be binary, symmetric, zero-diagonal."""
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=10,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.setup()

        assert dm._train_data is not None  # pyright: ignore[reportPrivateUsage]
        data = dm._train_data  # pyright: ignore[reportPrivateUsage]
        assert torch.all((data == 0) | (data == 1))
        assert torch.allclose(data, data.transpose(-2, -1))
        for i in range(data.shape[0]):
            assert torch.all(data[i].diagonal() == 0)

    def test_get_dataset_info(self) -> None:
        """get_dataset_info() should return expected metadata."""
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=50,
        )
        info = dm.get_dataset_info()
        assert info["num_graphs"] == 50
        assert info["num_nodes"] == 16
        assert info["graph_type"] == "sbm"

    def test_idempotent_setup(self) -> None:
        """Calling setup() twice should not regenerate data."""
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=10,
            num_graphs=20,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.setup()
        train_before = dm._train_data  # pyright: ignore[reportPrivateUsage]
        dm.setup()
        assert dm._train_data is train_before  # pyright: ignore[reportPrivateUsage]

    def test_setup_required_for_dataloaders(self) -> None:
        """Accessing dataloaders before setup() should raise RuntimeError."""
        dm = MultiGraphDataModule(
            graph_type="sbm",
            num_nodes=10,
            num_graphs=20,
        )
        with pytest.raises(RuntimeError, match="not setup|not set up"):
            dm.train_dataloader()
        with pytest.raises(RuntimeError, match="not setup|not set up"):
            dm.val_dataloader()
        with pytest.raises(RuntimeError, match="not setup|not set up"):
            dm.test_dataloader()


# ---------------------------------------------------------------------------
# SyntheticCategoricalDataModule lifecycle
# ---------------------------------------------------------------------------


class TestSyntheticCategoricalDataModuleContract:
    """Full lifecycle for SyntheticCategoricalDataModule (discrete generative)."""

    def test_sbm_lifecycle(self) -> None:
        """SBM: init → setup → dataloader → (X, E, y, node_mask) batch.

        Rationale: Discrete diffusion requires categorical tuple batches,
        NOT plain tensors. This is the fundamental format difference.
        """
        dm = SyntheticCategoricalDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=30,
            train_ratio=0.8,
            val_ratio=0.1,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        X, E, y, node_mask = batch

        assert X.shape == (4, 16, 2)  # (bs, n, dx)
        assert E.shape == (4, 16, 16, 2)  # (bs, n, n, de)
        assert y.shape == (4, 0)  # (bs, 0) — no global features
        assert node_mask.shape == (4, 16)  # (bs, n)
        assert node_mask.dtype == torch.bool

    def test_marginals_contract(self) -> None:
        """node_marginals and edge_marginals are valid probability vectors.

        Rationale: The discrete diffusion LightningModule reads these
        during its own setup() to construct transition matrices.
        """
        dm = SyntheticCategoricalDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=50,
            batch_size=4,
            num_workers=0,
            seed=42,
        )
        dm.setup()

        nm = dm.node_marginals
        em = dm.edge_marginals

        assert nm.shape == (2,)
        assert em.shape == (2,)
        assert torch.allclose(nm.sum(), torch.tensor(1.0), atol=1e-6)
        assert torch.allclose(em.sum(), torch.tensor(1.0), atol=1e-6)
        assert (nm >= 0).all()
        assert (em >= 0).all()

    def test_get_dataset_info(self) -> None:
        """get_dataset_info() should return categorical metadata."""
        dm = SyntheticCategoricalDataModule(
            graph_type="sbm",
            num_nodes=16,
            num_graphs=50,
        )
        info = dm.get_dataset_info()
        assert info["num_graphs"] == 50
        assert info["num_nodes"] == 16
        assert info["num_node_classes"] == 2
        assert info["num_edge_classes"] == 2

    def test_er_lifecycle(self) -> None:
        """Non-SBM graph types should also produce valid categorical data."""
        dm = SyntheticCategoricalDataModule(
            graph_type="er",
            num_nodes=12,
            num_graphs=20,
            batch_size=4,
            num_workers=0,
            seed=42,
            graph_config={"p": 0.3},
        )
        dm.setup()

        batch = next(iter(dm.train_dataloader()))
        X, E, y, node_mask = batch
        assert X.shape[1:] == (12, 2)
        assert E.shape[1:] == (12, 12, 2)


# ---------------------------------------------------------------------------
# Cross-cutting: reproducibility
# ---------------------------------------------------------------------------


class TestReproducibility:
    """Verify that all datamodules produce identical output given identical seeds."""

    def test_graph_distribution_reproducible(self) -> None:
        """Two MultiGraphDataModule instances with the same seed
        should produce identical training data."""

        def _make_gdm() -> MultiGraphDataModule:
            return MultiGraphDataModule(
                graph_type="sbm",
                num_nodes=10,
                num_graphs=20,
                batch_size=4,
                num_workers=0,
                seed=42,
            )

        dm1 = _make_gdm()
        dm1.setup()
        dm2 = _make_gdm()
        dm2.setup()

        assert dm1._train_data is not None and dm2._train_data is not None  # pyright: ignore[reportPrivateUsage]
        assert torch.equal(dm1._train_data, dm2._train_data)  # pyright: ignore[reportPrivateUsage]

    def test_categorical_reproducible(self) -> None:
        """Two SyntheticCategoricalDataModule instances with the same seed
        should produce identical marginals."""

        def _make_cat_dm() -> SyntheticCategoricalDataModule:
            return SyntheticCategoricalDataModule(
                graph_type="sbm",
                num_nodes=10,
                num_graphs=20,
                batch_size=4,
                num_workers=0,
                seed=42,
            )

        dm1 = _make_cat_dm()
        dm1.setup()
        dm2 = _make_cat_dm()
        dm2.setup()

        assert torch.equal(dm1.node_marginals, dm2.node_marginals)
        assert torch.equal(dm1.edge_marginals, dm2.edge_marginals)
