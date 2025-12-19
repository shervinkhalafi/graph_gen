"""Unit tests for SingleGraphDataModule with various graph types.

Tests verify that each supported graph type:
1. Generates valid symmetric binary adjacency matrices
2. Properly implements same_graph_all_splits protocol
3. Creates functional dataloaders with correct batch shapes

Test Rationale
--------------
The single-graph training protocol is central to stage 1 and stage 1.5 experiments.
Each graph type has different generation logic and potential edge cases:
- Synthetic graphs (ER, regular, tree): Use numpy-based generators
- NetworkX graphs (ring_of_cliques, LFR): External library with specific constraints
- PyG graphs: External datasets requiring download and format conversion

These tests ensure the SingleGraphDataModule abstraction works uniformly across
all graph types, catching integration issues early.
"""

import numpy as np
import pytest
import torch

from tmgg.experiment_utils.data.single_graph_data_module import SingleGraphDataModule


class TestSyntheticGraphs:
    """Test synthetic graph types that don't require external downloads."""

    @pytest.mark.parametrize(
        "graph_type,kwargs",
        [
            ("sbm", {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}),
            ("erdos_renyi", {"p": 0.1}),
            ("regular", {"d": 3}),
            ("tree", {}),
        ],
    )
    def test_synthetic_graph_generation(self, graph_type: str, kwargs: dict):
        """Test that synthetic graphs generate valid adjacency matrices."""
        dm = SingleGraphDataModule(
            graph_type=graph_type,
            n=50,
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
            **kwargs,
        )
        dm.setup()

        A = dm.get_train_graph()

        # Shape check
        assert A.shape == (50, 50), f"Expected (50, 50), got {A.shape}"

        # Symmetry check
        assert np.allclose(A, A.T), "Adjacency matrix should be symmetric"

        # Binary check
        unique_vals = set(np.unique(A))
        assert unique_vals.issubset({0.0, 1.0}), f"Expected binary, got {unique_vals}"

        # No self-loops
        assert np.allclose(np.diag(A), 0), "Diagonal should be zero (no self-loops)"

    @pytest.mark.parametrize(
        "graph_type,kwargs",
        [
            ("sbm", {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}),
            ("erdos_renyi", {"p": 0.1}),
            ("regular", {"d": 3}),
            ("tree", {}),
        ],
    )
    def test_same_graph_all_splits(self, graph_type: str, kwargs: dict):
        """Test that same_graph_all_splits=True uses identical graphs."""
        dm = SingleGraphDataModule(
            graph_type=graph_type,
            n=50,
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
            **kwargs,
        )
        dm.setup()

        train_graph = dm.get_train_graph()
        val_graph = dm.get_val_graph()
        test_graph = dm.get_test_graph()

        assert np.array_equal(
            train_graph, val_graph
        ), "Train and val should be identical"
        assert np.array_equal(
            train_graph, test_graph
        ), "Train and test should be identical"

    @pytest.mark.parametrize(
        "graph_type,kwargs",
        [
            ("sbm", {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}),
            ("erdos_renyi", {"p": 0.1}),
        ],
    )
    def test_different_graphs_when_disabled(self, graph_type: str, kwargs: dict):
        """Test that same_graph_all_splits=False uses different graphs."""
        dm = SingleGraphDataModule(
            graph_type=graph_type,
            n=50,
            same_graph_all_splits=False,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
            **kwargs,
        )
        dm.setup()

        train_graph = dm.get_train_graph()
        val_graph = dm.get_val_graph()
        test_graph = dm.get_test_graph()

        # Graphs should differ (with high probability for random graphs)
        assert not np.array_equal(train_graph, val_graph), "Train and val should differ"
        assert not np.array_equal(
            train_graph, test_graph
        ), "Train and test should differ"

    @pytest.mark.parametrize(
        "graph_type,kwargs",
        [
            ("sbm", {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}),
            ("erdos_renyi", {"p": 0.1}),
            ("regular", {"d": 3}),
            ("tree", {}),
        ],
    )
    def test_dataloader_batch_shape(self, graph_type: str, kwargs: dict):
        """Test that dataloaders return correctly shaped batches."""
        batch_size = 4
        dm = SingleGraphDataModule(
            graph_type=graph_type,
            n=50,
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=batch_size,
            **kwargs,
        )
        dm.setup()

        loader = dm.train_dataloader()
        batch = next(iter(loader))

        assert batch.shape == (
            batch_size,
            50,
            50,
        ), f"Expected ({batch_size}, 50, 50), got {batch.shape}"
        assert batch.dtype == torch.float32, f"Expected float32, got {batch.dtype}"


class TestNetworkXGraphs:
    """Test NetworkX-based graph types."""

    def test_ring_of_cliques_generation(self):
        """Test ring of cliques graph generation."""
        dm = SingleGraphDataModule(
            graph_type="ring_of_cliques",
            n=20,  # Ignored for this type
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
            num_cliques=4,
            clique_size=5,
        )
        dm.setup()

        A = dm.get_train_graph()

        # Total nodes = num_cliques * clique_size = 20
        assert A.shape == (20, 20), f"Expected (20, 20), got {A.shape}"
        assert np.allclose(A, A.T), "Should be symmetric"
        assert set(np.unique(A)).issubset({0.0, 1.0}), "Should be binary"

    def test_lfr_generation(self):
        """Test LFR benchmark graph generation."""
        dm = SingleGraphDataModule(
            graph_type="lfr",
            n=200,  # LFR needs large n for community constraints
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
            tau1=2.0,
            tau2=1.1,
            mu=0.1,
            average_degree=8,
            min_community=15,
        )
        dm.setup()

        A = dm.get_train_graph()

        assert A.shape == (200, 200), f"Expected (200, 200), got {A.shape}"
        assert np.allclose(A, A.T), "Should be symmetric"
        assert set(np.unique(A)).issubset({0.0, 1.0}), "Should be binary"

    def test_ring_of_cliques_same_graph_protocol(self):
        """Test same graph protocol for ring of cliques."""
        dm = SingleGraphDataModule(
            graph_type="ring_of_cliques",
            n=20,
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
            num_cliques=4,
            clique_size=5,
        )
        dm.setup()

        assert np.array_equal(dm.get_train_graph(), dm.get_val_graph())
        assert np.array_equal(dm.get_train_graph(), dm.get_test_graph())


class TestPyGGraphs:
    """Test PyTorch Geometric dataset loading.

    These tests require torch-geometric to be installed and may download
    datasets on first run. Marked as slow to allow skipping in fast test runs.
    """

    @pytest.mark.slow
    @pytest.mark.parametrize(
        "graph_type",
        [
            "pyg_enzymes",
            "pyg_proteins",
            # "pyg_qm9",  # QM9 is very large, skip by default
        ],
    )
    def test_pyg_graph_loading(self, graph_type: str):
        """Test that PyG datasets load and convert to valid adjacency matrices."""
        pytest.importorskip("torch_geometric", reason="torch-geometric not installed")

        dm = SingleGraphDataModule(
            graph_type=graph_type,
            n=0,  # Ignored for PyG
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
            graph_idx=0,
        )
        dm.setup()

        A = dm.get_train_graph()

        # Shape should be square
        assert A.shape[0] == A.shape[1], f"Expected square matrix, got {A.shape}"

        # Symmetry
        assert np.allclose(A, A.T), "Should be symmetric"

        # Binary
        assert set(np.unique(A)).issubset({0.0, 1.0}), "Should be binary"

    @pytest.mark.slow
    def test_pyg_same_graph_protocol(self):
        """Test same graph protocol for PyG datasets."""
        pytest.importorskip("torch_geometric", reason="torch-geometric not installed")

        dm = SingleGraphDataModule(
            graph_type="pyg_enzymes",
            n=0,
            same_graph_all_splits=True,
            num_train_samples=10,
            num_val_samples=5,
            num_test_samples=5,
            batch_size=4,
            graph_idx=0,
        )
        dm.setup()

        assert np.array_equal(dm.get_train_graph(), dm.get_val_graph())
        assert np.array_equal(dm.get_train_graph(), dm.get_test_graph())


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_unknown_graph_type_raises(self):
        """Test that unknown graph types raise ValueError."""
        dm = SingleGraphDataModule(
            graph_type="unknown_type",
            n=50,
            same_graph_all_splits=True,
        )
        with pytest.raises(ValueError, match="Unknown graph type"):
            dm.setup()

    def test_setup_required_before_access(self):
        """Test that accessing graphs before setup raises RuntimeError."""
        dm = SingleGraphDataModule(
            graph_type="sbm",
            n=50,
            same_graph_all_splits=True,
        )
        with pytest.raises(RuntimeError, match="Call setup"):
            dm.get_train_graph()

    def test_reproducibility_with_seed(self):
        """Test that the same seed produces identical graphs."""
        kwargs = {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}

        dm1 = SingleGraphDataModule(
            graph_type="sbm",
            n=50,
            train_seed=42,
            same_graph_all_splits=True,
            **kwargs,
        )
        dm1.setup()

        dm2 = SingleGraphDataModule(
            graph_type="sbm",
            n=50,
            train_seed=42,
            same_graph_all_splits=True,
            **kwargs,
        )
        dm2.setup()

        assert np.array_equal(dm1.get_train_graph(), dm2.get_train_graph())

    def test_different_seeds_produce_different_graphs(self):
        """Test that different seeds produce different graphs."""
        kwargs = {"p_intra": 0.7, "p_inter": 0.05, "num_blocks": 3}

        dm1 = SingleGraphDataModule(
            graph_type="sbm",
            n=50,
            train_seed=42,
            same_graph_all_splits=True,
            **kwargs,
        )
        dm1.setup()

        dm2 = SingleGraphDataModule(
            graph_type="sbm",
            n=50,
            train_seed=123,
            same_graph_all_splits=True,
            **kwargs,
        )
        dm2.setup()

        assert not np.array_equal(dm1.get_train_graph(), dm2.get_train_graph())
