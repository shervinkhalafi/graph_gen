"""Tests for SyntheticCategoricalDataModule.

Verifies that synthetic graph generation, categorical conversion, splitting,
collation, marginals computation, round-trip fidelity, and size distribution
interface all behave correctly.
"""

from __future__ import annotations

import pytest
import torch

from tmgg.data.batches import CategoricalBatch
from tmgg.experiment_utils.data.conversions import categorical_to_adjacency
from tmgg.experiment_utils.data.size_distribution import SizeDistribution
from tmgg.experiments.discrete_diffusion_generative.datamodule import (
    SyntheticCategoricalDataModule,
)

# -- Shared fixtures -------------------------------------------------------


@pytest.fixture()
def small_dm() -> SyntheticCategoricalDataModule:
    """A small SBM data module with 100 graphs of 20 nodes.

    Uses default SBM parameters (2 blocks, p_in=0.7, p_out=0.1) and an
    80/10/10 split, producing 80 train, 10 val, 10 test graphs.
    """
    dm = SyntheticCategoricalDataModule(
        graph_type="sbm",
        num_nodes=20,
        num_graphs=100,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=8,
        seed=123,
    )
    dm.setup()
    return dm


@pytest.fixture()
def er_dm() -> SyntheticCategoricalDataModule:
    """An Erdos-Renyi data module to test non-SBM graph types."""
    dm = SyntheticCategoricalDataModule(
        graph_type="er",
        num_nodes=15,
        num_graphs=50,
        train_ratio=0.8,
        val_ratio=0.1,
        batch_size=4,
        seed=99,
        graph_config={"p": 0.3},
    )
    dm.setup()
    return dm


# -- TestSetup --------------------------------------------------------------


class TestSetup:
    """Verify dataset generation produces correct split sizes and shapes."""

    def test_split_sizes(self, small_dm: SyntheticCategoricalDataModule) -> None:
        """The 80/10/10 split of 100 graphs should yield 80, 10, 10."""
        assert small_dm._train_data is not None
        assert small_dm._val_data is not None
        assert small_dm._test_data is not None
        assert len(small_dm._train_data) == 80
        assert len(small_dm._val_data) == 10
        assert len(small_dm._test_data) == 10

    def test_tuple_shapes(self, small_dm: SyntheticCategoricalDataModule) -> None:
        """Each stored tuple must contain (X, E, y, n) with correct shapes.

        X: (n, 2), E: (n, n, 2), y: (0,), n: int == num_nodes.
        """
        assert small_dm._train_data is not None
        X_i, E_i, y_i, n_i = small_dm._train_data[0]
        n = small_dm.num_nodes

        assert n_i == n
        assert X_i.shape == (n, 2)
        assert E_i.shape == (n, n, 2)
        assert y_i.shape == (0,)

    def test_idempotent_setup(self, small_dm: SyntheticCategoricalDataModule) -> None:
        """Calling setup() a second time should be a no-op (same data objects)."""
        train_before = small_dm._train_data
        small_dm.setup()
        assert small_dm._train_data is train_before

    def test_no_overlap_between_splits(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Train, val, and test splits should contain no duplicate graphs.

        We compare by computing a per-graph hash from X and E tensors.
        """
        assert small_dm._train_data is not None
        assert small_dm._val_data is not None
        assert small_dm._test_data is not None

        def _hash_sample(
            sample: tuple[torch.Tensor, torch.Tensor, torch.Tensor, int],
        ) -> int:
            X, E, _y, _n = sample
            return hash((X.numpy().tobytes(), E.numpy().tobytes()))

        train_hashes = {_hash_sample(s) for s in small_dm._train_data}
        val_hashes = {_hash_sample(s) for s in small_dm._val_data}
        test_hashes = {_hash_sample(s) for s in small_dm._test_data}

        assert train_hashes.isdisjoint(val_hashes)
        assert train_hashes.isdisjoint(test_hashes)
        assert val_hashes.isdisjoint(test_hashes)

    def test_er_generation(self, er_dm: SyntheticCategoricalDataModule) -> None:
        """Non-SBM types (Erdos-Renyi) should also produce valid data."""
        assert er_dm._train_data is not None
        assert len(er_dm._train_data) == 40  # 0.8 * 50


# -- TestDataloaders --------------------------------------------------------


class TestDataloaders:
    """Verify that DataLoaders yield properly collated batches."""

    def test_train_batch_shape(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """A training batch should be a (X, E, y, node_mask) tuple with
        correct shapes: X (bs, n, 2), E (bs, n, n, 2), y (bs, 0),
        node_mask (bs, n).
        """
        batch = next(iter(small_dm.train_dataloader()))
        X, E, y, node_mask = batch
        bs = X.shape[0]
        n = small_dm.num_nodes

        assert X.shape == (bs, n, 2)
        assert E.shape == (bs, n, n, 2)
        assert y.shape == (bs, 0)
        assert node_mask.shape == (bs, n)
        assert node_mask.dtype == torch.bool

    def test_val_batch_no_error(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Validation dataloader should produce at least one batch."""
        batch = next(iter(small_dm.val_dataloader()))
        assert batch is not None

    def test_test_batch_no_error(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Test dataloader should produce at least one batch."""
        batch = next(iter(small_dm.test_dataloader()))
        assert batch is not None

    def test_categorical_batch_from_collated(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """A CategoricalBatch should be constructible from collated output.

        This checks that the collation output matches CategoricalBatch's
        expected tensor shapes and dtypes.
        """
        X, E, y, node_mask = next(iter(small_dm.train_dataloader()))
        cb = CategoricalBatch(X=X, E=E, y=y, node_mask=node_mask)
        assert cb.dx == 2
        assert cb.de == 2


# -- TestMarginals ----------------------------------------------------------


class TestMarginals:
    """Verify that marginal distributions are valid probability vectors."""

    def test_node_marginals_shape(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """node_marginals should have shape (dx,) = (2,)."""
        assert small_dm.node_marginals.shape == (2,)

    def test_edge_marginals_shape(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """edge_marginals should have shape (de,) = (2,)."""
        assert small_dm.edge_marginals.shape == (2,)

    def test_node_marginals_sum_to_one(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Node marginals must form a valid probability distribution."""
        assert torch.allclose(
            small_dm.node_marginals.sum(), torch.tensor(1.0), atol=1e-6
        )

    def test_edge_marginals_sum_to_one(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Edge marginals must form a valid probability distribution."""
        assert torch.allclose(
            small_dm.edge_marginals.sum(), torch.tensor(1.0), atol=1e-6
        )

    def test_node_marginals_nonnegative(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """All entries of node_marginals should be >= 0."""
        assert (small_dm.node_marginals >= 0).all()

    def test_edge_marginals_nonnegative(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """All entries of edge_marginals should be >= 0."""
        assert (small_dm.edge_marginals >= 0).all()

    def test_node_marginals_are_nontrivial(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """For synthetic graphs all nodes are real, so class-1 (real node)
        should dominate: node_marginals[1] should be close to 1.0.
        """
        assert small_dm.node_marginals[1] > 0.99

    def test_marginals_not_available_before_setup(self) -> None:
        """Accessing marginals before setup() should raise RuntimeError."""
        dm = SyntheticCategoricalDataModule(num_graphs=10, num_nodes=5)
        with pytest.raises(RuntimeError, match="Call setup"):
            _ = dm.node_marginals
        with pytest.raises(RuntimeError, match="Call setup"):
            _ = dm.edge_marginals


# -- TestDatasetInfo --------------------------------------------------------


class TestDatasetInfo:
    """Verify get_dataset_info returns the expected metadata keys."""

    def test_keys(self, small_dm: SyntheticCategoricalDataModule) -> None:
        """All four required keys must be present."""
        info = small_dm.get_dataset_info()
        assert info["num_graphs"] == 100
        assert info["num_nodes"] == 20
        assert info["num_node_classes"] == 2
        assert info["num_edge_classes"] == 2


# -- TestRoundTrip ----------------------------------------------------------


class TestRoundTrip:
    """Verify adjacency -> categorical -> adjacency preserves structure.

    The round-trip invariant: converting a binary adjacency matrix to
    categorical one-hot and back should recover the original adjacency
    (modulo self-loops, which are zeroed by the conversion).
    """

    def test_single_graph_roundtrip(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Pick a training graph, convert to CategoricalBatch, convert back,
        and compare with the original adjacency.
        """
        assert small_dm._train_data is not None
        X_i, E_i, y_i, n_i = small_dm._train_data[0]

        # Reconstruct adjacency from categorical edge features
        recovered_adj = categorical_to_adjacency(E_i.unsqueeze(0)).squeeze(0)

        # Build original adjacency from the one-hot E
        original_adj = E_i[..., 1]  # class 1 = edge

        assert torch.allclose(recovered_adj, original_adj)

    def test_batch_roundtrip(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Collated batch -> CategoricalBatch -> adjacency should preserve
        the edge structure for all graphs in the batch.
        """
        X, E, y, node_mask = next(iter(small_dm.train_dataloader()))
        cb = CategoricalBatch(X=X, E=E, y=y, node_mask=node_mask)
        recovered = cb.to_adjacency()

        # The recovered adjacency should match E[..., 1] at real positions
        expected = E[..., 1]
        # Mask out padded positions for comparison
        mask_2d = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)
        assert torch.allclose(recovered * mask_2d.float(), expected * mask_2d.float())


# -- TestSizeDistribution ---------------------------------------------------


class TestSizeDistribution:
    """Verify the size distribution interface on SyntheticCategoricalDataModule.

    For fixed-size synthetic data, ``get_size_distribution`` returns a
    degenerate distribution and ``sample_n_nodes`` returns a constant
    tensor. These tests confirm both the datamodule integration and the
    contract that the generative pipeline relies on.
    """

    def test_default_is_degenerate(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Fixed-size data should produce a degenerate distribution."""
        dist = small_dm.get_size_distribution()
        assert dist.is_degenerate
        assert dist.sizes == (20,)
        assert dist.max_size == 20

    def test_train_split(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """Training split should also be degenerate for fixed-size data."""
        dist = small_dm.get_size_distribution("train")
        assert dist.is_degenerate
        assert dist.sizes == (20,)
        # All 80 training graphs have 20 nodes
        assert dist.counts == (80,)

    def test_val_split(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        dist = small_dm.get_size_distribution("val")
        assert dist.is_degenerate
        assert dist.counts == (10,)

    def test_test_split(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        dist = small_dm.get_size_distribution("test")
        assert dist.is_degenerate
        assert dist.counts == (10,)

    def test_invalid_split_raises(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        with pytest.raises(ValueError, match="Unknown split"):
            small_dm.get_size_distribution("predict")

    def test_before_setup_returns_fixed(self) -> None:
        """Before setup, should fall back to SizeDistribution.fixed(num_nodes)."""
        dm = SyntheticCategoricalDataModule(num_graphs=10, num_nodes=15)
        dist = dm.get_size_distribution("train")
        assert dist == SizeDistribution.fixed(15)

    def test_sample_n_nodes_shape(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """sample_n_nodes should return a 1-D tensor of the requested size."""
        result = small_dm.sample_n_nodes(4)
        assert result.shape == (4,)
        assert result.dtype == torch.long

    def test_sample_n_nodes_all_equal(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """For fixed-size data, all sampled sizes should equal num_nodes."""
        result = small_dm.sample_n_nodes(16)
        assert (result == 20).all()

    def test_serialization_roundtrip(
        self,
        small_dm: SyntheticCategoricalDataModule,
    ) -> None:
        """The distribution should survive a to_dict/from_dict round-trip."""
        dist = small_dm.get_size_distribution("train")
        restored = SizeDistribution.from_dict(dist.to_dict())
        assert restored == dist
