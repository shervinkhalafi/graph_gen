"""Tests for discrete diffusion evaluation pipeline (P1.6).

Testing strategy
----------------
Verifies the conversion utilities, end-to-end MMD evaluation, and the
validation hook integration. Uses the same tiny model configuration as
test_discrete_diffusion_module.py (2 layers, 16-dim hidden, dx=de=2,
dy=0, T=10) with small graph counts for fast execution.

The key invariants tested are:
- Categorical samples convert correctly to binary adjacency (class 0 = no edge)
- Multi-class edges collapse to binary via (edge_types > 0)
- Adjacencies are symmetric with zero diagonal
- MMD evaluation returns finite non-negative floats
- The validation hook logs MMD keys during training
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest
import pytorch_lightning as pl
import torch

from tmgg.experiments.discrete_diffusion_generative.datamodule import (
    SyntheticCategoricalDataModule,
)
from tmgg.experiments.discrete_diffusion_generative.evaluate import (
    categorical_samples_to_adjacencies,
    categorical_samples_to_graphs,
    evaluate_discrete_samples,
)
from tmgg.experiments.discrete_diffusion_generative.lightning_module import (
    DiscreteDiffusionLightningModule,
)
from tmgg.models.digress.discrete_transformer import DiscreteGraphTransformer
from tmgg.models.digress.noise_schedule import PredefinedNoiseScheduleDiscrete

_DX = 2
_DE = 2
_DY = 0
_T = 10
_NUM_NODES = 8
_NUM_GRAPHS = 40
_BATCH_SIZE = 4


# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------


@pytest.fixture()
def model() -> DiscreteGraphTransformer:
    return DiscreteGraphTransformer(
        n_layers=2,
        input_dims={"X": _DX, "E": _DE, "y": 1},
        hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
        hidden_dims={"dx": 16, "de": 16, "dy": 16, "n_head": 2},
        output_dims={"X": _DX, "E": _DE, "y": _DY},
    )


@pytest.fixture()
def noise_schedule() -> PredefinedNoiseScheduleDiscrete:
    return PredefinedNoiseScheduleDiscrete("cosine", _T)


@pytest.fixture()
def datamodule() -> SyntheticCategoricalDataModule:
    return SyntheticCategoricalDataModule(
        num_nodes=_NUM_NODES,
        num_graphs=_NUM_GRAPHS,
        batch_size=_BATCH_SIZE,
        seed=42,
    )


def _build_module(
    model: DiscreteGraphTransformer,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
    **kwargs: object,
) -> DiscreteDiffusionLightningModule:
    defaults = {
        "diffusion_steps": _T,
        "transition_type": "marginal",
        "eval_num_samples": 8,
    }
    defaults.update(kwargs)
    return DiscreteDiffusionLightningModule(
        model=model,
        noise_schedule=noise_schedule,
        **defaults,  # pyright: ignore[reportArgumentType]
    )


def _setup_with_trainer(
    module: DiscreteDiffusionLightningModule,
    dm: SyntheticCategoricalDataModule,
) -> None:
    """Attach a minimal trainer and run setup so transition model is built."""
    trainer = pl.Trainer(
        max_steps=1,
        enable_checkpointing=False,
        logger=False,
    )
    trainer.fit_loop.setup_data()  # pyright: ignore[reportAttributeAccessIssue]
    module.trainer = trainer  # pyright: ignore[reportAttributeAccessIssue]
    trainer.datamodule = dm  # pyright: ignore[reportAttributeAccessIssue]
    dm.setup("fit")
    module.setup("fit")


# ------------------------------------------------------------------
# Conversion tests
# ------------------------------------------------------------------


class TestCategoricalToAdjacencies:
    """Invariants: class 0 = no edge, symmetric, zero diagonal."""

    def test_binary_edges(self) -> None:
        """de=2: edge_types==1 should produce adjacency 1, edge_types==0 produces 0."""
        n = 5
        # Construct a sample with known edge pattern
        node_types = torch.zeros(n, dtype=torch.long)
        edge_types = torch.zeros(n, n, dtype=torch.long)
        edge_types[0, 1] = 1
        edge_types[1, 0] = 1
        edge_types[2, 3] = 1
        edge_types[3, 2] = 1

        samples = [(node_types, edge_types)]
        adjs = categorical_samples_to_adjacencies(samples)

        assert len(adjs) == 1
        adj = adjs[0]
        assert adj.shape == (n, n)
        assert adj[0, 1] == 1.0
        assert adj[1, 0] == 1.0
        assert adj[2, 3] == 1.0
        assert adj[0, 0] == 0.0  # zero diagonal
        assert adj.dtype == np.float32

    def test_multiclass_edges(self) -> None:
        """de>2: any class > 0 should produce adjacency 1."""
        n = 4
        node_types = torch.zeros(n, dtype=torch.long)
        edge_types = torch.zeros(n, n, dtype=torch.long)
        edge_types[0, 1] = 2  # class 2 edge
        edge_types[1, 0] = 3  # class 3 edge
        edge_types[0, 2] = 1  # class 1 edge
        edge_types[2, 0] = 1

        samples = [(node_types, edge_types)]
        adjs = categorical_samples_to_adjacencies(samples)
        adj = adjs[0]

        assert adj[0, 1] == 1.0
        assert adj[1, 0] == 1.0
        assert adj[0, 2] == 1.0
        # Class 0 entries remain 0
        assert adj[2, 3] == 0.0

    def test_symmetry_enforced(self) -> None:
        """Even if input is asymmetric, output adjacency is symmetric."""
        n = 4
        node_types = torch.zeros(n, dtype=torch.long)
        edge_types = torch.zeros(n, n, dtype=torch.long)
        edge_types[0, 1] = 1  # one direction only

        samples = [(node_types, edge_types)]
        adjs = categorical_samples_to_adjacencies(samples)
        adj = adjs[0]

        assert adj[0, 1] == adj[1, 0] == 1.0


class TestCategoricalToGraphs:
    def test_returns_networkx(self) -> None:
        n = 5
        node_types = torch.zeros(n, dtype=torch.long)
        edge_types = torch.zeros(n, n, dtype=torch.long)
        edge_types[0, 1] = 1
        edge_types[1, 0] = 1

        graphs = categorical_samples_to_graphs([(node_types, edge_types)])
        assert len(graphs) == 1
        assert isinstance(graphs[0], nx.Graph)
        assert graphs[0].number_of_nodes() == n
        assert graphs[0].number_of_edges() == 1  # single undirected edge


# ------------------------------------------------------------------
# End-to-end MMD evaluation
# ------------------------------------------------------------------


class TestEvaluateDiscreteSamples:
    def test_returns_finite_mmd(
        self,
        model: DiscreteGraphTransformer,
        noise_schedule: PredefinedNoiseScheduleDiscrete,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Sample from a tiny model and verify MMD values are finite and non-negative."""
        module = _build_module(model, noise_schedule, eval_num_samples=4)
        _setup_with_trainer(module, datamodule)

        # Generate a few samples
        samples = module.sample_batch(batch_size=4, num_nodes=_NUM_NODES)

        # Build reference graphs from the datamodule
        from tmgg.experiment_utils.mmd_metrics import adjacency_to_networkx

        datamodule.setup("fit")
        ref_graphs: list[nx.Graph] = []  # pyright: ignore[reportExplicitAny]
        dl = datamodule.val_dataloader()
        for batch in dl:
            X, E, _y, node_mask = batch
            adj_batch = E.argmax(dim=-1)
            for i in range(adj_batch.size(0)):
                n = int(node_mask[i].sum().item())
                adj_np = (adj_batch[i, :n, :n] > 0).cpu().numpy().astype("float32")
                ref_graphs.append(adjacency_to_networkx(adj_np))
                if len(ref_graphs) >= 4:
                    break
            if len(ref_graphs) >= 4:
                break

        mmd = evaluate_discrete_samples(ref_graphs, samples)
        assert np.isfinite(mmd.degree_mmd)
        assert np.isfinite(mmd.clustering_mmd)
        assert np.isfinite(mmd.spectral_mmd)
        assert mmd.degree_mmd >= 0
        assert mmd.clustering_mmd >= 0
        assert mmd.spectral_mmd >= 0


# ------------------------------------------------------------------
# Validation hook integration
# ------------------------------------------------------------------


class TestValidationHookMMD:
    """Verify that MMD keys appear in logged metrics after a training run.

    Rationale: the on_validation_epoch_end method should accumulate
    reference graphs during validation_step and compute MMD at epoch end.
    We use a short run (2 training steps, 2 val batches) to trigger this.
    """

    def test_mmd_keys_logged(
        self,
        model: DiscreteGraphTransformer,
        noise_schedule: PredefinedNoiseScheduleDiscrete,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        module = _build_module(model, noise_schedule, eval_num_samples=4)

        # Use max_epochs=1 (not max_steps) so validation runs at epoch end
        trainer = pl.Trainer(
            max_epochs=1,
            limit_val_batches=2,
            enable_checkpointing=False,
            logger=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, datamodule)

        # Check that MMD metrics were logged
        logged = trainer.logged_metrics
        assert (
            "val/degree_mmd" in logged
        ), f"Missing degree_mmd in {list(logged.keys())}"
        assert "val/clustering_mmd" in logged
        assert "val/spectral_mmd" in logged

        # Values should be finite
        assert np.isfinite(logged["val/degree_mmd"])
        assert np.isfinite(logged["val/clustering_mmd"])
        assert np.isfinite(logged["val/spectral_mmd"])


# ------------------------------------------------------------------
# CLI smoke test
# ------------------------------------------------------------------


class TestCLI:
    def test_help_exits_zero(self) -> None:
        """CLI --help should exit 0 (validates argparse setup)."""
        from tmgg.experiments.discrete_diffusion_generative.evaluate_cli import main

        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
