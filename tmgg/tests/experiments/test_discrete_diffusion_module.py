"""Tests for DiscreteDiffusionLightningModule.

Testing strategy
----------------
Each test uses a tiny model configuration (2 layers, 16-dim hidden, dx=de=2,
dy=0, T=10) paired with a small SyntheticCategoricalDataModule to keep
wall-clock time under a few seconds. The goal is to verify wiring correctness
(shapes, finiteness, mixin delegation) rather than training convergence.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl
import torch
from torch import Tensor

from tmgg.experiments.discrete_diffusion_generative.datamodule import (
    SyntheticCategoricalDataModule,
)
from tmgg.experiments.discrete_diffusion_generative.lightning_module import (
    DiscreteDiffusionLightningModule,
)
from tmgg.models.digress.discrete_transformer import DiscreteGraphTransformer
from tmgg.models.digress.noise_schedule import PredefinedNoiseScheduleDiscrete

# ------------------------------------------------------------------
# Shared fixtures
# ------------------------------------------------------------------

_DX = 2
_DE = 2
_DY = 0
_T = 10
_NUM_NODES = 8
_NUM_GRAPHS = 40
_BATCH_SIZE = 4


@pytest.fixture()
def model() -> DiscreteGraphTransformer:
    """Tiny transformer for fast tests."""
    return DiscreteGraphTransformer(
        n_layers=2,
        input_dims={"X": _DX, "E": _DE, "y": 1},  # +1 for timestep
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


@pytest.fixture()
def lightning_module(
    model: DiscreteGraphTransformer,
    noise_schedule: PredefinedNoiseScheduleDiscrete,
) -> DiscreteDiffusionLightningModule:
    return DiscreteDiffusionLightningModule(
        model=model,
        noise_schedule=noise_schedule,
        diffusion_steps=_T,
        transition_type="marginal",
    )


def _attach_trainer_and_setup(
    module: DiscreteDiffusionLightningModule,
    dm: SyntheticCategoricalDataModule,
) -> None:
    """Attach a minimal trainer and run setup so the transition model exists."""
    trainer = pl.Trainer(
        max_steps=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.datamodule = dm  # pyright: ignore[reportAttributeAccessIssue]
    module.trainer = trainer
    dm.setup("fit")
    module.setup("fit")


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


class TestSetup:
    """Verify that setup() constructs the transition model and limit dist."""

    def test_setup_constructs_transition(
        self,
        lightning_module: DiscreteDiffusionLightningModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """After setup, transition_model and limit_dist are populated.

        Starting state: fresh module with no transition model.
        Invariant: setup('fit') populates both attributes; limit_dist.X
        and limit_dist.E have the correct dimensionality.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        assert hasattr(lightning_module, "transition_model")
        assert hasattr(lightning_module, "limit_dist")
        assert lightning_module.limit_dist.X.shape == (_DX,)
        assert lightning_module.limit_dist.E.shape == (_DE,)

    def test_setup_is_idempotent(
        self,
        lightning_module: DiscreteDiffusionLightningModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Calling setup() twice does not error or reconstruct.

        Starting state: module already set up.
        Invariant: second call is a no-op; limit_dist is the same object.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)
        limit_dist_first = lightning_module.limit_dist

        lightning_module.setup("fit")
        assert lightning_module.limit_dist is limit_dist_first


class TestApplyNoise:
    """Verify forward diffusion produces correct shapes and symmetric edges."""

    def test_shapes_and_symmetry(
        self,
        lightning_module: DiscreteDiffusionLightningModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """apply_noise returns dict with correct shapes and symmetric E_t.

        Starting state: clean one-hot batch from the datamodule.
        Invariant: noisy X_t has same shape as X, E_t is symmetric,
        all returned tensors have a batch dimension equal to batch_size.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        batch = next(iter(datamodule.train_dataloader()))
        X, E, y, node_mask = batch

        lightning_module.train()
        noisy_data = lightning_module.apply_noise(X, E, y, node_mask)

        assert noisy_data["X_t"].shape == X.shape
        assert noisy_data["E_t"].shape == E.shape
        assert noisy_data["t"].shape[0] == X.shape[0]

        # E_t must be symmetric (undirected graph invariant)
        E_t = noisy_data["E_t"]
        assert torch.allclose(E_t, E_t.transpose(1, 2)), (
            f"E_t not symmetric: max dev "
            f"{(E_t - E_t.transpose(1, 2)).abs().max():.2e}"
        )


class TestForward:
    """Verify the forward pass returns normalised probability distributions."""

    def test_output_probabilities(
        self,
        lightning_module: DiscreteDiffusionLightningModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Forward output sums to ~1 along the class dimension.

        Starting state: noisy batch from apply_noise.
        Invariant: pred.X sums to 1 over dim=-1 for valid nodes;
        pred.E sums to 1 over dim=-1 for valid edges.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        batch = next(iter(datamodule.train_dataloader()))
        X, E, y, node_mask = batch

        lightning_module.eval()
        noisy_data = lightning_module.apply_noise(X, E, y, node_mask)
        extra_data = lightning_module.compute_extra_data(noisy_data)
        pred = lightning_module.forward(noisy_data, extra_data, node_mask)

        # Node predictions: check valid nodes sum to ~1
        valid_X_sums = pred.X[node_mask].sum(dim=-1)
        assert torch.allclose(valid_X_sums, torch.ones_like(valid_X_sums), atol=1e-5)

        # Edge predictions: check valid edges sum to ~1
        edge_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        valid_E_sums = pred.E[edge_mask].sum(dim=-1)
        assert torch.allclose(valid_E_sums, torch.ones_like(valid_E_sums), atol=1e-5)


class TestTraining:
    """Verify the training step produces finite positive loss."""

    def test_training_loss_finite_and_positive(
        self,
        lightning_module: DiscreteDiffusionLightningModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Noise → predict → loss pipeline yields a finite positive scalar.

        Starting state: set-up module, one batch from train dataloader.
        Invariant: cross-entropy loss is finite, positive, and a 0-d tensor.
        Note: we call the loss function directly rather than training_step
        to avoid requiring an active trainer logging loop.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        batch = next(iter(datamodule.train_dataloader()))
        X, E, y, node_mask = batch
        lightning_module.train()

        noisy_data = lightning_module.apply_noise(X, E, y, node_mask)
        extra_data = lightning_module.compute_extra_data(noisy_data)
        pred = lightning_module.forward(noisy_data, extra_data, node_mask)

        loss = lightning_module.train_loss_fn(
            pred_X=pred.X.clone(),
            pred_E=pred.E.clone(),
            true_X=X,
            true_E=E,
            node_mask=node_mask,
        )

        assert isinstance(loss, Tensor)
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"


class TestValidation:
    """Verify VLB components are finite."""

    def test_vlb_terms_finite(
        self,
        lightning_module: DiscreteDiffusionLightningModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Validation step produces finite VLB components.

        Starting state: set-up module, one validation batch.
        Invariant: the accumulated _val_nll, _val_kl_prior, _val_kl_diffusion,
        _val_reconstruction are all finite scalars after one step.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        batch = next(iter(datamodule.val_dataloader()))
        lightning_module.eval()

        result = lightning_module.validation_step(batch, batch_idx=0)

        assert "val_nll" in result
        assert torch.isfinite(
            result["val_nll"]
        ), f"val_nll not finite: {result['val_nll'].item()}"
        assert len(lightning_module._val_nll) == 1
        assert torch.isfinite(lightning_module._val_kl_prior[0])
        assert torch.isfinite(lightning_module._val_kl_diffusion[0])
        assert torch.isfinite(lightning_module._val_reconstruction[0])


class TestSampling:
    """Verify that sample_batch returns correctly shaped integer tensors."""

    def test_sample_batch_shapes(
        self,
        lightning_module: DiscreteDiffusionLightningModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Sampling generates graphs with the correct node/edge shapes.

        Starting state: set-up module.
        Invariant: each generated graph is a (node_types, edge_types) pair
        with shapes (n,) and (n, n) respectively, containing non-negative
        integer class indices.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)
        lightning_module.eval()

        n_samples = 2
        samples = lightning_module.sample_batch(
            batch_size=n_samples, num_nodes=_NUM_NODES
        )

        assert len(samples) == n_samples

        for node_types, edge_types in samples:
            assert node_types.shape == (_NUM_NODES,)
            assert edge_types.shape == (_NUM_NODES, _NUM_NODES)
            # Class indices should be non-negative integers
            assert (node_types >= 0).all()
            assert (edge_types >= 0).all()
            # Edge types should be symmetric (undirected graphs)
            assert (edge_types == edge_types.T).all(), "Sampled edges not symmetric"


class TestConfigureOptimizers:
    """Verify that optimizer configuration works."""

    def test_configure_optimizers(
        self,
        lightning_module: DiscreteDiffusionLightningModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """configure_optimizers returns an Adam-family optimizer.

        Starting state: set-up module with default optimizer_type='adamw'.
        Invariant: returned object is an AdamW optimizer.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        result = lightning_module.configure_optimizers()
        assert isinstance(result, torch.optim.AdamW)


class TestSmokeTraining:
    """End-to-end smoke test: a few training steps via the Trainer."""

    @pytest.mark.slow
    def test_trainer_fit_smoke(
        self,
        lightning_module: DiscreteDiffusionLightningModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Trainer.fit runs for a few steps without error.

        Starting state: fresh module and datamodule.
        Invariant: training completes without exceptions.
        """
        trainer = pl.Trainer(
            max_steps=3,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.fit(lightning_module, datamodule)

        # After fit, transition model should be initialised
        assert lightning_module._transition_initialized
