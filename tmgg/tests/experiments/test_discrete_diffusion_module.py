"""Tests for DiffusionModule with CategoricalNoiseProcess.

Testing strategy
----------------
Each test uses a tiny model configuration (2 layers, 16-dim hidden, dx=de=2,
dy=0, T=10) paired with a small SyntheticCategoricalDataModule to keep
wall-clock time under a few seconds. The goal is to verify wiring correctness
(shapes, finiteness, mixin delegation) rather than training convergence.

Exercises the DiffusionModule + CategoricalNoiseProcess + CategoricalSampler
pipeline end-to-end.
"""

from __future__ import annotations

from typing import Any

import pytest
import pytorch_lightning as pl
import torch
from torch import Tensor

from tmgg.data.datasets.graph_types import GraphData
from tmgg.diffusion.noise_process import CategoricalNoiseProcess
from tmgg.diffusion.sampler import CategoricalSampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.experiments.discrete_diffusion_generative.datamodule import (
    SyntheticCategoricalDataModule,
)
from tmgg.training.evaluation_metrics.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.training.lightning_modules.diffusion_module import (
    DiffusionModule,
)

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
def unified_schedule() -> NoiseSchedule:
    return NoiseSchedule("cosine_iddpm", timesteps=_T)


@pytest.fixture()
def noise_process(
    unified_schedule: NoiseSchedule,
) -> CategoricalNoiseProcess:
    return CategoricalNoiseProcess(
        noise_schedule=unified_schedule,
        x_classes=_DX,
        e_classes=_DE,
    )


@pytest.fixture()
def sampler(
    noise_process: CategoricalNoiseProcess,
    unified_schedule: NoiseSchedule,
) -> CategoricalSampler:
    return CategoricalSampler(
        noise_process=noise_process, noise_schedule=unified_schedule
    )


@pytest.fixture()
def evaluator() -> GraphEvaluator:
    return GraphEvaluator(eval_num_samples=8, kernel="gaussian", sigma=1.0)


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
    noise_process: CategoricalNoiseProcess,
    sampler: CategoricalSampler,
    unified_schedule: NoiseSchedule,
    evaluator: GraphEvaluator,
) -> DiffusionModule:
    from tmgg.models.digress.transformer_model import GraphTransformer

    model = GraphTransformer(
        n_layers=2,
        input_dims={"X": _DX, "E": _DE, "y": 0},
        hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
        hidden_dims={"dx": 16, "de": 16, "dy": 16, "n_head": 2},
        output_dims={"X": _DX, "E": _DE, "y": _DY},
        use_timestep=True,
    )
    return DiffusionModule(
        model=model,
        noise_process=noise_process,
        sampler=sampler,
        noise_schedule=unified_schedule,
        evaluator=evaluator,
        loss_type="cross_entropy",
        num_nodes=_NUM_NODES,
        eval_every_n_steps=1,
    )


def _attach_trainer_and_setup(
    module: DiffusionModule,
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
    """Verify that setup() constructs the transition model via CategoricalNoiseProcess."""

    def test_setup_constructs_transition(
        self,
        lightning_module: DiffusionModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """After setup, the noise process transition model is populated.

        Starting state: fresh module with no transition model (marginal).
        Invariant: setup('fit') populates the transition model on the noise process.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        assert isinstance(lightning_module.noise_process, CategoricalNoiseProcess)
        assert lightning_module.noise_process.transition_model is not None

    def test_setup_is_idempotent(
        self,
        lightning_module: DiffusionModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Calling setup() twice does not error or reconstruct.

        Starting state: module already set up.
        Invariant: second call is a no-op; transition model is the same object.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        assert isinstance(lightning_module.noise_process, CategoricalNoiseProcess)
        tm_first = lightning_module.noise_process.transition_model

        lightning_module.setup("fit")
        assert lightning_module.noise_process.transition_model is tm_first


class TestForward:
    """Verify the forward pass through the model returns sensible output."""

    def test_output_shapes(
        self,
        lightning_module: DiffusionModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Forward output has correct shapes.

        Starting state: noisy batch from noise process.
        Invariant: pred.X and pred.E match batch dimensions.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        batch: GraphData = next(iter(datamodule.train_dataloader()))

        lightning_module.eval()
        t_int = torch.randint(1, _T + 1, (batch.X.shape[0],))
        z_t = lightning_module.noise_process.apply(batch, t_int)
        t_norm = t_int.float() / _T
        pred = lightning_module.model(z_t, t=t_norm)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        assert pred.X.shape == batch.X.shape  # pyright: ignore[reportUnknownMemberType]
        assert pred.E.shape == batch.E.shape  # pyright: ignore[reportUnknownMemberType]


class TestTraining:
    """Verify the training step produces finite positive loss."""

    def test_training_loss_finite_and_positive(
        self,
        lightning_module: DiffusionModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Noise -> predict -> loss pipeline yields a finite positive scalar.

        Starting state: set-up module, one batch from train dataloader.
        Invariant: cross-entropy loss is finite, positive, and a 0-d tensor.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        batch: GraphData = next(iter(datamodule.train_dataloader()))

        lightning_module.train()
        bs = batch.X.shape[0]
        t_int = torch.randint(1, _T + 1, (bs,))
        z_t = lightning_module.noise_process.apply(batch, t_int)
        t_norm = t_int.float() / _T
        pred = lightning_module.model(z_t, t=t_norm)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        loss = lightning_module._compute_loss(pred, batch)  # pyright: ignore[reportUnknownVariableType]

        assert isinstance(loss, Tensor)
        assert loss.ndim == 0, f"Expected scalar loss, got shape {loss.shape}"  # pyright: ignore[reportUnknownMemberType]
        assert torch.isfinite(loss), f"Loss is not finite: {loss.item()}"  # pyright: ignore[reportUnknownMemberType]
        assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"  # pyright: ignore[reportUnknownMemberType]


class TestValidation:
    """Verify VLB components are finite via a trainer-driven validation pass."""

    def test_vlb_terms_finite(
        self,
        lightning_module: DiffusionModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """A 1-epoch run produces finite VLB metrics.

        Starting state: fresh module and datamodule.
        Invariant: val/epoch_NLL is logged and finite after one epoch.
        DiffusionModule.validation_step uses self.log(), which requires
        an active trainer loop, so we run through the trainer.
        """
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            accelerator="cpu",
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(lightning_module, datamodule)

        logged = trainer.logged_metrics
        assert "val/epoch_NLL" in logged, f"Missing val/epoch_NLL in {list(logged)}"
        assert torch.isfinite(
            torch.tensor(float(logged["val/epoch_NLL"]))
        ), f"val/epoch_NLL not finite: {logged['val/epoch_NLL']}"


class TestConfigureOptimizers:
    """Verify that optimizer configuration works."""

    def test_configure_optimizers(
        self,
        lightning_module: DiffusionModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """configure_optimizers returns an Adam-family optimizer.

        Starting state: set-up module with default optimizer_type='adam'.
        Invariant: returned object is an Adam optimizer.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        result = lightning_module.configure_optimizers()
        assert isinstance(result, torch.optim.Adam)


class TestSmokeTraining:
    """End-to-end smoke test: a few training steps via the Trainer."""

    @pytest.mark.slow
    def test_trainer_fit_smoke(
        self,
        lightning_module: DiffusionModule,
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

        # After fit, the categorical noise process transition model should be initialised
        assert isinstance(lightning_module.noise_process, CategoricalNoiseProcess)
        assert lightning_module.noise_process.transition_model is not None


class TestDiGressAlignment:
    """Regression tests for DiGress alignment (Tasks 1-5).

    These verify that TrainLossDiscrete delegation, padding masking,
    lambda_E weighting, reconstruction at t=1, and log_pN behaviour
    for degenerate size distributions all work correctly.
    """

    def test_training_loss_uses_train_loss_discrete(
        self,
        lightning_module: DiffusionModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """_compute_loss delegates to TrainLossDiscrete for the categorical path.

        Starting state: set-up module, one batch from train dataloader.
        Invariant: loss from _compute_loss matches a manual TrainLossDiscrete call
        with the same softmaxed+cloned predictions.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)
        batch = next(iter(datamodule.train_dataloader()))

        # Get model prediction
        t_int = torch.randint(1, _T + 1, (batch.X.shape[0],))
        z_t = lightning_module.noise_process.apply(batch, t_int)
        t_norm = t_int.float() / _T
        pred = lightning_module.model(z_t, t=t_norm)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        # Loss via DiffusionModule
        loss_module = lightning_module._compute_loss(pred, batch)  # pyright: ignore[reportUnknownVariableType]

        # Loss via direct TrainLossDiscrete call
        from tmgg.training.lightning_modules.train_loss_discrete import (
            TrainLossDiscrete,
        )

        tld = TrainLossDiscrete(lambda_E=5.0)
        pred_X = torch.nn.functional.softmax(pred.X, dim=-1).clone()  # pyright: ignore[reportUnknownMemberType]
        pred_E = torch.nn.functional.softmax(pred.E, dim=-1).clone()  # pyright: ignore[reportUnknownMemberType]
        loss_direct = tld(pred_X, pred_E, batch.X, batch.E, batch.node_mask)

        assert torch.allclose(loss_module, loss_direct, atol=1e-6)  # pyright: ignore[reportUnknownMemberType]

    def test_training_loss_masks_padding(
        self,
        lightning_module: DiffusionModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Padding positions do not contribute to the training loss.

        Starting state: set-up module, batch with last 2 nodes masked as padding.
        Invariant: corrupting predictions at padding positions does not change the loss.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)
        batch = next(iter(datamodule.train_dataloader()))

        # Create a batch with some padding: zero out last 2 nodes' masks
        batch_padded = GraphData(
            X=batch.X.clone(),
            E=batch.E.clone(),
            y=batch.y,
            node_mask=batch.node_mask.clone(),
        )
        batch_padded.node_mask[:, -2:] = 0  # Mark last 2 nodes as padding

        # Get predictions
        t_int = torch.randint(1, _T + 1, (batch.X.shape[0],))
        z_t = lightning_module.noise_process.apply(batch_padded, t_int)
        t_norm = t_int.float() / _T
        pred = lightning_module.model(z_t, t=t_norm)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        loss1 = lightning_module._compute_loss(pred, batch_padded)  # pyright: ignore[reportUnknownVariableType]

        # Corrupt padding positions in prediction — should NOT affect loss
        pred_corrupted = GraphData(
            X=pred.X.clone(),  # pyright: ignore[reportUnknownMemberType]
            E=pred.E.clone(),  # pyright: ignore[reportUnknownMemberType]
            y=pred.y,  # pyright: ignore[reportUnknownMemberType]
            node_mask=pred.node_mask,  # pyright: ignore[reportUnknownMemberType]
        )
        pred_corrupted.X[:, -2:, :] = (
            torch.randn_like(pred_corrupted.X[:, -2:, :]) * 100
        )
        pred_corrupted.E[:, -2:, :, :] = (
            torch.randn_like(pred_corrupted.E[:, -2:, :, :]) * 100
        )
        pred_corrupted.E[:, :, -2:, :] = (
            torch.randn_like(pred_corrupted.E[:, :, -2:, :]) * 100
        )

        loss2 = lightning_module._compute_loss(pred_corrupted, batch_padded)  # pyright: ignore[reportUnknownVariableType]

        assert torch.allclose(loss1, loss2, atol=1e-5), (  # pyright: ignore[reportUnknownMemberType]
            f"Padding positions affected loss: {loss1.item()} vs {loss2.item()}"  # pyright: ignore[reportUnknownMemberType]
        )

    def test_lambda_E_weighting(
        self,
        noise_process: CategoricalNoiseProcess,
        sampler: CategoricalSampler,
        unified_schedule: NoiseSchedule,
        evaluator: GraphEvaluator,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """Different lambda_E values produce different loss magnitudes.

        Starting state: two modules with identical weights but lambda_E=1 vs 10.
        Invariant: same predictions yield different scalar losses (unless edge
        loss is exactly zero, which is astronomically unlikely).
        """
        from tmgg.models.digress.transformer_model import GraphTransformer

        def _make_gt_model() -> GraphTransformer:
            return GraphTransformer(
                n_layers=2,
                input_dims={"X": _DX, "E": _DE, "y": 0},
                hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
                hidden_dims={"dx": 16, "de": 16, "dy": 16, "n_head": 2},
                output_dims={"X": _DX, "E": _DE, "y": _DY},
                use_timestep=True,
            )

        common_kwargs: dict[str, Any] = dict(  # pyright: ignore[reportExplicitAny]
            noise_process=noise_process,
            sampler=sampler,
            noise_schedule=unified_schedule,
            evaluator=evaluator,
            loss_type="cross_entropy",
            num_nodes=_NUM_NODES,
            eval_every_n_steps=1,
        )
        mod_low = DiffusionModule(model=_make_gt_model(), **common_kwargs, lambda_E=1.0)  # pyright: ignore[reportCallIssue]
        mod_high = DiffusionModule(
            model=_make_gt_model(), **common_kwargs, lambda_E=10.0
        )  # pyright: ignore[reportCallIssue]

        # Copy weights so both models produce identical predictions
        mod_high.load_state_dict(mod_low.state_dict())

        # Set up both modules
        _attach_trainer_and_setup(mod_low, datamodule)
        # mod_high shares the same noise_process (already set up by mod_low)
        trainer = pl.Trainer(
            max_steps=1,
            enable_checkpointing=False,
            logger=False,
            enable_progress_bar=False,
        )
        trainer.datamodule = datamodule  # pyright: ignore[reportAttributeAccessIssue]
        mod_high.trainer = trainer
        mod_high.setup("fit")

        batch = next(iter(datamodule.train_dataloader()))
        t_int = torch.randint(1, _T + 1, (batch.X.shape[0],))
        z_t = noise_process.apply(batch, t_int)
        t_norm = t_int.float() / _T
        pred = mod_low.model(z_t, t=t_norm)  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]

        loss_low = mod_low._compute_loss(pred, batch)  # pyright: ignore[reportUnknownVariableType]
        loss_high = mod_high._compute_loss(pred, batch)  # pyright: ignore[reportUnknownVariableType]

        # With same predictions, higher lambda_E should produce a different loss
        assert not torch.allclose(loss_low, loss_high, atol=1e-6), (  # pyright: ignore[reportUnknownMemberType]
            f"lambda_E weighting had no effect: {loss_low.item()} vs {loss_high.item()}"  # pyright: ignore[reportUnknownMemberType]
        )

    def test_reconstruction_evaluates_at_t1(
        self,
        lightning_module: DiffusionModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """_compute_reconstruction_at_t1 returns finite per-graph log-probabilities.

        Starting state: set-up module, one clean batch.
        Invariant: output is a finite tensor of shape (bs,).
        """
        _attach_trainer_and_setup(lightning_module, datamodule)
        batch = next(iter(datamodule.train_dataloader()))

        recon = lightning_module._compute_reconstruction_at_t1(batch)

        assert isinstance(recon, torch.Tensor)
        assert recon.shape == (batch.X.shape[0],), f"Expected (bs,), got {recon.shape}"
        assert torch.isfinite(recon).all(), f"Non-finite reconstruction values: {recon}"

    def test_log_pN_degenerate_is_zero(
        self,
        lightning_module: DiffusionModule,
        datamodule: SyntheticCategoricalDataModule,
    ) -> None:
        """For a fixed-size dataset, log_pN should be zero (log(1) = 0).

        Starting state: set-up module using SyntheticCategoricalDataModule
        (all graphs have the same size).
        Invariant: size distribution is degenerate and log_prob of the
        fixed size equals 0.
        """
        _attach_trainer_and_setup(lightning_module, datamodule)

        assert lightning_module._size_distribution is not None
        assert lightning_module._size_distribution.is_degenerate

        node_count = torch.tensor([_NUM_NODES])
        log_p = lightning_module._size_distribution.log_prob(node_count)
        assert torch.allclose(
            log_p, torch.zeros(1), atol=1e-6
        ), f"Expected log_prob=0 for degenerate distribution, got {log_p.item()}"
