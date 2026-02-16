"""Single-step and full-flow tests for DigressDenoisingLightningModule.

Test rationale
--------------
Prior to these tests, the DiGress LightningModule path was never exercised:
``tests/debug/test_digress_sanity.py`` tested the raw ``GraphTransformer``
model directly, bypassing Lightning wiring (noise sampling from the
datamodule, loss computation through the base class, validation/test
metrics). These tests close that gap with both a fast single-step check
and a full train-validate-test flow using ``Trainer.fit`` + ``Trainer.test``.

Invariants
~~~~~~~~~~
- ``training_step`` produces a finite, positive scalar loss.
- Gradients propagate to model parameters after backward.
- A 1-epoch ``Trainer.fit`` followed by ``Trainer.test`` completes
  without error on a tiny SBM dataset.
"""

from unittest.mock import MagicMock, patch

import pytest
import pytorch_lightning as pl
import torch

from tmgg.experiment_utils.data.data_module import GraphDataModule
from tmgg.experiments.digress_denoising.lightning_module import (
    DigressDenoisingLightningModule,
)

N_NODES = 16


@pytest.fixture
def sample_batch() -> torch.Tensor:
    """4 binary symmetric adjacency matrices with block structure, 16 nodes."""
    batch = torch.zeros(4, N_NODES, N_NODES)
    half = N_NODES // 2
    batch[:, :half, :half] = torch.bernoulli(torch.full((4, half, half), 0.7))
    batch[:, half:, half:] = torch.bernoulli(torch.full((4, half, half), 0.5))
    batch = (batch + batch.transpose(-2, -1)) / 2
    batch = (batch > 0.5).float()
    batch.diagonal(dim1=-2, dim2=-1).zero_()
    return batch


@pytest.fixture
def data_module() -> GraphDataModule:
    return GraphDataModule(
        dataset_name="sbm",
        dataset_config={"num_nodes": N_NODES, "num_graphs": 8},
        batch_size=4,
        noise_levels=[0.1, 0.2],
    )


@pytest.fixture
def module() -> DigressDenoisingLightningModule:
    """Small DiGress module suitable for fast CPU tests."""
    return DigressDenoisingLightningModule(
        k=8,
        n_layers=2,
        hidden_mlp_dims={"X": 32, "E": 16, "y": 32},
        hidden_dims={"dx": 32, "de": 16, "dy": 32, "n_head": 2},
        output_dims={"X": 0, "E": 1, "y": 0},
        loss_type="MSE",
        noise_type="digress",
        learning_rate=1e-3,
    )


class TestDigressSingleStep:
    """Lightweight checks that training_step runs and produces valid output."""

    def test_training_step_finite_loss(
        self,
        module: DigressDenoisingLightningModule,
        data_module: GraphDataModule,
        sample_batch: torch.Tensor,
    ) -> None:
        """training_step should return a finite positive loss."""
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_trainer.datamodule = data_module
        module._trainer = mock_trainer  # pyright: ignore[reportPrivateUsage]

        with patch.object(module, "log", return_value=None):
            result = module.training_step(sample_batch, batch_idx=0)

        assert torch.isfinite(result["loss"])  # pyright: ignore[reportArgumentType]
        assert result["loss"] > 0  # pyright: ignore[reportOperatorIssue]

    def test_gradients_flow(
        self,
        module: DigressDenoisingLightningModule,
        data_module: GraphDataModule,
        sample_batch: torch.Tensor,
    ) -> None:
        """Gradients should propagate to model parameters after backward."""
        mock_trainer = MagicMock(spec=pl.Trainer)
        mock_trainer.datamodule = data_module
        module._trainer = mock_trainer  # pyright: ignore[reportPrivateUsage]

        with patch.object(module, "log", return_value=None):
            result = module.training_step(sample_batch, batch_idx=0)

        result["loss"].backward()
        params_with_grad = [p for p in module.model.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0, "No gradients flowed to model parameters"


class TestDigressFullFlow:
    """Full train -> validation -> test flow through the Lightning Trainer."""

    def test_train_val_test_completes(
        self,
        module: DigressDenoisingLightningModule,
        data_module: GraphDataModule,
    ) -> None:
        """One epoch of fit + test should complete without error.

        Starting state: fresh module and datamodule (SBM, 16 nodes, 8 graphs).
        Invariant: Trainer completes fit and test without exceptions.
        """
        trainer = pl.Trainer(
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            limit_test_batches=1,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
            num_sanity_val_steps=0,
        )
        trainer.fit(module, data_module)
        trainer.test(module, data_module)

        assert trainer.current_epoch == 1
