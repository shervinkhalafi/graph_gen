"""Tiny-model full-flow tests: train -> validation -> test for every experiment type.

Test rationale
--------------
Prior to these tests, no test in the suite called ``trainer.test()``, leaving
``test_step`` and ``on_test_epoch_end`` completely unverified. Several
experiment types also lacked any ``trainer.fit()`` coverage through their
LightningModule (only raw model tests existed). This file ensures every
experiment type survives a complete 1-batch train/val/test cycle with a
deliberately tiny model and dataset, catching wiring bugs (wrong shapes,
missing attributes, broken callbacks) without the cost of real training.

Invariants
~~~~~~~~~~
- ``trainer.fit()`` and ``trainer.test()`` complete without exceptions.
- Each test finishes within ~15 seconds on CPU.

Each experiment type gets its own test class so failures are immediately
attributable to a specific experiment.
"""

from __future__ import annotations

import pytest
import pytorch_lightning as pl

from tmgg.experiment_utils.data.data_module import GraphDataModule

N_NODES = 16
N_GRAPHS = 8
BATCH_SIZE = 4
NOISE_LEVELS = [0.1, 0.2]


def _tiny_trainer() -> pl.Trainer:
    """Trainer configured for a single tiny batch through train/val/test."""
    return pl.Trainer(
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


@pytest.fixture
def denoising_data_module() -> GraphDataModule:
    """Tiny SBM data module shared by all denoising experiments."""
    return GraphDataModule(
        dataset_name="sbm",
        dataset_config={"num_nodes": N_NODES, "num_graphs": N_GRAPHS},
        batch_size=BATCH_SIZE,
        noise_levels=NOISE_LEVELS,
    )


# ── Denoising experiments ─────────────────────────────────────────────


class TestSpectralArchFullFlow:
    """spectral_arch_denoising: filter_bank through the full Lightning path."""

    def test_train_val_test(self, denoising_data_module: GraphDataModule) -> None:
        from tmgg.experiments.spectral_arch_denoising.lightning_module import (
            SpectralDenoisingLightningModule,
        )

        module = SpectralDenoisingLightningModule(
            model_type="filter_bank",
            k=4,
            max_nodes=N_NODES,
            learning_rate=1e-3,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, denoising_data_module)
        trainer.test(module, denoising_data_module)
        assert trainer.current_epoch == 1


class TestGNNFullFlow:
    """gnn_denoising: GNN model through the full Lightning path."""

    def test_train_val_test(self, denoising_data_module: GraphDataModule) -> None:
        from tmgg.experiments.gnn_denoising.lightning_module import (
            GNNDenoisingLightningModule,
        )

        module = GNNDenoisingLightningModule(
            model_type="GNN",
            num_layers=1,
            num_terms=2,
            feature_dim_in=8,
            feature_dim_out=4,
            learning_rate=1e-3,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, denoising_data_module)
        trainer.test(module, denoising_data_module)
        assert trainer.current_epoch == 1


class TestGNNTransformerFullFlow:
    """gnn_transformer_denoising: hybrid GNN+Transformer through Lightning."""

    def test_train_val_test(self, denoising_data_module: GraphDataModule) -> None:
        from tmgg.experiments.gnn_transformer_denoising.lightning_module import (
            HybridDenoisingLightningModule,
        )

        module = HybridDenoisingLightningModule(
            gnn_num_layers=1,
            gnn_num_terms=2,
            gnn_feature_dim_in=8,
            gnn_feature_dim_out=4,
            transformer_num_layers=1,
            transformer_num_heads=2,
            learning_rate=1e-3,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, denoising_data_module)
        trainer.test(module, denoising_data_module)
        assert trainer.current_epoch == 1


class TestDigressFullFlow:
    """digress_denoising: GraphTransformer through the full Lightning path."""

    def test_train_val_test(self, denoising_data_module: GraphDataModule) -> None:
        from tmgg.experiments.digress_denoising.lightning_module import (
            DigressDenoisingLightningModule,
        )

        module = DigressDenoisingLightningModule(
            k=4,
            n_layers=2,
            hidden_mlp_dims={"X": 16, "E": 8, "y": 16},
            hidden_dims={"dx": 16, "de": 8, "dy": 16, "n_head": 2},
            output_dims={"X": 0, "E": 1, "y": 0},
            loss_type="MSE",
            noise_type="digress",
            learning_rate=1e-3,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, denoising_data_module)
        trainer.test(module, denoising_data_module)
        assert trainer.current_epoch == 1


class TestBaselineFullFlow:
    """lin_mlp_baseline_denoising: both linear and mlp through Lightning."""

    @pytest.mark.parametrize("model_type", ["linear", "mlp"])
    def test_train_val_test(
        self, model_type: str, denoising_data_module: GraphDataModule
    ) -> None:
        from tmgg.experiments.lin_mlp_baseline_denoising.lightning_module import (
            BaselineLightningModule,
        )

        module = BaselineLightningModule(
            model_type=model_type,
            max_nodes=N_NODES,
            hidden_dim=32,
            num_layers=1,
            learning_rate=1e-3,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, denoising_data_module)
        trainer.test(module, denoising_data_module)
        assert trainer.current_epoch == 1


# ── Generative experiments ─────────────────────────────────────────────


class TestGaussianGenerativeFullFlow:
    """gaussian_diffusion_generative: diffusion training + MMD validation."""

    def test_train_val_test(self) -> None:
        from tmgg.experiments.gaussian_diffusion_generative.datamodule import (
            GraphDistributionDataModule,
        )
        from tmgg.experiments.gaussian_diffusion_generative.lightning_module import (
            GenerativeLightningModule,
        )

        module = GenerativeLightningModule(
            model_type="self_attention",
            model_config={"k": 4},
            num_diffusion_steps=5,
            eval_num_samples=4,
            learning_rate=1e-3,
        )
        datamodule = GraphDistributionDataModule(
            dataset_type="sbm",
            num_nodes=N_NODES,
            num_graphs=N_GRAPHS,
            batch_size=BATCH_SIZE,
            dataset_config={"num_blocks": 2, "p_in": 0.7, "p_out": 0.1},
            seed=42,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, datamodule)
        trainer.test(module, datamodule)
        assert trainer.current_epoch == 1


class TestDiscreteGenerativeFullFlow:
    """discrete_diffusion_generative: categorical diffusion through Lightning."""

    def test_train_val_test(self) -> None:
        from tmgg.experiments.discrete_diffusion_generative.datamodule import (
            SyntheticCategoricalDataModule,
        )
        from tmgg.experiments.discrete_diffusion_generative.lightning_module import (
            DiscreteDiffusionLightningModule,
        )
        from tmgg.models.digress.discrete_transformer import (
            DiscreteGraphTransformer,
        )
        from tmgg.models.digress.noise_schedule import (
            PredefinedNoiseScheduleDiscrete,
        )

        diffusion_steps = 5
        dx, de = 2, 2

        model = DiscreteGraphTransformer(
            n_layers=2,
            input_dims={"X": dx, "E": de, "y": 1},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 16, "dy": 16, "n_head": 2},
            output_dims={"X": dx, "E": de, "y": 0},
        )
        noise_schedule = PredefinedNoiseScheduleDiscrete("cosine", diffusion_steps)

        module = DiscreteDiffusionLightningModule(
            model=model,
            noise_schedule=noise_schedule,
            diffusion_steps=diffusion_steps,
            transition_type="marginal",
            eval_num_samples=4,
        )
        datamodule = SyntheticCategoricalDataModule(
            num_nodes=N_NODES,
            num_graphs=N_GRAPHS,
            batch_size=BATCH_SIZE,
            seed=42,
        )
        trainer = _tiny_trainer()
        trainer.fit(module, datamodule)
        trainer.test(module, datamodule)
        assert trainer.current_epoch == 1
