"""E2E integration tests for GenerativeLightningModule (diffusion training).

Test rationale:
    Verifies the complete diffusion-based graph generation pipeline:
    1. Model instantiation for all architectures
    2. Training step with diffusion noise and loss computation
    3. Validation with MMD metric computation
    4. Graph sampling with structural validity checks
    5. Full training loop across all dataset types

Invariants:
    - Model output shape matches input shape (adjacency matrix reconstruction)
    - Loss is finite and positive
    - Generated graphs are binary, symmetric, with zero diagonal
    - MMD metrics are non-negative floats
"""

from unittest.mock import patch

import pytest
import pytorch_lightning as pl
import torch

from tmgg.experiments.generative.datamodule import GraphDistributionDataModule
from tmgg.experiments.generative.lightning_module import GenerativeLightningModule

# Dataset configurations for parametrized tests (excludes LFR due to generation complexity)
DATASET_CONFIGS: dict[str, dict] = {
    "sbm": {"num_blocks": 2, "p_in": 0.7, "p_out": 0.1},
    "erdos_renyi": {"p": 0.3},
    "regular": {"d": 3},
    "tree": {},
    "watts_strogatz": {"k": 4, "p": 0.3},
    "random_geometric": {"radius": 0.4},
}


class TestGenerativeModuleInstantiation:
    """Verify all model architectures instantiate correctly."""

    @pytest.mark.parametrize(
        "model_type",
        [
            "linear_pe",
            "filter_bank",
            "self_attention",
            "self_attention_mlp",
            "multilayer_attention",
            "gnn",
            "gnn_sym",
            "hybrid",
        ],
    )
    def test_model_type_instantiates(self, model_type: str) -> None:
        """Each supported model_type should produce a valid model."""
        module = GenerativeLightningModule(
            model_type=model_type,
            model_config={"k": 8},
            num_diffusion_steps=10,
        )
        assert module.model is not None
        assert hasattr(module.model, "forward")

    def test_invalid_model_type_raises(self) -> None:
        """Unknown model_type should raise ValueError with informative message."""
        with pytest.raises(ValueError, match="model_type must be one of"):
            GenerativeLightningModule(model_type="invalid_type")


class TestGenerativeTrainingStep:
    """Verify training step produces valid loss and gradients."""

    @pytest.fixture
    def module_and_batch(self) -> tuple[GenerativeLightningModule, torch.Tensor]:
        """Create module and a batch of simple block-structured graphs."""
        module = GenerativeLightningModule(
            model_type="self_attention",
            model_config={"k": 8},
            num_diffusion_steps=10,
        )
        batch = torch.zeros(4, 16, 16)
        # Create simple block structure
        batch[:, :8, :8] = torch.bernoulli(torch.full((4, 8, 8), 0.6))
        batch = (batch + batch.transpose(-2, -1)) / 2
        batch = (batch > 0.5).float()
        batch.diagonal(dim1=-2, dim2=-1).zero_()
        return module, batch

    def test_training_step_finite_loss(
        self, module_and_batch: tuple[GenerativeLightningModule, torch.Tensor]
    ) -> None:
        """Training step should produce a finite scalar loss."""
        module, batch = module_and_batch
        with patch.object(module, "log"):
            result = module.training_step(batch, batch_idx=0)
        assert torch.isfinite(result["loss"])  # pyright: ignore[reportArgumentType]
        assert result["loss"] > 0  # pyright: ignore[reportOperatorIssue]

    def test_gradients_flow(
        self, module_and_batch: tuple[GenerativeLightningModule, torch.Tensor]
    ) -> None:
        """Gradients should propagate to model parameters after backward pass."""
        module, batch = module_and_batch
        with patch.object(module, "log"):
            result = module.training_step(batch, batch_idx=0)
        result["loss"].backward()
        params_with_grad = [p for p in module.model.parameters() if p.grad is not None]
        assert len(params_with_grad) > 0, "No gradients flowed to model parameters"


class TestGenerativeSampling:
    """Verify sample() produces structurally valid graphs."""

    @pytest.fixture
    def module(self) -> GenerativeLightningModule:
        """Create a minimal module for sampling tests."""
        return GenerativeLightningModule(
            model_type="self_attention",
            model_config={"k": 8},
            num_diffusion_steps=10,
        )

    def test_sample_shapes(self, module: GenerativeLightningModule) -> None:
        """Sampled graphs should have correct count and shape."""
        samples = module.sample(num_graphs=3, num_nodes=16, num_steps=5)
        assert len(samples) == 3
        assert all(s.shape == (16, 16) for s in samples)

    def test_samples_binary(self, module: GenerativeLightningModule) -> None:
        """Sampled adjacency matrices should contain only 0s and 1s."""
        samples = module.sample(num_graphs=3, num_nodes=16, num_steps=5)
        for s in samples:
            assert torch.all((s == 0) | (s == 1)), "Non-binary values in sampled graph"

    def test_samples_symmetric(self, module: GenerativeLightningModule) -> None:
        """Sampled adjacency matrices should be symmetric."""
        samples = module.sample(num_graphs=3, num_nodes=16, num_steps=5)
        for s in samples:
            assert torch.allclose(s, s.T), "Sampled graph is not symmetric"

    def test_samples_zero_diagonal(self, module: GenerativeLightningModule) -> None:
        """Sampled adjacency matrices should have zero diagonal (no self-loops)."""
        samples = module.sample(num_graphs=3, num_nodes=16, num_steps=5)
        for s in samples:
            assert torch.all(s.diagonal() == 0), "Sampled graph has self-loops"


class TestGenerativeValidation:
    """Verify validation step and MMD metric computation."""

    @pytest.fixture
    def module_and_batch(self) -> tuple[GenerativeLightningModule, torch.Tensor]:
        """Create module with small eval samples for validation tests."""
        module = GenerativeLightningModule(
            model_type="self_attention",
            model_config={"k": 8},
            num_diffusion_steps=10,
            eval_num_samples=4,  # Small for fast test
        )
        batch = torch.zeros(4, 16, 16)
        batch[:, :8, :8] = torch.bernoulli(torch.full((4, 8, 8), 0.6))
        batch = (batch + batch.transpose(-2, -1)) / 2
        batch = (batch > 0.5).float()
        batch.diagonal(dim1=-2, dim2=-1).zero_()
        return module, batch

    def test_validation_accumulates_ref_graphs(
        self, module_and_batch: tuple[GenerativeLightningModule, torch.Tensor]
    ) -> None:
        """Validation step should accumulate reference graphs for MMD."""
        module, batch = module_and_batch
        with patch.object(module, "log"):
            module.validation_step(batch, batch_idx=0)
        # Access private attribute to verify accumulation
        assert len(module._ref_graphs) > 0  # pyright: ignore[reportPrivateUsage]

    def test_validation_epoch_end_computes_mmd(
        self, module_and_batch: tuple[GenerativeLightningModule, torch.Tensor]
    ) -> None:
        """on_validation_epoch_end should compute and log MMD metrics."""
        module, batch = module_and_batch
        logged_metrics: dict[str, float] = {}

        def capture_log(name: str, value: float, **_: object) -> None:
            logged_metrics[name] = value

        with patch.object(module, "log", side_effect=capture_log):
            module.validation_step(batch, batch_idx=0)
            module.on_validation_epoch_end()

        # Check that MMD metrics were logged
        assert "val/degree_mmd" in logged_metrics
        assert "val/clustering_mmd" in logged_metrics
        assert "val/spectral_mmd" in logged_metrics

        # All MMD values should be non-negative
        for key in ["val/degree_mmd", "val/clustering_mmd", "val/spectral_mmd"]:
            assert logged_metrics[key] >= 0, f"{key} is negative"


class TestGenerativeE2EPerDataset:
    """Full training loop across all dataset types."""

    @pytest.mark.parametrize("dataset_type", list(DATASET_CONFIGS.keys()))
    def test_short_training_run(self, dataset_type: str, tmp_path: object) -> None:
        """Short training run should complete without errors and produce finite loss.

        Runs 2 epochs with small graphs (16 nodes) and batches (4 graphs).
        """
        module = GenerativeLightningModule(
            model_type="self_attention",
            model_config={"k": 8},
            num_diffusion_steps=10,
            eval_num_samples=8,
        )

        datamodule = GraphDistributionDataModule(
            dataset_type=dataset_type,
            num_nodes=16,
            num_graphs=20,
            batch_size=4,
            dataset_config=DATASET_CONFIGS[dataset_type],
            seed=42,
        )

        trainer = pl.Trainer(
            max_epochs=2,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_model_summary=False,
            logger=False,
            enable_checkpointing=False,
        )

        trainer.fit(module, datamodule)
        assert trainer.current_epoch == 2


class TestGenerativeDataModule:
    """Verify datamodule generates valid data for all graph types."""

    @pytest.mark.parametrize("dataset_type", list(DATASET_CONFIGS.keys()))
    def test_datamodule_setup(self, dataset_type: str) -> None:
        """DataModule should set up train/val/test splits correctly."""
        datamodule = GraphDistributionDataModule(
            dataset_type=dataset_type,
            num_nodes=16,
            num_graphs=20,
            batch_size=4,
            dataset_config=DATASET_CONFIGS[dataset_type],
            seed=42,
        )

        datamodule.setup("fit")

        # Verify data was created
        assert datamodule._train_data is not None  # pyright: ignore[reportPrivateUsage]
        assert datamodule._val_data is not None  # pyright: ignore[reportPrivateUsage]

        # Verify shapes
        train_data = datamodule._train_data  # pyright: ignore[reportPrivateUsage]
        assert train_data.dim() == 3
        assert train_data.shape[1] == 16  # num_nodes
        assert train_data.shape[2] == 16

    @pytest.mark.parametrize("dataset_type", list(DATASET_CONFIGS.keys()))
    def test_generated_graphs_are_valid(self, dataset_type: str) -> None:
        """Generated graphs should be binary, symmetric, with zero diagonal."""
        datamodule = GraphDistributionDataModule(
            dataset_type=dataset_type,
            num_nodes=16,
            num_graphs=10,
            batch_size=4,
            dataset_config=DATASET_CONFIGS[dataset_type],
            seed=42,
        )

        datamodule.setup("fit")
        train_data = datamodule._train_data  # pyright: ignore[reportPrivateUsage]
        assert train_data is not None

        # Check binary values
        assert torch.all((train_data == 0) | (train_data == 1)), "Non-binary adjacency"

        # Check symmetry
        assert torch.allclose(
            train_data, train_data.transpose(-2, -1)
        ), "Adjacency not symmetric"

        # Check zero diagonal
        for i in range(train_data.shape[0]):
            assert torch.all(train_data[i].diagonal() == 0), "Self-loops present"
