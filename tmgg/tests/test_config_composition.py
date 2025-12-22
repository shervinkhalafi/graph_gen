"""Integration tests for config composition and instantiation.

Test rationale:
    These tests verify that all Hydra config combinations compose correctly
    and produce valid, instantiable objects. They catch:
    - Missing required config keys
    - Invalid interpolation references
    - Incompatible config compositions
    - Model/data module instantiation errors
    - Forward pass shape mismatches

Assumptions:
    - Tests run on CPU with minimal data
    - Hydra state is cleared between tests
    - Each config produces a valid LightningModule and DataModule

Invariants:
    - hydra.compose() succeeds for all base configs
    - hydra.utils.instantiate(config.data) returns a DataModule
    - hydra.utils.instantiate(config.model) returns a LightningModule
    - Model forward pass returns tensor with correct shape
"""

from pathlib import Path

import hydra
import pytest
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

# Base configs to test with their default model/data compositions
BASE_CONFIGS = [
    "base_config_spectral",
    "base_config_attention",
    "base_config_gnn",
    "base_config_hybrid",
    "base_config_digress",
]


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear Hydra global state before each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.fixture
def exp_config_path() -> Path:
    """Path to experiment configs."""
    return Path(__file__).parent.parent / "src" / "tmgg" / "exp_configs"


def get_minimal_overrides(tmp_path: Path) -> list[str]:
    """Generate minimal overrides for testing config composition.

    These overrides ensure configs can be resolved without runtime
    dependencies (Hydra output dir, S3, W&B).
    """
    return [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        "trainer.max_steps=2",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        # Disable logger config (tensorboard/wandb)
        "~logger",
        # Minimal data for fast instantiation
        "data.batch_size=2",
        "data.num_workers=0",
        # Override interpolations that depend on Hydra runtime
        f"hydra.run.dir={tmp_path}",
    ]


@pytest.mark.integration
@pytest.mark.config
class TestConfigComposition:
    """Tests for config loading and composition."""

    @pytest.mark.parametrize("base_config", BASE_CONFIGS)
    def test_config_composes_successfully(
        self, base_config: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify that base config composes without errors.

        This tests the Hydra composition process, including:
        - Defaults resolution
        - Package directives
        - Interpolation resolution
        """
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        # Verify required keys exist
        assert "model" in cfg, f"Config {base_config} missing 'model' key"
        assert "data" in cfg, f"Config {base_config} missing 'data' key"
        assert "trainer" in cfg, f"Config {base_config} missing 'trainer' key"
        assert "paths" in cfg, f"Config {base_config} missing 'paths' key"

        # Verify _target_ is set for instantiation
        assert "_target_" in cfg.model, f"Config {base_config} model missing _target_"
        assert "_target_" in cfg.data, f"Config {base_config} data missing _target_"

    @pytest.mark.parametrize("base_config", BASE_CONFIGS)
    def test_model_instantiation(
        self, base_config: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify that model can be instantiated from config."""
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        model = hydra.utils.instantiate(cfg.model)

        # Verify model is a LightningModule
        import pytorch_lightning as pl

        assert isinstance(
            model, pl.LightningModule
        ), f"Config {base_config} model is not a LightningModule: {type(model)}"

    @pytest.mark.parametrize("base_config", BASE_CONFIGS)
    def test_data_module_instantiation(
        self, base_config: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify that data module can be instantiated from config."""
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        data_module = hydra.utils.instantiate(cfg.data)

        # Verify data module is a LightningDataModule
        import pytorch_lightning as pl

        assert isinstance(
            data_module, pl.LightningDataModule
        ), f"Config {base_config} data is not a LightningDataModule: {type(data_module)}"


# Configs that use standard adjacency matrix input
ADJACENCY_INPUT_CONFIGS = [
    "base_config_spectral",
    "base_config_attention",
    "base_config_gnn",
    "base_config_hybrid",
]

# DiGress uses eigenvector input by default, requiring special handling
EIGENVECTOR_INPUT_CONFIGS = [
    "base_config_digress",
]


@pytest.mark.integration
@pytest.mark.config
class TestForwardPass:
    """Tests for model forward pass with composed configs."""

    @pytest.mark.parametrize("base_config", ADJACENCY_INPUT_CONFIGS)
    def test_forward_pass_produces_valid_output(
        self,
        base_config: str,
        exp_config_path: Path,
        tmp_path: Path,
        sample_adjacency_batch: torch.Tensor,
    ) -> None:
        """Verify model forward pass produces tensor of correct shape.

        This tests that the full config composition produces a model
        that can process input data without errors.
        """
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        model = hydra.utils.instantiate(cfg.model)
        model.eval()

        with torch.no_grad():
            output = model(sample_adjacency_batch)

        # Output should have same shape as input (denoising task)
        assert output.shape == sample_adjacency_batch.shape, (
            f"Config {base_config}: output shape {output.shape} != "
            f"input shape {sample_adjacency_batch.shape}"
        )

        # Output should be finite (no NaN/Inf)
        assert torch.isfinite(
            output
        ).all(), f"Config {base_config}: output contains non-finite values"

    @pytest.mark.parametrize("base_config", EIGENVECTOR_INPUT_CONFIGS)
    def test_digress_forward_pass(
        self,
        base_config: str,
        exp_config_path: Path,
        tmp_path: Path,
    ) -> None:
        """Verify DiGress model forward pass.

        DiGress models with use_eigenvectors=True internally compute
        eigenvectors from the adjacency matrix. The input must be a
        square adjacency matrix matching the model's node_feature_dim
        for eigenvector extraction.
        """
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        model = hydra.utils.instantiate(cfg.model)
        model.eval()

        # Get the node feature dimension (used as k for eigenvector extraction)
        node_feature_dim = getattr(model, "_node_feature_dim", 50)
        batch_size = 4
        # For eigenvector extraction, n_nodes must be >= node_feature_dim
        n_nodes = max(node_feature_dim, 50)

        # Generate appropriately sized adjacency matrix
        adjacency_batch = torch.zeros(batch_size, n_nodes, n_nodes)
        for i in range(batch_size):
            A = torch.bernoulli(torch.full((n_nodes, n_nodes), 0.3))
            A = (A + A.T) / 2
            A.fill_diagonal_(0)
            adjacency_batch[i] = (A > 0.5).float()

        with torch.no_grad():
            output = model(adjacency_batch)

        # Output should be (batch, n_nodes, n_nodes) for edge prediction
        assert output.shape == (batch_size, n_nodes, n_nodes), (
            f"Config {base_config}: output shape {output.shape} != "
            f"expected ({batch_size}, {n_nodes}, {n_nodes})"
        )

        # Output should be finite (no NaN/Inf)
        assert torch.isfinite(
            output
        ).all(), f"Config {base_config}: output contains non-finite values"


@pytest.mark.integration
@pytest.mark.config
class TestTrainerInstantiation:
    """Tests for trainer instantiation from config."""

    @pytest.mark.parametrize("base_config", BASE_CONFIGS)
    def test_trainer_instantiation(
        self, base_config: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify trainer can be instantiated from config.

        Tests that trainer config includes all required parameters
        and produces a valid Trainer object.
        """
        overrides = get_minimal_overrides(tmp_path)

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name=base_config, overrides=overrides)

        import pytorch_lightning as pl

        trainer = hydra.utils.instantiate(cfg.trainer)

        assert isinstance(
            trainer, pl.Trainer
        ), f"Config {base_config} trainer is not a Trainer: {type(trainer)}"


@pytest.mark.integration
@pytest.mark.config
class TestStageConfigComposition:
    """Tests for stage config composition with base spectral config."""

    STAGE_CONFIGS = [
        "stage1_poc",
        "stage1_sanity",
        "stage2_validation",
        "stage3_diversity",
        "stage4_benchmarks",
        "stage5_full",
    ]

    @pytest.mark.parametrize("stage_name", STAGE_CONFIGS)
    def test_stage_config_composes_with_base(
        self, stage_name: str, exp_config_path: Path, tmp_path: Path
    ) -> None:
        """Verify stage config composes correctly with base spectral config."""
        overrides = get_minimal_overrides(tmp_path)
        overrides.append(f"+stage={stage_name}")

        with initialize_config_dir(
            version_base=None,
            config_dir=str(exp_config_path),
        ):
            cfg = compose(config_name="base_config_spectral", overrides=overrides)

        # Verify stage was applied
        assert (
            cfg.get("stage") == stage_name
        ), f"Stage {stage_name} not applied correctly"

        # Verify essential keys still exist
        assert "model" in cfg
        assert "data" in cfg
        assert "trainer" in cfg
