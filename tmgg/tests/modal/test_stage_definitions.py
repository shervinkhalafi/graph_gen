"""Tests for modal stage definitions.

Test Rationale
--------------
Stage definitions drive GPU experiments on Modal. Config composition or
instantiation errors waste GPU hours. These tests catch issues locally:
- Missing config keys or invalid interpolations
- Model architecture incompatibilities
- Training step failures (gradient flow, shape mismatches)

Two modes:
- Quick (default): One config per architecture for fast feedback
- Exhaustive (marked with @pytest.mark.exhaustive): All HP combinations × seeds

Invariants:
- All stage definitions load without error
- All architecture configs exist and can be loaded
- All models instantiate as LightningModules
- Single training step executes with gradient flow

Note on Hydra limitations:
    Hydra's compose() defaults override doesn't work with path-based imports
    like `models/spectral/linear_pe@model`. This test uses the same approach
    as Modal's generate_configs.py: load model configs directly and merge
    with the base config to test instantiation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import hydra
import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

# Mock modal before importing tmgg.modal modules
mock_modal = MagicMock()
mock_modal.exception = MagicMock()
mock_modal.exception.NotFoundError = type("NotFoundError", (Exception,), {})
sys.modules["modal"] = mock_modal
sys.modules["modal.exception"] = mock_modal.exception

from tmgg.modal.stage_definitions import (  # noqa: E402
    list_stages,
    load_stage_definition,
)

# Discover all stages dynamically
ALL_STAGES = sorted(list_stages())

# Path to experiment configs
EXP_CONFIGS_PATH = Path(__file__).parent.parent.parent / "src" / "tmgg" / "exp_configs"


@pytest.fixture(autouse=True)
def clear_hydra():
    """Clear Hydra global state before and after each test."""
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def get_unique_architectures(stage_name: str) -> list[str]:
    """Get unique architecture paths from a stage definition."""
    stage_def = load_stage_definition(stage_name)
    return stage_def["architectures"]


def get_all_unique_architectures() -> list[tuple[str, str]]:
    """Get all unique (stage, architecture) pairs across all stages."""
    pairs = []
    for stage_name in ALL_STAGES:
        for arch in get_unique_architectures(stage_name):
            pairs.append((stage_name, arch))
    return pairs


def load_model_config(arch_path: str) -> DictConfig:
    """Load a model config YAML file directly.

    Parameters
    ----------
    arch_path
        Architecture path like "models/spectral/linear_pe".

    Returns
    -------
    DictConfig
        The model configuration.
    """
    model_config_path = EXP_CONFIGS_PATH / f"{arch_path}.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    return OmegaConf.load(model_config_path)  # type: ignore[return-value]


def compose_base_config(base_config: str, tmp_path: Path) -> DictConfig:
    """Compose a base config with minimal overrides.

    Parameters
    ----------
    base_config
        Base config name like "base_config_spectral".
    tmp_path
        Temporary directory for output paths.

    Returns
    -------
    DictConfig
        Composed config with resolved interpolations.
    """
    overrides = [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        "trainer.max_steps=1",
        "trainer.accelerator=cpu",
        "~logger",
        "data.batch_size=2",
        "data.num_workers=0",
        "seed=1",
        f"hydra.run.dir={tmp_path}",
    ]

    with initialize_config_dir(
        version_base=None,
        config_dir=str(EXP_CONFIGS_PATH),
    ):
        cfg = compose(config_name=base_config, overrides=overrides)

    return cfg


def merge_model_config(base_cfg: DictConfig, model_cfg: DictConfig) -> dict[str, Any]:
    """Merge model config with base config model section.

    Follows the same approach as Modal's generate_configs.py: start with
    resolved base config values, then override with non-interpolated values
    from the architecture-specific config.

    Parameters
    ----------
    base_cfg
        Composed base config.
    model_cfg
        Architecture-specific model config.

    Returns
    -------
    dict
        Merged model config as plain dict.
    """
    # Start with base model config values
    base_model_dict = OmegaConf.to_container(base_cfg.model, resolve=True)
    merged_dict: dict[str, Any] = dict(base_model_dict)  # type: ignore[arg-type]

    # Override with non-interpolation values from arch config
    for key in model_cfg:
        str_key = str(key)
        node = model_cfg._get_node(str_key)
        # Skip interpolations (they reference parent config)
        if node is not None and not OmegaConf.is_interpolation(model_cfg, str_key):
            value = model_cfg[str_key]
            # Convert OmegaConf objects to containers, keep primitives as-is
            if OmegaConf.is_config(value):
                merged_dict[str_key] = OmegaConf.to_container(value, resolve=False)
            else:
                merged_dict[str_key] = value

    return merged_dict


def create_test_batch_for_model(model_cfg: dict[str, Any]) -> torch.Tensor:
    """Create minimal test batch appropriate for model type.

    Parameters
    ----------
    model_cfg
        Model config dict.

    Returns
    -------
    torch.Tensor
        Batch of symmetric adjacency matrices.
    """
    target = model_cfg.get("_target_", "")

    # DiGress needs larger graphs for eigenvector extraction
    if "digress" in target.lower():
        k = model_cfg.get("k", 50)
        n = max(k, 50)
    else:
        n = 20

    batch_size = 2
    A = torch.rand(batch_size, n, n)
    A = (A + A.transpose(-1, -2)) / 2  # Symmetrize
    for i in range(batch_size):
        A[i].fill_diagonal_(0)

    return A


class TestStageDefinitionsLoad:
    """Verify all stage definitions load without errors."""

    @pytest.mark.parametrize("stage_name", ALL_STAGES)
    def test_stage_loads(self, stage_name: str) -> None:
        """Stage definition YAML loads and contains required keys."""
        defn = load_stage_definition(stage_name)

        assert "name" in defn, f"Stage {stage_name} missing 'name'"
        assert "base_config" in defn, f"Stage {stage_name} missing 'base_config'"
        assert "architectures" in defn, f"Stage {stage_name} missing 'architectures'"
        assert (
            len(defn["architectures"]) > 0
        ), f"Stage {stage_name} has no architectures"
        assert (
            "hyperparameters" in defn
        ), f"Stage {stage_name} missing 'hyperparameters'"
        assert "seeds" in defn, f"Stage {stage_name} missing 'seeds'"
        assert (
            "run_id_template" in defn
        ), f"Stage {stage_name} missing 'run_id_template'"


@pytest.mark.integration
@pytest.mark.config
class TestStageArchitectureConfigs:
    """Verify all architecture configs referenced by stages exist and are valid."""

    @pytest.mark.parametrize("stage_name", ALL_STAGES)
    def test_architecture_configs_exist(self, stage_name: str) -> None:
        """All architecture configs referenced by stage exist as files."""
        for arch in get_unique_architectures(stage_name):
            model_config_path = EXP_CONFIGS_PATH / f"{arch}.yaml"
            assert (
                model_config_path.exists()
            ), f"Stage {stage_name} references missing config: {arch}.yaml"

    @pytest.mark.parametrize("stage_name", ALL_STAGES)
    def test_architecture_configs_load(self, stage_name: str) -> None:
        """All architecture configs can be loaded as valid YAML."""
        for arch in get_unique_architectures(stage_name):
            model_cfg = load_model_config(arch)
            assert (
                "_target_" in model_cfg
            ), f"Architecture {arch} missing '_target_' key"


@pytest.mark.integration
@pytest.mark.config
class TestStageConfigComposition:
    """Verify base configs compose with resolved interpolations."""

    @pytest.mark.parametrize("stage_name", ALL_STAGES)
    def test_base_config_composes(self, stage_name: str, tmp_path: Path) -> None:
        """Base config for stage composes without unresolved interpolations."""
        stage_def = load_stage_definition(stage_name)
        base_config = stage_def["base_config"]

        cfg = compose_base_config(base_config, tmp_path)

        # Verify essential keys exist
        assert "model" in cfg, "Config missing 'model' key"
        assert "data" in cfg, "Config missing 'data' key"
        assert "trainer" in cfg, "Config missing 'trainer' key"

        # Verify config can be resolved (no unresolved interpolations in data/trainer)
        OmegaConf.to_container(cfg.data, resolve=True)
        OmegaConf.to_container(cfg.trainer, resolve=True)


@pytest.mark.integration
class TestStageModelInstantiation:
    """Verify models can be instantiated from stage configs."""

    @pytest.mark.parametrize("stage_name", ALL_STAGES)
    def test_architecture_instantiation(self, stage_name: str, tmp_path: Path) -> None:
        """Each architecture in stage instantiates correctly."""
        stage_def = load_stage_definition(stage_name)
        base_config = stage_def["base_config"]

        # Compose base config once
        base_cfg = compose_base_config(base_config, tmp_path)

        for arch in stage_def["architectures"]:
            # Load and merge model config
            model_cfg = load_model_config(arch)
            merged_model = merge_model_config(base_cfg, model_cfg)

            # Instantiate model
            model = hydra.utils.instantiate(merged_model)

            assert isinstance(
                model, pl.LightningModule
            ), f"Model from {arch} is not a LightningModule: {type(model)}"


@pytest.mark.integration
@pytest.mark.slow
class TestStageSingleStep:
    """Verify single training step executes without error."""

    @pytest.mark.parametrize("stage_name", ALL_STAGES)
    def test_architecture_single_step(self, stage_name: str, tmp_path: Path) -> None:
        """Single training step succeeds for each architecture in stage."""
        stage_def = load_stage_definition(stage_name)
        base_config = stage_def["base_config"]

        # Compose base config once
        base_cfg = compose_base_config(base_config, tmp_path)

        for arch in stage_def["architectures"]:
            # Load and merge model config
            model_cfg = load_model_config(arch)
            merged_model = merge_model_config(base_cfg, model_cfg)

            # Instantiate and prepare model
            model = hydra.utils.instantiate(merged_model)
            model.train()
            model.zero_grad()

            # Create appropriate test batch
            A_noisy = create_test_batch_for_model(merged_model)
            A_clean = A_noisy.clone()

            # Run forward pass and compute loss
            output = model(A_noisy)
            loss = torch.nn.functional.mse_loss(output, A_clean)
            loss.backward()

            # Verify gradients flow to at least some parameters
            has_grad = any(
                p.grad is not None and p.grad.abs().sum() > 0
                for p in model.parameters()
                if p.requires_grad
            )
            assert has_grad, (
                f"No gradients in model from {arch}. "
                "Check that forward pass is differentiable."
            )
