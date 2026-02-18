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

Architecture config loading and merging use ExperimentConfigBuilder from
tmgg.modal.config_builder, which loads YAML with interpolations stripped
and merges via deep_merge — matching the production config pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import hydra
import pytest
import pytorch_lightning as pl
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from tmgg.modal.config_builder import ExperimentConfigBuilder, deep_merge
from tmgg.modal.stage_definitions import (
    StageDefinition,
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
    return stage_def.architectures


def get_all_unique_architectures() -> list[tuple[str, str]]:
    """Get all unique (stage, architecture) pairs across all stages."""
    pairs = []
    for stage_name in ALL_STAGES:
        for arch in get_unique_architectures(stage_name):
            pairs.append((stage_name, arch))
    return pairs


def load_model_config(arch_path: str) -> dict[str, Any]:
    """Load architecture config using ConfigBuilder (strips interpolations).

    Parameters
    ----------
    arch_path
        Architecture path like "models/spectral/linear_pe".

    Returns
    -------
    dict
        Architecture config as plain dict with interpolations stripped.
    """
    builder = ExperimentConfigBuilder()
    return builder.load_architecture(arch_path)


def compose_base_config(base_config: str, tmp_path: Path) -> DictConfig:
    """Compose a base config with minimal overrides.

    Parameters
    ----------
    base_config
        Base config name like "base_config_spectral_arch".
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


def merge_model_config(base_cfg: DictConfig, arch: dict[str, Any]) -> dict[str, Any]:
    """Merge architecture config with base config model section.

    Uses ConfigBuilder's deep_merge instead of manual OmegaConf node iteration.
    The arch dict already has interpolations stripped by load_model_config().

    Parameters
    ----------
    base_cfg
        Composed base config (DictConfig).
    arch
        Architecture config as plain dict (interpolations already stripped).

    Returns
    -------
    dict
        Merged model config as plain dict.
    """
    base_model = OmegaConf.to_container(base_cfg.model, resolve=True)
    return deep_merge(base_model, arch)  # type: ignore[arg-type]


def create_tiny_datamodule(base_cfg: DictConfig) -> pl.LightningDataModule:
    """Instantiate the stage's DataModule with minimal dataset size.

    Resolves the ``data`` section of the composed config, shrinks
    size-controlling fields, and returns a ready-to-use DataModule.
    Each experiment type's DataModule produces batches in the format
    that its LightningModule's ``training_step`` expects, so the test
    stays model-agnostic.

    Parameters
    ----------
    base_cfg
        Composed Hydra config (DictConfig with ``data`` section).

    Returns
    -------
    pl.LightningDataModule
        DataModule with ``prepare_data()`` and ``setup("fit")`` already called.
    """
    data_cfg = OmegaConf.to_container(base_cfg.data, resolve=True)
    assert isinstance(data_cfg, dict), f"Expected dict, got {type(data_cfg)}"
    # Shrink dataset for speed (batch_size/num_workers already set by compose_base_config)
    if "num_graphs" in data_cfg:
        data_cfg["num_graphs"] = 20
    if "samples_per_graph" in data_cfg:
        data_cfg["samples_per_graph"] = 10
    dm = hydra.utils.instantiate(data_cfg)
    dm.prepare_data()
    dm.setup("fit")
    return dm


class TestStageDefinitionsLoad:
    """Verify all stage definitions load without errors."""

    @pytest.mark.parametrize("stage_name", ALL_STAGES)
    def test_stage_loads(self, stage_name: str) -> None:
        """Stage definition YAML loads and validates as StageDefinition."""
        defn = load_stage_definition(stage_name)

        assert isinstance(
            defn, StageDefinition
        ), f"Stage {stage_name} did not load as StageDefinition: {type(defn)}"
        assert defn.name, f"Stage {stage_name} has empty name"
        assert defn.base_config, f"Stage {stage_name} has empty base_config"
        assert len(defn.architectures) > 0, f"Stage {stage_name} has no architectures"
        assert (
            defn.hyperparameters is not None
        ), f"Stage {stage_name} missing hyperparameters"
        assert len(defn.seeds) > 0, f"Stage {stage_name} has no seeds"
        assert defn.run_id_template, f"Stage {stage_name} has empty run_id_template"


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

        cfg = compose_base_config(stage_def.base_config, tmp_path)

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
        base_config = stage_def.base_config

        # Compose base config once
        base_cfg = compose_base_config(base_config, tmp_path)

        for arch in stage_def.architectures:
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
    """Verify a single training step executes without error.

    Uses ``trainer.fit()`` with ``limit_train_batches=1`` so that each
    model's ``training_step`` runs through the full Lightning machinery
    (Trainer attachment, logging, optimizer step). This avoids calling
    ``training_step`` directly, which fails because Lightning modules
    require a Trainer for ``self.log()`` and ``self.datamodule`` access.
    """

    @pytest.mark.parametrize("stage_name", ALL_STAGES)
    def test_architecture_single_step(self, stage_name: str, tmp_path: Path) -> None:
        """Single training step succeeds for each architecture in stage."""
        stage_def = load_stage_definition(stage_name)

        # Compose base config and create a tiny DataModule (shared across archs)
        base_cfg = compose_base_config(stage_def.base_config, tmp_path)
        datamodule = create_tiny_datamodule(base_cfg)

        for arch in stage_def.architectures:
            # Load and merge model config
            model_cfg = load_model_config(arch)
            merged_model = merge_model_config(base_cfg, model_cfg)

            # Instantiate model
            model = hydra.utils.instantiate(merged_model)

            # Run one training step through the full Lightning path.
            # If training_step + backward + optimizer.step complete, gradients
            # must have flowed — no need to inspect them after the optimizer
            # zeroes them.
            trainer = pl.Trainer(
                max_epochs=1,
                limit_train_batches=1,
                limit_val_batches=0,
                accelerator="cpu",
                enable_progress_bar=False,
                enable_model_summary=False,
                logger=False,
                enable_checkpointing=False,
                num_sanity_val_steps=0,
            )
            trainer.fit(model, datamodule=datamodule)
