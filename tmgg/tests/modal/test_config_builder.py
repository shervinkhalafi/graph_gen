"""Tests for ExperimentConfigBuilder and its helper functions.

Test rationale
--------------
ConfigBuilder replaces the entangled OmegaConf-based config generation in
generate_configs.py with a two-phase approach: Hydra composition (Phase 1)
followed by plain-dict manipulation (Phase 2). The regression test at the
end verifies that the new pipeline produces byte-identical output to the
old one for a known stage definition.

Invariants
----------
- ``strip_interpolations`` removes *only* ``${...}``-valued entries.
- ``deep_merge`` never mutates its inputs.
- ``compose_base`` returns a dict with zero unresolved interpolations.
- ``build`` output matches ``generate_configs_for_stage`` for the same inputs.
"""

from __future__ import annotations

import itertools
from typing import Any

import pytest

from tmgg.modal.config_builder import (
    ExperimentConfigBuilder,
    deep_merge,
    get_nested,
    set_nested,
    strip_interpolations,
)

# ── Helper function unit tests ────────────────────────────────────


class TestStripInterpolations:
    """Verify that strip_interpolations removes exactly the right entries."""

    def test_removes_dollar_values(self):
        """Top-level interpolation strings are dropped."""
        result = strip_interpolations({"a": "${x}", "b": 5})
        assert result == {"b": 5}

    def test_nested(self):
        """Interpolations inside nested dicts are dropped recursively."""
        inp = {"outer": {"a": "${data.noise_type}", "b": "cosine"}}
        result = strip_interpolations(inp)
        assert result == {"outer": {"b": "cosine"}}

    def test_preserves_non_interpolation_strings(self):
        """Concrete strings that happen to contain special chars survive."""
        inp = {
            "schedule": "cosine",
            "opt": "adamw",
            "target": "tmgg.models.Foo",
        }
        result = strip_interpolations(inp)
        assert result == inp

    def test_handles_lists(self):
        """Bare interpolation items are removed from lists; others kept."""
        inp = {"items": ["${a}", 42, "real", "${b}"]}
        result = strip_interpolations(inp)
        assert result == {"items": [42, "real"]}

    def test_handles_nested_lists_with_dicts(self):
        """Dicts inside lists have their interpolations stripped recursively."""
        inp = {"items": [{"a": "${x}", "b": 1}, {"c": "real"}]}
        result = strip_interpolations(inp)
        assert result == {"items": [{"b": 1}, {"c": "real"}]}

    def test_does_not_strip_partial_interpolation(self):
        """A string that *contains* but is not entirely ``${...}`` survives.

        This matches how arch configs never embed interpolations within
        larger strings, so partial-match stripping would be unsafe.
        """
        inp = {"name": "prefix_${x}_suffix"}
        result = strip_interpolations(inp)
        assert result == inp


class TestDeepMerge:
    """deep_merge must produce correct merges without mutating inputs."""

    def test_override_wins(self):
        assert deep_merge({"a": 1}, {"a": 2}) == {"a": 2}

    def test_base_preserved(self):
        assert deep_merge({"a": 1, "b": 2}, {"a": 3}) == {"a": 3, "b": 2}

    def test_recursive(self):
        base = {"x": {"y": 1, "z": 2}}
        over = {"x": {"y": 99}}
        assert deep_merge(base, over) == {"x": {"y": 99, "z": 2}}

    def test_list_replacement(self):
        """Lists are replaced entirely, not merged element-wise."""
        base = {"items": [1, 2, 3]}
        over = {"items": [4, 5]}
        assert deep_merge(base, over) == {"items": [4, 5]}

    def test_no_mutation(self):
        base = {"x": {"y": 1}}
        over = {"x": {"y": 2}}
        _ = deep_merge(base, over)
        assert base == {"x": {"y": 1}}
        assert over == {"x": {"y": 2}}


class TestSetNested:
    """set_nested must create intermediate dicts and overwrite leaves."""

    def test_creates_intermediates(self):
        d: dict[str, Any] = {}
        set_nested(d, "model.noise_schedule.timesteps", 100)
        assert d == {"model": {"noise_schedule": {"timesteps": 100}}}

    def test_overwrites_existing(self):
        d: dict[str, Any] = {"model": {"k": 16}}
        set_nested(d, "model.k", 8)
        assert d["model"]["k"] == 8

    def test_raises_on_non_dict_intermediate(self):
        """Setting through a scalar intermediate must raise, not silently overwrite."""
        d: dict[str, Any] = {"model": "a_string"}
        with pytest.raises(TypeError, match="not a dict"):
            set_nested(d, "model.k", 8)


class TestGetNested:
    """get_nested must traverse safely and return None on missing keys."""

    def test_returns_none_on_missing(self):
        assert get_nested({"a": 1}, ("b",)) is None
        assert get_nested({"a": {"b": 1}}, ("a", "c")) is None

    def test_returns_value(self):
        d = {"a": {"b": {"c": 42}}}
        assert get_nested(d, ("a", "b", "c")) == 42


# ── ExperimentConfigBuilder method tests ──────────────────────────


class TestComposeBase:
    """compose_base should resolve all interpolations and extract W&B."""

    builder = ExperimentConfigBuilder()

    def test_resolves_interpolations(self):
        """No ``${`` string should remain anywhere in the resolved output."""
        cfg = self.builder.compose_base("base_config_spectral_arch", ["~logger"])
        _assert_no_interpolations(cfg)

    def test_extracts_wandb(self):
        cfg = self.builder.compose_base("base_config_spectral_arch", ["~logger"])
        assert "_wandb_config" in cfg
        wc = cfg["_wandb_config"]
        assert "entity" in wc
        assert "project" in wc

    def test_removes_logger(self):
        cfg = self.builder.compose_base("base_config_spectral_arch", ["~logger"])
        assert "logger" not in cfg


class TestLoadArchitecture:
    """load_architecture should strip interpolations, keep concrete values."""

    builder = ExperimentConfigBuilder()

    def test_strips_interpolations(self):
        arch = self.builder.load_architecture("models/spectral/linear_pe")
        assert "learning_rate" not in arch
        assert "weight_decay" not in arch
        assert "noise_type" not in arch

    def test_preserves_target(self):
        arch = self.builder.load_architecture("models/spectral/linear_pe")
        assert "_target_" in arch

    def test_preserves_concrete_values(self):
        arch = self.builder.load_architecture("models/spectral/linear_pe")
        assert arch["k"] == 16
        assert arch["max_nodes"] == 200
        assert arch["use_bias"] is True

    def test_discrete_preserves_nested_dicts(self):
        """Nested sub-dicts (model, noise_schedule) should survive intact."""
        arch = self.builder.load_architecture("models/discrete/discrete_default")
        assert "model" in arch
        assert "noise_schedule" in arch
        assert arch["model"]["n_layers"] == 5
        assert arch["noise_schedule"]["timesteps"] == 500

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            self.builder.load_architecture("models/nonexistent/nope")


class TestMergeArchitecture:
    """merge_architecture should overlay arch onto config['model']."""

    builder = ExperimentConfigBuilder()

    def test_preserves_training_params(self):
        """Base model values for stripped keys survive the merge."""
        config: dict[str, Any] = {
            "model": {"learning_rate": 0.001, "k": 4, "_target_": "Old"},
        }
        arch = {"_target_": "New", "k": 16, "max_nodes": 200}
        self.builder.merge_architecture(config, arch)
        assert config["model"]["learning_rate"] == 0.001

    def test_overrides_arch_params(self):
        config: dict[str, Any] = {
            "model": {"_target_": "Old", "k": 4},
        }
        arch = {"_target_": "New", "k": 16}
        self.builder.merge_architecture(config, arch)
        assert config["model"]["_target_"] == "New"
        assert config["model"]["k"] == 16


class TestApplyHyperparameters:
    """apply_hyperparameters must route dotted keys correctly."""

    builder = ExperimentConfigBuilder()

    def test_dotted_path(self):
        config: dict[str, Any] = {"model": {"k": 16}}
        self.builder.apply_hyperparameters(config, {"model.k": 8})
        assert config["model"]["k"] == 8

    def test_plus_prefix(self):
        config: dict[str, Any] = {"model": {}}
        self.builder.apply_hyperparameters(config, {"+model.k": 8})
        assert config["model"]["k"] == 8

    def test_top_level(self):
        config: dict[str, Any] = {"learning_rate": 0.001}
        self.builder.apply_hyperparameters(config, {"learning_rate": 0.0001})
        assert config["learning_rate"] == 0.0001


class TestSyncCoupledParams:
    """sync_coupled_params should propagate diffusion_steps to timesteps."""

    builder = ExperimentConfigBuilder()

    def test_syncs(self):
        config: dict[str, Any] = {
            "model": {
                "diffusion_steps": 100,
                "noise_schedule": {"timesteps": 500},
            }
        }
        self.builder.sync_coupled_params(config)
        assert config["model"]["noise_schedule"]["timesteps"] == 100

    def test_skips_when_missing(self):
        """No error when noise_schedule is absent (e.g. spectral models)."""
        config: dict[str, Any] = {"model": {"diffusion_steps": 100}}
        self.builder.sync_coupled_params(config)  # should not raise
        assert config["model"]["diffusion_steps"] == 100


class TestStripForRemote:
    """strip_for_remote should null paths and drop hydra."""

    builder = ExperimentConfigBuilder()

    def test_nulls_paths_and_drops_hydra(self):
        config: dict[str, Any] = {
            "paths": {"output_dir": "/tmp/foo", "results_dir": "/tmp/bar"},
            "hydra": {"run": {"dir": "outputs/"}},
            "model": {"_target_": "X"},
        }
        self.builder.strip_for_remote(config)
        assert config["paths"]["output_dir"] is None
        assert config["paths"]["results_dir"] is None
        assert "hydra" not in config


class TestValidate:
    """validate should reject incomplete configs loudly."""

    builder = ExperimentConfigBuilder()

    def test_missing_model_raises(self):
        with pytest.raises(ValueError, match="model"):
            self.builder.validate({"data": {}, "trainer": {}})

    def test_missing_target_raises(self):
        with pytest.raises(ValueError, match="_target_"):
            self.builder.validate({"model": {}, "data": {}, "trainer": {}})

    def test_passes_valid_config(self):
        self.builder.validate({"model": {"_target_": "X"}, "data": {}, "trainer": {}})


# ── Run ID tests ─────────────────────────────────────────────────


class TestGenerateRunId:
    """generate_run_id must format HP values consistently."""

    builder = ExperimentConfigBuilder()

    def test_formatting(self):
        run_id = self.builder.generate_run_id(
            template="exp_{arch}_{diffusion_steps}_{lr}_s{seed}",
            arch="models/discrete/discrete_default",
            dataset=None,
            hp_combo={
                "learning_rate": 1e-4,
                "model.diffusion_steps": 100,
            },
            seed=1,
        )
        assert run_id == "exp_discrete_default_T100_lr1e-4_s1"

    def test_with_dataset(self):
        run_id = self.builder.generate_run_id(
            template="{arch}_{data}_s{seed}",
            arch="models/spectral/linear_pe",
            dataset="data/sbm_default",
            hp_combo={},
            seed=42,
        )
        assert run_id == "linear_pe_sbm_default_s42"

    def test_weight_decay_formatting(self):
        run_id = self.builder.generate_run_id(
            template="{lr}_{wd}",
            arch="x",
            dataset=None,
            hp_combo={"learning_rate": 1e-4, "weight_decay": 1e-2},
            seed=0,
        )
        assert run_id == "lr1e-4_wd0.01"


# ── Integration tests (real Hydra configs) ────────────────────────


class TestBuildEndToEnd:
    """Full pipeline integration tests with real configs from the codebase."""

    builder = ExperimentConfigBuilder()

    def test_build_spectral(self):
        config = self.builder.build(
            config_name="base_config_spectral_arch",
            arch_path="models/spectral/linear_pe",
            overrides=["~logger", "learning_rate=0.001", "seed=42"],
            hp_overrides={"model.k": 8},
            seed=42,
            run_id_template="test_{arch}_{k}_s{seed}",
        )
        assert config["model"]["_target_"].endswith("SpectralDenoisingLightningModule")
        assert config["model"]["k"] == 8
        assert config["seed"] == 42
        assert "_wandb_config" in config
        assert "run_id" in config
        assert config["paths"]["output_dir"] is None
        _assert_no_interpolations(config)

    def test_build_discrete(self):
        config = self.builder.build(
            config_name="base_config_discrete_diffusion_generative",
            arch_path="models/discrete/discrete_default",
            overrides=["~logger", "learning_rate=0.0001", "seed=1"],
            hp_overrides={
                "learning_rate": 1e-4,
                "model.diffusion_steps": 100,
            },
            seed=1,
            run_id_template="discrete_gen_{arch}_{diffusion_steps}_{lr}_s{seed}",
        )
        assert config["model"]["diffusion_steps"] == 100
        assert config["model"]["noise_schedule"]["timesteps"] == 100
        assert config["run_id"] == "discrete_gen_discrete_default_T100_lr1e-4_s1"
        _assert_no_interpolations(config)

    def test_build_digress(self):
        config = self.builder.build(
            config_name="base_config_digress",
            arch_path="models/digress/digress_transformer",
            overrides=["~logger", "learning_rate=0.001", "seed=42"],
            hp_overrides={"learning_rate": 0.001},
            seed=42,
            run_id_template="digress_{arch}_{lr}_s{seed}",
        )
        assert config["model"]["_target_"].endswith("DigressDenoisingLightningModule")
        assert config["model"]["use_eigenvectors"] is True
        assert config["model"]["k"] == 50
        _assert_no_interpolations(config)


# ── Regression test (old pipeline vs new) ─────────────────────────


class TestBuildConsistency:
    """Verify that ``build()`` and ``generate_configs_for_stage()`` agree.

    Both code paths now use ``ExperimentConfigBuilder`` internally, so this
    is a consistency check rather than a regression test against the old
    OmegaConf pipeline (which was deleted in commit dbb4009).
    """

    def test_build_via_generate_configs_consistent(self):
        from tmgg.modal.cli.generate_configs import generate_configs_for_stage
        from tmgg.modal.stage_definitions import load_stage_definition

        stage_def = load_stage_definition("stage_discrete_gen")

        # --- Old pipeline: generate all configs, pick the first one ---
        old_configs = generate_configs_for_stage(stage_def)
        old_cfg = old_configs[0]

        # --- New pipeline: replicate the same inputs ---
        # The first combo from stage_discrete_gen is:
        #   learning_rate=1e-4, model.diffusion_steps=100, seed=1
        hp_keys = list(stage_def.hyperparameters.keys())
        hp_values = [stage_def.hyperparameters[k] for k in hp_keys]
        first_combo = dict(
            zip(hp_keys, next(itertools.product(*hp_values)), strict=False)
        )
        first_seed = stage_def.seeds[0]
        first_arch = stage_def.architectures[0]

        # Build Hydra overrides the same way the old code does
        overrides = ["~logger"]
        for key, value in first_combo.items():
            if not key.startswith("model.") and not key.startswith("+model."):
                overrides.append(f"{key}={value}")
        overrides.append(f"seed={first_seed}")

        builder = ExperimentConfigBuilder()
        new_cfg = builder.build(
            config_name=stage_def.base_config,
            arch_path=first_arch,
            overrides=overrides,
            hp_overrides=first_combo,
            seed=first_seed,
            run_id_template=stage_def.run_id_template,
        )

        # --- Compare ---
        assert (
            new_cfg["run_id"] == old_cfg["run_id"]
        ), f"run_id mismatch: {new_cfg['run_id']!r} != {old_cfg['run_id']!r}"

        # Compare every key (excluding run_id which we already checked)
        _assert_dicts_equal(old_cfg, new_cfg, path="root")


# ── Test utilities ────────────────────────────────────────────────


def _assert_no_interpolations(obj: Any, path: str = "") -> None:
    """Walk a nested structure and fail if any ``${`` string is found."""
    if isinstance(obj, str) and "${" in obj:
        raise AssertionError(f"Unresolved interpolation at {path}: {obj}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _assert_no_interpolations(v, f"{path}.{k}")
    if isinstance(obj, list):
        for i, v in enumerate(obj):
            _assert_no_interpolations(v, f"{path}[{i}]")


def _assert_dicts_equal(
    expected: dict[str, Any],
    actual: dict[str, Any],
    path: str = "",
) -> None:
    """Deep comparison of two dicts with path-aware error messages."""
    expected_keys = set(expected.keys())
    actual_keys = set(actual.keys())

    missing = expected_keys - actual_keys
    extra = actual_keys - expected_keys
    if missing:
        raise AssertionError(f"Missing keys at {path}: {missing}")
    if extra:
        raise AssertionError(f"Extra keys at {path}: {extra}")

    for key in expected_keys:
        e_val = expected[key]
        a_val = actual[key]
        sub = f"{path}.{key}"

        if isinstance(e_val, dict) and isinstance(a_val, dict):
            _assert_dicts_equal(e_val, a_val, sub)
        elif isinstance(e_val, float) and isinstance(a_val, float):
            # Float comparison with tolerance for floating-point differences
            assert e_val == pytest.approx(
                a_val
            ), f"Value mismatch at {sub}: {e_val!r} != {a_val!r}"
        else:
            assert e_val == a_val, f"Value mismatch at {sub}: {e_val!r} != {a_val!r}"
