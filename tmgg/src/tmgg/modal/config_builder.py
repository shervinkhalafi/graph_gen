"""Two-phase experiment config builder: Hydra composition then plain-dict transforms.

Why two phases
--------------
Hydra's ``defaults:`` list composes a single model config per run, but batch
generation needs to iterate over multiple architectures. Hydra's
``model=models/spectral/linear_pe`` override does not work here because the
architecture YAMLs live under path-based ``@model`` group syntax that Hydra
cannot resolve across groups in programmatic compose mode. Phase 1 therefore
uses Hydra only to compose the *base* config (resolving all interpolations and
defaults), then converts the result to a plain dict. Phase 2 loads each
architecture YAML separately, strips it, deep-merges it, and applies HP
overrides -- all as plain dict operations that are easy to test and serialise.

Architecture config backward compatibility
------------------------------------------
Architecture YAML files contain ``${learning_rate}``, ``${noise_levels}``, etc.
so they remain usable with Hydra's CLI for local single-run invocations.
``strip_interpolations()`` removes these entries before merging, so the resolved
values from Phase 1 survive and no architecture files need editing.

Coupled parameters
------------------
Some OmegaConf interpolations (e.g. ``timesteps: ${model.diffusion_steps}``)
link values across config sections. These linkages are lost when Phase 1
converts to a plain dict. ``COUPLED_PARAMS`` on ``ExperimentConfigBuilder``
explicitly propagates each source path to its target after all merges and
overrides are applied. To add a new coupling, append a
``(source_path_tuple, target_path_tuple)`` to that list.

Adding a new architecture
-------------------------
Create a YAML file under ``exp_configs/models/<family>/``, then add its path
(e.g. ``models/spectral/my_new_arch``) to the ``architectures`` list in the
relevant stage definition. ``ConfigBuilder`` handles loading, stripping, and
merging automatically.

Adding a new experiment type
----------------------------
1. Create a base config YAML with Hydra ``defaults:`` under ``exp_configs/``.
2. Create one or more model configs under ``exp_configs/models/<family>/``.
3. Create a stage definition YAML under ``stage_definitions/``.
4. Run ``generate_configs --stage <name> --output-dir <dir>``.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, cast

import yaml
from omegaconf import DictConfig, OmegaConf, open_dict

from tmgg.experiment_utils.task import _extract_wandb_config
from tmgg.modal.config_compose import compose_config
from tmgg.modal.paths import get_exp_configs_path

# ── Helper functions ──────────────────────────────────────────────


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursive dict merge where override wins on conflicts.

    Non-dict values (including lists) are replaced entirely, not merged
    element-wise. Neither input is mutated.

    Parameters
    ----------
    base
        The base dictionary.
    override
        The override dictionary whose values take precedence.

    Returns
    -------
    dict
        A new dictionary containing the merged result.
    """
    result = deepcopy(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def set_nested(d: dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set a value at a dotted path, creating intermediate dicts as needed.

    Parameters
    ----------
    d
        The dictionary to mutate.
    dotted_key
        A dot-separated key path like ``"noise_schedule.timesteps"``.
    value
        The value to set at the leaf.
    """
    parts = dotted_key.split(".")
    current = d
    for part in parts[:-1]:
        if part not in current:
            current[part] = {}
        elif not isinstance(current[part], dict):
            raise TypeError(
                f"Cannot set '{dotted_key}': intermediate key '{part}' "
                f"is a {type(current[part]).__name__}, not a dict"
            )
        current = current[part]
    current[parts[-1]] = value


def get_nested(d: dict[str, Any], path: tuple[str, ...]) -> Any | None:
    """Get a value at a tuple path, returning None if any key is missing.

    Parameters
    ----------
    d
        The dictionary to traverse.
    path
        A tuple of keys like ``("model", "diffusion_steps")``.

    Returns
    -------
    Any or None
        The value at the path, or None if any intermediate key is absent.
    """
    current: Any = d
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


_INTERPOLATION_PATTERN = re.compile(r"^\$\{[^}]+\}$")

# Matches strings that represent numbers but are loaded as str by YAML
# (e.g. "1e-4", "1e-3"). Hydra and OmegaConf parse these as floats, so
# we must do the same when applying HP overrides from YAML-loaded stages.
_NUMERIC_PATTERN = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


def strip_interpolations(obj: Any) -> Any:
    """Recursively remove dict entries whose values are OmegaConf interpolation strings.

    A value is an interpolation string when the entire string matches ``${...}``.
    Concrete strings like ``"cosine"`` or ``"tmgg.models.Foo"`` are preserved.

    Parameters
    ----------
    obj
        A nested structure of dicts, lists, and scalars (as returned by
        ``yaml.safe_load``).

    Returns
    -------
    Any
        A new structure with interpolation-valued entries removed. Dicts and
        lists are copied; other types are returned as-is.
    """
    if isinstance(obj, dict):
        result = {}
        for key, value in obj.items():
            if isinstance(value, str) and _INTERPOLATION_PATTERN.match(value):
                continue
            result[key] = strip_interpolations(value)
        return result
    if isinstance(obj, list):
        return [
            strip_interpolations(item)
            for item in obj
            if not (isinstance(item, str) and _INTERPOLATION_PATTERN.match(item))
        ]
    return obj


# ── Builder ───────────────────────────────────────────────────────


class ExperimentConfigBuilder:
    """Build experiment configs: Hydra for base composition, plain dicts for the rest.

    The two-phase pipeline keeps OmegaConf confined to ``compose_base`` and
    performs all subsequent transformations on plain dicts, which are easier
    to reason about, test, and serialise.
    """

    COUPLED_PARAMS: list[tuple[tuple[str, ...], tuple[str, ...]]] = [
        (("model", "diffusion_steps"), ("model", "noise_schedule", "timesteps")),
    ]

    def compose_base(self, config_name: str, overrides: list[str]) -> dict[str, Any]:
        """Phase 1: Hydra compose, W&B extraction, full interpolation resolution.

        This is the only method that touches OmegaConf. The returned dict is
        fully resolved with no interpolation strings remaining.

        Parameters
        ----------
        config_name
            Hydra config name (without ``.yaml``).
        overrides
            Hydra CLI-style overrides (e.g. ``["~logger", "seed=42"]``).

        Returns
        -------
        dict
            Plain dict with all interpolations resolved. Contains
            ``_wandb_config`` if the config defines W&B settings.
        """
        cfg: DictConfig = compose_config(config_name, overrides)

        # Extract W&B metadata before we strip anything — needs the DictConfig
        wandb_config = _extract_wandb_config(cfg)

        # Null out path interpolations that depend on Hydra runtime, and drop
        # the hydra section entirely; both are execution-time concerns.
        with open_dict(cfg):
            if "paths" not in cfg:
                cfg.paths = OmegaConf.create({})  # type: ignore[assignment]
            cfg.paths.output_dir = None  # type: ignore[union-attr]
            cfg.paths.results_dir = None  # type: ignore[union-attr]
            if "hydra" in cfg:
                cfg.pop("hydra")  # pyright: ignore[reportArgumentType]
            if "logger" in cfg:
                cfg.pop("logger")  # pyright: ignore[reportArgumentType]

        raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(raw, dict):
            raise TypeError(
                f"Expected dict from OmegaConf.to_container, got {type(raw)}"
            )
        # OmegaConf.to_container returns Dict[DictKeyType, Any] but Hydra configs
        # always use string keys. The isinstance check above narrows to dict but
        # pyright cannot infer the key type.
        resolved = cast(dict[str, Any], raw)

        if wandb_config is not None:
            resolved["_wandb_config"] = wandb_config

        return resolved

    def load_architecture(self, arch_path: str) -> dict[str, Any]:
        """Load an architecture YAML and strip all interpolation-valued entries.

        Parameters
        ----------
        arch_path
            Path relative to ``exp_configs/``, without extension, e.g.
            ``"models/spectral/linear_pe"``.

        Returns
        -------
        dict
            Architecture config with only concrete (non-interpolation) values.

        Raises
        ------
        FileNotFoundError
            If the YAML file does not exist.
        """
        exp_configs = get_exp_configs_path()
        full_path = exp_configs / f"{arch_path}.yaml"
        if not full_path.exists():
            raise FileNotFoundError(f"Architecture config not found: {full_path}")

        with open(full_path) as f:
            raw: dict[str, Any] = yaml.safe_load(f)

        return strip_interpolations(raw)

    def merge_architecture(
        self, config: dict[str, Any], arch: dict[str, Any]
    ) -> dict[str, Any]:
        """Deep-merge architecture values into ``config["model"]``.

        Architecture-specific concrete values override the base model values.
        Interpolation-valued entries were already stripped by ``load_architecture``,
        so base values for those keys (resolved from Hydra) are preserved.

        Parameters
        ----------
        config
            The full experiment config (mutated in place for efficiency).
        arch
            Stripped architecture config.

        Returns
        -------
        dict
            The same ``config`` reference, with ``config["model"]`` updated.
        """
        config["model"] = deep_merge(config.get("model", {}), arch)
        return config

    def apply_hyperparameters(
        self, config: dict[str, Any], hp_overrides: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply hyperparameter overrides at dotted paths.

        Keys prefixed with ``model.`` or ``+model.`` target
        ``config["model"]``; other keys target the config root.

        Parameters
        ----------
        config
            The experiment config (mutated in place).
        hp_overrides
            Mapping of dotted keys to values.

        Returns
        -------
        dict
            The same ``config`` reference.
        """
        for raw_key, raw_value in hp_overrides.items():
            # Strip optional Hydra "+" add-key prefix
            key = raw_key.lstrip("+")
            # Coerce numeric-looking strings from YAML (e.g. "1e-4" → 0.0001)
            value = _coerce_numeric(raw_value)

            if key.startswith("model."):
                sub_key = key[len("model.") :]
                if "model" not in config:
                    config["model"] = {}
                set_nested(config["model"], sub_key, value)
            else:
                set_nested(config, key, value)

        return config

    def sync_coupled_params(self, config: dict[str, Any]) -> dict[str, Any]:
        """Copy source values to targets for coupled parameter pairs.

        If the source exists but the target's parent dict is absent, the pair
        is silently skipped (not every experiment type has all sections).

        Parameters
        ----------
        config
            The experiment config (mutated in place).

        Returns
        -------
        dict
            The same ``config`` reference.
        """
        for source_path, target_path in self.COUPLED_PARAMS:
            source_val = get_nested(config, source_path)
            if source_val is None:
                continue
            # Check that the target's parent exists
            target_parent = get_nested(config, target_path[:-1])
            if target_parent is None or not isinstance(target_parent, dict):
                continue
            target_parent[target_path[-1]] = source_val

        return config

    def strip_for_remote(self, config: dict[str, Any]) -> dict[str, Any]:
        """Null out execution-time paths and remove the hydra section.

        Parameters
        ----------
        config
            The experiment config (mutated in place).

        Returns
        -------
        dict
            The same ``config`` reference.
        """
        if "paths" in config and isinstance(config["paths"], dict):
            config["paths"]["output_dir"] = None
            config["paths"]["results_dir"] = None
        if "hydra" in config:
            del config["hydra"]
        return config

    def validate(self, config: dict[str, Any]) -> None:
        """Check that the config contains required structure for training.

        Parameters
        ----------
        config
            The experiment config to validate.

        Raises
        ------
        ValueError
            If required keys are missing.
        """
        missing = [k for k in ("model", "data", "trainer") if k not in config]
        if missing:
            raise ValueError(f"Config is missing required top-level keys: {missing}")
        model = config["model"]
        if not isinstance(model, dict) or "_target_" not in model:
            raise ValueError("config['model'] must be a dict containing '_target_'")

    def generate_run_id(
        self,
        template: str,
        arch: str,
        dataset: str | None,
        hp_combo: dict[str, Any],
        seed: int,
    ) -> str:
        """Generate a run ID from a template and hyperparameter values.

        Template variables: ``{arch}``, ``{data}``, ``{lr}``, ``{wd}``,
        ``{k}``, ``{diffusion_steps}``, ``{seed}``.

        Parameters
        ----------
        template
            A Python format string with named placeholders.
        arch
            Architecture path (the last segment is used).
        dataset
            Dataset path (the last segment is used), or None.
        hp_combo
            Mapping of HP keys to values.
        seed
            Random seed.

        Returns
        -------
        str
            The formatted run ID.
        """
        arch_name = arch.split("/")[-1]
        data_name = dataset.split("/")[-1] if dataset else ""

        hp_formatted: dict[str, str] = {}
        for key, value in hp_combo.items():
            formatted = _format_hp_value(key, value)
            if key == "learning_rate":
                hp_formatted["lr"] = formatted
            elif key == "weight_decay":
                hp_formatted["wd"] = formatted
            elif key.endswith(".k"):
                hp_formatted["k"] = formatted
            elif key.endswith("diffusion_steps"):
                hp_formatted["diffusion_steps"] = formatted
            else:
                hp_formatted[key.split(".")[-1]] = formatted

        return template.format(
            arch=arch_name,
            data=data_name,
            seed=seed,
            **hp_formatted,
        )

    def build(
        self,
        config_name: str,
        arch_path: str,
        overrides: list[str],
        hp_overrides: dict[str, Any],
        seed: int,
        run_id_template: str,
        dataset: str | None = None,
    ) -> dict[str, Any]:
        """Execute the full two-phase config generation pipeline.

        Parameters
        ----------
        config_name
            Hydra config name (e.g. ``"base_config_spectral_arch"``).
        arch_path
            Architecture YAML path (e.g. ``"models/spectral/linear_pe"``).
        overrides
            Hydra overrides applied during composition.
        hp_overrides
            Hyperparameter overrides applied after architecture merge.
        seed
            Random seed (already included in ``overrides``; used here for
            run ID generation).
        run_id_template
            Format string for the run ID.
        dataset
            Optional dataset path for run ID.

        Returns
        -------
        dict
            A fully resolved, validated config dict ready for serialisation.
        """
        config = self.compose_base(config_name, overrides)
        arch = self.load_architecture(arch_path)
        self.merge_architecture(config, arch)
        self.apply_hyperparameters(config, hp_overrides)
        self.sync_coupled_params(config)
        self.strip_for_remote(config)
        self.validate(config)

        run_id = self.generate_run_id(
            run_id_template, arch_path, dataset, hp_overrides, seed
        )
        config["run_id"] = run_id

        return config


# ── Private helpers ───────────────────────────────────────────────


def _coerce_numeric(value: Any) -> Any:
    """Coerce string values to int or float when they look numeric.

    YAML's ``safe_load`` parses ``1e-4`` as a string (YAML 1.1 requires a
    decimal point for float recognition), but Hydra and OmegaConf treat
    such values as numbers. This function bridges that gap so that HP
    overrides loaded from stage YAML files match the types Hydra produces.

    Parameters
    ----------
    value
        Any value. Only strings are candidates for coercion.

    Returns
    -------
    Any
        The original value if not a numeric-looking string, otherwise
        the parsed int or float.
    """
    if not isinstance(value, str):
        return value
    if not _NUMERIC_PATTERN.match(value):
        return value
    # Try int first (exact integers like "100"), then float
    try:
        as_int = int(value)
        # Only return int if the string has no decimal/exponent notation
        if "." not in value and "e" not in value.lower():
            return as_int
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _format_hp_value(key: str, value: float | int | str) -> str:
    """Format a hyperparameter value for embedding in a run ID.

    Parameters
    ----------
    key
        The hyperparameter key (e.g. ``"learning_rate"``, ``"model.k"``).
    value
        The hyperparameter value.

    Returns
    -------
    str
        A compact string like ``"lr1e-4"`` or ``"k8"``.
    """
    if key == "learning_rate":
        prefix = "lr"
    elif key == "weight_decay":
        prefix = "wd"
    elif key.endswith(".k"):
        prefix = "k"
    elif key.endswith("diffusion_steps"):
        prefix = "T"
    else:
        prefix = key.split(".")[-1][:3]

    if isinstance(value, float) and value < 0.01:
        val_str = f"{value:.0e}".replace("e-0", "e-")
    else:
        val_str = str(value)

    return f"{prefix}{val_str}"
