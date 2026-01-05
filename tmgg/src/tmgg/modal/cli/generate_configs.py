"""Generate experiment config JSON files from stage definitions.

Runs locally (no Modal) to produce config files that can be passed to run_single.py.

Usage
-----
uv run python -m tmgg.modal.cli.generate_configs --stage stage1 --output-dir ./configs/stage1/2026-01-05/
uv run python -m tmgg.modal.cli.generate_configs --stage stage2 --output-dir ./configs/stage2/2026-01-05/
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from tmgg.experiment_utils.task import prepare_config_for_remote
from tmgg.modal.config_compose import compose_config
from tmgg.modal.paths import get_exp_configs_path
from tmgg.modal.stage_definitions import list_stages, load_stage_definition


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

    Raises
    ------
    FileNotFoundError
        If the config file doesn't exist.
    """
    exp_configs = get_exp_configs_path()
    if exp_configs is None:
        raise RuntimeError("Could not find tmgg exp_configs path")

    model_config_path = exp_configs / f"{arch_path}.yaml"
    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    return OmegaConf.load(model_config_path)  # type: ignore[return-value]


def format_hp_value(key: str, value: float | int) -> str:
    """Format hyperparameter value for run_id.

    Examples: lr=1e-4 -> 'lr1e-4', weight_decay=1e-2 -> 'wd1e-2', +model.k=8 -> 'k8'
    """
    # Extract short name
    if key == "learning_rate":
        prefix = "lr"
    elif key == "weight_decay":
        prefix = "wd"
    elif key.endswith(".k"):
        prefix = "k"
    else:
        prefix = key.split(".")[-1][:3]

    # Format value
    if isinstance(value, float) and value < 0.01:
        val_str = f"{value:.0e}".replace("e-0", "e-")
    else:
        val_str = str(value)

    return f"{prefix}{val_str}"


def generate_run_id(
    template: str,
    arch: str,
    dataset: str | None,
    hp_combo: dict[str, Any],
    seed: int,
) -> str:
    """Generate run_id from template and values.

    Template variables: {arch}, {data}, {lr}, {wd}, {k}, {seed}
    """
    arch_name = arch.split("/")[-1]
    data_name = dataset.split("/")[-1] if dataset else ""

    # Format HP values
    hp_formatted = {}
    for key, value in hp_combo.items():
        formatted = format_hp_value(key, value)
        # Extract the short key for template
        if key == "learning_rate":
            hp_formatted["lr"] = formatted
        elif key == "weight_decay":
            hp_formatted["wd"] = formatted
        elif key.endswith(".k"):
            hp_formatted["k"] = formatted
        else:
            hp_formatted[key.split(".")[-1]] = formatted

    return template.format(
        arch=arch_name,
        data=data_name,
        seed=seed,
        **hp_formatted,
    )


def generate_configs_for_stage(stage_def: dict[str, Any]) -> list[dict[str, Any]]:
    """Generate all experiment configs for a stage definition.

    Parameters
    ----------
    stage_def
        Stage definition loaded from YAML.

    Returns
    -------
    list[dict]
        List of config dictionaries ready for JSON serialization.

    Notes
    -----
    Model configs are loaded and merged manually because Hydra's defaults
    override system doesn't work with path-based imports like
    `models/spectral/linear_pe@model`. The base config provides all settings
    except model, which is then merged from the architecture-specific YAML.
    """
    base_config = stage_def["base_config"]
    architectures = stage_def["architectures"]
    datasets = stage_def.get("datasets", [None])  # None means use base config default
    hyperparameters = stage_def["hyperparameters"]
    seeds = stage_def["seeds"]
    run_id_template = stage_def["run_id_template"]

    # Generate hyperparameter combinations
    hp_keys = list(hyperparameters.keys())
    hp_values = [hyperparameters[k] for k in hp_keys]
    hp_combos = list(itertools.product(*hp_values))

    configs = []

    for arch in architectures:
        # Load model config separately (Hydra's defaults override doesn't work
        # with path-based imports like models/spectral/linear_pe@model)
        model_cfg = load_model_config(arch)

        for dataset in datasets:
            for hp_combo_tuple in hp_combos:
                hp_combo = dict(zip(hp_keys, hp_combo_tuple, strict=False))

                for seed in seeds:
                    # Build Hydra overrides (without model override)
                    overrides = [
                        # Disable logger (requires TMGG_S3_BUCKET env var)
                        "~logger",
                    ]
                    if dataset:
                        overrides.append(f"data={dataset}")

                    # Apply hyperparameters (skip model.* ones, applied later)
                    for key, value in hp_combo.items():
                        if not key.startswith("model.") and not key.startswith(
                            "+model."
                        ):
                            overrides.append(f"{key}={value}")
                    overrides.append(f"seed={seed}")

                    # Compose base config with Hydra (keeps default model)
                    cfg = compose_config(base_config, overrides)

                    # Merge model config manually: start with resolved base config values,
                    # then override with non-interpolated values from arch-specific config.
                    # The arch config has interpolations like ${learning_rate} that can't
                    # be resolved in isolation, so we skip those and use the base values.
                    base_model_dict = OmegaConf.to_container(cfg.model, resolve=True)
                    merged_dict: dict[str, Any] = dict(base_model_dict)  # type: ignore[arg-type]

                    # Extract non-interpolation values from arch config
                    for key in model_cfg:
                        str_key = str(key)
                        node = model_cfg._get_node(str_key)
                        # Skip interpolations (they reference parent config)
                        if node is not None and not OmegaConf.is_interpolation(
                            model_cfg, str_key
                        ):
                            value = model_cfg[str_key]
                            # Convert OmegaConf objects to containers, keep primitives as-is
                            if OmegaConf.is_config(value):
                                merged_dict[str_key] = OmegaConf.to_container(
                                    value, resolve=False
                                )
                            else:
                                merged_dict[str_key] = value

                    merged_model = OmegaConf.create(merged_dict)

                    # Apply model-specific hyperparameters
                    for key, value in hp_combo.items():
                        if key.startswith("model."):
                            sub_key = key[6:]  # Remove "model." prefix
                            OmegaConf.update(merged_model, sub_key, value)
                        elif key.startswith("+model."):
                            sub_key = key[7:]  # Remove "+model." prefix
                            OmegaConf.update(merged_model, sub_key, value)

                    # Replace model in config
                    cfg.model = merged_model

                    # Prepare for remote (strips paths, env-dependent values)
                    config_dict = prepare_config_for_remote(cfg)

                    # Generate run_id
                    run_id = generate_run_id(
                        run_id_template, arch, dataset, hp_combo, seed
                    )
                    config_dict["run_id"] = run_id

                    configs.append(config_dict)

    return configs


def save_configs(configs: list[dict[str, Any]], output_dir: Path) -> None:
    """Save configs as individual JSON files.

    Parameters
    ----------
    configs
        List of config dictionaries.
    output_dir
        Directory to save JSON files to.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for config in configs:
        run_id = config["run_id"]
        output_path = output_dir / f"{run_id}.json"
        with open(output_path, "w") as f:
            json.dump(config, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate experiment config JSON files from stage definitions."
    )
    parser.add_argument(
        "--stage",
        required=True,
        help=f"Stage name (available: {', '.join(list_stages())})",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to save config JSON files",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config count without saving files",
    )

    args = parser.parse_args()

    # Load stage definition
    stage_def = load_stage_definition(args.stage)
    print(f"Loaded stage definition: {stage_def['name']}")

    # Generate configs
    configs = generate_configs_for_stage(stage_def)
    print(f"Generated {len(configs)} configs")

    if args.dry_run:
        print("\nDry run - sample configs:")
        for i, cfg in enumerate(configs[:5]):
            print(f"  [{i}] {cfg.get('run_id', 'unknown')}")
        if len(configs) > 5:
            print(f"  ... and {len(configs) - 5} more")
        return

    # Save configs
    save_configs(configs, args.output_dir)
    print(f"Saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
