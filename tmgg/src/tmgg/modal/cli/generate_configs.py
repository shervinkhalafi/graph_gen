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

from tmgg.modal.config_builder import ExperimentConfigBuilder
from tmgg.modal.stage_definitions import (
    StageDefinition,
    list_stages,
    load_stage_definition,
)


def generate_configs_for_stage(stage_def: StageDefinition) -> list[dict[str, Any]]:
    """Generate all experiment configs for a stage definition.

    Parameters
    ----------
    stage_def
        Validated stage definition.

    Returns
    -------
    list[dict]
        List of config dictionaries ready for JSON serialization.
    """
    builder = ExperimentConfigBuilder()
    base_config = stage_def.base_config
    architectures = stage_def.architectures
    datasets = stage_def.datasets or [None]
    hyperparameters = stage_def.hyperparameters
    seeds = stage_def.seeds
    run_id_template = stage_def.run_id_template

    hp_keys = list(hyperparameters.keys())
    hp_values = [hyperparameters[k] for k in hp_keys]
    hp_combos = [
        dict(zip(hp_keys, combo, strict=False))
        for combo in itertools.product(*hp_values)
    ]

    configs = []
    for arch in architectures:
        for dataset in datasets:
            for hp_combo in hp_combos:
                for seed in seeds:
                    # Non-model HPs go to Hydra overrides so they are resolved
                    # during composition. Model-prefixed HPs are applied later
                    # by ConfigBuilder.apply_hyperparameters.
                    overrides = ["~logger"]
                    if dataset:
                        overrides.append(f"data={dataset}")
                    for key, value in hp_combo.items():
                        if not key.startswith("model.") and not key.startswith(
                            "+model."
                        ):
                            overrides.append(f"{key}={value}")
                    overrides.append(f"seed={seed}")

                    config = builder.build(
                        config_name=base_config,
                        arch_path=arch,
                        overrides=overrides,
                        hp_overrides=hp_combo,
                        seed=seed,
                        run_id_template=run_id_template,
                        dataset=dataset,
                    )
                    configs.append(config)
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
    print(f"Loaded stage definition: {stage_def.name}")

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
