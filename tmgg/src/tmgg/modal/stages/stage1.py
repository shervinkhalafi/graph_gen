"""Stage 1: Proof of Concept - Modal execution.

Runs Linear PE, Graph Filter + Sigmoid, Self-Attention on SBM n=50.
Budget: 4.4 GPU-hours
"""

from __future__ import annotations

import re
from typing import Any

from tmgg.modal.app import app
from tmgg.modal.runner import (
    experiment_image,
    run_single_experiment,
    tigris_secret,
    wandb_secret,
)
from tmgg.modal.storage import get_storage_from_env


def validate_prefix(prefix: str) -> str:
    """Validate prefix contains only alphanumeric, dash, underscore.

    Parameters
    ----------
    prefix
        Storage path prefix to validate.

    Returns
    -------
    str
        The validated prefix (unchanged if valid).

    Raises
    ------
    ValueError
        If prefix contains invalid characters.
    """
    if prefix and not re.match(r"^[a-zA-Z0-9_-]+$", prefix):
        raise ValueError(
            f"Invalid prefix '{prefix}': must contain only alphanumeric, dash, underscore"
        )
    return prefix


# Stage 1 configuration
STAGE1_ARCHITECTURES = [
    "models/spectral/linear_pe",
    "models/spectral/filter_bank_nonlinear",
    "models/spectral/self_attention",
]

STAGE1_HYPERPARAMETERS = {
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-2, 1e-3],
    # Use + prefix to force-add model.k (struct mode blocks plain override)
    "+model.k": [8, 16],
}

STAGE1_SEEDS = [1, 2, 3]


def generate_stage1_configs() -> list[dict[str, Any]]:
    """Generate all experiment configurations for Stage 1.

    Uses Hydra composition to properly resolve config defaults and interpolations.

    Returns
    -------
    list[dict]
        List of configuration dictionaries with all values resolved.
    """
    import itertools

    from omegaconf import OmegaConf

    from tmgg.modal.config_compose import compose_config

    configs = []

    # Generate hyperparameter combinations
    hp_keys = list(STAGE1_HYPERPARAMETERS.keys())
    hp_values = [STAGE1_HYPERPARAMETERS[k] for k in hp_keys]
    hp_combos = list(itertools.product(*hp_values))

    for arch in STAGE1_ARCHITECTURES:
        for hp_combo in hp_combos:
            for seed in STAGE1_SEEDS:
                # Build Hydra overrides for this config variant
                overrides = [f"model={arch}"]
                for key, value in zip(hp_keys, hp_combo, strict=False):
                    overrides.append(f"{key}={value}")
                overrides.append(f"seed={seed}")

                # Compose config with Hydra (resolves all defaults and interpolations)
                config = compose_config("base_config_spectral", overrides)

                # Generate run ID
                arch_name = arch.split("/")[-1]
                lr_str = f"lr{hp_combo[0]:.0e}".replace("e-0", "e-")
                wd_str = f"wd{hp_combo[1]:.0e}".replace("e-0", "e-")
                k_str = f"k{hp_combo[2]}"
                run_id = f"stage1_{arch_name}_{lr_str}_{wd_str}_{k_str}_s{seed}"

                # Convert to dict and add run_id
                config_dict = OmegaConf.to_container(config, resolve=True)
                config_dict["run_id"] = run_id  # type: ignore[index]

                configs.append(config_dict)

    return configs


@app.function(
    image=experiment_image,
    secrets=[tigris_secret, wandb_secret],
    timeout=3600,  # 1 hour for orchestration
)
def run_stage1(
    gpu_type: str = "debug",
    dry_run: bool = False,
    additional_tags: list[str] | None = None,
    path_prefix: str = "",
) -> dict[str, Any]:
    """Run Stage 1: Proof of Concept experiments on Modal.

    Parameters
    ----------
    gpu_type
        GPU tier for experiments.
    dry_run
        If True, print configs without running.
    additional_tags
        Extra W&B tags to add to all experiments.
    path_prefix
        Storage path prefix for run isolation (e.g., "2025-01-05").

    Returns
    -------
    dict
        Stage results summary.
    """
    _ = additional_tags  # TODO: Integrate with TaskInput when stage1 is refactored
    from datetime import datetime

    configs = generate_stage1_configs()
    print(f"Stage 1: Generated {len(configs)} experiment configurations")
    if path_prefix:
        print(f"Using storage prefix: {path_prefix}")

    if dry_run:
        print("Dry run - configurations:")
        for i, cfg in enumerate(configs[:5]):
            print(f"  [{i}] {cfg.get('run_id', 'unknown')}")
        if len(configs) > 5:
            print(f"  ... and {len(configs) - 5} more")
        return {"status": "dry_run", "num_configs": len(configs)}

    # Check for completed runs with fine-grained status
    from tmgg.modal.result_status import (
        ResultStatus,
        filter_configs_by_status,
        summarize_status_map,
    )

    storage = get_storage_from_env(path_prefix=path_prefix)
    configs, status_map = filter_configs_by_status(
        storage,
        configs,
        skip_statuses={ResultStatus.COMPLETE},
        required_metrics=["best_val_loss"],
    )
    print(f"Result status: {summarize_status_map(status_map)}")

    if not configs:
        print("All experiments completed!")
        return {"status": "completed", "message": "All experiments already done"}

    print(f"Running {len(configs)} experiments")
    started_at = datetime.now().isoformat()

    # Run experiments in parallel using Modal's map
    results = list(run_single_experiment.map(configs))

    # Aggregate results
    completed_results = [r for r in results if r.get("status") == "completed"]
    failed_results = [r for r in results if r.get("status") == "failed"]

    # Log storage warnings from individual runs
    for result in completed_results:
        warnings = result.get("storage_warnings")
        if warnings:
            for w in warnings:
                print(f"  Warning: {w}")

    # Validate metrics and find best result
    def get_val_loss(result: dict) -> float | None:
        """Extract validation loss, returning None if missing or invalid."""
        metrics = result.get("metrics", {})
        loss = metrics.get("best_val_loss")
        if loss is None or loss == float("inf"):
            return None
        return loss

    # Separate results with valid metrics from those without
    results_with_metrics = []
    results_without_metrics = []
    for r in completed_results:
        loss = get_val_loss(r)
        if loss is not None:
            results_with_metrics.append((r, loss))
        else:
            results_without_metrics.append(r)

    if results_without_metrics:
        print(
            f"Warning: {len(results_without_metrics)} completed runs missing best_val_loss metric:"
        )
        for r in results_without_metrics[:5]:
            print(f"  - {r.get('run_id', 'unknown')}")
        if len(results_without_metrics) > 5:
            print(f"  ... and {len(results_without_metrics) - 5} more")

    # Find best result from those with valid metrics
    best_result = None
    if results_with_metrics:
        best_result, _ = min(results_with_metrics, key=lambda x: x[1])
    elif completed_results:
        print(
            "Warning: No completed runs have valid metrics, cannot determine best result"
        )

    # Upload stage summary
    summary = {
        "stage": "stage1_poc",
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
        "total_experiments": len(results),
        "completed": len(completed_results),
        "failed": len(failed_results),
        "best_run_id": best_result.get("run_id") if best_result else None,
        "best_val_loss": best_result.get("metrics", {}).get("best_val_loss")
        if best_result
        else None,
    }

    if storage:
        storage.upload_metrics(summary, "stage1_poc_summary")

    print(f"Stage 1 complete: {len(completed_results)}/{len(results)} succeeded")
    if best_result:
        print(
            f"Best run: {best_result.get('run_id')} with val_loss={best_result.get('metrics', {}).get('best_val_loss', 'N/A')}"
        )

    return summary


@app.local_entrypoint()
def main(
    gpu: str = "standard",
    dry_run: bool = False,
    additional_tags: str = "",
    prefix: str = "",
):
    """Local entry point for Stage 1.

    Usage
    -----
    modal run tmgg_modal/stages/stage1.py --gpu standard
    modal run tmgg_modal/stages/stage1.py --additional-tags "experiment,v2"
    modal run tmgg_modal/stages/stage1.py --prefix 2025-01-05
    """
    # Validate prefix before sending to Modal
    path_prefix = validate_prefix(prefix)
    tags = [t.strip() for t in additional_tags.split(",") if t.strip()]
    result = run_stage1.remote(
        gpu_type=gpu,
        dry_run=dry_run,
        additional_tags=tags or None,
        path_prefix=path_prefix,
    )
    print(f"Stage 1 result: {result}")
