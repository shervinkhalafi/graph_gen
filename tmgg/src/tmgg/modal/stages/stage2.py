"""Stage 2: Core Validation - Modal execution.

Validates generalization across configurations and compares with DiGress.
Budget: 166.5 GPU-hours
"""

from __future__ import annotations

import copy
from typing import Any

from tmgg.modal.app import app
from tmgg.modal.runner import (
    experiment_image,
    run_single_experiment,
    run_single_experiment_fast,
    tigris_secret,
    wandb_secret,
)
from tmgg.modal.stages.stage1 import validate_prefix
from tmgg.modal.storage import get_storage_from_env

# Stage 2 configuration (module-level defaults)
STAGE2_ARCHITECTURES = [
    "models/spectral/filter_bank_nonlinear",
    "models/spectral/self_attention",
    "models/digress/digress_transformer",
]

STAGE2_DATASETS = [
    "data/sbm_default",
    # "data/sbm_n100",  # Uncomment when config exists
]

STAGE2_DEFAULT_HYPERPARAMETERS = {
    "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-1, 1e-2, 1e-3],
    # Use + prefix to force-add model.k (struct mode blocks plain override)
    "+model.k": [4, 8, 16, 32],
}

STAGE2_SEEDS = [1, 2, 3]


def generate_stage2_configs(
    best_stage1_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate experiment configurations for Stage 2.

    Uses Hydra composition to properly resolve config defaults and interpolations.

    Parameters
    ----------
    best_stage1_config
        Best configuration from Stage 1 to narrow hyperparameter search.

    Returns
    -------
    list[dict]
        List of configuration dictionaries with all values resolved.
    """
    import itertools

    from omegaconf import OmegaConf

    from tmgg.modal.config_compose import compose_config

    configs = []

    # Copy default hyperparameters to avoid mutating module-level dict
    hyperparameters = copy.deepcopy(STAGE2_DEFAULT_HYPERPARAMETERS)

    # Narrow hyperparameter search around Stage 1 best if available
    if best_stage1_config:
        best_lr = best_stage1_config.get("learning_rate", 1e-4)
        best_wd = best_stage1_config.get("weight_decay", 1e-2)
        hyperparameters["learning_rate"] = [best_lr / 5, best_lr, best_lr * 5]
        hyperparameters["weight_decay"] = [best_wd / 10, best_wd, best_wd * 10]

    # Generate hyperparameter combinations
    hp_keys = list(hyperparameters.keys())
    hp_values = [hyperparameters[k] for k in hp_keys]
    hp_combos = list(itertools.product(*hp_values))

    for arch in STAGE2_ARCHITECTURES:
        for dataset in STAGE2_DATASETS:
            for hp_combo in hp_combos:
                for seed in STAGE2_SEEDS:
                    # Build Hydra overrides for this config variant
                    overrides = [f"model={arch}", f"data={dataset}"]
                    for key, value in zip(hp_keys, hp_combo, strict=False):
                        overrides.append(f"{key}={value}")
                    overrides.append(f"seed={seed}")

                    # Compose config with Hydra (resolves all defaults and interpolations)
                    config = compose_config("base_config_spectral", overrides)

                    # Generate run ID
                    arch_name = arch.split("/")[-1]
                    data_name = dataset.split("/")[-1]
                    lr_str = f"lr{hp_combo[0]:.0e}".replace("e-0", "e-")
                    wd_str = f"wd{hp_combo[1]:.0e}".replace("e-0", "e-")
                    k_str = f"k{hp_combo[2]}"
                    run_id = f"stage2_{arch_name}_{data_name}_{lr_str}_{wd_str}_{k_str}_s{seed}"

                    # Convert to dict and add run_id
                    config_dict = OmegaConf.to_container(config, resolve=True)
                    config_dict["run_id"] = run_id  # type: ignore[index]

                    configs.append(config_dict)

    return configs


@app.function(
    image=experiment_image,
    secrets=[tigris_secret, wandb_secret],
    timeout=7200,  # 2 hours for orchestration
)
def run_stage2(
    gpu_type: str = "debug",
    use_stage1_best: bool = True,
    dry_run: bool = False,
    additional_tags: list[str] | None = None,
    path_prefix: str = "",
    stage1_prefix: str = "",
) -> dict[str, Any]:
    """Run Stage 2: Core Validation experiments on Modal.

    Parameters
    ----------
    gpu_type
        GPU tier for experiments.
    use_stage1_best
        If True, use Stage 1 best config to narrow search.
    dry_run
        If True, print configs without running.
    additional_tags
        Extra W&B tags to add to all experiments.
    path_prefix
        Storage path prefix for Stage 2 results (e.g., "2025-01-05").
    stage1_prefix
        Storage path prefix for Stage 1 results. Defaults to path_prefix.

    Returns
    -------
    dict
        Stage results summary.
    """
    _ = additional_tags  # TODO: Integrate with TaskInput when stage2 is refactored
    from datetime import datetime

    # Resolve stage1_prefix: defaults to path_prefix if not specified
    effective_stage1_prefix = stage1_prefix or path_prefix
    if path_prefix:
        print(f"Using storage prefix: {path_prefix}")
    if effective_stage1_prefix and effective_stage1_prefix != path_prefix:
        print(f"Using Stage 1 prefix: {effective_stage1_prefix}")

    # Load Stage 1 best config if requested
    best_stage1 = None
    if use_stage1_best:
        stage1_storage = get_storage_from_env(path_prefix=effective_stage1_prefix)
        if stage1_storage:
            try:
                stage1_summary = stage1_storage.download_metrics("stage1_poc_summary")
                best_run_id = stage1_summary.get("best_run_id")
                if best_run_id:
                    best_result = stage1_storage.download_metrics(
                        f"results/{best_run_id}"
                    )
                    best_stage1 = best_result.get("config")
                    print(f"Using Stage 1 best config: {best_run_id}")
            except Exception as e:
                print(f"Could not load Stage 1 results: {e}")

    configs = generate_stage2_configs(best_stage1)
    print(f"Stage 2: Generated {len(configs)} experiment configurations")

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

    # Select function based on GPU tier
    # DiGress benefits from faster GPUs
    if gpu_type in ("fast", "multi", "h100"):
        results = list(run_single_experiment_fast.map(configs))
    else:
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

    # Validate metrics helper
    def get_val_loss(result: dict) -> float | None:
        """Extract validation loss, returning None if missing or invalid."""
        metrics = result.get("metrics", {})
        loss = metrics.get("best_val_loss")
        if loss is None or loss == float("inf"):
            return None
        return loss

    # Track results without valid metrics
    results_without_metrics = []

    # Find best result by architecture (only considering valid metrics)
    best_by_arch = {}
    for result in completed_results:
        val_loss = get_val_loss(result)
        if val_loss is None:
            results_without_metrics.append(result)
            continue

        arch = result.get("config", {}).get("model", "unknown")
        if arch not in best_by_arch or val_loss < best_by_arch[arch]["val_loss"]:
            best_by_arch[arch] = {
                "run_id": result.get("run_id"),
                "val_loss": val_loss,
            }

    if results_without_metrics:
        print(
            f"Warning: {len(results_without_metrics)} completed runs missing best_val_loss metric:"
        )
        for r in results_without_metrics[:5]:
            print(f"  - {r.get('run_id', 'unknown')}")
        if len(results_without_metrics) > 5:
            print(f"  ... and {len(results_without_metrics) - 5} more")

    # Upload stage summary
    summary = {
        "stage": "stage2_validation",
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(),
        "total_experiments": len(results),
        "completed": len(completed_results),
        "failed": len(failed_results),
        "best_by_architecture": best_by_arch,
    }

    if storage:
        storage.upload_metrics(summary, "stage2_validation_summary")

    print(f"Stage 2 complete: {len(completed_results)}/{len(results)} succeeded")
    print("Best results by architecture:")
    for arch, data in best_by_arch.items():
        print(f"  {arch}: {data['run_id']} (val_loss={data['val_loss']:.6f})")

    return summary


@app.local_entrypoint()
def main(
    gpu: str = "standard",
    use_stage1_best: bool = True,
    dry_run: bool = False,
    additional_tags: str = "",
    prefix: str = "",
    stage1_prefix: str = "",
):
    """Local entry point for Stage 2.

    Usage
    -----
    modal run tmgg_modal/stages/stage2.py --gpu fast
    modal run tmgg_modal/stages/stage2.py --additional-tags "experiment,v2"
    modal run tmgg_modal/stages/stage2.py --prefix 2025-01-05
    modal run tmgg_modal/stages/stage2.py --prefix 2025-01-05 --stage1-prefix 2025-01-04
    """
    # Validate prefixes before sending to Modal
    path_prefix = validate_prefix(prefix)
    s1_prefix = validate_prefix(stage1_prefix)

    tags = [t.strip() for t in additional_tags.split(",") if t.strip()]
    result = run_stage2.remote(
        gpu_type=gpu,
        use_stage1_best=use_stage1_best,
        dry_run=dry_run,
        additional_tags=tags or None,
        path_prefix=path_prefix,
        stage1_prefix=s1_prefix,
    )
    print(f"Stage 2 result: {result}")
