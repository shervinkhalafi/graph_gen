"""Stage 1: Proof of Concept - Modal execution.

Runs Linear PE, Graph Filter + Sigmoid, Self-Attention on SBM n=50.
Budget: 4.4 GPU-hours
"""

from __future__ import annotations

from typing import Any

from omegaconf import OmegaConf

from tmgg_modal.app import app  # pyright: ignore[reportImplicitRelativeImport]
from tmgg_modal.runner import (  # pyright: ignore[reportImplicitRelativeImport]
    experiment_image,
    tigris_secret,
    wandb_secret,
    run_single_experiment,
)
from tmgg_modal.storage import get_storage_from_env  # pyright: ignore[reportImplicitRelativeImport]


# Stage 1 configuration
STAGE1_ARCHITECTURES = [
    "models/spectral/linear_pe",
    "models/spectral/filter_bank_nonlinear",
    "models/spectral/self_attention",
]

STAGE1_HYPERPARAMETERS = {
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-2, 1e-3],
    "model.k": [8, 16],
}

STAGE1_SEEDS = [1, 2, 3]


def generate_stage1_configs() -> list[dict[str, Any]]:
    """Generate all experiment configurations for Stage 1.

    Returns
    -------
    list[dict]
        List of configuration dictionaries.
    """
    import itertools
    from tmgg_modal.paths import get_exp_configs_path  # pyright: ignore[reportImplicitRelativeImport]

    configs = []

    # Load base config using robust path resolution
    exp_configs = get_exp_configs_path()
    if exp_configs is not None:
        base_config_path = exp_configs / "base_config_spectral.yaml"
        if base_config_path.exists():
            base_config = OmegaConf.load(base_config_path)
        else:
            raise FileNotFoundError(
                f"Base config not found at {base_config_path}. "
                "Ensure tmgg exp_configs are properly set up."
            )
    else:
        raise RuntimeError(
            "Could not find tmgg package. Set TMGG_PATH or ensure "
            "modal/ and tmgg/ are siblings."
        )

    # Generate hyperparameter combinations
    hp_keys = list(STAGE1_HYPERPARAMETERS.keys())
    hp_values = [STAGE1_HYPERPARAMETERS[k] for k in hp_keys]
    hp_combos = list(itertools.product(*hp_values))

    for arch in STAGE1_ARCHITECTURES:
        for hp_combo in hp_combos:
            for seed in STAGE1_SEEDS:
                config = OmegaConf.create(
                    OmegaConf.to_container(base_config, resolve=True)
                )

                # Set architecture
                config.model = arch

                # Set hyperparameters
                for key, value in zip(hp_keys, hp_combo):
                    OmegaConf.update(config, key, value)

                # Set seed
                config.seed = seed

                # Generate run ID
                arch_name = arch.split("/")[-1]
                lr_str = f"lr{hp_combo[0]:.0e}".replace("e-0", "e-")
                wd_str = f"wd{hp_combo[1]:.0e}".replace("e-0", "e-")
                k_str = f"k{hp_combo[2]}"
                run_id = f"stage1_{arch_name}_{lr_str}_{wd_str}_{k_str}_s{seed}"
                config.run_id = run_id

                configs.append(OmegaConf.to_container(config, resolve=True))

    return configs


@app.function(
    image=experiment_image,
    secrets=[tigris_secret, wandb_secret],
    timeout=3600,  # 1 hour for orchestration
)
def run_stage1(
    parallelism: int = 4,
    gpu_type: str = "standard",
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run Stage 1: Proof of Concept experiments on Modal.

    Parameters
    ----------
    parallelism
        Maximum concurrent experiments.
    gpu_type
        GPU tier for experiments.
    dry_run
        If True, print configs without running.

    Returns
    -------
    dict
        Stage results summary.
    """
    from datetime import datetime

    configs = generate_stage1_configs()
    print(f"Stage 1: Generated {len(configs)} experiment configurations")

    if dry_run:
        print("Dry run - configurations:")
        for i, cfg in enumerate(configs[:5]):
            print(f"  [{i}] {cfg.get('run_id', 'unknown')}")
        if len(configs) > 5:
            print(f"  ... and {len(configs) - 5} more")
        return {"status": "dry_run", "num_configs": len(configs)}

    # Check for completed runs with fine-grained status
    from tmgg_modal.result_status import (  # pyright: ignore[reportImplicitRelativeImport]
        ResultStatus,
        filter_configs_by_status,
        summarize_status_map,
    )

    storage = get_storage_from_env()
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

    print(f"Running {len(configs)} experiments with parallelism={parallelism}")
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
    def get_val_loss(result: dict[str, Any]) -> float | None:
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
    parallelism: int = 4,
    gpu: str = "standard",
    dry_run: bool = False,
):
    """Local entry point for Stage 1.

    Usage
    -----
    modal run tmgg_modal/stages/stage1.py --parallelism 4 --gpu standard
    """
    result = run_stage1.remote(
        parallelism=parallelism,
        gpu_type=gpu,
        dry_run=dry_run,
    )
    print(f"Stage 1 result: {result}")
