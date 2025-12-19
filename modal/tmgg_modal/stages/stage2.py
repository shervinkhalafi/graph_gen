"""Stage 2: Core Validation - Modal execution.

Validates generalization across configurations and compares with DiGress.
Budget: 166.5 GPU-hours
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
    run_single_experiment_fast,
)
from tmgg_modal.storage import get_storage_from_env  # pyright: ignore[reportImplicitRelativeImport]


# Stage 2 configuration
STAGE2_ARCHITECTURES = [
    "models/spectral/filter_bank_nonlinear",
    "models/spectral/self_attention",
    "models/digress/digress_transformer",
]

STAGE2_DATASETS = [
    "data/sbm_default",
    # "data/sbm_n100",  # Uncomment when config exists
]

STAGE2_HYPERPARAMETERS = {
    "learning_rate": [1e-5, 5e-5, 1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-1, 1e-2, 1e-3],
    "model.k": [4, 8, 16, 32],
}

STAGE2_SEEDS = [1, 2, 3]


def generate_stage2_configs(
    best_stage1_config: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Generate experiment configurations for Stage 2.

    Parameters
    ----------
    best_stage1_config
        Best configuration from Stage 1 to include.

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

    # Use best config from Stage 1 if available
    if best_stage1_config:
        # Narrow hyperparameter search around best values
        best_lr = best_stage1_config.get("learning_rate", 1e-4)
        best_wd = best_stage1_config.get("weight_decay", 1e-2)
        STAGE2_HYPERPARAMETERS["learning_rate"] = [best_lr / 5, best_lr, best_lr * 5]
        STAGE2_HYPERPARAMETERS["weight_decay"] = [best_wd / 10, best_wd, best_wd * 10]

    # Generate hyperparameter combinations
    hp_keys = list(STAGE2_HYPERPARAMETERS.keys())
    hp_values = [STAGE2_HYPERPARAMETERS[k] for k in hp_keys]
    hp_combos = list(itertools.product(*hp_values))

    for arch in STAGE2_ARCHITECTURES:
        for dataset in STAGE2_DATASETS:
            for hp_combo in hp_combos:
                for seed in STAGE2_SEEDS:
                    config = OmegaConf.create(
                        OmegaConf.to_container(base_config, resolve=True)
                    )

                    # Set architecture and dataset
                    config.model = arch
                    config.data = dataset

                    # Set hyperparameters
                    for key, value in zip(hp_keys, hp_combo):
                        OmegaConf.update(config, key, value)

                    # Set seed
                    config.seed = seed

                    # Generate run ID
                    arch_name = arch.split("/")[-1]
                    data_name = dataset.split("/")[-1]
                    lr_str = f"lr{hp_combo[0]:.0e}".replace("e-0", "e-")
                    wd_str = f"wd{hp_combo[1]:.0e}".replace("e-0", "e-")
                    k_str = f"k{hp_combo[2]}"
                    run_id = f"stage2_{arch_name}_{data_name}_{lr_str}_{wd_str}_{k_str}_s{seed}"
                    config.run_id = run_id

                    configs.append(OmegaConf.to_container(config, resolve=True))

    return configs


@app.function(
    image=experiment_image,
    secrets=[tigris_secret, wandb_secret],
    timeout=7200,  # 2 hours for orchestration
)
def run_stage2(
    parallelism: int = 4,
    gpu_type: str = "standard",
    use_stage1_best: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Run Stage 2: Core Validation experiments on Modal.

    Parameters
    ----------
    parallelism
        Maximum concurrent experiments.
    gpu_type
        GPU tier for experiments.
    use_stage1_best
        If True, use Stage 1 best config to narrow search.
    dry_run
        If True, print configs without running.

    Returns
    -------
    dict
        Stage results summary.
    """
    from datetime import datetime

    # Load Stage 1 best config if requested
    best_stage1 = None
    if use_stage1_best:
        storage = get_storage_from_env()
        if storage:
            try:
                stage1_summary = storage.download_metrics("stage1_poc_summary")
                best_run_id = stage1_summary.get("best_run_id")
                if best_run_id:
                    best_result = storage.download_metrics(f"results/{best_run_id}")
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
    def get_val_loss(result: dict[str, Any]) -> float | None:
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
    parallelism: int = 4,
    gpu: str = "standard",
    use_stage1_best: bool = True,
    dry_run: bool = False,
):
    """Local entry point for Stage 2.

    Usage
    -----
    modal run tmgg_modal/stages/stage2.py --parallelism 4 --gpu fast
    """
    result = run_stage2.remote(
        parallelism=parallelism,
        gpu_type=gpu,
        use_stage1_best=use_stage1_best,
        dry_run=dry_run,
    )
    print(f"Stage 2 result: {result}")
