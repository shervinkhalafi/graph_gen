"""Unified CLI entry point for TMGG experiments.

Replaces individual stage entry points with a single command that accepts
stage overrides via Hydra's config groups:

    tmgg-experiment +stage=stage1_poc
    tmgg-experiment +stage=stage2_validation
    tmgg-experiment +stage=stage1_poc --multirun model=...

For sweeps, use Hydra's native --multirun with the custom TmggLauncher:

    tmgg-experiment +stage=stage1_poc --multirun \\
        hydra/launcher=tmgg_modal \\
        model=models/spectral/linear_pe,models/spectral/filter_bank
"""

from pathlib import Path
from typing import Any

import hydra
from omegaconf import DictConfig

from tmgg.experiment_utils.cloud.base import LocalRunner
from tmgg.experiment_utils.cloud.coordinator import ExperimentCoordinator, StageConfig
from tmgg.modal.runner import ModalRunner

# Config path relative to this file
TMGG_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = str(TMGG_ROOT / "exp_configs")


def _run_stage(cfg: DictConfig, stage_name: str) -> dict[str, Any]:
    """Stage execution logic for sweep mode.

    When sweep=true, uses the ExperimentCoordinator to run a full sweep
    defined in the stage config.

    Parameters
    ----------
    cfg
        Hydra configuration with settings:
        - sweep: bool - Run full sweep (default: False)
        - run_on_modal: bool - Use Modal GPUs (default: False)
        - detach: bool - Fire-and-forget (default: False)
        - parallelism: int - Max concurrent experiments (default: 4)
        - resume: bool - Skip completed experiments (default: True)
    stage_name
        Stage identifier used for config lookup.

    Returns
    -------
    dict
        Stage results (empty if detach=true).
    """
    # Check if running in sweep mode or single experiment mode
    if cfg.get("sweep", False):
        # Use stage name from config if available
        actual_stage_name = cfg.get("stage", stage_name)
        stage_config_path = (
            TMGG_ROOT / "exp_configs" / "stage" / f"{actual_stage_name}.yaml"
        )
        stage_config = StageConfig.from_yaml(stage_config_path)

        # Select runner based on config
        runner_instance = (
            ModalRunner() if cfg.get("run_on_modal", False) else LocalRunner()
        )

        # Initialize coordinator with runner
        coordinator = ExperimentCoordinator(
            runner=runner_instance,
            base_config_path=Path(CONFIG_PATH),
        )

        # Detached mode: spawn and exit immediately
        if cfg.get("detach", False):
            if not cfg.get("run_on_modal", False):
                raise ValueError("detach=true requires run_on_modal=true")

            spawned = coordinator.spawn_stage(
                stage_config,
                cfg,
                resume=cfg.get("resume", True),
            )

            if not spawned:
                print("All experiments already completed (or none to run).")
                return {}

            print(f"Spawned {len(spawned)} experiments on Modal (detached):")
            print(f"  GPU tier: {stage_config.gpu_type}")
            for task in spawned:
                print(f"  - {task.run_id}")
            print(
                "\nExperiments running in background. Check Modal dashboard for status:"
            )
            print("  https://modal.com/apps")

            return {"spawned_run_ids": [t.run_id for t in spawned]}

        # Blocking mode: run and wait for results
        result = coordinator.run_stage(
            stage_config,
            cfg,
            resume=cfg.get("resume", True),
        )

        return result.to_dict()
    else:
        # Single experiment mode - use standard run_experiment
        from tmgg.experiment_utils.run_experiment import run_experiment

        return run_experiment(cfg)


@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="base_config_spectral",
)
def main(cfg: DictConfig) -> dict[str, Any]:
    """Unified experiment entry point.

    Runs a single experiment or coordinates a sweep based on configuration.
    Stage configs are loaded via Hydra's config groups using +stage=<name>.

    Usage
    -----
    Single experiment:
        tmgg-experiment +stage=stage1_poc

    Single experiment with model override:
        tmgg-experiment +stage=stage1_poc model=models/spectral/filter_bank

    Sweep (using stage's _sweep_config):
        tmgg-experiment +stage=stage1_poc sweep=true

    Multirun via Hydra (local):
        tmgg-experiment +stage=stage1_poc --multirun \\
            model=models/spectral/linear_pe,models/spectral/filter_bank

    Multirun via Modal:
        tmgg-experiment +stage=stage1_poc --multirun \\
            hydra/launcher=tmgg_modal \\
            model=models/spectral/linear_pe,models/spectral/filter_bank

    Parameters
    ----------
    cfg
        Hydra configuration composed from base config + stage override.

    Returns
    -------
    dict
        Experiment results.
    """
    # Get stage name from config (set via +stage=<name>)
    stage_name = cfg.get("stage", "stage1_poc")
    return _run_stage(cfg, stage_name)


if __name__ == "__main__":
    main()
