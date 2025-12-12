"""CLI entry points for experimental stages.

Each stage function is decorated with @hydra.main to provide a
standalone entry point that can be invoked via:
    tmgg-stage1 [overrides]
    tmgg-stage1-sanity [overrides]
    etc.

Stage configs are loaded via Hydra's config groups. Each CLI command
automatically injects +stage=<stage_name> to compose the stage config
on top of base_config_spectral.
"""

import sys
from functools import wraps
from pathlib import Path

import hydra
from omegaconf import DictConfig

# Config path relative to this file
TMGG_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH = str(TMGG_ROOT / "exp_configs")


def _inject_stage_override(stage_name: str):
    """Decorator that injects +stage=<stage_name> into CLI args before Hydra runs.

    This allows each CLI command (tmgg-stage1, tmgg-stage1-sanity, etc.) to
    automatically load its corresponding stage config without requiring the
    user to specify it manually.
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Inject stage override if not already present
            stage_override = f"+stage={stage_name}"
            if stage_override not in sys.argv:
                # Check if any stage override is already specified
                has_stage = any(arg.startswith("+stage=") or arg.startswith("stage=")
                               for arg in sys.argv)
                if not has_stage:
                    sys.argv.append(stage_override)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def _run_stage(cfg: DictConfig, stage_name: str) -> dict:
    """Common stage execution logic.

    Runs the stage either locally (default) or via cloud coordinator
    if cloud configuration is provided.
    """
    from tmgg.experiment_utils.cloud import ExperimentCoordinator
    from tmgg.experiment_utils.cloud.coordinator import StageConfig

    # Check if running in sweep mode or single experiment mode
    if cfg.get("sweep", False):
        # Use stage name from config if available (allows override via +stage=...)
        actual_stage_name = cfg.get("stage", stage_name)
        stage_config_path = TMGG_ROOT / "exp_configs" / "stage" / f"{actual_stage_name}.yaml"
        stage_config = StageConfig.from_yaml(stage_config_path)

        # Initialize coordinator
        coordinator = ExperimentCoordinator(
            base_config_path=Path(CONFIG_PATH),
        )

        # Run the stage sweep
        result = coordinator.run_stage(
            stage_config,
            cfg,
            parallelism=cfg.get("parallelism", 4),
            resume=cfg.get("resume", True),
        )

        return result.to_dict()
    else:
        # Single experiment mode - use standard run_experiment
        from tmgg.experiment_utils.run_experiment import run_experiment

        return run_experiment(cfg)


@_inject_stage_override("stage1_poc")
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="base_config_spectral",
)
def stage1(cfg: DictConfig) -> dict:
    """Stage 1: Proof of Concept.

    Validates that spectral PE architectures can denoise graphs.
    Tests Linear PE, Graph Filter + Sigmoid, Self-Attention on SBM n=50.

    Budget: 4.4 GPU-hours

    Usage
    -----
    Single experiment:
        tmgg-stage1

    Full sweep:
        tmgg-stage1 sweep=true

    With overrides:
        tmgg-stage1 model.k=16 learning_rate=5e-4
    """
    return _run_stage(cfg, "stage1_poc")


@_inject_stage_override("stage1_sanity")
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="base_config_spectral",
)
def stage1_sanity(cfg: DictConfig) -> dict:
    """Stage 1 Sanity Check: Constant Noise Memorization.

    Tests whether models can memorize a fixed noisy→clean mapping.
    Uses fixed_noise_seed to ensure identical noise every training step.
    Expected: ~99% accuracy if architecture and gradient flow work.

    This validates:
    - Model has sufficient capacity to represent the mapping
    - Gradient flow works correctly through all layers
    - Optimizer can update weights in the right direction

    If this fails, something fundamental is broken.

    Budget: < 20 GPU-minutes

    Usage
    -----
    Single experiment:
        tmgg-stage1-sanity

    Full sweep:
        tmgg-stage1-sanity sweep=true

    With overrides:
        tmgg-stage1-sanity model.k=8 learning_rate=5e-3
    """
    return _run_stage(cfg, "stage1_sanity")


@_inject_stage_override("stage1_5_crossdata")
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="base_config_spectral",
)
def stage1_5(cfg: DictConfig) -> dict:
    """Stage 1.5: Cross-Dataset Validation.

    Validates single-graph denoising across diverse graph families:
    - Synthetic: ER, d-regular, tree, ring of cliques, LFR
    - PyG benchmarks: QM9, ENZYMES, PROTEINS

    Uses high LR / zero weight decay settings matching stage 1.

    Budget: ~20 GPU-hours

    Usage
    -----
    Single experiment:
        tmgg-stage1-5

    Full sweep:
        tmgg-stage1-5 sweep=true

    With specific dataset:
        tmgg-stage1-5 data=data/er_single_graph
    """
    return _run_stage(cfg, "stage1_5_crossdata")


@_inject_stage_override("stage2_validation")
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="base_config_spectral",
)
def stage2(cfg: DictConfig) -> dict:
    """Stage 2: Core Validation.

    Validates generalization across configurations and compares with DiGress.
    Tests best architectures on multiple SBM configs.

    Budget: 166.5 GPU-hours

    Usage
    -----
    Single experiment:
        tmgg-stage2

    Full sweep:
        tmgg-stage2 sweep=true

    With specific architecture:
        tmgg-stage2 model=models/spectral/filter_bank_nonlinear
    """
    return _run_stage(cfg, "stage2_validation")


@_inject_stage_override("stage3_diversity")
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="base_config_spectral",
)
def stage3(cfg: DictConfig) -> dict:
    """Stage 3: Dataset Diversity (future work).

    Validates across all graph families.

    Budget: 400 GPU-hours (beyond initial 200h budget)

    Note: This stage is deferred to future work per experimental design.
    """
    return _run_stage(cfg, "stage3_diversity")


@_inject_stage_override("stage4_benchmarks")
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="base_config_spectral",
)
def stage4(cfg: DictConfig) -> dict:
    """Stage 4: Real-World Benchmarks (future work).

    Validates on practical benchmark datasets (QM9, ENZYMES, PROTEINS).

    Budget: 300 GPU-hours (beyond initial 200h budget)

    Note: This stage is deferred to future work per experimental design.
    """
    return _run_stage(cfg, "stage4_benchmarks")


@_inject_stage_override("stage5_full")
@hydra.main(
    version_base=None,
    config_path=CONFIG_PATH,
    config_name="base_config_spectral",
)
def stage5(cfg: DictConfig) -> dict:
    """Stage 5: Full Validation (future work).

    Comprehensive ablations and robustness analysis for publication.

    Budget: 1500 GPU-hours (beyond initial 200h budget)

    Note: This stage is deferred to future work per experimental design.
    """
    return _run_stage(cfg, "stage5_full")


if __name__ == "__main__":
    # Default to stage1 when run directly
    stage1()
