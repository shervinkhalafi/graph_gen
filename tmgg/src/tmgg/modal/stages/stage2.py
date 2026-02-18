"""Stage 2: Core Validation — configuration generation.

.. deprecated::
    The ``run_stage2`` Modal function and ``main`` entrypoint have been
    removed. Use YAML ``stage_definitions/`` + ``generate_configs`` +
    ``launch_sweep`` pipeline instead.

    ``generate_stage2_configs`` is retained for downstream use.
"""

from __future__ import annotations

import copy
import warnings
from typing import Any

warnings.warn(
    "tmgg.modal.stages.stage2 is deprecated. "
    "Use YAML stage_definitions/ + generate_configs + launch_sweep pipeline.",
    DeprecationWarning,
    stacklevel=2,
)

# Stage 2 configuration (module-level defaults)
STAGE2_ARCHITECTURES = [
    "models/spectral/filter_bank",
    "models/spectral/self_attention",
    "models/digress/digress_transformer",
]

STAGE2_DATASETS = [
    "data/sbm_default",
    "data/sbm_n100",
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

    Uses Hydra composition to resolve config defaults and interpolations.

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
                    config = compose_config("base_config_spectral_arch", overrides)

                    # Generate run ID
                    arch_name = arch.split("/")[-1]
                    data_name = dataset.split("/")[-1]
                    lr_str = f"lr{hp_combo[0]:.0e}".replace("e-0", "e-")
                    wd_str = f"wd{hp_combo[1]:.0e}".replace("e-0", "e-")
                    k_str = f"k{hp_combo[2]}"
                    run_id = f"stage2_{arch_name}_{data_name}_{lr_str}_{wd_str}_{k_str}_s{seed}"

                    # Convert to dict and add run_id
                    config_dict: dict[str, Any] = OmegaConf.to_container(
                        config, resolve=True
                    )  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
                    config_dict["run_id"] = run_id

                    configs.append(config_dict)

    return configs
