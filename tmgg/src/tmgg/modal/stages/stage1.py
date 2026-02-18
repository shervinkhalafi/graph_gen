"""Stage 1: Proof of Concept — configuration generation.

.. deprecated::
    The ``run_stage1`` Modal function and ``main`` entrypoint have been
    removed. Use YAML ``stage_definitions/`` + ``generate_configs`` +
    ``launch_sweep`` pipeline instead.

    ``validate_prefix`` and ``generate_stage1_configs`` are retained for
    downstream use.
"""

from __future__ import annotations

import re
import warnings
from typing import Any

warnings.warn(
    "tmgg.modal.stages.stage1 is deprecated. "
    "Use YAML stage_definitions/ + generate_configs + launch_sweep pipeline.",
    DeprecationWarning,
    stacklevel=2,
)


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
    "models/spectral/filter_bank",
    "models/spectral/self_attention",
]

STAGE1_HYPERPARAMETERS = {
    "learning_rate": [1e-4, 5e-4, 1e-3],
    "weight_decay": [1e-2, 1e-3],
    "+model.k": [8, 16],
}

STAGE1_SEEDS = [1, 2, 3]


def generate_stage1_configs() -> list[dict[str, Any]]:
    """Generate all experiment configurations for Stage 1.

    Uses Hydra composition to resolve config defaults and interpolations.

    Returns
    -------
    list[dict]
        List of configuration dictionaries with all values resolved.
    """
    import itertools

    from omegaconf import OmegaConf

    from tmgg.modal.config_compose import compose_config

    configs = []

    hp_keys = list(STAGE1_HYPERPARAMETERS.keys())
    hp_values = [STAGE1_HYPERPARAMETERS[k] for k in hp_keys]
    hp_combos = list(itertools.product(*hp_values))

    for arch in STAGE1_ARCHITECTURES:
        for hp_combo in hp_combos:
            for seed in STAGE1_SEEDS:
                overrides = [f"model={arch}"]
                for key, value in zip(hp_keys, hp_combo, strict=False):
                    overrides.append(f"{key}={value}")
                overrides.append(f"seed={seed}")

                config = compose_config("base_config_spectral_arch", overrides)

                arch_name = arch.split("/")[-1]
                lr_str = f"lr{hp_combo[0]:.0e}".replace("e-0", "e-")
                wd_str = f"wd{hp_combo[1]:.0e}".replace("e-0", "e-")
                k_str = f"k{hp_combo[2]}"
                run_id = f"stage1_{arch_name}_{lr_str}_{wd_str}_{k_str}_s{seed}"

                config_dict: dict[str, Any] = OmegaConf.to_container(
                    config, resolve=True
                )  # type: ignore[assignment]  # pyright: ignore[reportAssignmentType]
                config_dict["run_id"] = run_id

                configs.append(config_dict)

    return configs
