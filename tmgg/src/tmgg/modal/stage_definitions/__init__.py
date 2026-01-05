"""Stage definitions for experiment sweeps.

Stage definitions are YAML files that specify:
- architectures to test
- hyperparameter grid
- seeds for replication
- run_id template
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

STAGE_DEFINITIONS_DIR = Path(__file__).parent


def load_stage_definition(stage_name: str) -> dict[str, Any]:
    """Load a stage definition YAML file.

    Parameters
    ----------
    stage_name
        Name of the stage (e.g., 'stage1', 'stage2').

    Returns
    -------
    dict
        Stage definition with keys: name, base_config, architectures,
        hyperparameters, seeds, run_id_template.

    Raises
    ------
    FileNotFoundError
        If the stage definition file doesn't exist.
    """
    stage_path = STAGE_DEFINITIONS_DIR / f"{stage_name}.yaml"
    if not stage_path.exists():
        raise FileNotFoundError(
            f"Stage definition not found: {stage_path}\n"
            f"Available stages: {list_stages()}"
        )

    with open(stage_path) as f:
        return yaml.safe_load(f)


def list_stages() -> list[str]:
    """List available stage definitions.

    Returns
    -------
    list[str]
        Names of available stages (without .yaml extension).
    """
    return [
        p.stem
        for p in STAGE_DEFINITIONS_DIR.glob("*.yaml")
        if not p.name.startswith("_")
    ]
