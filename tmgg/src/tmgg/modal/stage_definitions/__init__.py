"""Stage definitions for experiment sweeps.

Stage definitions are YAML files that specify:
- architectures to test
- hyperparameter grid
- seeds for replication
- run_id template
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

STAGE_DEFINITIONS_DIR = Path(__file__).parent


@dataclass
class StageDefinition:
    """Validated stage definition for experiment sweeps.

    Parameters
    ----------
    name
        Human-readable stage name.
    base_config
        Hydra config name (without .yaml), e.g. "base_config_spectral_arch".
    architectures
        Architecture config paths, e.g. ["models/spectral/linear_pe"].
    hyperparameters
        Grid of hyperparameter values, e.g. {"learning_rate": [1e-3, 1e-4]}.
    seeds
        Random seeds for replication.
    run_id_template
        Python format string for run IDs.
    datasets
        Dataset config paths. None means use the base config default.
    """

    name: str
    base_config: str
    architectures: list[str]
    hyperparameters: dict[str, list[Any]]
    seeds: list[int]
    run_id_template: str
    datasets: list[str] | None = None

    def __post_init__(self) -> None:
        if not self.architectures:
            raise ValueError(f"Stage '{self.name}' has no architectures")
        if not self.seeds:
            raise ValueError(f"Stage '{self.name}' has no seeds")

    @classmethod
    def from_yaml(cls, path: Path) -> StageDefinition:
        """Load and validate a stage definition from a YAML file.

        Parameters
        ----------
        path
            Path to the YAML file.

        Returns
        -------
        StageDefinition
            Validated stage definition.

        Raises
        ------
        ValueError
            If required keys are missing or validation fails.
        FileNotFoundError
            If the YAML file doesn't exist.
        """
        with open(path) as f:
            raw = yaml.safe_load(f)
        required = {
            "name",
            "base_config",
            "architectures",
            "hyperparameters",
            "seeds",
            "run_id_template",
        }
        missing = required - set(raw.keys())
        if missing:
            raise ValueError(f"Stage definition {path.name} missing keys: {missing}")
        # Only pass known fields to the dataclass
        known_fields = set(cls.__dataclass_fields__)
        return cls(**{k: raw[k] for k in known_fields if k in raw})


def load_stage_definition(stage_name: str) -> StageDefinition:
    """Load a stage definition YAML file.

    Parameters
    ----------
    stage_name
        Name of the stage (e.g., 'stage1', 'stage2').

    Returns
    -------
    StageDefinition
        Validated stage definition.

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
    return StageDefinition.from_yaml(stage_path)


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
