"""Cloud execution abstractions for distributed experiment runs.

This module provides backend-agnostic interfaces for running experiments
on cloud platforms (Modal, local) with unified storage and coordination.

    >>> from tmgg.experiment_utils.cloud import LocalRunner
    >>> runner = LocalRunner()  # Sequential, single-process
    >>> result = runner.run_experiment(config)
"""

from tmgg.experiment_utils.cloud.base import (
    CloudRunner,
    ExperimentResult,
    LocalRunner,
    SpawnedTask,
)
from tmgg.experiment_utils.cloud.coordinator import ExperimentCoordinator
from tmgg.experiment_utils.cloud.storage import CloudStorage, LocalStorage, S3Storage

__all__ = [
    "CloudRunner",
    "ExperimentResult",
    "LocalRunner",
    "SpawnedTask",
    "CloudStorage",
    "LocalStorage",
    "S3Storage",
    "ExperimentCoordinator",
]
