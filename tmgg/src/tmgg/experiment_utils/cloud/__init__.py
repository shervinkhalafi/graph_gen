"""Cloud execution abstractions for distributed experiment runs.

This module provides backend-agnostic interfaces for running experiments
on cloud platforms (Modal, Ray, etc.) with unified storage and coordination.

Use CloudRunnerFactory.create() to get a runner for a specific backend:

    >>> from tmgg.experiment_utils.cloud import CloudRunnerFactory
    >>> runner = CloudRunnerFactory.create("local")  # or "modal" if installed
    >>> result = runner.run_experiment(config)
"""

from tmgg.experiment_utils.cloud.base import CloudRunner, ExperimentResult, LocalRunner
from tmgg.experiment_utils.cloud.factory import CloudRunnerFactory
from tmgg.experiment_utils.cloud.storage import CloudStorage, S3Storage
from tmgg.experiment_utils.cloud.coordinator import ExperimentCoordinator

__all__ = [
    "CloudRunner",
    "CloudRunnerFactory",
    "ExperimentResult",
    "LocalRunner",
    "CloudStorage",
    "S3Storage",
    "ExperimentCoordinator",
]
