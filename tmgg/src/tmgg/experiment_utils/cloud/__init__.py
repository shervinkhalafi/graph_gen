"""Cloud execution abstractions for distributed experiment runs.

This module provides backend-agnostic interfaces for running experiments
on cloud platforms (Modal, Ray, local) with unified storage and coordination.

Use LocalRunner for sequential execution, RayRunner for local parallelism,
or ModalRunner for cloud execution:

    >>> from tmgg.experiment_utils.cloud import LocalRunner, RayRunner
    >>> runner = LocalRunner()  # Sequential, single-process
    >>> runner = RayRunner(max_concurrent=4)  # Parallel, local Ray cluster
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

# RayRunner is optional (requires ray package)
try:
    from tmgg.experiment_utils.cloud.ray_runner import (  # noqa: F401
        RayRunner,
        RaySpawnedTask,
    )

    _RAY_AVAILABLE = True
except ImportError:
    _RAY_AVAILABLE = False

# SlurmRunner (requires SLURM cluster access)
from tmgg.experiment_utils.cloud.slurm_runner import (  # noqa: F401
    SlurmConfig,
    SlurmRunner,
    SlurmSpawnedTask,
)

__all__ = [
    "CloudRunner",
    "ExperimentResult",
    "LocalRunner",
    "SpawnedTask",
    "CloudStorage",
    "LocalStorage",
    "S3Storage",
    "ExperimentCoordinator",
    # SLURM
    "SlurmRunner",
    "SlurmConfig",
    "SlurmSpawnedTask",
]

if _RAY_AVAILABLE:
    __all__.extend(["RayRunner", "RaySpawnedTask"])
