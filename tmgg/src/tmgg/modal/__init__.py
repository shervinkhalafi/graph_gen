"""Modal cloud execution for TMGG spectral denoising experiments.

This package provides Modal-specific implementations for running
TMGG experiments on cloud GPUs with Tigris S3 storage.
"""

from tmgg.modal.app import GPU_CONFIGS, app
from tmgg.modal.runner import (
    ModalNotDeployedError,
    ModalRunner,
    SpawnedTask,
    check_modal_deployment,
    modal_execute_task,
)
from tmgg.modal.storage import TigrisStorage

__all__ = [
    "app",
    "GPU_CONFIGS",
    "ModalRunner",
    "ModalNotDeployedError",
    "SpawnedTask",
    "TigrisStorage",
    "check_modal_deployment",
    "modal_execute_task",
]
