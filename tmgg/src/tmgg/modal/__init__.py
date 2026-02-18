"""Modal cloud execution for TMGG spectral denoising experiments.

This package provides Modal-specific implementations for running
TMGG experiments on cloud GPUs with Tigris S3 storage.

No ``import modal`` at package level. The ``modal.App`` and all
``@app.function`` decorators live in ``_functions.py``, the sole
deployment entry-point. Runtime code uses ``modal.Function.from_name()``
lazily inside method bodies.
"""

from tmgg.modal.app import GPU_CONFIGS, MODAL_APP_NAME
from tmgg.modal.runner import (
    ModalNotDeployedError,
    ModalRunner,
    ModalSpawnedTask,
    check_modal_deployment,
    create_runner,
)
from tmgg.modal.storage import TigrisStorage

__all__ = [
    "GPU_CONFIGS",
    "MODAL_APP_NAME",
    "ModalNotDeployedError",
    "ModalRunner",
    "ModalSpawnedTask",
    "TigrisStorage",
    "check_modal_deployment",
    "create_runner",
]
