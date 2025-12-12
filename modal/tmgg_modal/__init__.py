"""Modal cloud execution for TMGG spectral denoising experiments.

This package provides Modal-specific implementations for running
TMGG experiments on cloud GPUs with Tigris S3 storage.
"""

from tmgg_modal.app import app, GPU_CONFIGS
from tmgg_modal.runner import ModalRunner
from tmgg_modal.storage import TigrisStorage

__all__ = ["app", "GPU_CONFIGS", "ModalRunner", "TigrisStorage"]
