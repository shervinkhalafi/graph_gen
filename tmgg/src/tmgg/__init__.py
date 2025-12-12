"""TMGG: Graph denoising experiments with Hydra and PyTorch Lightning."""

__version__ = "0.1.0"

# Register custom OmegaConf resolver for version-based output paths
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("tmgg_version", lambda: __version__, replace=True)

from . import models
from . import experiment_utils

__all__ = ["models", "experiment_utils"]


def hello() -> str:
    return "Hello from tmgg!"
