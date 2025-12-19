"""TMGG: Graph denoising experiments with Hydra and PyTorch Lightning."""

from omegaconf import OmegaConf

from . import experiment_utils, models

__version__ = "0.1.0"
__all__ = ["models", "experiment_utils"]

# Register custom OmegaConf resolver for version-based output paths
OmegaConf.register_new_resolver("tmgg_version", lambda: __version__, replace=True)


def hello() -> str:
    return "Hello from tmgg!"
