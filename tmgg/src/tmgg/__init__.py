"""TMGG: Graph denoising experiments with Hydra and PyTorch Lightning."""

__version__ = "0.1.0"

from . import models
from . import experiment_utils

__all__ = ["models", "experiment_utils"]


def hello() -> str:
    return "Hello from tmgg!"
