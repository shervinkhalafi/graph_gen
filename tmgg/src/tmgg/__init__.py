"""TMGG: Graph denoising experiments with Hydra and PyTorch Lightning."""

from omegaconf import OmegaConf

from . import experiment_utils, models

__version__ = "0.1.0"
__all__ = ["models", "experiment_utils"]

# Register custom OmegaConf resolvers
OmegaConf.register_new_resolver("tmgg_version", lambda: __version__, replace=True)
OmegaConf.register_new_resolver(
    "if_",
    lambda cond, true_val, false_val: true_val if cond else false_val,
    replace=True,
)
OmegaConf.register_new_resolver("not_", lambda x: not x, replace=True)


def hello() -> str:
    return "Hello from tmgg!"
