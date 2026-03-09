"""Discrete diffusion experiment for graph generation.

Evaluation utilities (``evaluate_discrete_samples``,
``categorical_samples_to_graphs``) live in the ``evaluate`` submodule
and should be imported directly from there to avoid an import cycle.

``DiffusionModule`` is the active training module for discrete diffusion.
"""

from tmgg.experiments._shared_utils.lightning_modules.diffusion_module import (
    DiffusionModule,
)
from tmgg.experiments.discrete_diffusion_generative.datamodule import (
    SyntheticCategoricalDataModule,
)

__all__ = [
    "DiffusionModule",
    "SyntheticCategoricalDataModule",
]
