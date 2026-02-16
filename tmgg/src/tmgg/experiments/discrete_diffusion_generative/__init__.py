"""Discrete diffusion experiment for graph generation.

Evaluation utilities (``evaluate_discrete_samples``,
``categorical_samples_to_graphs``) live in the ``evaluate`` submodule
and should be imported directly from there to avoid an import cycle
with ``lightning_module``.
"""

from tmgg.experiments.discrete_diffusion_generative.datamodule import (
    SyntheticCategoricalDataModule,
)
from tmgg.experiments.discrete_diffusion_generative.lightning_module import (
    DiscreteDiffusionLightningModule,
)

__all__ = [
    "DiscreteDiffusionLightningModule",
    "SyntheticCategoricalDataModule",
]
