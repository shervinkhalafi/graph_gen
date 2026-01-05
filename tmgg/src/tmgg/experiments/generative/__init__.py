"""Generative graph modeling experiment with MMD evaluation.

This module provides a complete pipeline for training generative graph models
using discrete denoising diffusion and evaluating them with MMD metrics on
graph-theoretic properties (degree distribution, clustering, spectral).
"""

from tmgg.experiments.generative.datamodule import GraphDistributionDataModule
from tmgg.experiments.generative.lightning_module import GenerativeLightningModule

__all__ = [
    "GenerativeLightningModule",
    "GraphDistributionDataModule",
]
