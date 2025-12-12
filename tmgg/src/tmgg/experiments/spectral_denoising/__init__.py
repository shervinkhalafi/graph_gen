"""Spectral denoising experiment module.

Provides a unified experiment interface for training spectral graph denoising
architectures: Linear PE, Graph Filter Bank, and Self-Attention denoisers.

Usage
-----
Run via Hydra:
    python -m tmgg.experiments.spectral_denoising.runner

Or import directly:
    from tmgg.experiments.spectral_denoising import SpectralDenoisingLightningModule
"""

from tmgg.experiments.spectral_denoising.lightning_module import (
    SpectralDenoisingLightningModule,
)

__all__ = ["SpectralDenoisingLightningModule"]
