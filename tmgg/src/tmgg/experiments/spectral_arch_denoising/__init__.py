"""Spectral denoising experiment module.

Provides a unified experiment interface for training spectral graph denoising
architectures: Linear PE, Graph Filter Bank, and Self-Attention denoisers.

Usage
-----
Run via Hydra:
    python -m tmgg.experiments.spectral_arch_denoising.runner
"""

from tmgg.training.lightning_modules.denoising_module import (
    SingleStepDenoisingModule,
)

__all__ = ["SingleStepDenoisingModule"]
