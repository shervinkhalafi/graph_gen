"""Baseline experiments for sanity checking training pipelines."""

from tmgg.experiments._shared_utils.lightning_modules.denoising_module import (
    SingleStepDenoisingModule,
)

__all__ = ["SingleStepDenoisingModule"]
