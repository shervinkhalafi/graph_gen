"""Diffusion components for discrete and continuous graph generation.

The pipeline composes four kinds of object:

- **NoiseSchedule** precomputes ``beta``, ``alpha``, and ``alpha_bar`` tensors
  that control how quickly signal decays over ``T`` diffusion steps.
- **NoiseProcess** (categorical or continuous) applies forward noise and
  computes VLB terms. Categorical processes delegate to a **TransitionModel**
  for the discrete transition matrices.
- **Sampler** runs the reverse diffusion loop, denoising from the limit
  distribution back to clean graphs.
- **TransitionModel** (protocol) defines the row-stochastic matrices ``Q_t``
  and ``Q̄_t`` used by the categorical noise process and sampler.
"""

from .graph_types import LimitDistribution, TransitionMatrices
from .noise_process import (
    CategoricalNoiseProcess,
    ContinuousNoiseProcess,
    NoiseProcess,
)
from .protocols import TransitionModel
from .sampler import (
    CategoricalSampler,
    ContinuousSampler,
    Sampler,
)
from .schedule import NoiseSchedule
from .transitions import DiscreteUniformTransition, MarginalUniformTransition

__all__ = [
    "CategoricalNoiseProcess",
    "CategoricalSampler",
    "ContinuousNoiseProcess",
    "ContinuousSampler",
    "DiscreteUniformTransition",
    "LimitDistribution",
    "MarginalUniformTransition",
    "NoiseProcess",
    "NoiseSchedule",
    "Sampler",
    "TransitionMatrices",
    "TransitionModel",
]
