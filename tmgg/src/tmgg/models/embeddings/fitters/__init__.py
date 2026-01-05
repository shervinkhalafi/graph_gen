"""Fitters for graph embedding optimization.

This module provides different fitting strategies for learning graph embeddings:
- GradientFitter: Adam optimizer with BCE/MSE loss and early stopping
- SpectralFitter: SVD/eigendecomposition for closed-form initialization/fitting
- GaugeStabilizedFitter: Hadamard-based gauge stabilization for LPCA embeddings
"""

from tmgg.models.embeddings.fitters.gauge_stabilized import (
    GaugeStabilizedConfig,
    GaugeStabilizedFitter,
    canonicalize_eigenvectors,
    compute_hadamard_anchors,
    compute_hadamard_theta,
    interpolate_embeddings_naive,
    interpolated_svd_init,
    interpolated_theta_init,
)
from tmgg.models.embeddings.fitters.gradient import GradientFitter
from tmgg.models.embeddings.fitters.spectral import SpectralFitter

__all__ = [
    "GradientFitter",
    "SpectralFitter",
    "GaugeStabilizedFitter",
    "GaugeStabilizedConfig",
    "canonicalize_eigenvectors",
    "compute_hadamard_theta",
    "compute_hadamard_anchors",
    "interpolated_theta_init",
    "interpolated_svd_init",
    "interpolate_embeddings_naive",
]
