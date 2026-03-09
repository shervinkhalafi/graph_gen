"""Spectral analysis primitives for eigenstructure comparison.

Pure linear-algebra utilities for computing Laplacians, eigenvector subspace
distances, Procrustes rotations, and spectral deltas between graph pairs.
"""

from tmgg.experiments._shared_utils.spectral_utils.laplacian import compute_laplacian
from tmgg.experiments._shared_utils.spectral_utils.spectral_deltas import (
    compute_alg_connectivity_delta,
    compute_eigengap_delta,
    compute_eigenvalue_drift,
    compute_spectral_deltas,
    compute_subspace_distance_from_eigenvectors,
)
from tmgg.experiments._shared_utils.spectral_utils.subspace import (
    compute_procrustes_rotation,
    compute_procrustes_rotation_batch,
)

__all__ = [
    "compute_alg_connectivity_delta",
    "compute_eigengap_delta",
    "compute_eigenvalue_drift",
    "compute_laplacian",
    "compute_procrustes_rotation",
    "compute_procrustes_rotation_batch",
    "compute_spectral_deltas",
    "compute_subspace_distance_from_eigenvectors",
]
