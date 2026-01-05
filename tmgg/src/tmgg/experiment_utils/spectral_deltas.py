"""Spectral delta metrics for comparing graphs during training.

Provides efficient batched computation of spectral structure differences
between clean and noisy/denoised graphs. Designed for integration with
Lightning training loops to log spectral delta metrics at validation time.

Four metrics are computed:
1. Eigengap delta: Relative change in spectral gap (lambda_max - lambda_{max-1})
2. Algebraic connectivity delta: Relative change in Fiedler value (lambda_2)
3. Eigenvalue drift: Relative L2 distance ||lambda_other - lambda_clean||_2
4. Subspace distance: Projection Frobenius ||P_clean - P_other||_F
"""

from __future__ import annotations

import torch

from tmgg.experiment_utils.eigenstructure_study.laplacian import compute_laplacian


def compute_spectral_deltas(
    A_clean: torch.Tensor,
    A_other: torch.Tensor,
    k: int = 4,
    compute_rotation: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Compute spectral delta metrics between clean and other (noisy/denoised) graphs.

    Parameters
    ----------
    A_clean : torch.Tensor
        Clean adjacency matrices, shape (n, n) or (batch, n, n).
    A_other : torch.Tensor
        Noisy or denoised adjacency matrices, same shape as A_clean.
    k : int
        Number of top eigenvectors for subspace comparison.
    compute_rotation : bool
        If True, also compute Procrustes rotation angle and residual metrics.

    Returns
    -------
    dict[str, torch.Tensor]
        Delta metrics, each of shape (batch,). For single graphs (2D input),
        returns shape (1,) - never scalars - for consistent downstream handling.
        - eigengap_delta: Relative change in spectral gap
        - alg_conn_delta: Relative change in algebraic connectivity
        - eigenvalue_drift: Relative L2 drift of eigenvalues
        - subspace_distance: Frobenius norm of projection difference
        - rotation_angle: (if compute_rotation=True) Procrustes rotation angle
        - rotation_residual: (if compute_rotation=True) Frobenius residual after alignment

    Notes
    -----
    Positive eigengap_delta means the gap increased in A_other.
    Positive alg_conn_delta means connectivity increased in A_other.
    For denoised graphs, we expect deltas closer to 0 than for noisy graphs.
    """
    # Handle 2D input (single graph) by adding batch dimension
    if A_clean.dim() == 2:
        A_clean = A_clean.unsqueeze(0)
        A_other = A_other.unsqueeze(0)

    # Compute eigendecompositions for adjacency matrices
    eig_clean, vec_clean = torch.linalg.eigh(A_clean)
    eig_other, vec_other = torch.linalg.eigh(A_other)

    # Compute Laplacian eigendecompositions
    L_clean = compute_laplacian(A_clean)
    L_other = compute_laplacian(A_other)
    lap_eig_clean, _ = torch.linalg.eigh(L_clean)
    lap_eig_other, _ = torch.linalg.eigh(L_other)

    # 1. Eigengap delta (relative)
    gap_clean = eig_clean[:, -1] - eig_clean[:, -2]
    gap_other = eig_other[:, -1] - eig_other[:, -2]
    eigengap_delta = (gap_other - gap_clean) / (gap_clean.abs() + 1e-10)

    # 2. Algebraic connectivity delta (relative)
    alg_conn_clean = lap_eig_clean[:, 1]
    alg_conn_other = lap_eig_other[:, 1]
    alg_conn_delta = (alg_conn_other - alg_conn_clean) / (alg_conn_clean.abs() + 1e-10)

    # 3. Eigenvalue drift (relative L2)
    eig_diff = torch.norm(eig_other - eig_clean, dim=-1)
    eig_norm = torch.norm(eig_clean, dim=-1) + 1e-10
    eigenvalue_drift = eig_diff / eig_norm

    # 4. Subspace distance (projection Frobenius)
    # Use top-k eigenvectors by largest eigenvalue (last k columns)
    V_clean_k = vec_clean[:, :, -k:]  # (batch, n, k)
    V_other_k = vec_other[:, :, -k:]
    P_clean = V_clean_k @ V_clean_k.transpose(-2, -1)  # (batch, n, n)
    P_other = V_other_k @ V_other_k.transpose(-2, -1)
    subspace_distance = torch.norm(P_clean - P_other, p="fro", dim=(-2, -1))

    result = {
        "eigengap_delta": eigengap_delta,
        "alg_conn_delta": alg_conn_delta,
        "eigenvalue_drift": eigenvalue_drift,
        "subspace_distance": subspace_distance,
    }

    # 5. Optional: Procrustes rotation metrics
    if compute_rotation:
        from tmgg.experiment_utils.eigenstructure_study.analyzer import (
            compute_procrustes_rotation_batch,
        )

        proc = compute_procrustes_rotation_batch(vec_clean, vec_other, k)
        result["rotation_angle"] = proc["angles"]
        result["rotation_residual"] = proc["residuals"]

    return result


def compute_spectral_deltas_summary(
    A_clean: torch.Tensor,
    A_other: torch.Tensor,
    k: int = 4,
    compute_rotation: bool = False,
) -> dict[str, float]:
    """
    Compute mean spectral delta metrics for a batch.

    Convenience wrapper around compute_spectral_deltas that returns
    batch-averaged scalar values suitable for direct logging.

    Parameters
    ----------
    A_clean : torch.Tensor
        Clean adjacency matrices, shape (batch, n, n).
    A_other : torch.Tensor
        Noisy or denoised adjacency matrices, same shape.
    k : int
        Number of top eigenvectors for subspace comparison.
    compute_rotation : bool
        If True, also compute Procrustes rotation metrics.

    Returns
    -------
    dict[str, float]
        Mean delta values across the batch.
    """
    deltas = compute_spectral_deltas(
        A_clean, A_other, k, compute_rotation=compute_rotation
    )
    return {key: val.mean().item() for key, val in deltas.items()}
