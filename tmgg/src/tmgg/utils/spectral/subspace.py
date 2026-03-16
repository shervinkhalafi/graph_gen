"""Subspace comparison via orthogonal Procrustes rotation.

Quantifies how much the eigenvector basis rotates when a graph is
perturbed (by noise, denoising, etc.).  The batch implementation is the
canonical one; the single-graph variant delegates to it.
"""

from __future__ import annotations

import torch


def compute_procrustes_rotation_batch(
    V1_batch: torch.Tensor,
    V2_batch: torch.Tensor,
    k: int,
    *,
    return_rotation: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute optimal orthogonal Procrustes rotation for a batch of eigenvector pairs.

    Finds the rotation R minimizing ``||U1 - U2 @ R||_F`` per sample, where
    U1/U2 are the top-*k* eigenvector subspaces (last *k* columns in
    ascending-eigenvalue order).

    Parameters
    ----------
    V1_batch, V2_batch : torch.Tensor
        Batched eigenvector matrices of shape ``(batch, n, n)``.
    k : int
        Number of top eigenvectors to consider (last *k* columns).
    return_rotation : bool
        If True, also return the ``(batch, k, k)`` rotation matrices.

    Returns
    -------
    dict[str, torch.Tensor]
        Always contains ``angles`` and ``residuals``, each ``(batch,)``.
        When *return_rotation* is True, also contains ``rotations`` of
        shape ``(batch, k, k)``.
    """
    U1 = V1_batch[:, :, -k:]  # (batch, n, k)
    U2 = V2_batch[:, :, -k:]
    M = U1.transpose(-2, -1) @ U2  # (batch, k, k)
    U, _, Vt = torch.linalg.svd(M)
    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)  # (batch, k, k)

    trace_R = torch.diagonal(R, dim1=-2, dim2=-1).sum(dim=-1)
    cos_theta = ((trace_R - 1) / max(k - 1, 1)).clamp(-1.0, 1.0)
    angles = torch.acos(cos_theta)
    residuals = torch.norm(U1 - U2 @ R, p="fro", dim=(-2, -1))

    result: dict[str, torch.Tensor] = {"angles": angles, "residuals": residuals}
    if return_rotation:
        result["rotations"] = R
    return result


def compute_procrustes_rotation(
    V1: torch.Tensor, V2: torch.Tensor, k: int
) -> dict[str, torch.Tensor]:
    """Compute Procrustes rotation for a single pair of eigenvector matrices.

    Thin wrapper around :func:`compute_procrustes_rotation_batch` that
    accepts unbatched ``(n, n)`` inputs and returns the rotation matrix,
    angle, and residual as scalar tensors.

    Parameters
    ----------
    V1, V2 : torch.Tensor
        Eigenvector matrices of shape ``(n, n)``, columns sorted by
        ascending eigenvalue.
    k : int
        Number of top eigenvectors to consider.

    Returns
    -------
    dict[str, torch.Tensor]
        ``rotation`` (k, k), ``angle`` (scalar), ``residual`` (scalar).
    """
    batch = compute_procrustes_rotation_batch(
        V1.unsqueeze(0), V2.unsqueeze(0), k, return_rotation=True
    )
    return {
        "rotation": batch["rotations"][0],
        "angle": batch["angles"][0],
        "residual": batch["residuals"][0],
    }
