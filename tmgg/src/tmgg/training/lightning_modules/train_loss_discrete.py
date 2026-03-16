"""Discrete diffusion training loss."""

from __future__ import annotations

import torch
from torch import Tensor

from tmgg.diffusion.diffusion_sampling import mask_distributions


class TrainLossDiscrete:
    """Masked cross-entropy loss for discrete diffusion training.

    Computes separate node-level and edge-level cross-entropy losses,
    applying masking to exclude invalid positions (padding nodes and
    diagonal edges). The edge loss is weighted by ``lambda_E`` relative
    to the node loss.

    Parameters
    ----------
    lambda_E
        Weight for edge loss relative to node loss. Default is 5.0,
        following DiGress convention.
    """

    def __init__(self, lambda_E: float = 5.0) -> None:
        self.lambda_E = lambda_E

    def __call__(
        self,
        pred_X: Tensor,
        pred_E: Tensor,
        true_X: Tensor,
        true_E: Tensor,
        node_mask: Tensor,
    ) -> Tensor:
        """Compute masked cross-entropy loss.

        Parameters
        ----------
        pred_X
            Predicted node class probabilities, shape ``(bs, n, dx)``.
        pred_E
            Predicted edge class probabilities, shape ``(bs, n, n, de)``.
        true_X
            True node class distribution (one-hot or soft targets),
            shape ``(bs, n, dx)``.
        true_E
            True edge class distribution, shape ``(bs, n, n, de)``.
        node_mask
            Boolean mask for valid nodes, shape ``(bs, n)``.

        Returns
        -------
        Tensor
            Scalar loss value, averaged over the batch.
        """
        # mask_distributions sets invalid positions to uniform and adds eps
        true_X, true_E, pred_X, pred_E = mask_distributions(
            true_X, true_E, pred_X, pred_E, node_mask
        )

        # Cross-entropy: -sum(target * log(pred)) per position
        loss_X = -torch.sum(
            true_X * torch.log(pred_X.clamp(min=1e-10)), dim=-1
        )  # (bs, n)
        loss_E = -torch.sum(
            true_E * torch.log(pred_E.clamp(min=1e-10)), dim=-1
        )  # (bs, n, n)

        # Average node loss over valid positions
        num_nodes = node_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (bs, 1)
        loss_X = (loss_X * node_mask).sum(dim=-1) / num_nodes.squeeze(-1)  # (bs,)

        # Build edge mask: valid node pairs, excluding the diagonal
        diag_mask = ~torch.eye(
            node_mask.size(1), device=node_mask.device, dtype=torch.bool
        ).unsqueeze(0)
        edge_mask = (
            node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask
        )  # (bs, n, n)
        num_edges = edge_mask.sum(dim=(-1, -2)).clamp(min=1)  # (bs,)
        loss_E = (loss_E * edge_mask).sum(dim=(-1, -2)) / num_edges  # (bs,)

        return (loss_X + self.lambda_E * loss_E).mean()
