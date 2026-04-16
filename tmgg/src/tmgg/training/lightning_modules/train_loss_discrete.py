"""Discrete diffusion training loss — per-field masked CE helpers.

The Wave 5.1 refactor splits the original joint ``(X, E)`` loss into per-field
helpers so :class:`~tmgg.training.lightning_modules.diffusion_module.DiffusionModule`
can iterate over ``noise_process.fields``. :class:`TrainLossDiscrete` is now a
thin compatibility wrapper that composes the node and edge helpers with a
``lambda_E`` weighting, keeping upstream DiGress parity tests that call it
directly green. New callers should prefer the helpers below.
"""

from __future__ import annotations

import torch
from torch import Tensor


def _node_mask_fill_uniform(tensor: Tensor, node_mask: Tensor) -> Tensor:
    """Return ``tensor`` with masked node rows replaced by a ``(1, 0, ...)`` pad.

    Matches the semantics of
    :func:`tmgg.diffusion.diffusion_sampling.mask_distributions` for the node
    view: invalid positions get a deterministic distribution that is later
    excluded from the averaging, so they contribute exactly zero to the loss.
    """
    row = torch.zeros(tensor.size(-1), dtype=torch.float, device=tensor.device)
    row[0] = 1.0
    out = tensor.clone()
    out[~node_mask] = row
    return out


def _edge_mask_fill_uniform(tensor: Tensor, node_mask: Tensor) -> Tensor:
    """Return ``tensor`` with masked edge positions replaced by a ``(1, 0, ...)`` pad.

    Padding covers both padding-node pairs and the diagonal; the returned
    tensor can then be epsilon-smoothed without corrupting valid entries.
    """
    row = torch.zeros(tensor.size(-1), dtype=torch.float, device=tensor.device)
    row[0] = 1.0
    diag_mask = ~torch.eye(
        node_mask.size(1), device=node_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    valid = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask
    out = tensor.clone()
    out[~valid, :] = row
    return out


def _epsilon_renormalise(tensor: Tensor, eps: float = 1e-7) -> Tensor:
    """Add a small ``eps`` and renormalise along the last axis.

    Mirrors the renormalisation inside ``mask_distributions`` so per-field
    masked CE values stay numerically equal to the legacy dual-field path.
    """
    out = tensor + eps
    return out / out.sum(dim=-1, keepdim=True)


def masked_node_ce(
    pred_X: Tensor,
    true_X: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Per-graph masked cross-entropy over a node-shaped categorical field.

    Parameters
    ----------
    pred_X
        Predicted PMF, shape ``(bs, n, dx)``. Values should already be
        post-softmax.
    true_X
        Target PMF (typically one-hot), shape ``(bs, n, dx)``.
    node_mask
        Boolean mask of valid nodes, shape ``(bs, n)``.

    Returns
    -------
    Tensor
        Scalar CE loss averaged over valid node positions per graph, then
        mean-reduced over the batch.
    """
    true_X = _node_mask_fill_uniform(true_X, node_mask)
    pred_X = _node_mask_fill_uniform(pred_X, node_mask)

    true_X = _epsilon_renormalise(true_X)
    pred_X = _epsilon_renormalise(pred_X)

    loss = -torch.sum(true_X * torch.log(pred_X.clamp(min=1e-10)), dim=-1)  # (bs, n)
    num_nodes = node_mask.sum(dim=-1, keepdim=True).clamp(min=1)  # (bs, 1)
    per_graph = (loss * node_mask).sum(dim=-1) / num_nodes.squeeze(-1)  # (bs,)
    return per_graph.mean()


def masked_edge_ce(
    pred_E: Tensor,
    true_E: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Per-graph masked cross-entropy over an edge-shaped categorical field.

    Parameters
    ----------
    pred_E
        Predicted PMF, shape ``(bs, n, n, de)``. Already post-softmax.
    true_E
        Target PMF, shape ``(bs, n, n, de)``.
    node_mask
        Boolean mask of valid nodes, shape ``(bs, n)``.

    Returns
    -------
    Tensor
        Scalar CE loss averaged over valid off-diagonal edge positions per
        graph, then mean-reduced over the batch.
    """
    true_E = _edge_mask_fill_uniform(true_E, node_mask)
    pred_E = _edge_mask_fill_uniform(pred_E, node_mask)

    true_E = _epsilon_renormalise(true_E)
    pred_E = _epsilon_renormalise(pred_E)

    loss = -torch.sum(true_E * torch.log(pred_E.clamp(min=1e-10)), dim=-1)  # (bs, n, n)

    diag_mask = ~torch.eye(
        node_mask.size(1), device=node_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask
    num_edges = edge_mask.sum(dim=(-1, -2)).clamp(min=1)  # (bs,)
    per_graph = (loss * edge_mask).sum(dim=(-1, -2)) / num_edges  # (bs,)
    return per_graph.mean()


def masked_node_mse(
    pred_X: Tensor,
    true_X: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Per-graph masked MSE over a node-shaped continuous field.

    Zeros out contributions at padding positions; remaining positions are
    averaged per graph and then over the batch.
    """
    diff_sq = (pred_X - true_X) ** 2  # (bs, n, d)
    diff_sum = diff_sq.sum(dim=-1)  # (bs, n)
    num_nodes = node_mask.sum(dim=-1).clamp(min=1).to(diff_sum.dtype)  # (bs,)
    per_graph = (diff_sum * node_mask).sum(dim=-1) / num_nodes  # (bs,)
    return per_graph.mean()


def masked_edge_mse(
    pred_E: Tensor,
    true_E: Tensor,
    node_mask: Tensor,
) -> Tensor:
    """Per-graph masked MSE over an edge-shaped continuous field.

    Zeros contributions at padding edges and along the diagonal; remaining
    off-diagonal entries are averaged per graph and then over the batch.
    """
    diff_sq = (pred_E - true_E) ** 2  # (bs, n, n, d)
    diff_sum = diff_sq.sum(dim=-1)  # (bs, n, n)
    diag_mask = ~torch.eye(
        node_mask.size(1), device=node_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    edge_mask = (node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask).to(
        diff_sum.dtype
    )
    num_edges = edge_mask.sum(dim=(-1, -2)).clamp(min=1)  # (bs,)
    per_graph = (diff_sum * edge_mask).sum(dim=(-1, -2)) / num_edges  # (bs,)
    return per_graph.mean()


class TrainLossDiscrete:
    """Masked cross-entropy loss for dual-field discrete diffusion training.

    Thin compatibility wrapper around :func:`masked_node_ce` and
    :func:`masked_edge_ce`. New code should call the helpers directly via the
    :class:`~tmgg.training.lightning_modules.diffusion_module.DiffusionModule`
    per-field loss loop; this class remains so the upstream DiGress parity
    regression tests stay green without changes.

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
        loss_x = masked_node_ce(pred_X, true_X, node_mask)
        loss_e = masked_edge_ce(pred_E, true_E, node_mask)
        return loss_x + self.lambda_E * loss_e
