"""Categorical sampling and masking utilities for graph diffusion."""

# pyright: reportAttributeAccessIssue=false
# F.one_hot exists at runtime; pyright cannot resolve it from the functional stub.

import torch
from torch.nn import functional as F

from tmgg.data.datasets.graph_data_fields import FieldName
from tmgg.data.datasets.graph_types import GraphData


def sample_discrete_features(
    probX: torch.Tensor,
    probE: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    field: FieldName | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample node and edge class indices from per-position categorical PMFs.

    Parameters
    ----------
    probX : torch.Tensor
        Node class PMFs, shape ``(bs, n, dx_out)``.
    probE : torch.Tensor
        Edge class PMFs, shape ``(bs, n, n, de_out)``.
    node_mask : torch.Tensor
        Boolean node-validity mask, shape ``(bs, n)``.
    field : FieldName | None
        Informational only. Documents which
        :class:`tmgg.data.datasets.graph_types.GraphData` field the caller
        intends to populate with the returned draw; the helper is
        field-neutral.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(X_idx, E_idx)`` integer class indices with shapes
        ``(bs, n)`` and ``(bs, n, n)`` respectively. Callers that need
        one-hot tensors wrap the result via
        :func:`torch.nn.functional.one_hot`.
    """
    _ = field  # Informational only.
    # Clone to avoid in-place mutation of gradient-tracked tensors
    probX = probX.clone()
    probE = probE.clone()

    bs, n, _ = probX.shape
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)
    X_t = probX.multinomial(1)  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)
    E_t = probE.multinomial(1).reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    return X_t, E_t


def compute_posterior_distribution_per_x0(
    M_t: torch.Tensor,
    Qt_M: torch.Tensor,
    Qsb_M: torch.Tensor,
    Qtb_M: torch.Tensor,
    *,
    field: FieldName,
) -> torch.Tensor:
    """Per-x0 reverse posterior tensor — upstream-DiGress marginalisation form.

    Returns ``p(z_s = k | z_t, x_0 = c)`` for every position, every
    ``x_0`` class ``c``, and every ``z_s`` class ``k``. The caller
    marginalises over ``x_0`` by contracting along ``c`` against the
    model's predicted ``p(x_0 = c | z_t)``. Mirrors upstream
    ``compute_batched_over0_posterior_distribution`` in
    ``DiGress/src/diffusion/diffusion_utils.py``.

    Parameters
    ----------
    M_t
        Noisy one-hot at timestep ``t``. Shape ``(bs, n, d)`` for nodes
        or ``(bs, n, n, d)`` for edges.
    Qt_M, Qsb_M, Qtb_M
        Single-step, signal-retention, and cumulative transition
        kernels. All ``(bs, d, d)``.
    field
        :data:`FieldName` naming which GraphData field the caller is
        sampling for (``"X_class"`` for nodes, ``"E_class"`` for
        edges). The math is field-neutral — it operates on ``M_t``
        regardless of semantics — so the parameter is informational
        today. It becomes load-bearing in Wave 9 when the legacy
        ``X`` / ``E`` aliases are removed and callers must explicitly
        name every field they sample.

    Returns
    -------
    torch.Tensor
        Shape ``(bs, N, d, d)`` where ``N = n`` (nodes) or ``n*n``
        (edges), the third axis indexes ``x_0`` classes ``c``, and the
        fourth indexes ``z_s`` classes ``k``.
    """
    _ = field  # Informational only during the transition.
    M_t_flat = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # (bs, N, d)
    Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)
    left_term = M_t_flat @ Qt_M_T  # (bs, N, d) -- common across x0 classes
    left_term = left_term.unsqueeze(2)  # (bs, N, 1, d)

    right_term = Qsb_M.unsqueeze(1)  # (bs, 1, d, d) -- per-x0 signal kernel
    numerator = left_term * right_term  # (bs, N, d, d)

    # Per-x0 normaliser z_t @ Qtb_M[c, :].T summed over the z_t support.
    # Equivalent to (Qtb_M @ M_t.T).T evaluated per-x0 row.
    M_t_transposed = M_t_flat.transpose(-1, -2)  # (bs, d, N)
    prod = (Qtb_M @ M_t_transposed).transpose(-1, -2)  # (bs, N, d)
    denominator = prod.unsqueeze(-1)  # (bs, N, d, 1)
    # Numerical floor for masked / impossible classes; matches upstream.
    denominator = denominator.clone()
    denominator[denominator == 0] = 1e-6

    return numerator / denominator


def compute_posterior_distribution(
    M: torch.Tensor,
    M_t: torch.Tensor,
    Qt_M: torch.Tensor,
    Qsb_M: torch.Tensor,
    Qtb_M: torch.Tensor,
    *,
    field: FieldName,
) -> torch.Tensor:
    """Direct reverse-posterior tensor used when ``x_0`` is already one-hot.

    Computes ``xt @ Qt.T * x_0 @ Qsb / (x_0 @ Qtb @ xt.T)`` — the
    DiGress Bayes-rule form that treats ``M`` (``x_0``) as a
    distribution over classes (commonly the one-hot clean target, or
    the softmax of a model prediction).

    Parameters
    ----------
    M, M_t
        Clean and noisy categorical tensors. ``M`` is either a one-hot
        target or a soft prediction; ``M_t`` is the noisy state at
        timestep ``t``. Shapes ``(bs, n, d)`` for node fields,
        ``(bs, n, n, d)`` for edge fields.
    Qt_M, Qsb_M, Qtb_M
        Single-step, signal-retention, and cumulative transition
        kernels. All ``(bs, d, d)``.
    field
        :data:`FieldName` naming which GraphData field the caller is
        sampling for (``"X_class"`` for nodes, ``"E_class"`` for
        edges). Informational today; see
        :func:`compute_posterior_distribution_per_x0` for the rationale.
    """
    _ = field  # Informational only during the transition.
    # Flatten feature tensors
    M = M.flatten(start_dim=1, end_dim=-2).to(
        torch.float32
    )  # (bs, N, d) with N = n or n * n
    M_t = M_t.flatten(start_dim=1, end_dim=-2).to(torch.float32)  # same

    Qt_M_T = torch.transpose(Qt_M, -2, -1)  # (bs, d, d)

    left_term = M_t @ Qt_M_T  # (bs, N, d)
    right_term = M @ Qsb_M  # (bs, N, d)
    product = left_term * right_term  # (bs, N, d)

    denom = M @ Qtb_M  # (bs, N, d) @ (bs, d, d) = (bs, N, d)
    denom = (denom * M_t).sum(dim=-1)  # (bs, N, d) * (bs, N, d) + sum = (bs, N)
    denom = denom.clamp(min=1e-6)  # Prevent division by near-zero

    prob = product / denom.unsqueeze(-1)  # (bs, N, d)

    return prob


def mask_distributions(
    true_X: torch.Tensor,
    true_E: torch.Tensor,
    pred_X: torch.Tensor,
    pred_E: torch.Tensor,
    node_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Set masked rows to arbitrary distributions, so it doesn't contribute to loss
    :param true_X: bs, n, dx_out
    :param true_E: bs, n, n, de_out
    :param pred_X: bs, n, dx_out
    :param pred_E: bs, n, n, de_out
    :param node_mask: bs, n
    :return: same sizes as input
    """

    row_X = torch.zeros(true_X.size(-1), dtype=torch.float, device=true_X.device)
    row_X[0] = 1.0
    row_E = torch.zeros(true_E.size(-1), dtype=torch.float, device=true_E.device)
    row_E[0] = 1.0

    diag_mask = ~torch.eye(
        node_mask.size(1), device=node_mask.device, dtype=torch.bool
    ).unsqueeze(0)
    true_X[~node_mask] = row_X
    pred_X[~node_mask] = row_X
    true_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E
    pred_E[~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2) * diag_mask), :] = row_E

    # Epsilon prevents log(0) = -inf downstream (reconstruction_logp, KL).
    # Masked positions are already set to one-hot; epsilon only affects
    # non-masked entries and is removed by renormalization.
    true_X = true_X + 1e-7
    pred_X = pred_X + 1e-7
    true_E = true_E + 1e-7
    pred_E = pred_E + 1e-7

    true_X = true_X / torch.sum(true_X, dim=-1, keepdim=True)
    pred_X = pred_X / torch.sum(pred_X, dim=-1, keepdim=True)
    true_E = true_E / torch.sum(true_E, dim=-1, keepdim=True)
    pred_E = pred_E / torch.sum(pred_E, dim=-1, keepdim=True)

    return true_X, true_E, pred_X, pred_E


def sample_discrete_feature_noise(
    x_limit: torch.Tensor,
    e_limit: torch.Tensor,
    node_mask: torch.Tensor,
    y_limit: torch.Tensor | None = None,
) -> GraphData:
    """Sample a one-hot categorical prior ``GraphData`` from stationary PMFs.

    Returns
    -------
    GraphData
        Instance with ``X_class`` and ``E_class`` populated as one-hot
        samples drawn from the supplied stationary PMFs and masked
        against ``node_mask``.
    """
    bs, n_max = node_mask.shape
    x_limit = x_limit.to(node_mask.device)[None, None, :].expand(bs, n_max, -1)
    e_limit = e_limit.to(node_mask.device)[None, None, None, :].expand(
        bs, n_max, n_max, -1
    )
    ux_idx = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    ue_idx = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)

    if y_limit is None or y_limit.numel() == 0:
        uy = torch.empty((bs, 0), device=node_mask.device)
    else:
        y_limit = y_limit.to(node_mask.device)[None, :].expand(bs, -1)
        uy_idx = y_limit.multinomial(1).squeeze(-1)
        uy = F.one_hot(uy_idx, num_classes=y_limit.shape[-1]).float()

    long_mask = node_mask.long()
    ux_idx = ux_idx.type_as(long_mask)
    ue_idx = ue_idx.type_as(long_mask)
    uy = uy.type_as(long_mask)

    ux_one_hot = F.one_hot(ux_idx, num_classes=x_limit.shape[-1]).float()
    ue_one_hot = F.one_hot(ue_idx, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(ue_one_hot)
    indices = torch.triu_indices(
        row=ue_one_hot.size(1), col=ue_one_hot.size(2), offset=1
    )
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    ue_one_hot = ue_one_hot * upper_triangular_mask
    ue_one_hot = ue_one_hot + torch.transpose(ue_one_hot, 1, 2)

    if not (torch.transpose(ue_one_hot, 1, 2) == ue_one_hot).all():
        raise AssertionError("Edge noise is not symmetric")

    return GraphData(
        y=uy,
        node_mask=node_mask,
        X_class=ux_one_hot,
        E_class=ue_one_hot,
    ).mask()
