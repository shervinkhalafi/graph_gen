"""Categorical sampling and masking utilities for graph diffusion."""

# pyright: reportAttributeAccessIssue=false
# F.one_hot exists at runtime; pyright cannot resolve it from the functional stub.

import torch
from torch.nn import functional as F

from tmgg.data.datasets.graph_types import GraphData


def sample_discrete_features(
    probX: torch.Tensor, probE: torch.Tensor, node_mask: torch.Tensor
) -> GraphData:
    """Sample features from multinomial distribution with given probabilities (probX, probE, proby)
    :param probX: bs, n, dx_out        node features
    :param probE: bs, n, n, de_out     edge features
    :param proby: bs, dy_out           global features.
    """
    # Clone to avoid in-place mutation of gradient-tracked tensors
    probX = probX.clone()
    probE = probE.clone()

    bs, n, _ = probX.shape
    # Noise X
    # The masked rows should define probability distributions as well
    probX[~node_mask] = 1 / probX.shape[-1]

    # Flatten the probability tensor to sample with multinomial
    probX = probX.reshape(bs * n, -1)  # (bs * n, dx_out)

    # Sample X
    X_t = probX.multinomial(1)  # (bs * n, 1)
    X_t = X_t.reshape(bs, n)  # (bs, n)

    # Noise E
    # The masked rows should define probability distributions as well
    inverse_edge_mask = ~(node_mask.unsqueeze(1) * node_mask.unsqueeze(2))
    diag_mask = torch.eye(n).unsqueeze(0).expand(bs, -1, -1)

    probE[inverse_edge_mask] = 1 / probE.shape[-1]
    probE[diag_mask.bool()] = 1 / probE.shape[-1]

    probE = probE.reshape(bs * n * n, -1)  # (bs * n * n, de_out)

    # Sample E
    E_t = probE.multinomial(1).reshape(bs, n, n)  # (bs, n, n)
    E_t = torch.triu(E_t, diagonal=1)
    E_t = E_t + torch.transpose(E_t, 1, 2)

    return GraphData(
        X=X_t, E=E_t, y=torch.zeros(bs, 0).type_as(X_t), node_mask=node_mask
    )


def compute_posterior_distribution(
    M: torch.Tensor,
    M_t: torch.Tensor,
    Qt_M: torch.Tensor,
    Qsb_M: torch.Tensor,
    Qtb_M: torch.Tensor,
) -> torch.Tensor:
    """M: X or E
    Compute xt @ Qt.T * x0 @ Qsb / x0 @ Qtb @ xt.T
    """
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
    """Sample one-hot categorical noise from stationary class PMFs."""
    bs, n_max = node_mask.shape
    x_limit = x_limit.to(node_mask.device)[None, None, :].expand(bs, n_max, -1)
    e_limit = e_limit.to(node_mask.device)[None, None, None, :].expand(
        bs, n_max, n_max, -1
    )
    U_X = x_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max)
    U_E = e_limit.flatten(end_dim=-2).multinomial(1).reshape(bs, n_max, n_max)

    if y_limit is None or y_limit.numel() == 0:
        U_y = torch.empty((bs, 0), device=node_mask.device)
    else:
        y_limit = y_limit.to(node_mask.device)[None, :].expand(bs, -1)
        U_y = y_limit.multinomial(1).squeeze(-1)
        U_y = F.one_hot(U_y, num_classes=y_limit.shape[-1]).float()

    long_mask = node_mask.long()
    U_X = U_X.type_as(long_mask)
    U_E = U_E.type_as(long_mask)
    U_y = U_y.type_as(long_mask)

    U_X = F.one_hot(U_X, num_classes=x_limit.shape[-1]).float()
    U_E = F.one_hot(U_E, num_classes=e_limit.shape[-1]).float()

    # Get upper triangular part of edge noise, without main diagonal
    upper_triangular_mask = torch.zeros_like(U_E)
    indices = torch.triu_indices(row=U_E.size(1), col=U_E.size(2), offset=1)
    upper_triangular_mask[:, indices[0], indices[1], :] = 1

    U_E = U_E * upper_triangular_mask
    U_E = U_E + torch.transpose(U_E, 1, 2)

    if not (torch.transpose(U_E, 1, 2) == U_E).all():
        raise AssertionError("Edge noise is not symmetric")

    return GraphData(X=U_X, E=U_E, y=U_y, node_mask=node_mask).mask()
