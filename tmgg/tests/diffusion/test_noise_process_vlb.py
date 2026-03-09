"""Tests for VLB methods on CategoricalNoiseProcess.

Test rationale
--------------
``compute_Lt`` and ``reconstruction_logp`` are the two remaining VLB
terms ported from the old ``DiscreteDiffusionLightningModule``. They are
pure noise-process math with no model or Lightning dependency.

Starting state: a uniform ``CategoricalNoiseProcess`` with small
dimensions (dx=2, de=2, T=10, bs=3, n=6).

Invariants tested:
- Output shapes are ``(bs,)`` for both methods.
- Outputs are finite (no NaN/Inf from division or log).
- Perfect predictions yield near-zero loss: KL(p||p) ~ 0 for
  ``compute_Lt``, and ``log(1) = 0`` for ``reconstruction_logp``.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from tmgg.data.datasets.graph_types import GraphData
from tmgg.diffusion.noise_process import CategoricalNoiseProcess
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.diffusion.transitions import DiscreteUniformTransition

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

BS, N, DX, DE, T = 3, 6, 2, 2, 10


@pytest.fixture()
def schedule() -> NoiseSchedule:
    return NoiseSchedule(schedule_type="cosine_iddpm", timesteps=T)


@pytest.fixture()
def proc(schedule: NoiseSchedule) -> CategoricalNoiseProcess:
    return CategoricalNoiseProcess(
        noise_schedule=schedule,
        x_classes=DX,
        e_classes=DE,
        transition_model=DiscreteUniformTransition(DX, DE, 0),
    )


@pytest.fixture()
def clean_data() -> GraphData:
    """One-hot clean graph data with all nodes valid."""
    torch.manual_seed(0)

    x_idx = torch.randint(0, DX, (BS, N))
    X = F.one_hot(x_idx, num_classes=DX).float()  # pyright: ignore[reportAttributeAccessIssue]

    e_idx = torch.randint(0, DE, (BS, N, N))
    e_idx = torch.triu(e_idx, diagonal=1)
    e_idx = e_idx + e_idx.transpose(1, 2)
    E = torch.zeros(BS, N, N, DE)
    E.scatter_(3, e_idx.unsqueeze(-1), 1.0)
    diag = torch.arange(N)
    E[:, diag, diag, :] = 0
    E[:, diag, diag, 0] = 1.0

    y = torch.zeros(BS, 0)
    node_mask = torch.ones(BS, N, dtype=torch.bool)
    return GraphData(X=X, E=E, y=y, node_mask=node_mask)


# ---------------------------------------------------------------------------
# compute_Lt tests
# ---------------------------------------------------------------------------


class TestComputeLt:
    """Tests for the diffusion KL term L_t."""

    def test_output_shape(
        self, proc: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """compute_Lt returns shape (bs,)."""
        t_int = torch.tensor([5, 3, 7])
        noisy = proc.apply(clean_data, t_int)
        # Use uniform predictions as a simple stand-in
        pred = GraphData(
            X=torch.ones_like(clean_data.X) / DX,
            E=torch.ones_like(clean_data.E) / DE,
            y=clean_data.y,
            node_mask=clean_data.node_mask,
        )
        result = proc.compute_Lt(clean_data, pred, noisy, t_int)
        assert result.shape == (BS,)

    def test_output_finite(
        self, proc: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """compute_Lt produces finite values."""
        t_int = torch.tensor([5, 3, 7])
        noisy = proc.apply(clean_data, t_int)
        pred = GraphData(
            X=torch.ones_like(clean_data.X) / DX,
            E=torch.ones_like(clean_data.E) / DE,
            y=clean_data.y,
            node_mask=clean_data.node_mask,
        )
        result = proc.compute_Lt(clean_data, pred, noisy, t_int)
        assert torch.isfinite(result).all()

    def test_perfect_prediction_near_zero(
        self, proc: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """When pred == clean, KL should be near zero."""
        t_int = torch.tensor([5, 3, 7])
        noisy = proc.apply(clean_data, t_int)
        # Perfect prediction: pass the actual clean data as pred
        result = proc.compute_Lt(clean_data, clean_data, noisy, t_int)
        assert result.shape == (BS,)
        # KL(p || p) = 0, but mask_distributions adds eps so allow small tolerance
        assert (result.abs() < 0.5).all(), f"Expected near-zero KL, got {result}"


# ---------------------------------------------------------------------------
# reconstruction_logp tests
# ---------------------------------------------------------------------------


class TestReconstructionLogp:
    """Tests for the reconstruction log-probability term."""

    def test_output_shape(
        self, proc: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """reconstruction_logp returns shape (bs,)."""
        # pred_probs is a probability distribution over classes
        pred_probs = GraphData(
            X=torch.ones_like(clean_data.X) / DX,
            E=torch.ones_like(clean_data.E) / DE,
            y=clean_data.y,
            node_mask=clean_data.node_mask,
        )
        result = proc.reconstruction_logp(clean_data, pred_probs)
        assert result.shape == (BS,)

    def test_output_finite(
        self, proc: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """reconstruction_logp produces finite values."""
        pred_probs = GraphData(
            X=torch.ones_like(clean_data.X) / DX,
            E=torch.ones_like(clean_data.E) / DE,
            y=clean_data.y,
            node_mask=clean_data.node_mask,
        )
        result = proc.reconstruction_logp(clean_data, pred_probs)
        assert torch.isfinite(result).all()

    def test_perfect_prediction_near_zero(
        self, proc: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """When pred_probs match clean with small epsilon floor, loss is ~0.

        In the real pipeline, ``_reconstruction_logp`` ensures masked
        positions have probability 1 and the model never outputs exact 0.
        We simulate that by clamping to a small epsilon.
        """
        # Add tiny epsilon to avoid 0*log(0) = NaN, then renormalise
        eps = 1e-7
        X_safe = clean_data.X + eps
        X_safe = X_safe / X_safe.sum(dim=-1, keepdim=True)
        E_safe = clean_data.E + eps
        E_safe = E_safe / E_safe.sum(dim=-1, keepdim=True)
        pred = GraphData(
            X=X_safe, E=E_safe, y=clean_data.y, node_mask=clean_data.node_mask
        )
        result = proc.reconstruction_logp(clean_data, pred)
        assert result.shape == (BS,)
        # log(~1) is near 0 at one-hot positions, and 0 * log(eps) = ~0
        assert (result.abs() < 0.1).all(), f"Expected near-zero, got {result}"

    def test_negative_for_imperfect_prediction(
        self, proc: CategoricalNoiseProcess, clean_data: GraphData
    ) -> None:
        """Log-prob of a uniform distribution under one-hot targets is negative."""
        pred_probs = GraphData(
            X=torch.ones_like(clean_data.X) / DX,
            E=torch.ones_like(clean_data.E) / DE,
            y=clean_data.y,
            node_mask=clean_data.node_mask,
        )
        result = proc.reconstruction_logp(clean_data, pred_probs)
        # log(1/K) < 0, so sum should be negative
        assert (result < 0).all(), f"Expected negative log-prob, got {result}"
