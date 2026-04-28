"""Stage 2 telemetry: cheap optimizer-health metrics.

The ``DiffusionModule._log_grad_health_by_block`` method bundles four
calculations: per-block grad cosine vs the previous step, per-block
grad SNR ``mean²/var``, global ``effective_lr = lr × ‖∇‖ / ‖θ‖``, and
the trailing update-to-weight ratio. These tests pin the *math* in
isolation; full integration with the Lightning hook ordering is
covered by the existing ``DiffusionModule`` tests when they exercise
``on_before_optimizer_step`` / ``on_train_batch_end`` end-to-end.
"""

from __future__ import annotations

import torch


def _cosine(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return (a * b).sum() / (a.norm() * b.norm()).clamp(min=eps)


def test_grad_cosine_identical_grads_returns_one() -> None:
    """Two identical gradient vectors ⇒ cosine = 1."""
    g = torch.tensor([1.0, -2.0, 3.0])
    assert torch.isclose(_cosine(g, g), torch.tensor(1.0))


def test_grad_cosine_opposite_grads_returns_minus_one() -> None:
    """Antiparallel ⇒ cosine = -1. Catches a sign-handling bug."""
    g = torch.tensor([1.0, -2.0, 3.0])
    assert torch.isclose(_cosine(g, -g), torch.tensor(-1.0))


def test_grad_cosine_orthogonal_grads_returns_zero() -> None:
    """Two orthogonal gradient vectors ⇒ cosine = 0."""
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert torch.isclose(_cosine(a, b), torch.tensor(0.0), atol=1e-7)


def test_grad_cosine_zero_norm_does_not_nan() -> None:
    """Cosine of a zero gradient against any other ⇒ 0, not NaN.

    The clamp on the denominator is what guarantees this: a literal
    ``0/0`` would otherwise propagate NaN through the dashboard.
    """
    z = torch.zeros(3)
    g = torch.tensor([1.0, 2.0, 3.0])
    cos = _cosine(z, g)
    assert torch.isfinite(cos)


def test_grad_snr_constant_grad_is_high() -> None:
    """Constant-valued gradient ⇒ var → 0 ⇒ SNR clamps to a large value.

    With ``var.clamp(min=1e-12)`` the ratio is ``mean² / 1e-12 = 1e12``
    for ``mean = 1.0``. Functions as the "all elements consistent"
    qualitative signal we want.
    """
    g = torch.full((100,), 1.0)
    n = g.numel()
    mean = g.sum() / n
    var = g.pow(2).sum() / n - mean.pow(2)
    snr = mean.pow(2) / var.clamp(min=1e-12)
    assert snr.item() > 1e9


def test_grad_snr_zero_mean_is_zero() -> None:
    """Zero-mean gradient (Σg = 0) gives SNR = 0 regardless of variance.

    Catches "noise dominates" cases — the metric is doing its job
    when it reads ~0 in this regime.
    """
    g = torch.tensor([1.0, -1.0, 1.0, -1.0])
    n = g.numel()
    mean = g.sum() / n
    var = g.pow(2).sum() / n - mean.pow(2)
    snr = mean.pow(2) / var.clamp(min=1e-12)
    assert snr.item() < 1e-6


def test_update_to_weight_zero_when_no_step() -> None:
    """If ``θ_after == θ_before`` the ratio is 0 — sanity check on the
    sign and the use of ``clamp(min=1e-12)`` in the denominator."""
    theta = torch.randn(10)
    delta = (theta - theta).norm()
    weight_norm = theta.norm()
    ratio = delta / weight_norm.clamp(min=1e-12)
    assert ratio.item() == 0.0


def test_effective_lr_formula_matches_documented_definition() -> None:
    """``lr × ‖∇‖ / ‖θ‖`` agrees with hand-computed values."""
    lr = 0.01
    g = torch.tensor([3.0, 4.0])  # ‖∇‖ = 5
    theta = torch.tensor([0.0, 1.0, 0.0, 0.0])  # ‖θ‖ = 1
    eff = lr * g.norm() / theta.norm().clamp(min=1e-12)
    assert torch.isclose(eff, torch.tensor(0.05))
