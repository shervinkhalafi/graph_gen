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


def test_optimizer_health_metrics_survive_short_epochs() -> None:
    """End-to-end: short epochs (11 batches) + cadence=5 + log_every_n_steps=2.

    Regression test for the bug where logs written from
    ``on_before_optimizer_step`` got wiped by ``reset_results()`` at
    every epoch boundary because the cadence iter (one past the
    flush boundary) never coincided with a flush before the next reset.

    Fix: ``DiffusionModule`` computes grad-derived stats in
    ``on_before_optimizer_step`` but stashes them in
    ``_opt_health_payload``; ``on_train_batch_end`` (whose cadence iter
    *does* coincide with a flush boundary) drains the payload via
    ``self.log``.

    This test instantiates a minimal ``LightningModule`` that mirrors
    the same stash/drain pattern and verifies that all expected keys
    show up in the logger's records across at least one cadence
    boundary that crosses an epoch end.
    """
    import pytorch_lightning as pl

    class _Model(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.linear = torch.nn.Linear(4, 1)
            self._opt_health_payload: dict[str, torch.Tensor] | None = None
            self._cadence = 5

        def training_step(
            self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
        ) -> torch.Tensor:
            x, y = batch
            loss = ((self.linear(x) - y) ** 2).mean()
            self.log("train/loss", loss, on_step=True)
            return loss

        def on_before_optimizer_step(self, optimizer: torch.optim.Optimizer) -> None:
            # Mirror the production cadence trigger.
            if (self.global_step + 1) % self._cadence == 0:
                self._opt_health_payload = {
                    "train/grad_snr/_total": torch.tensor(2.0 + self.global_step),
                    "train/effective_lr": torch.tensor(0.001 + self.global_step / 1000),
                }

        def on_train_batch_end(
            self,
            outputs: object,
            batch: object,
            batch_idx: int,
        ) -> None:
            if self.global_step % self._cadence == 0 and self._opt_health_payload:
                for k, v in self._opt_health_payload.items():
                    self.log(k, v, on_step=True)
                self._opt_health_payload = None

        def configure_optimizers(self) -> torch.optim.Optimizer:
            return torch.optim.Adam(self.parameters(), lr=1e-3)

    from pytorch_lightning.loggers import Logger as _PLLogger

    class _Capture(_PLLogger):
        def __init__(self) -> None:
            super().__init__()
            self.records: list[tuple[int | None, dict[str, float]]] = []

        @property
        def name(self) -> str:
            return "capture"

        @property
        def version(self) -> str:
            return "1"

        def log_metrics(
            self,
            metrics: dict[str, float] | None = None,
            step: int | None = None,
            **kw: object,
        ) -> None:
            assert metrics is not None
            self.records.append((step, dict(metrics)))

        def log_hyperparams(self, *a: object, **k: object) -> None:
            pass

        def save(self) -> None:
            pass

    torch.manual_seed(0)
    # 33 samples / batch_size 3 = 11 batches per epoch — same as production
    # smoke-run dataset, where the bug originally manifested.
    ds = torch.utils.data.TensorDataset(torch.randn(33, 4), torch.randn(33, 1))
    dl = torch.utils.data.DataLoader(ds, batch_size=3)
    cap = _Capture()
    trainer = pl.Trainer(
        max_steps=33,
        log_every_n_steps=2,
        logger=cap,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
    )
    trainer.fit(_Model(), train_dataloaders=dl)

    # Cadence triggers at global_step ∈ {5, 10, 15, 20, 25, 30}.
    # Each one should appear in at least one flush event.
    flushed_keys = {k for _, m in cap.records for k in m}
    assert "train/grad_snr/_total" in flushed_keys, (
        "regression: grad_snr metric was wiped by reset_results between "
        "the cadence trigger and the next log_every_n_steps flush"
    )
    assert "train/effective_lr" in flushed_keys, (
        "regression: effective_lr metric was wiped by reset_results "
        "between the cadence trigger and the next log_every_n_steps flush"
    )
