"""Lightning callback that maintains and swaps EMA model weights.

At fit start, snapshot the live model's parameters into an
:class:`~tmgg.training.ema.ExponentialMovingAverage`. After every
training batch, update the shadow. Around validation, swap EMA weights
into the live model so validation and sampling see smoothed weights,
then restore the live weights for the next training step.

Mirrors upstream DiGress's gating pattern (``main.py:181-183``:
``cfg.train.ema_decay > 0``). Configure via ``ema_decay > 0`` in the
trainer-side config; ``run_experiment.create_callbacks`` registers the
callback only when the gate fires.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, override

from pytorch_lightning.callbacks import Callback

from tmgg.training.ema import ExponentialMovingAverage

if TYPE_CHECKING:
    import pytorch_lightning as pl


class EMACallback(Callback):
    """Track EMA weights and swap them in around validation.

    Parameters
    ----------
    decay
        Smoothing constant in ``(0, 1]``. Pass ``decay`` close to 1
        (e.g. ``0.999``) for a slowly-changing shadow. ``decay = 0``
        is rejected — disable EMA by not registering the callback at
        all rather than passing a zero gate here.
    """

    def __init__(self, decay: float) -> None:
        super().__init__()
        if not 0.0 < decay <= 1.0:
            raise ValueError(
                f"EMACallback decay must be in (0, 1]; got {decay}. "
                "Disable EMA by not registering the callback rather "
                "than passing decay=0."
            )
        self.decay = decay
        self.ema: ExponentialMovingAverage | None = None

    @override
    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        # ``pl_module.model`` is the convention used by
        # BaseGraphModule subclasses; we EMA the wrapped backbone, not
        # the LightningModule itself, so optimizer states and other
        # non-trainable scalars stay untouched.
        self.ema = ExponentialMovingAverage(
            pl_module.model.parameters(),  # pyright: ignore[reportAttributeAccessIssue]
            decay=self.decay,
        )

    @override
    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        assert self.ema is not None, "on_train_batch_end fired before on_fit_start"
        self.ema.update(pl_module.model.parameters())  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.ema is None:
            # Lightning runs a sanity-check validation pass before
            # on_fit_start in some configurations; skip swapping when
            # no shadow exists yet.
            return
        self.ema.store(pl_module.model.parameters())  # pyright: ignore[reportAttributeAccessIssue]
        self.ema.copy_to(pl_module.model.parameters())  # pyright: ignore[reportAttributeAccessIssue]

    @override
    def on_validation_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.ema is None:
            return
        if self.ema._stored_params is None:
            # store() was skipped (sanity check before fit) — nothing
            # to restore.
            return
        self.ema.restore(pl_module.model.parameters())  # pyright: ignore[reportAttributeAccessIssue]
