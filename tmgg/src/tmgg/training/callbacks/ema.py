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


def _require_backbone_parameters(pl_module: pl.LightningModule) -> Any:
    """Return ``pl_module.model.parameters()``, raising on missing backbone.

    EMACallback only needs a parameter-bearing ``.model`` backbone to
    shadow. Lightning's callback hooks type ``pl_module`` as the loose
    :class:`pytorch_lightning.LightningModule` base, which doesn't
    declare ``.model``; rather than silencing the resulting pyright
    attribute error, do the lookup via ``getattr`` (whose ``Any``
    return type pyright accepts) and raise informatively if the
    attribute is missing. Fail-loud at registration time per CLAUDE.md
    instead of erroring deep inside an EMA shadow update. A
    ``runtime_checkable`` ``Protocol`` was tried first but Python's
    isinstance check on protocols with annotated attributes only sees
    class-level annotations — instance-set ``self.model = ...``
    attributes (the universal pattern in BaseGraphModule and our test
    fixtures) are not detected.
    """
    backbone = getattr(pl_module, "model", None)
    if backbone is None:
        raise TypeError(
            "EMACallback requires a LightningModule with a `.model` backbone "
            f"to shadow; got {type(pl_module).__name__} which does not "
            "expose one."
        )
    return backbone.parameters()


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
        # EMA the wrapped backbone, not the LightningModule itself, so
        # optimizer states and other non-trainable scalars stay
        # untouched.
        self.ema = ExponentialMovingAverage(
            _require_backbone_parameters(pl_module),
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
        self.ema.update(_require_backbone_parameters(pl_module))

    @override
    def on_validation_start(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        if self.ema is None:
            # Lightning runs a sanity-check validation pass before
            # on_fit_start in some configurations; skip swapping when
            # no shadow exists yet.
            return
        params = _require_backbone_parameters(pl_module)
        self.ema.store(params)
        # ``parameters()`` returns a fresh generator on every call; the
        # second iteration above would be empty if we re-used ``params``,
        # so re-fetch for copy_to.
        self.ema.copy_to(_require_backbone_parameters(pl_module))

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
        self.ema.restore(_require_backbone_parameters(pl_module))
