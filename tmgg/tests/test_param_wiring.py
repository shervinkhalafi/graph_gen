"""Guard: Lightning module __init__ params must match the base class.

Rationale: Three modules accepted noise_levels but the base class
DenoisingLightningModule has eval_noise_levels. The param was silently
absorbed by **kwargs. Also, gnn_transformer defaulted to loss_type="BCE"
which the base class does not accept (only "MSE" and "BCEWithLogits").
"""

import inspect


def test_no_noise_levels_param_in_denoising_modules():
    """Denoising modules must use eval_noise_levels, not noise_levels."""
    from tmgg.experiments.digress_denoising.lightning_module import (
        DigressDenoisingLightningModule,
    )
    from tmgg.experiments.gnn_denoising.lightning_module import (
        GNNDenoisingLightningModule,
    )
    from tmgg.experiments.gnn_transformer_denoising.lightning_module import (
        HybridDenoisingLightningModule,
    )

    for cls in [
        GNNDenoisingLightningModule,
        HybridDenoisingLightningModule,
        DigressDenoisingLightningModule,
    ]:
        sig = inspect.signature(cls.__init__)
        assert "noise_levels" not in sig.parameters, (
            f"{cls.__name__}.__init__ still accepts 'noise_levels'. "
            "Use 'eval_noise_levels' to match the base class."
        )


def test_gnn_transformer_loss_type_default_is_valid():
    """HybridDenoisingLightningModule default loss_type must be MSE or BCEWithLogits."""
    from tmgg.experiments.gnn_transformer_denoising.lightning_module import (
        HybridDenoisingLightningModule,
    )

    sig = inspect.signature(HybridDenoisingLightningModule.__init__)
    default = sig.parameters["loss_type"].default
    assert default in (
        "MSE",
        "BCEWithLogits",
    ), f"loss_type default is {default!r}, must be 'MSE' or 'BCEWithLogits'"
