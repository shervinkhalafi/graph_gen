"""Model type validation must use the factory registry, not hardcoded sets.

Rationale
---------
Both ``SpectralDenoisingLightningModule`` and ``GenerativeLightningModule``
previously maintained hardcoded sets (``VALID_MODEL_TYPES`` /
``SUPPORTED_ARCHITECTURES``) that duplicated the authoritative source of truth
in ``MODEL_REGISTRY``.  When a new model was registered in ``factory.py`` but
not added to these sets, the lightning module rejected it with a confusing
``ValueError`` even though the factory was perfectly capable of constructing it.

After the fix, validation is delegated to ``create_model`` which checks against
``MODEL_REGISTRY`` directly.  These tests verify that:

1. Invalid model types raise ``ValueError`` mentioning the registry.
2. The hardcoded class attributes no longer exist.
"""

import re

import pytest

from tmgg.models.factory import MODEL_REGISTRY


def test_spectral_no_hardcoded_valid_model_types():
    """VALID_MODEL_TYPES class attribute must not exist after the fix."""
    from tmgg.experiments.spectral_arch_denoising.lightning_module import (
        SpectralDenoisingLightningModule,
    )

    assert not hasattr(SpectralDenoisingLightningModule, "VALID_MODEL_TYPES"), (
        "VALID_MODEL_TYPES should have been removed -- "
        "validation is now delegated to MODEL_REGISTRY via create_model"
    )


def test_generative_no_hardcoded_supported_architectures():
    """SUPPORTED_ARCHITECTURES class attribute must not exist after the fix."""
    from tmgg.experiments.gaussian_diffusion_generative.lightning_module import (
        GenerativeLightningModule,
    )

    assert not hasattr(GenerativeLightningModule, "SUPPORTED_ARCHITECTURES"), (
        "SUPPORTED_ARCHITECTURES should have been removed -- "
        "validation is now delegated to MODEL_REGISTRY via create_model"
    )


def test_spectral_rejects_unknown_model_type():
    """SpectralDenoisingLightningModule must reject unknown model types.

    The error should mention ``Registered types`` to confirm that the factory
    registry is the source of truth.
    """
    from tmgg.experiments.spectral_arch_denoising.lightning_module import (
        SpectralDenoisingLightningModule,
    )

    with pytest.raises(ValueError, match=re.escape("Registered types")):
        SpectralDenoisingLightningModule(model_type="nonexistent_model", k=8)


def test_generative_rejects_unknown_model_type():
    """GenerativeLightningModule must reject unknown model types.

    The error should mention ``Registered types`` to confirm that the factory
    registry is the source of truth.
    """
    from tmgg.experiments.gaussian_diffusion_generative.lightning_module import (
        GenerativeLightningModule,
    )

    with pytest.raises(ValueError, match=re.escape("Registered types")):
        GenerativeLightningModule(model_type="nonexistent_model")


def test_all_registry_keys_are_constructible():
    """Every key in MODEL_REGISTRY must produce a model without error.

    Guards against stale or broken registrations.
    """
    for name, factory_fn in MODEL_REGISTRY.items():
        # Use an empty config -- factories fall back to defaults via .get()
        model = factory_fn({})
        assert model is not None, f"Factory for {name!r} returned None"
