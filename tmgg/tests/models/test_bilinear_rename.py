"""Regression test: BilinearDenoiser exists and old factory keys still work.

Verifies that the rename from SelfAttentionDenoiser to BilinearDenoiser
is complete and backward-compatible. The class computes QK^T/sqrt(d_k)
without softmax or value projection -- a scaled bilinear form, not
self-attention.
"""

import torch


def test_bilinear_denoiser_importable():
    """The renamed class must be importable."""
    from tmgg.models.spectral_denoisers import BilinearDenoiser

    assert BilinearDenoiser is not None


def test_bilinear_denoiser_with_mlp_importable():
    from tmgg.models.spectral_denoisers import BilinearDenoiserWithMLP

    assert BilinearDenoiserWithMLP is not None


def test_factory_creates_bilinear():
    """New 'bilinear' factory key works."""
    from tmgg.models.factory import create_model

    model = create_model("bilinear", {"k": 4, "d_k": 8})
    assert type(model).__name__ == "BilinearDenoiser"


def test_factory_self_attention_still_works():
    """Existing 'self_attention' key must still work (backward compat).

    After Task 1b this will return the correct SelfAttentionDenoiser,
    but for now it returns BilinearDenoiser.
    """
    from tmgg.models.factory import create_model

    model = create_model("self_attention", {"k": 4, "d_k": 8})
    assert model is not None


def test_bilinear_output_shape():
    """BilinearDenoiser produces (batch, n, n) adjacency logits."""
    from tmgg.models.spectral_denoisers import BilinearDenoiser

    model = BilinearDenoiser(k=4, d_k=8)
    A = torch.randn(2, 10, 10)
    A = (A + A.transpose(-1, -2)) / 2
    out = model(A)
    assert out.shape == (2, 10, 10)


def test_bilinear_has_no_softmax():
    """BilinearDenoiser must NOT apply softmax -- it's a raw bilinear form."""
    from tmgg.models.spectral_denoisers import BilinearDenoiser

    model = BilinearDenoiser(k=4, d_k=8)
    A = torch.randn(2, 10, 10)
    A = (A + A.transpose(-1, -2)) / 2
    out = model(A)
    # Raw logits can be negative and > 1 -- softmax output would be in [0,1]
    # with rows summing to 1. Bilinear should have neither property.
    assert out.min() < 0 or out.max() > 1, "Output looks softmax-bounded"
