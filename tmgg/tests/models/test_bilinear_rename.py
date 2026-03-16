"""Regression test: BilinearDenoiser exists and can be constructed directly.

Verifies that the rename from SelfAttentionDenoiser to BilinearDenoiser
is complete. The class computes QK^T/sqrt(d_k) without softmax or value
projection -- a scaled bilinear form, not self-attention.
"""

import torch

from tmgg.data.datasets.graph_types import GraphData


def test_bilinear_denoiser_importable():
    """The renamed class must be importable."""
    from tmgg.models.spectral_denoisers import BilinearDenoiser

    assert BilinearDenoiser is not None


def test_bilinear_denoiser_with_mlp_importable():
    from tmgg.models.spectral_denoisers import BilinearDenoiserWithMLP

    assert BilinearDenoiserWithMLP is not None


def test_bilinear_denoiser_constructible():
    """BilinearDenoiser can be constructed directly."""
    from tmgg.models.spectral_denoisers import BilinearDenoiser

    model = BilinearDenoiser(k=4, d_k=8)
    assert type(model).__name__ == "BilinearDenoiser"


def test_self_attention_denoiser_constructible():
    """SelfAttentionDenoiser can be constructed directly."""
    from tmgg.models.spectral_denoisers.self_attention import SelfAttentionDenoiser

    model = SelfAttentionDenoiser(k=4, d_k=8)
    assert model is not None


def test_bilinear_output_shape():
    """BilinearDenoiser produces GraphData with correct adjacency shape."""
    from tmgg.models.spectral_denoisers import BilinearDenoiser

    model = BilinearDenoiser(k=4, d_k=8)
    A = torch.randn(2, 10, 10)
    A = (A + A.transpose(-1, -2)) / 2
    result = model(GraphData.from_adjacency(A))
    assert isinstance(result, GraphData)
    assert result.to_adjacency().shape == (2, 10, 10)


def test_bilinear_has_no_softmax():
    """BilinearDenoiser must NOT apply softmax -- it's a raw bilinear form."""
    from tmgg.models.spectral_denoisers import BilinearDenoiser

    model = BilinearDenoiser(k=4, d_k=8)
    A = torch.randn(2, 10, 10)
    A = (A + A.transpose(-1, -2)) / 2
    result = model(GraphData.from_adjacency(A))
    # Extract raw adjacency values from edge features (channel 1 = edge probability)
    raw_adj = result.E[:, :, :, 1]
    # Raw logits can be negative and > 1 -- softmax output would be in [0,1]
    # with rows summing to 1. Bilinear should have neither property.
    assert raw_adj.min() < 0 or raw_adj.max() > 1, "Output looks softmax-bounded"
