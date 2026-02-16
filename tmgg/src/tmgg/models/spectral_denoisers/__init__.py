"""Spectral denoising models for graph diffusion tasks.

This package provides denoising architectures that operate on the eigenspace
of noisy adjacency matrices. All models follow a common pattern: extract
top-k eigenvectors, apply a learnable transformation, reconstruct adjacency.

All models output raw logits. Use model.predict(logits) to get [0,1] probabilities.

Models
------
LinearPE
    Simple linear transformation: Â = V W V^T + bias
GraphFilterBank
    Spectral polynomial filter: W = Σ Λ^ℓ H^{(ℓ)}
SelfAttentionDenoiser
    Vaswani et al. (2017) self-attention with softmax + value projection,
    followed by bilinear readout: Â = (attn @ Val) readout
BilinearDenoiser
    Scaled bilinear form: Â = Q K^T / √d_k (no softmax, no values)
BilinearDenoiserWithMLP
    Bilinear with MLP post-processing: Â = MLP(Q K^T / √d_k)
MultiLayerBilinearDenoiser
    Stacked transformer blocks on eigenvectors with bilinear readout

Shrinkage Wrappers (Experimental)
---------------------------------
StrictShrinkageWrapper
    SVD-based shrinkage with sigmoid gating: S_mod = sigmoid(α) * S
RelaxedShrinkageWrapper
    FiLM-style modulation: S_mod = scale * S + shift

Note: Shrinkage wrappers are experimental and not yet used in standard
experiment sweeps. They wrap an inner SpectralDenoiser to extract features
for predicting shrinkage coefficients.

Layers
------
TopKEigenLayer
    Extract top-k eigenvectors with sign normalization

Base Classes
------------
SpectralDenoiser
    Abstract base for all spectral denoising models
"""

from tmgg.models.spectral_denoisers.base_spectral import SpectralDenoiser
from tmgg.models.spectral_denoisers.bilinear import (
    BilinearDenoiser,
    BilinearDenoiserWithMLP,
    MultiLayerBilinearDenoiser,
)
from tmgg.models.spectral_denoisers.filter_bank import GraphFilterBank
from tmgg.models.spectral_denoisers.linear_pe import LinearPE
from tmgg.models.spectral_denoisers.self_attention import SelfAttentionDenoiser
from tmgg.models.spectral_denoisers.shrinkage_wrapper import (
    RelaxedShrinkageWrapper,
    ShrinkageSVDLayer,
    ShrinkageWrapper,
    StrictShrinkageWrapper,
)
from tmgg.models.spectral_denoisers.topk_eigen import TopKEigenLayer

# Backward-compat aliases until correct replacements are built
SelfAttentionDenoiserWithMLP = BilinearDenoiserWithMLP
MultiLayerSelfAttentionDenoiser = MultiLayerBilinearDenoiser

__all__ = [
    # Base classes
    "SpectralDenoiser",
    # Layers
    "TopKEigenLayer",
    # Models
    "LinearPE",
    "GraphFilterBank",
    "BilinearDenoiser",
    "BilinearDenoiserWithMLP",
    "MultiLayerBilinearDenoiser",
    "SelfAttentionDenoiser",
    # Shrinkage wrappers (experimental)
    "ShrinkageWrapper",
    "StrictShrinkageWrapper",
    "RelaxedShrinkageWrapper",
    "ShrinkageSVDLayer",
]
