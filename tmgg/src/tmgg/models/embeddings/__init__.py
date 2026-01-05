"""Graph embedding models for dimension analysis.

This module provides various graph embedding methods for finding minimal
dimension representations that allow near-exact graph reconstruction.

Embedding types:
- LPCA: Logistic PCA, A ≈ σ(X·Xᵀ)
- DotProduct: Low-rank factorization, A ≈ X·Xᵀ
- DotThreshold: Thresholded inner product, A_ij = 1 iff xᵢ·xⱼ > τ
- DistanceThreshold: Distance-based, A_ij = 1 iff ‖xᵢ-xⱼ‖ < τ
- Orthogonal: Orthogonal representation, ū⊥v̄ ⟺ (u,v)∈E
"""

from tmgg.models.embeddings.base import (
    AsymmetricEmbedding,
    EmbeddingResult,
    GraphEmbedding,
    SymmetricEmbedding,
)
from tmgg.models.embeddings.dimension_search import (
    DimensionResult,
    DimensionSearcher,
    EmbeddingType,
)
from tmgg.models.embeddings.distance_threshold import DistanceThresholdSymmetric
from tmgg.models.embeddings.dot_product import (
    DotProductAsymmetric,
    DotProductSymmetric,
)
from tmgg.models.embeddings.dot_threshold import (
    DotThresholdAsymmetric,
    DotThresholdSymmetric,
)
from tmgg.models.embeddings.lpca import LPCAAsymmetric, LPCASymmetric
from tmgg.models.embeddings.orthogonal import OrthogonalRepSymmetric

__all__ = [
    # Base classes
    "GraphEmbedding",
    "SymmetricEmbedding",
    "AsymmetricEmbedding",
    "EmbeddingResult",
    # LPCA
    "LPCASymmetric",
    "LPCAAsymmetric",
    # Dot product
    "DotProductSymmetric",
    "DotProductAsymmetric",
    # Dot threshold
    "DotThresholdSymmetric",
    "DotThresholdAsymmetric",
    # Distance threshold
    "DistanceThresholdSymmetric",
    # Orthogonal representation
    "OrthogonalRepSymmetric",
    # Dimension search
    "DimensionSearcher",
    "DimensionResult",
    "EmbeddingType",
]
