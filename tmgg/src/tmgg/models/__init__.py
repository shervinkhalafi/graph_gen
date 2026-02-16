"""Shared model architectures for graph denoising experiments."""

from .attention import MultiLayerAttention
from .base import DenoisingModel
from .factory import MODEL_REGISTRY, create_model, register_model
from .gnn import (
    GNN,
    GNNSymmetric,
    NodeVarGNN,
)
from .hybrid import SequentialDenoisingModel, create_sequential_model
from .layers import (
    EigenEmbedding,
    GraphConvolutionLayer,
    MultiHeadAttention,
    NodeVarGraphConvolutionLayer,
    masked_softmax,
)

__all__ = [
    "DenoisingModel",
    "MODEL_REGISTRY",
    "create_model",
    "register_model",
    "MultiHeadAttention",
    "MultiLayerAttention",
    "EigenEmbedding",
    "GraphConvolutionLayer",
    "NodeVarGraphConvolutionLayer",
    "GNN",
    "NodeVarGNN",
    "GNNSymmetric",
    "SequentialDenoisingModel",
    "create_sequential_model",
    "masked_softmax",
]
