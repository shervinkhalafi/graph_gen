"""Abstract base classes and structural protocols for graph models."""

from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from tmgg.data.datasets.graph_types import GraphData


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Model that produces node embeddings for hybrid composition.

    Satisfied by any model with an ``embeddings`` method (e.g. ``GNN``)
    and the standard ``get_config`` introspection hook.
    """

    def embeddings(self, data: GraphData) -> tuple[Tensor, Tensor]:
        """Compute (X, Y) node embeddings from graph data."""
        ...

    def get_config(self) -> dict[str, Any]:
        """Return model configuration dictionary."""
        ...


class BaseModel(nn.Module, ABC):
    """Base class for all models with configuration support."""

    @abstractmethod
    def get_config(self) -> dict[str, Any]:
        """Return model hyperparameters for logging and serialization."""
        pass

    def parameter_count(self) -> dict[str, Any]:
        """Count trainable parameters in this module and its children.

        Returns
        -------
        dict[str, Any]
            Hierarchical counts: ``"total"`` for the full module,
            ``"self"`` for parameters not in children, and one entry
            per named child module.
        """
        counts: dict[str, Any] = {"total": 0, "self": 0}

        # Count parameters directly owned by this module (not in children)
        for _, param in self.named_parameters(recurse=False):
            if param.requires_grad:
                counts["self"] += param.numel()

        # Recursively count child modules
        for name, module in self.named_children():
            if hasattr(module, "parameter_count") and callable(module.parameter_count):
                # Child has parameter_count method - use it for recursive counting
                child_counts = module.parameter_count()
                counts[name] = child_counts
                counts["total"] += child_counts["total"]
            else:
                # Standard PyTorch module - count its parameters
                child_total = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                if child_total > 0:
                    counts[name] = {"total": child_total}
                    counts["total"] += child_total

        # Add self parameters to total
        counts["total"] += counts["self"]

        return counts


class GraphModel(BaseModel, ABC):
    """Unified model interface: all models accept and return GraphData.

    This is the standard base class for graph models in the training loop.
    """

    @abstractmethod
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Forward pass operating on typed graph data.

        Parameters
        ----------
        data
            Batched graph features (X, E, y, node_mask).
        t
            Normalised diffusion timestep, or ``None`` for unconditional
            models.

        Returns
        -------
        GraphData
            Predicted graph features.
        """
        ...
