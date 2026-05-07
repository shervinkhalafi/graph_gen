"""Abstract base classes and structural protocols for graph models."""

from abc import ABC, abstractmethod
from typing import Any, Literal, Protocol, runtime_checkable

import torch
import torch.nn as nn
from torch import Tensor

from tmgg.data.datasets.graph_types import GraphData

EdgeSource = Literal["class", "feat"]
"""Discriminator selecting which split edge field a denoising-style
architecture reads from. Defined here so every architecture family can
import a single canonical literal type."""


def read_edge_scalar(data: GraphData, source: EdgeSource) -> Tensor:
    """Return a dense scalar adjacency from the requested split edge field.

    Thin wrapper over :meth:`GraphData.to_edge_scalar` that translates the
    architecture-family ``EdgeSource`` literal into the split-field name.

    Parameters
    ----------
    data
        Input graph data.
    source
        Which split edge field to prefer (``"class"`` reads ``E_class``;
        ``"feat"`` reads ``E_feat``).

    Returns
    -------
    Tensor
        Dense scalar adjacency, ``(bs, n, n)`` or ``(n, n)``.
    """
    return data.to_edge_scalar(source=source)


def write_edge_scalar(
    data: GraphData,
    *,
    edge_scalar: Tensor,
    target: EdgeSource,
) -> GraphData:
    """Return a copy of ``data`` with a scalar prediction written to ``target``.

    Wave 7 output-side helper. The single scalar adjacency is packed into
    the requested split edge field of a fresh :class:`GraphData` that
    shares ``data.node_mask`` and carries ``data.y`` through unchanged.
    Unrelated split fields are left ``None``; downstream code reads
    ``edge_source``-selected tensors via :func:`read_edge_scalar`.

    Parameters
    ----------
    data
        Input graph; ``node_mask`` and ``y`` propagate unchanged.
    edge_scalar
        Dense scalar adjacency, ``(bs, n, n)`` or ``(n, n)``.
    target
        Which split edge field to populate (``"feat"`` writes ``E_feat``
        with shape ``(..., 1)``; ``"class"`` writes a two-channel
        ``[1 - adj, adj]`` ``E_class``).

    Returns
    -------
    GraphData
        New instance carrying the prediction in the selected split-edge
        field.
    """
    if target == "feat":
        decoded = GraphData.from_structure_only(data.node_mask, edge_scalar)
    else:  # target == "class"
        decoded = GraphData.from_edge_scalar(
            edge_scalar, node_mask=data.node_mask, target="E_class"
        )
    return decoded.replace(y=data.y)


def append_timestep_to_y(y: Tensor, t: Tensor | None) -> Tensor:
    """Append a normalised timestep scalar to a global feature tensor.

    Mirrors the two-line pattern used by :class:`GraphTransformer`'s
    ``use_timestep`` path: when ``t`` is provided, concatenate it as a
    final ``y`` channel; otherwise return ``y`` unchanged.
    """
    if t is None:
        return y
    return torch.cat([y, t.unsqueeze(-1)], dim=-1)


type ParameterCountTree = dict[str, int | ParameterCountTree]
"""Recursive parameter-count structure returned by ``BaseModel.parameter_count``."""


def get_parameter_count_int(counts: ParameterCountTree, key: str) -> int:
    """Return a required integer leaf from a parameter-count tree.

    Parameters
    ----------
    counts
        Hierarchical parameter-count mapping.
    key
        Name of the leaf entry to extract.

    Returns
    -------
    int
        Integer parameter count stored at ``counts[key]``.

    Raises
    ------
    TypeError
        If ``counts[key]`` is itself a nested subtree instead of an integer
        leaf. Internal callers use this to fail fast when the expected count
        shape drifts.
    """
    value = counts[key]
    if not isinstance(value, int):
        raise TypeError(f"Expected integer parameter count at key '{key}'")
    return value


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

    def parameter_count(self) -> ParameterCountTree:
        """Count trainable parameters in this module and its children.

        Returns
        -------
        ParameterCountTree
            Hierarchical counts: ``"total"`` for the full module,
            ``"self"`` for parameters not in children, and one entry
            per named child module.
        """
        self_total = 0
        total = 0
        child_breakdown: ParameterCountTree = {}

        # Count parameters directly owned by this module (not in children)
        for _, param in self.named_parameters(recurse=False):
            if param.requires_grad:
                self_total += param.numel()

        # Recursively count child modules
        for name, module in self.named_children():
            if isinstance(module, BaseModel):
                # Child has parameter_count method - use it for recursive counting
                counts = module.parameter_count()
                child_breakdown[name] = counts
                total += get_parameter_count_int(counts, "total")
            else:
                # Standard PyTorch module - count its parameters
                child_total = sum(
                    p.numel() for p in module.parameters() if p.requires_grad
                )
                if child_total > 0:
                    child_breakdown[name] = {"total": child_total}
                    total += child_total

        return {
            "total": total + self_total,
            "self": self_total,
            **child_breakdown,
        }


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
