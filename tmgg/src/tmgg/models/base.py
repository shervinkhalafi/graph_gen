"""Abstract base classes for denoising models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import torch
import torch.nn as nn


class BaseModel(nn.Module, ABC):
    """Base class for all models with configuration support."""

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:
        """
        Get model configuration for logging/saving.

        Returns:
            Dictionary containing model hyperparameters
        """
        pass

    def parameter_count(self) -> Dict[str, Any]:
        """
        Count trainable parameters in this module and its children.
        
        Returns a hierarchical dictionary with:
        - "total": Total trainable parameters in this module and all children
        - "self": Parameters directly owned by this module (not in children)
        - Child module counts with their names as keys
        
        Returns:
            Dictionary with parameter counts
        """
        counts: Dict[str, Any] = {"total": 0, "self": 0}
        
        # Count parameters directly owned by this module (not in children)
        for name, param in self.named_parameters(recurse=False):
            if param.requires_grad:
                counts["self"] += param.numel()
        
        # Recursively count child modules
        for name, module in self.named_children():
            if hasattr(module, 'parameter_count') and callable(getattr(module, 'parameter_count')):
                # Child has parameter_count method - use it for recursive counting
                child_counts = module.parameter_count()
                counts[name] = child_counts
                counts["total"] += child_counts["total"]
            else:
                # Standard PyTorch module - count its parameters
                child_total = sum(p.numel() for p in module.parameters() if p.requires_grad)
                if child_total > 0:
                    counts[name] = {"total": child_total}
                    counts["total"] += child_total
        
        # Add self parameters to total
        counts["total"] += counts["self"]
        
        return counts


class DenoisingModel(BaseModel):  # pyright: ignore[reportImplicitAbstractClass]
    """Abstract base class for graph denoising models."""

    def __init__(self):
        """Initialize denoising model."""
        super().__init__()

    def transform_for_loss(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return output and target directly for loss computation.

        Parameters
        ----------
        output
            Model output tensor (denoised adjacency).
        target
            Target tensor (clean adjacency matrix).

        Returns
        -------
        tuple
            (output, target) unchanged.
        """
        return output, target

    def logits_to_graph(self, logits: torch.Tensor) -> torch.Tensor:
        """Convert logits to binary graph predictions.

        For BCE loss, sigmoid(x) > 0.5 is equivalent to x > 0, so we threshold
        logits directly. Subclasses may override for different thresholding.

        Parameters
        ----------
        logits
            Raw model output from forward().

        Returns
        -------
        torch.Tensor
            Binary predictions (0 or 1).
        """
        return (logits > 0).float()

    def predict(self, logits: torch.Tensor, zero_diag: bool = True) -> torch.Tensor:
        """Convert model output (logits) to binary graph predictions.

        Uses logits_to_graph() for thresholding, then optionally zeros diagonal
        (adjacency matrices have no self-loops).

        Parameters
        ----------
        logits
            Raw model output from forward().
        zero_diag
            If True, zero out diagonal entries. Default True since adjacency
            matrices typically have no self-loops.

        Returns
        -------
        torch.Tensor
            Binary predictions with zero diagonal (if zero_diag=True).
        """
        graph = self.logits_to_graph(logits)
        if zero_diag:
            graph = self._zero_diagonal(graph)
        return graph

    def _zero_diagonal(self, A: torch.Tensor) -> torch.Tensor:
        """Zero out diagonal entries of adjacency matrix.

        Parameters
        ----------
        A
            Adjacency matrix of shape (n, n) or (batch, n, n).

        Returns
        -------
        torch.Tensor
            Matrix with zeros on diagonal.
        """
        if A.ndim == 2:
            return A.fill_diagonal_(0)
        elif A.ndim == 3:
            # Batch case: zero diagonal for each matrix
            mask = torch.eye(A.shape[-1], device=A.device, dtype=torch.bool)
            A = A.masked_fill(mask.unsqueeze(0), 0)
            return A
        else:
            return A

    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, tuple]:
        """Forward pass for denoising.

        Parameters
        ----------
        x
            Input tensor (typically noisy adjacency matrix).

        Returns
        -------
        torch.Tensor
            Raw logits (unbounded). Use predict() for [0, 1] probabilities.
        """
        pass


class EmbeddingModel(BaseModel):  # pyright: ignore[reportImplicitAbstractClass]
    """Abstract base class for graph embedding models."""

    @abstractmethod
    def forward(self, A: torch.Tensor) -> Union[torch.Tensor, tuple]:
        """
        Forward pass for embedding generation.

        Args:
            A: Adjacency matrix tensor

        Returns:
            Node embeddings tensor or tuple of embedding tensors
        """
        pass
