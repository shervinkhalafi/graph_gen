"""Abstract base classes for denoising models."""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Union


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


class DenoisingModel(BaseModel):
    """Abstract base class for graph denoising models."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, tuple]:
        """
        Forward pass for denoising.
        
        Args:
            x: Input tensor (typically noisy adjacency matrix or embeddings)
            
        Returns:
            Denoised output tensor or tuple of tensors
        """
        pass


class EmbeddingModel(BaseModel):
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