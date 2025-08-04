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
    
    def __init__(self, domain: str = "standard"):
        """
        Initialize denoising model.
        
        Args:
            domain: Domain for adjacency matrix processing
                   - "standard": Direct adjacency matrix processing
                   - "inv-sigmoid": Apply logistic transformation before processing
        """
        super().__init__()
        if domain not in ["standard", "inv-sigmoid"]:
            raise ValueError(f"Invalid domain '{domain}'. Must be 'standard' or 'inv-sigmoid'")
        self.domain = domain
    
    def _apply_domain_transform(self, A: torch.Tensor) -> torch.Tensor:
        """
        Apply domain transformation to adjacency matrix.
        
        Args:
            A: Input adjacency matrix
            
        Returns:
            Transformed adjacency matrix
        """
        if self.domain == "inv-sigmoid":
            # Apply logistic transformation: logit(A) = log(A / (1 - A))
            # Clamp values to avoid numerical issues
            A_clamped = torch.clamp(A, 1e-7, 1-1e-7)
            return torch.logit(A_clamped)
        else:
            return A
    
    def _apply_output_transform(self, output: torch.Tensor) -> torch.Tensor:
        """
        Apply output transformation based on domain and training mode.
        
        Args:
            output: Raw model output (logits)
            
        Returns:
            Transformed output
        """
        if self.domain == "inv-sigmoid" and not self.training:
            # Convert logits back to probabilities during evaluation
            return torch.sigmoid(output)
        else:
            # Return raw logits for training or standard domain
            return output
    
    def _apply_target_transform(self, target: torch.Tensor) -> torch.Tensor:
        """
        Apply target transformation to match output domain during training.
        
        Args:
            target: Target tensor (adjacency matrix in probability space)
            
        Returns:
            Transformed target tensor
        """
        if self.domain == "inv-sigmoid" and self.training:
            # Convert target to logit space for training with inv-sigmoid domain
            target_clamped = torch.clamp(target, 1e-7, 1-1e-7)
            return torch.logit(target_clamped)
        else:
            # Return original target for standard domain or evaluation
            return target
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, tuple]:
        """
        Forward pass for denoising.
        
        Args:
            x: Input tensor (typically noisy adjacency matrix or embeddings)
            
        Returns:
            Denoised output tensor or tuple of tensors (raw logits)
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