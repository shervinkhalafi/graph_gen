"""Graph Neural Network models for graph denoising."""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import numpy as np

from .base import BaseModel, DenoisingModel


class EigenDecompositionError(Exception):
    """Custom exception for eigendecomposition failures with debugging context."""
    
    def __init__(self, matrix_idx: int, matrix: torch.Tensor, original_error: Exception):
        self.matrix_idx = matrix_idx
        self.matrix = matrix
        self.original_error = original_error
        
        # Compute debugging metrics
        self.debugging_context = self._compute_debugging_metrics(matrix)
        
        super().__init__(self._format_message())
    
    def _compute_debugging_metrics(self, A: torch.Tensor) -> dict:
        """Compute key metrics for debugging ill-conditioned matrices."""
        with torch.no_grad():
            # Convert to numpy for condition number calculation
            A_np = A.cpu().numpy()
            
            # 1. Condition number (ratio of largest to smallest singular value)
            try:
                singular_values = np.linalg.svd(A_np, compute_uv=False)
                condition_number = singular_values[0] / (singular_values[-1] + 1e-10)
            except:
                condition_number = float('inf')
            
            # 2. Frobenius norm (matrix magnitude)
            frobenius_norm = torch.norm(A, p='fro').item()
            
            # 3. Trace (sum of diagonal elements)
            trace = torch.trace(A).item()
            
            # 4. Check for NaN/Inf values
            has_nan = torch.isnan(A).any().item()
            has_inf = torch.isinf(A).any().item()
            
            # 5. Diagonal dominance metric (how dominant the diagonal is)
            diagonal = torch.diag(A)
            off_diagonal_sum = torch.sum(torch.abs(A), dim=1) - torch.abs(diagonal)
            min_diagonal_dominance = torch.min(torch.abs(diagonal) - off_diagonal_sum).item()
            
            return {
                'condition_number': condition_number,
                'frobenius_norm': frobenius_norm,
                'trace': trace,
                'has_nan': has_nan,
                'has_inf': has_inf,
                'min_diagonal_dominance': min_diagonal_dominance,
                'matrix_shape': list(A.shape),
            }
    
    def _format_message(self) -> str:
        """Format error message with debugging context."""
        ctx = self.debugging_context
        return (
            f"Eigendecomposition failed for matrix {self.matrix_idx}. "
            f"Debugging context: "
            f"condition_number={ctx['condition_number']:.2e}, "
            f"frobenius_norm={ctx['frobenius_norm']:.2e}, "
            f"trace={ctx['trace']:.2e}, "
            f"has_nan={ctx['has_nan']}, "
            f"has_inf={ctx['has_inf']}, "
            f"min_diagonal_dominance={ctx['min_diagonal_dominance']:.2e}, "
            f"shape={ctx['matrix_shape']}. "
            f"Original error: {str(self.original_error)}"
        )


class GaussianEmbedding(nn.Module):
    """Gaussian embedding layer using powers of adjacency matrix."""
    
    def __init__(self, num_terms: int, num_channels: int):
        """
        Initialize Gaussian embedding.
        
        Args:
            num_terms: Number of terms in the polynomial expansion
            num_channels: Number of output channels
        """
        super(GaussianEmbedding, self).__init__()
        self.num_terms = num_terms
        self.num_channels = num_channels

        self.h = nn.Parameter(torch.randn(num_terms + 1, num_channels))
        nn.init.xavier_uniform_(self.h)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of Gaussian embedding.
        
        Args:
            A: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
            
        Returns:
            Node embeddings of shape (batch_size, num_nodes, num_channels)
        """
        batch_size, num_nodes, _ = A.shape
        Y_hat = torch.zeros(batch_size, num_nodes, self.num_channels, device=A.device)
        
        for c in range(self.num_channels):
            result = self.h[0, c] * torch.eye(num_nodes, device=A.device).unsqueeze(0).expand(batch_size, -1, -1)
            for i in range(1, self.num_terms + 1):
                A_power_i = torch.matrix_power(A, i)
                result += self.h[i, c] * A_power_i
            Y_hat[..., c] = torch.diagonal(result, dim1=-2, dim2=-1)
            
        return Y_hat


class EigenEmbedding(nn.Module):
    """Embedding layer using eigenvectors of adjacency matrix."""
    
    def __init__(self):
        super(EigenEmbedding, self).__init__()

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of eigen embedding.
        
        Args:
            A: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)
            
        Returns:
            Eigenvector embeddings of shape (batch_size, num_nodes, num_nodes)
            
        Raises:
            EigenDecompositionError: If eigendecomposition fails with debugging context
        """
        eigenvectors = []
        for i in range(A.shape[0]):
            try:
                _, V = torch.linalg.eigh(A[i])
                eigenvectors.append(V)
            except torch._C._LinAlgError as e:
                # Propagate with debugging context
                raise EigenDecompositionError(i, A[i], e)
        return torch.stack(eigenvectors, dim=0)


class GraphConvolutionLayer(nn.Module):
    """Graph convolution layer using polynomial filters."""
    
    def __init__(self, num_terms: int, num_channels: int):
        """
        Initialize graph convolution layer.
        
        Args:
            num_terms: Number of terms in polynomial filter
            num_channels: Number of input/output channels
        """
        super(GraphConvolutionLayer, self).__init__()
        self.num_terms = num_terms
        self.num_channels = num_channels

        self.H = nn.Parameter(torch.randn(num_terms + 1, num_channels, num_channels))
        nn.init.xavier_uniform_(self.H)

        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of graph convolution.
        
        Args:
            A: Adjacency matrix
            X: Input features
            
        Returns:
            Convolved features
        """
        # Ensure X has the same dtype as model parameters
        X = X.to(self.H.dtype)
        A = A.to(self.H.dtype)
        
        Y_hat = X @ self.H[0] 
        for i in range(1, self.num_terms + 1):
            A_power_i = torch.matrix_power(A, i)
            Y_hat += torch.bmm(A_power_i, X) @ self.H[i]
        Y_hat = self.layer_norm(Y_hat)
        Y_hat = self.activation(Y_hat)
        return Y_hat


class NodeVarGraphConvolutionLayer(nn.Module):
    """Node-variant graph convolution layer."""
    
    def __init__(self, num_terms: int, num_channels: int, num_nodes: int):
        """
        Initialize node-variant graph convolution layer.
        
        Args:
            num_terms: Number of terms in polynomial filter
            num_channels: Number of output channels
            num_nodes: Number of nodes in the graph
        """
        super(NodeVarGraphConvolutionLayer, self).__init__()
        self.num_terms = num_terms
        self.num_channels = num_channels
        self.num_nodes = num_nodes

        self.h = nn.Parameter(torch.randn(num_terms + 1, num_channels, num_nodes))
        nn.init.xavier_uniform_(self.h)

        self.activation = nn.Tanh()
        self.layer_norm = nn.LayerNorm(num_channels)

    def forward(self, A: torch.Tensor, X: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of node-variant graph convolution.
        
        Args:
            A: Adjacency matrix
            X: Input features
            
        Returns:
            Convolved features
        """
        batch_size, num_nodes, num_channels_in = X.shape
        Y_hat = torch.zeros(batch_size, num_nodes, self.num_channels, device=A.device)
        
        for c in range(self.num_channels):
            result = torch.zeros(batch_size, num_nodes, device=A.device)
            # Ensure self.h[0, c] has correct size for current num_nodes
            if self.h.shape[2] != num_nodes:
                # Resize h to match current num_nodes by truncating or padding
                h_vals = self.h[0, c]
                if h_vals.shape[0] > num_nodes:
                    h_vals = h_vals[:num_nodes]
                else:
                    h_vals = torch.nn.functional.pad(h_vals, (0, num_nodes - h_vals.shape[0]))
                h_diag = torch.diag_embed(h_vals)
            else:
                h_diag = torch.diag_embed(self.h[0, c])
            
            # Expand h_diag to batch size
            h_diag = h_diag.unsqueeze(0).expand(batch_size, -1, -1)
            A_w = h_diag
            
            for ch in range(num_channels_in):
                result += torch.bmm(A_w, X[..., ch].unsqueeze(-1)).squeeze(-1)

            for i in range(1, self.num_terms + 1):
                A_power_i = torch.matrix_power(A, i)
                
                # Handle dimension mismatch for h
                if self.h.shape[2] != num_nodes:
                    h_vals = self.h[i, c]
                    if h_vals.shape[0] > num_nodes:
                        h_vals = h_vals[:num_nodes]
                    else:
                        h_vals = torch.nn.functional.pad(h_vals, (0, num_nodes - h_vals.shape[0]))
                    h_diag = torch.diag_embed(h_vals)
                else:
                    h_diag = torch.diag_embed(self.h[i, c])
                    
                # Expand h_diag to batch size
                h_diag = h_diag.unsqueeze(0).expand(batch_size, -1, -1)
                A_w = torch.bmm(h_diag, A_power_i)
                
                for ch in range(num_channels_in):
                    result += torch.bmm(A_w, X[..., ch].unsqueeze(-1)).squeeze(-1)
            Y_hat[..., c] = result
            
        Y_hat = self.layer_norm(Y_hat)
        Y_hat = self.activation(Y_hat)
        return Y_hat


class GNN(DenoisingModel):
    """Standard Graph Neural Network for adjacency matrix reconstruction."""
    
    def __init__(self, num_layers: int, num_terms: int = 3, 
                 feature_dim_in: int = 10, feature_dim_out: int = 10):
        """
        Initialize GNN.
        
        Args:
            num_layers: Number of graph convolution layers
            num_terms: Number of terms in polynomial filters
            feature_dim_in: Input feature dimension
            feature_dim_out: Output feature dimension
        """
        super(GNN, self).__init__()
        
        self.num_layers = num_layers
        self.num_terms = num_terms
        self.feature_dim_in = feature_dim_in
        self.feature_dim_out = feature_dim_out

        self.embedding_layer = EigenEmbedding()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphConvolutionLayer(num_terms, feature_dim_in))

        self.out_x = nn.Linear(feature_dim_in, feature_dim_out)
        self.out_y = nn.Linear(feature_dim_in, feature_dim_out)

    def forward(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass returning node embeddings.
        
        Args:
            A: Adjacency matrix
            
        Returns:
            Tuple of (X_embeddings, Y_embeddings)
        """
        Z = self.embedding_layer(A)
        # Take only the first feature_dim_in columns from eigenvectors
        # But ensure we don't exceed the available columns
        actual_feature_dim = min(Z.shape[2], self.feature_dim_in)
        Z = Z[:, :, :actual_feature_dim]
        
        # If we have fewer features than expected, pad with zeros
        if actual_feature_dim < self.feature_dim_in:
            padding = torch.zeros(Z.shape[0], Z.shape[1], self.feature_dim_in - actual_feature_dim, 
                                device=Z.device, dtype=Z.dtype)
            Z = torch.cat([Z, padding], dim=2)
        for layer in self.layers:
            Z = layer(A, Z)
        X = self.out_x(Z)
        Y = self.out_y(Z)
        return X, Y
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim_in": self.feature_dim_in,
            "feature_dim_out": self.feature_dim_out,
        }


class GNNSymmetric(DenoisingModel):
    """Symmetric GNN using same embedding for both X and Y."""
    
    def __init__(self, num_layers: int, num_terms: int = 3, 
                 feature_dim_in: int = 10, feature_dim_out: int = 10):
        super(GNNSymmetric, self).__init__()
        
        self.num_layers = num_layers
        self.num_terms = num_terms
        self.feature_dim_in = feature_dim_in
        self.feature_dim_out = feature_dim_out

        self.embedding_layer = EigenEmbedding()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(GraphConvolutionLayer(num_terms, feature_dim_in))

        self.out_x = nn.Linear(feature_dim_in, feature_dim_out)

    def forward(self, A: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with symmetric embeddings.
        
        Args:
            A: Adjacency matrix
            
        Returns:
            Tuple of (reconstructed_adjacency, X_embeddings)
        """
        Z = self.embedding_layer(A)
        # Take only the first feature_dim_in columns from eigenvectors
        # But ensure we don't exceed the available columns
        actual_feature_dim = min(Z.shape[2], self.feature_dim_in)
        Z = Z[:, :, :actual_feature_dim]
        
        # If we have fewer features than expected, pad with zeros
        if actual_feature_dim < self.feature_dim_in:
            padding = torch.zeros(Z.shape[0], Z.shape[1], self.feature_dim_in - actual_feature_dim, 
                                device=Z.device, dtype=Z.dtype)
            Z = torch.cat([Z, padding], dim=2)
        for layer in self.layers:
            Z = layer(A, Z)
        X = self.out_x(Z)

        outer = torch.bmm(X, X.transpose(1, 2))
        return torch.sigmoid(outer), X
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim_in": self.feature_dim_in,
            "feature_dim_out": self.feature_dim_out,
        }


class NodeVarGNN(DenoisingModel):
    """Node-variant Graph Neural Network."""
    
    def __init__(self, num_layers: int, num_terms: int = 3, feature_dim: int = 10):
        """
        Initialize Node-variant GNN.
        
        Args:
            num_layers: Number of layers
            num_terms: Number of polynomial terms
            feature_dim: Feature dimension
        """
        super(NodeVarGNN, self).__init__()
        
        self.num_layers = num_layers
        self.num_terms = num_terms
        self.feature_dim = feature_dim

        self.embedding_layer = EigenEmbedding()

        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(NodeVarGraphConvolutionLayer(num_terms, feature_dim, feature_dim))

        self.out_x = nn.Linear(feature_dim, feature_dim)
        self.out_y = nn.Linear(feature_dim, feature_dim)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning reconstructed adjacency matrix.
        
        Args:
            A: Input adjacency matrix
            
        Returns:
            Reconstructed adjacency matrix
        """
        Z = self.embedding_layer(A)
        # Take only the first feature_dim columns from eigenvectors
        # But ensure we don't exceed the available columns
        actual_feature_dim = min(Z.shape[2], self.feature_dim)
        Z = Z[:, :, :actual_feature_dim]
        
        # If we have fewer features than expected, pad with zeros
        if actual_feature_dim < self.feature_dim:
            padding = torch.zeros(Z.shape[0], Z.shape[1], self.feature_dim - actual_feature_dim, 
                                device=Z.device, dtype=Z.dtype)
            Z = torch.cat([Z, padding], dim=2)
        
        # Dynamically create layers if needed with correct num_nodes
        if len(self.layers) == 0 or self.layers[0].num_nodes != Z.shape[1]:
            self.layers = nn.ModuleList()
            for _ in range(self.num_layers):
                self.layers.append(NodeVarGraphConvolutionLayer(
                    self.num_terms, self.feature_dim, Z.shape[1]
                ))
                
        for layer in self.layers:
            Z = layer(A, Z)
        X = self.out_x(Z)
        Y = self.out_y(Z)
        outer = torch.bmm(X, Y.transpose(1, 2))
        return torch.sigmoid(outer)
    
    def get_config(self) -> Dict[str, Any]:
        """Get model configuration."""
        return {
            "num_layers": self.num_layers,
            "num_terms": self.num_terms,
            "feature_dim": self.feature_dim,
        }