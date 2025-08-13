import torch
import torch.nn as nn


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
        X = X.to(self.H.dtype)  # pyright: ignore[reportConstantRedefinition]
        A = A.to(self.H.dtype)  # pyright: ignore[reportConstantRedefinition]

        Y_hat = X @ self.H[0]
        for i in range(1, self.num_terms + 1):
            A_power_i = torch.matrix_power(A, i)
            Y_hat += torch.bmm(A_power_i, X) @ self.H[i]
        Y_hat = self.layer_norm(Y_hat)
        Y_hat = self.activation(Y_hat)
        return Y_hat
