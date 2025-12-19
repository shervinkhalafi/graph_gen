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
        super().__init__()
        self.num_terms = num_terms
        self.num_channels = num_channels

        self.H = nn.Parameter(torch.randn(num_terms + 1, num_channels, num_channels))
        nn.init.xavier_uniform_(self.H)

        # GELU provides better nonlinearity than Tanh after LayerNorm,
        # since LayerNorm already normalizes to ~N(0,1) where Tanh is nearly linear
        self.activation = nn.GELU()
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

        # Symmetric normalization: D^{-1/2} A D^{-1/2} bounds spectral radius to [-1, 1],
        # preventing overflow in matrix powers for dense graphs
        D = A.sum(dim=-1)  # (batch, n) - degree vector
        D_inv_sqrt = torch.where(D > 0, D.pow(-0.5), torch.zeros_like(D))
        # Create diagonal matrices and apply normalization
        D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt)  # (batch, n, n)
        A_norm = torch.bmm(torch.bmm(D_inv_sqrt_mat, A), D_inv_sqrt_mat)

        Y_hat = X @ self.H[0]
        for i in range(1, self.num_terms + 1):
            A_power_i = torch.matrix_power(A_norm, i)
            Y_hat += torch.bmm(A_power_i, X) @ self.H[i]
        Y_hat = self.layer_norm(Y_hat)
        Y_hat = self.activation(Y_hat)
        return Y_hat
