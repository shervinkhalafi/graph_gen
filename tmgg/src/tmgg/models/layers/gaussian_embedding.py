import torch
import torch.nn as nn


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
            result = self.h[0, c] * torch.eye(num_nodes, device=A.device).unsqueeze(
                0
            ).expand(batch_size, -1, -1)
            for i in range(1, self.num_terms + 1):
                A_power_i = torch.matrix_power(A, i)
                result += self.h[i, c] * A_power_i
            Y_hat[..., c] = torch.diagonal(result, dim1=-2, dim2=-1)

        return Y_hat
