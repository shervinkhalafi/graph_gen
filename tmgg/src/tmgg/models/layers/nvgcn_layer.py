import torch
import torch.nn as nn


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
                    h_vals = torch.nn.functional.pad(
                        h_vals, (0, num_nodes - h_vals.shape[0])
                    )
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
                        h_vals = torch.nn.functional.pad(
                            h_vals, (0, num_nodes - h_vals.shape[0])
                        )
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
