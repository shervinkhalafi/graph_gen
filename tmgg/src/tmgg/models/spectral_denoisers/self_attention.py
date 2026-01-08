"""Self-Attention denoiser using query/key projections on eigenvectors.

Implements a scaled dot-product attention mechanism operating on the
eigenspace of the noisy adjacency matrix.
"""

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from tmgg.models.spectral_denoisers.base_spectral import SpectralDenoiser


class SelfAttentionDenoiser(SpectralDenoiser):
    """Self-Attention denoiser with query/key projections.

    Reconstructs the adjacency matrix using scaled dot-product attention on
    the eigenvector embeddings:
        Q = V W_Q
        K = V W_K
        Â = Q K^T / √d_k

    where V ∈ R^{n×k} are the top-k eigenvectors, W_Q, W_K ∈ R^{k×d_k} are
    learnable projection matrices, and d_k is the key dimension.

    The 1/√d_k scaling stabilizes gradients following standard practice in
    transformer architectures.

    Outputs raw logits. Use model.predict(logits) for [0,1] probabilities.

    Parameters
    ----------
    k : int
        Number of eigenvectors to use as input dimension.
    d_k : int, optional
        Key/query dimension for attention. Default is 64.

    Notes
    -----
    Unlike transformer attention which uses softmax over the sequence
    dimension, this produces a symmetric attention matrix representing edge
    logits. The model.predict() method applies sigmoid to get probabilities.

    Examples
    --------
    >>> model = SelfAttentionDenoiser(k=8, d_k=32)
    >>> A_noisy = torch.randn(4, 50, 50)
    >>> A_noisy = (A_noisy + A_noisy.transpose(-1, -2)) / 2
    >>> logits = model(A_noisy)
    >>> predictions = model.predict(logits)  # apply sigmoid
    >>> predictions.shape
    torch.Size([4, 50, 50])
    """

    def __init__(
        self,
        k: int,
        d_k: int = 64,
    ):
        super().__init__(k=k)
        self.d_k = d_k
        self.scale = d_k**-0.5  # 1/√d_k

        # Query and Key projection matrices
        self.W_Q = nn.Parameter(torch.empty(k, d_k))
        self.W_K = nn.Parameter(torch.empty(k, d_k))

        # Initialize with Xavier for better gradient flow
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)

    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Compute scaled dot-product attention.

        Parameters
        ----------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k).
        Lambda : torch.Tensor
            Eigenvalues (unused in this model).
        A : torch.Tensor
            Input adjacency (unused).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency logits.
        """
        unbatched = V.ndim == 2
        if unbatched:
            V = V.unsqueeze(0)

        # Compute Q = V W_Q and K = V W_K
        # V: (batch, n, k), W_Q: (k, d_k) -> Q: (batch, n, d_k)
        Q = torch.matmul(V, self.W_Q)
        K = torch.matmul(V, self.W_K)

        # Scaled dot-product: Q K^T / √d_k
        # (batch, n, d_k) @ (batch, d_k, n) -> (batch, n, n)
        A_reconstructed = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["d_k"] = self.d_k
        return config

    def get_features(self, A: torch.Tensor) -> torch.Tensor:
        """Extract key projections as node features.

        Returns the K = V @ W_K projections, which capture learned
        representations of node similarity for adjacency reconstruction.

        Parameters
        ----------
        A : torch.Tensor
            Adjacency matrix of shape (batch, n, n) or (n, n).

        Returns
        -------
        torch.Tensor
            Key features of shape (batch, n, d_k) or (n, d_k).
        """
        # Get embeddings (eigenvectors or PEARL) via base class
        V = super().get_features(A)

        # Return key projections
        K = torch.matmul(V, self.W_K)
        return K


class SelfAttentionDenoiserWithMLP(SpectralDenoiser):
    """Self-Attention denoiser with MLP post-processing.

    Extends the attention mechanism by adding an MLP that transforms the
    attention output before producing adjacency logits:
        Q = V W_Q
        K = V W_K
        H = Q K^T / √d_k
        Â = MLP(H)

    The MLP operates element-wise on the n×n attention output, enabling
    non-linear transformations of edge logits.

    Parameters
    ----------
    k : int
        Number of eigenvectors to use as input dimension.
    d_k : int, optional
        Key/query dimension for attention. Default is 64.
    mlp_hidden_dim : int, optional
        Hidden dimension of the MLP layers. Default is 128.
    mlp_num_layers : int, optional
        Number of MLP layers (including output). Default is 2.
        With mlp_num_layers=2: Linear→ReLU→Linear

    Notes
    -----
    The MLP treats each edge independently (no cross-edge information flow
    within the MLP). This preserves the permutation equivariance of the model.
    """

    def __init__(
        self,
        k: int,
        d_k: int = 64,
        mlp_hidden_dim: int = 128,
        mlp_num_layers: int = 2,
    ):
        super().__init__(k=k)
        self.d_k = d_k
        self.mlp_hidden_dim = mlp_hidden_dim
        self.mlp_num_layers = mlp_num_layers
        self.scale = d_k**-0.5  # 1/√d_k

        # Query and Key projection matrices (same as SelfAttentionDenoiser)
        self.W_Q = nn.Parameter(torch.empty(k, d_k))
        self.W_K = nn.Parameter(torch.empty(k, d_k))
        nn.init.xavier_uniform_(self.W_Q)
        nn.init.xavier_uniform_(self.W_K)

        # Build MLP: input is scalar (each edge), output is scalar (edge logit)
        # Uses expand/contract pattern for element-wise transformation
        layers: list[nn.Module] = []
        in_dim = 1
        for _ in range(mlp_num_layers - 1):
            layers.append(nn.Linear(in_dim, mlp_hidden_dim))
            layers.append(nn.ReLU())
            in_dim = mlp_hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention followed by MLP transformation.

        Parameters
        ----------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k).
        Lambda : torch.Tensor
            Eigenvalues (unused in this model).
        A : torch.Tensor
            Input adjacency (unused).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency logits.
        """
        unbatched = V.ndim == 2
        if unbatched:
            V = V.unsqueeze(0)

        # Attention: Q K^T / √d_k
        Q = torch.matmul(V, self.W_Q)
        K = torch.matmul(V, self.W_K)
        H = torch.matmul(Q, K.transpose(-1, -2)) * self.scale  # (batch, n, n)

        # MLP: reshape to (batch*n*n, 1), apply MLP, reshape back
        batch, n, _ = H.shape
        H_flat = H.reshape(-1, 1)  # (batch*n*n, 1)
        A_flat = self.mlp(H_flat)  # (batch*n*n, 1)
        A_reconstructed = A_flat.reshape(batch, n, n)

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["d_k"] = self.d_k
        config["mlp_hidden_dim"] = self.mlp_hidden_dim
        config["mlp_num_layers"] = self.mlp_num_layers
        return config

    def get_features(self, A: torch.Tensor) -> torch.Tensor:
        """Extract key projections as node features.

        Parameters
        ----------
        A : torch.Tensor
            Adjacency matrix of shape (batch, n, n) or (n, n).

        Returns
        -------
        torch.Tensor
            Key features of shape (batch, n, d_k) or (n, d_k).
        """
        # Get embeddings (eigenvectors or PEARL) via base class
        V = super().get_features(A)

        K = torch.matmul(V, self.W_K)
        return K


class _TransformerBlock(nn.Module):
    """Single transformer block with attention, residual, and optional MLP.

    Uses post-norm style: LN(x + sublayer(x)).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        use_mlp: bool = True,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.use_mlp = use_mlp
        self.dropout_p = dropout

        # Multi-head self-attention
        self.d_k = d_model // num_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Optional MLP
        if use_mlp:
            mlp_dim = mlp_hidden_dim if mlp_hidden_dim is not None else 4 * d_model
            self.mlp = nn.Sequential(
                nn.Linear(d_model, mlp_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp_dim, d_model),
            )
            self.norm2 = nn.LayerNorm(d_model)
            self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer block.

        Parameters
        ----------
        x : torch.Tensor
            Input of shape (batch, n, d_model).

        Returns
        -------
        torch.Tensor
            Output of shape (batch, n, d_model).
        """
        batch, n, _ = x.shape

        # Multi-head self-attention
        Q = self.W_q(x).view(batch, n, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch, n, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch, n, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scale = 1.0 / math.sqrt(self.d_k)
        attn_weights = torch.matmul(Q, K.transpose(-1, -2)) * scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)

        attn_out = torch.matmul(attn_weights, V)
        attn_out = attn_out.transpose(1, 2).reshape(batch, n, self.d_model)
        attn_out = self.W_o(attn_out)

        # Residual + LayerNorm (post-norm)
        x = self.norm1(x + self.dropout1(attn_out))

        # Optional MLP with residual
        if self.use_mlp:
            mlp_out = self.mlp(x)
            x = self.norm2(x + self.dropout2(mlp_out))

        return x


class MultiLayerSelfAttentionDenoiser(SpectralDenoiser):
    """Multilayer self-attention denoiser operating on eigenvectors.

    Stacks transformer-style blocks on eigenvector embeddings, maintaining
    size-agnosticism through the spectral decomposition:

        1. Project V (n×k) → H (n×d_model)
        2. Apply L transformer blocks (attention + residual + optional MLP)
        3. Final Q/K projections → adjacency logits Â (n×n)

    This is the multilayer extension of SelfAttentionDenoiser, adding depth
    through stacked attention blocks while preserving variable graph size
    support.

    Parameters
    ----------
    k : int
        Number of eigenvectors to use as input dimension.
    d_model : int
        Hidden dimension for transformer blocks. Default is 64.
    num_heads : int
        Number of attention heads. Must divide d_model evenly. Default is 4.
    num_layers : int
        Number of stacked transformer blocks. Default is 2.
    use_mlp : bool
        Whether to include MLP sublayer in each block. Default is True.
    mlp_hidden_dim : int | None
        MLP hidden dimension. Defaults to 4 * d_model if None.
    dropout : float
        Dropout probability. Default is 0.0.

    Notes
    -----
    Unlike the simple SelfAttentionDenoiser which directly computes QK^T from
    eigenvectors, this model first transforms the eigenvector representation
    through multiple attention layers, allowing the network to learn more
    complex spectral-to-adjacency mappings.

    Examples
    --------
    >>> model = MultiLayerSelfAttentionDenoiser(k=8, d_model=64, num_layers=3)
    >>> A_noisy = torch.randn(4, 50, 50)
    >>> A_noisy = (A_noisy + A_noisy.transpose(-1, -2)) / 2
    >>> logits = model(A_noisy)
    >>> logits.shape
    torch.Size([4, 50, 50])
    """

    def __init__(
        self,
        k: int,
        d_model: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        use_mlp: bool = True,
        mlp_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
            )

        super().__init__(k=k)
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.use_mlp = use_mlp
        self.mlp_hidden_dim = (
            mlp_hidden_dim if mlp_hidden_dim is not None else 4 * d_model
        )
        self.dropout = dropout

        # Input projection: eigenvectors (k) → hidden (d_model)
        self.input_proj = nn.Linear(k, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [
                _TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    use_mlp=use_mlp,
                    mlp_hidden_dim=self.mlp_hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

        # Final Q/K projections for adjacency reconstruction
        self.W_Q = nn.Linear(d_model, d_model, bias=False)
        self.W_K = nn.Linear(d_model, d_model, bias=False)
        self.scale = d_model**-0.5

    def _spectral_forward(
        self,
        V: torch.Tensor,
        Lambda: torch.Tensor,
        A: torch.Tensor,
    ) -> torch.Tensor:
        """Transform eigenvectors through attention layers to adjacency.

        Parameters
        ----------
        V : torch.Tensor
            Top-k eigenvectors of shape (batch, n, k) or (n, k).
        Lambda : torch.Tensor
            Eigenvalues (unused in this model).
        A : torch.Tensor
            Input adjacency (unused).

        Returns
        -------
        torch.Tensor
            Reconstructed adjacency logits.
        """
        unbatched = V.ndim == 2
        v = V.unsqueeze(0) if unbatched else V

        # Project to hidden dimension
        h = self.input_proj(v)  # (batch, n, d_model)

        # Pass through transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Final Q/K attention for adjacency reconstruction
        Q = self.W_Q(h)  # (batch, n, d_model)
        K = self.W_K(h)  # (batch, n, d_model)
        A_reconstructed = torch.matmul(Q, K.transpose(-1, -2)) * self.scale

        if unbatched:
            A_reconstructed = A_reconstructed.squeeze(0)

        return A_reconstructed

    def get_config(self) -> dict[str, Any]:
        config = super().get_config()
        config["d_model"] = self.d_model
        config["num_heads"] = self.num_heads
        config["num_layers"] = self.num_layers
        config["use_mlp"] = self.use_mlp
        config["mlp_hidden_dim"] = self.mlp_hidden_dim
        config["dropout"] = self.dropout
        return config

    def get_features(self, A: torch.Tensor) -> torch.Tensor:
        """Extract key projections after transformer layers.

        Returns K = W_K(h) where h is the output of the transformer stack,
        providing the most refined learned representation for each node.

        Parameters
        ----------
        A : torch.Tensor
            Adjacency matrix of shape (batch, n, n) or (n, n).

        Returns
        -------
        torch.Tensor
            Key features of shape (batch, n, d_model) or (n, d_model).
        """
        # Get embeddings (eigenvectors or PEARL) via base class
        V = SpectralDenoiser.get_features(self, A)

        unbatched = V.ndim == 2
        v = V.unsqueeze(0) if unbatched else V

        # Project to hidden dimension
        h = self.input_proj(v)  # (batch, n, d_model)

        # Pass through transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Return key projections
        K = self.W_K(h)

        if unbatched:
            K = K.squeeze(0)

        return K
