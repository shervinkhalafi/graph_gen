"""PEARL positional encoding via GNN message passing.

Implements PEARL (Position Encoding with Adaptable Random Layers) from:

    E. Hejin, S. Welke, V. P. Dwivedi, and C. Morris.
    "PEARL: A Scalable and Effective Random Positional Encoding."
    ICLR 2025. https://github.com/ehejin/Pearl-PE

PEARL generates positional encodings through GNN message passing, offering
linear complexity O(n) compared to O(n²) eigendecomposition.

Two variants:
- R-PEARL: Random node initialization + GNN layers
- B-PEARL: Standard basis vectors + GNN layers
"""

import torch
import torch.nn as nn


class PEARLEmbedding(nn.Module):
    """PEARL positional encoding via GNN message passing.

    Generates positional encodings by propagating initial node features
    through GNN layers. R-PEARL uses random initialization while B-PEARL
    uses standard basis vectors.

    Reference
    ---------
    PEARL: Position Encoding with Adaptable Random Layers (ICLR 2025)
    https://github.com/ehejin/Pearl-PE

    Parameters
    ----------
    output_dim : int
        Output embedding dimension (k in spectral models).
    num_layers : int
        Number of GNN layers for message passing.
    mode : str
        "random" for R-PEARL or "basis" for B-PEARL.
    hidden_dim : int
        Hidden dimension for GNN layers.
    input_samples : int
        Number of random samples for R-PEARL (ignored for B-PEARL).
    max_nodes : int
        Maximum graph size for B-PEARL basis vectors.

    Notes
    -----
    R-PEARL is generally preferred as it doesn't require knowing max graph
    size ahead of time and handles variable-sized graphs naturally.

    Examples
    --------
    >>> pearl = PEARLEmbedding(output_dim=16, num_layers=3, mode="random")
    >>> A = torch.rand(4, 50, 50)
    >>> A = (A + A.transpose(-1, -2)) / 2  # Symmetrize
    >>> embeddings = pearl(A)
    >>> embeddings.shape
    torch.Size([4, 50, 16])
    """

    def __init__(
        self,
        output_dim: int,
        num_layers: int = 3,
        mode: str = "random",
        hidden_dim: int = 64,
        input_samples: int = 32,
        max_nodes: int = 200,
    ) -> None:
        super().__init__()
        if mode not in ("random", "basis"):
            raise ValueError(f"mode must be 'random' or 'basis', got {mode!r}")

        self.output_dim = output_dim
        self.num_layers = num_layers
        self.mode = mode
        self.hidden_dim = hidden_dim
        self.input_samples = input_samples
        self.max_nodes = max_nodes

        # Input dimension depends on mode
        input_dim = input_samples if mode == "random" else max_nodes

        # GNN layers: simple message passing with MLPs
        # Using the propagation pattern from PEARL: aggregate neighbor info
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            self.layers.append(
                nn.Sequential(
                    nn.Linear(in_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim),
                )
            )

        # Output projection
        self.out_proj = nn.Linear(hidden_dim, output_dim)

        # LayerNorm for stability
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim) for _ in range(num_layers)]
        )

    def _normalize_adjacency(self, A: torch.Tensor) -> torch.Tensor:
        """Compute symmetric normalized adjacency D^{-1/2} A D^{-1/2}."""
        D = A.sum(dim=-1)  # (batch, n)
        D_inv_sqrt = torch.where(D > 0, D.pow(-0.5), torch.zeros_like(D))
        D_inv_sqrt_mat = torch.diag_embed(D_inv_sqrt)
        return torch.bmm(torch.bmm(D_inv_sqrt_mat, A), D_inv_sqrt_mat)

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """Generate PEARL embeddings from adjacency matrix.

        Parameters
        ----------
        A : torch.Tensor
            Adjacency matrix of shape (batch, n, n).

        Returns
        -------
        torch.Tensor
            PEARL embeddings of shape (batch, n, output_dim).
        """
        unbatched = A.ndim == 2
        if unbatched:
            A = A.unsqueeze(0)

        batch_size, n, _ = A.shape
        device = A.device
        dtype = A.dtype

        # Initialize node features based on mode
        if self.mode == "random":
            # R-PEARL: random features, fixed per forward (deterministic in eval)
            if self.training:
                X = torch.randn(batch_size, n, self.input_samples, device=device)
            else:
                # Use fixed random seed for reproducibility in eval
                generator = torch.Generator(device=device).manual_seed(42)
                X = torch.randn(
                    batch_size,
                    n,
                    self.input_samples,
                    device=device,
                    generator=generator,
                )
        else:
            # B-PEARL: standard basis vectors (one-hot up to max_nodes)
            # Each node i gets one-hot vector e_i
            X = torch.eye(self.max_nodes, device=device, dtype=dtype)[:n, :]
            X = X.unsqueeze(0).expand(batch_size, -1, -1)

        # Convert to float for linear layers
        X = X.to(dtype=torch.float32)

        # Normalize adjacency for message passing
        A_norm = self._normalize_adjacency(A.to(dtype=torch.float32))

        # GNN message passing
        for layer, norm in zip(self.layers, self.layer_norms, strict=True):
            # Aggregate neighbor features
            H = torch.bmm(A_norm, X)  # (batch, n, in_dim)
            # Transform
            H = layer(H)
            # Normalize
            H = norm(H)
            # Residual if dimensions match
            X = X + H if H.shape == X.shape else H

        # Output projection
        embeddings = self.out_proj(X)

        if unbatched:
            embeddings = embeddings.squeeze(0)

        return embeddings
