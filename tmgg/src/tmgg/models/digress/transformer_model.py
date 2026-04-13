# pyright: reportAttributeAccessIssue=false
# torch.nn.functional.pad is not fully typed in the PyTorch stubs; the pad
# function exists at runtime but pyright cannot resolve it via the stub.

import math
from typing import Any, override

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.linear import Linear
from torch.nn.modules.normalization import LayerNorm

from tmgg.data.datasets.graph_types import GraphData, GraphStructure

from ..base import GraphModel
from ..layers import BareGraphConvolutionLayer, SpectralProjectionLayer
from ..layers.masked_softmax import masked_softmax
from .extra_features import ExtraFeaturesProvider
from .layers import Etoy, Xtoy


def _assert_correctly_masked(variable: torch.Tensor, node_mask: torch.Tensor) -> None:
    """Verify that masked positions are near-zero."""
    max_val = (variable * (1 - node_mask.long())).abs().max().item()
    if not max_val < 1e-4:
        raise AssertionError("Variables not masked properly.")


class XEyTransformerLayer(nn.Module):
    """Transformer that updates node, edge and global features
    d_x: node features
    d_e: edge features
    dz : global features
    n_head: the number of heads in the multi_head_attention
    dim_feedforward: the dimension of the feedforward network model after self-attention
    dropout: dropout probablility. 0 to disable
    layer_norm_eps: eps value in layer normalizations.
    """

    self_attn: nn.Module
    linX1: Linear
    linX2: Linear
    normX1: LayerNorm
    normX2: LayerNorm
    dropoutX1: Dropout
    dropoutX2: Dropout
    dropoutX3: Dropout
    linE1: Linear
    linE2: Linear
    normE1: LayerNorm
    normE2: LayerNorm
    dropoutE1: Dropout
    dropoutE2: Dropout
    dropoutE3: Dropout
    lin_y1: Linear
    lin_y2: Linear
    norm_y1: LayerNorm
    norm_y2: LayerNorm
    dropout_y1: Dropout
    dropout_y2: Dropout
    dropout_y3: Dropout

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        dim_ffX: int = 2048,
        dim_ffE: int = 128,
        dim_ffy: int = 2048,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        use_gnn_q: bool = False,
        use_gnn_k: bool = False,
        use_gnn_v: bool = False,
        gnn_num_terms: int = 2,
        use_spectral_q: bool = False,
        use_spectral_k: bool = False,
        use_spectral_v: bool = False,
        spectral_k: int = 16,
        spectral_num_terms: int = 3,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.self_attn = NodeEdgeBlock(
            dx,
            de,
            dy,
            n_head,
            use_gnn_q=use_gnn_q,
            use_gnn_k=use_gnn_k,
            use_gnn_v=use_gnn_v,
            gnn_num_terms=gnn_num_terms,
            use_spectral_q=use_spectral_q,
            use_spectral_k=use_spectral_k,
            use_spectral_v=use_spectral_v,
            spectral_k=spectral_k,
            spectral_num_terms=spectral_num_terms,
            device=device,
            dtype=dtype,
        )

        self.linX1 = Linear(dx, dim_ffX, device=device, dtype=dtype)
        self.linX2 = Linear(dim_ffX, dx, device=device, dtype=dtype)
        self.normX1 = LayerNorm(dx, eps=layer_norm_eps, device=device, dtype=dtype)
        self.normX2 = LayerNorm(dx, eps=layer_norm_eps, device=device, dtype=dtype)
        self.dropoutX1 = Dropout(dropout)
        self.dropoutX2 = Dropout(dropout)
        self.dropoutX3 = Dropout(dropout)

        self.linE1 = Linear(de, dim_ffE, device=device, dtype=dtype)
        self.linE2 = Linear(dim_ffE, de, device=device, dtype=dtype)
        self.normE1 = LayerNorm(de, eps=layer_norm_eps, device=device, dtype=dtype)
        self.normE2 = LayerNorm(de, eps=layer_norm_eps, device=device, dtype=dtype)
        self.dropoutE1 = Dropout(dropout)
        self.dropoutE2 = Dropout(dropout)
        self.dropoutE3 = Dropout(dropout)

        self.lin_y1 = Linear(dy, dim_ffy, device=device, dtype=dtype)
        self.lin_y2 = Linear(dim_ffy, dy, device=device, dtype=dtype)
        self.norm_y1 = LayerNorm(dy, eps=layer_norm_eps, device=device, dtype=dtype)
        self.norm_y2 = LayerNorm(dy, eps=layer_norm_eps, device=device, dtype=dtype)
        self.dropout_y1 = Dropout(dropout)
        self.dropout_y2 = Dropout(dropout)
        self.dropout_y3 = Dropout(dropout)

    def forward(
        self,
        X: Tensor,
        E: Tensor,
        y: Tensor,
        node_mask: Tensor,
        structure: GraphStructure,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Pass the input through the encoder layer.

        Parameters
        ----------
        X
            Node features (bs, n, d).
        E
            Edge features (bs, n, n, d).
        y
            Global features (bs, dy).
        node_mask
            Mask for the src keys per batch (bs, n).
        structure
            Pre-computed structural context (adjacency, eigenvectors,
            eigenvalues) from the input graph.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Updated (X, E, y) with the same shapes.
        """
        newX, newE, new_y = self.self_attn(
            X, E, y, node_mask=node_mask, structure=structure
        )

        newX_d = self.dropoutX1(newX)
        X = self.normX1(X + newX_d)

        newE_d = self.dropoutE1(newE)
        E = self.normE1(E + newE_d)

        new_y_d = self.dropout_y1(new_y)
        y = self.norm_y1(y + new_y_d)

        ff_outputX = self.linX2(self.dropoutX2(F.relu(self.linX1(X))))
        ff_outputX = self.dropoutX3(ff_outputX)
        X = self.normX2(X + ff_outputX)

        ff_outputE = self.linE2(self.dropoutE2(F.relu(self.linE1(E))))
        ff_outputE = self.dropoutE3(ff_outputE)
        E = self.normE2(E + ff_outputE)

        ff_output_y = self.lin_y2(self.dropout_y2(F.relu(self.lin_y1(y))))
        ff_output_y = self.dropout_y3(ff_output_y)
        y = self.norm_y2(y + ff_output_y)

        return X, E, y


class NodeEdgeBlock(nn.Module):
    """Self attention layer that also updates the representations on the edges.

    Supports three projection modes for Q/K/V:
    - Linear: Standard linear projection (default)
    - GNN: Polynomial graph convolution using adjacency matrix
    - Spectral: Eigenvalue-polynomial filtering using eigenvectors

    Parameters
    ----------
    dx
        Node feature dimension.
    de
        Edge feature dimension.
    dy
        Global feature dimension.
    n_head
        Number of attention heads.
    use_gnn_q
        If True, use polynomial graph convolution for query projection.
    use_gnn_k
        If True, use polynomial graph convolution for key projection.
    use_gnn_v
        If True, use polynomial graph convolution for value projection.
    gnn_num_terms
        Number of polynomial terms for GNN projections (default 2).
    use_spectral_q
        If True, use spectral filter bank for query projection.
    use_spectral_k
        If True, use spectral filter bank for key projection.
    use_spectral_v
        If True, use spectral filter bank for value projection.
    spectral_k
        Number of eigenvectors for spectral projections (default 16).
    spectral_num_terms
        Number of polynomial terms for spectral filter (default 3).
    device
        Device for layer parameters.
    dtype
        Data type for layer parameters.

    Notes
    -----
    GNN and spectral projections are mutually exclusive per projection type.
    For example, use_gnn_q and use_spectral_q cannot both be True.
    """

    dx: int
    de: int
    dy: int
    df: int
    n_head: int
    q: nn.Module  # Linear, BareGraphConvolutionLayer, or SpectralProjectionLayer
    k: nn.Module  # Linear, BareGraphConvolutionLayer, or SpectralProjectionLayer
    v: nn.Module  # Linear, BareGraphConvolutionLayer, or SpectralProjectionLayer
    e_add: Linear
    e_mul: Linear
    y_e_mul: Linear
    y_e_add: Linear
    y_x_mul: Linear
    y_x_add: Linear
    y_y: Linear
    x_y: Xtoy
    e_y: Etoy
    x_out: Linear
    e_out: Linear
    y_out: nn.Sequential

    def __init__(
        self,
        dx: int,
        de: int,
        dy: int,
        n_head: int,
        use_gnn_q: bool = False,
        use_gnn_k: bool = False,
        use_gnn_v: bool = False,
        gnn_num_terms: int = 2,
        use_spectral_q: bool = False,
        use_spectral_k: bool = False,
        use_spectral_v: bool = False,
        spectral_k: int = 16,
        spectral_num_terms: int = 3,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if not dx % n_head == 0:
            raise AssertionError(f"dx: {dx} -- nhead: {n_head}")

        # Validate mutually exclusive projection modes
        if use_gnn_q and use_spectral_q:
            raise ValueError("use_gnn_q and use_spectral_q are mutually exclusive")
        if use_gnn_k and use_spectral_k:
            raise ValueError("use_gnn_k and use_spectral_k are mutually exclusive")
        if use_gnn_v and use_spectral_v:
            raise ValueError("use_gnn_v and use_spectral_v are mutually exclusive")

        self.dx = dx
        self.de = de
        self.dy = dy
        self.df = int(dx / n_head)
        self.n_head = n_head

        # Store projection mode flags for forward pass
        self._use_gnn_q = use_gnn_q
        self._use_gnn_k = use_gnn_k
        self._use_gnn_v = use_gnn_v
        self._use_spectral_q = use_spectral_q
        self._use_spectral_k = use_spectral_k
        self._use_spectral_v = use_spectral_v

        # Attention projections (Linear, GNN, or Spectral)
        if use_gnn_q:
            self.q = BareGraphConvolutionLayer(gnn_num_terms, dx)
        elif use_spectral_q:
            self.q = SpectralProjectionLayer(
                k=spectral_k, out_dim=dx, num_terms=spectral_num_terms
            )
        else:
            self.q = Linear(dx, dx, device=device, dtype=dtype)

        if use_gnn_k:
            self.k = BareGraphConvolutionLayer(gnn_num_terms, dx)
        elif use_spectral_k:
            self.k = SpectralProjectionLayer(
                k=spectral_k, out_dim=dx, num_terms=spectral_num_terms
            )
        else:
            self.k = Linear(dx, dx, device=device, dtype=dtype)

        if use_gnn_v:
            self.v = BareGraphConvolutionLayer(gnn_num_terms, dx)
        elif use_spectral_v:
            self.v = SpectralProjectionLayer(
                k=spectral_k, out_dim=dx, num_terms=spectral_num_terms
            )
        else:
            self.v = Linear(dx, dx, device=device, dtype=dtype)

        # FiLM E to X
        self.e_add = Linear(de, dx, device=device, dtype=dtype)
        self.e_mul = Linear(de, dx, device=device, dtype=dtype)

        # FiLM y to E
        self.y_e_mul = Linear(
            dy, dx, device=device, dtype=dtype
        )  # Warning: here it's dx and not de
        self.y_e_add = Linear(dy, dx, device=device, dtype=dtype)

        # FiLM y to X
        self.y_x_mul = Linear(dy, dx, device=device, dtype=dtype)
        self.y_x_add = Linear(dy, dx, device=device, dtype=dtype)

        # Process y
        self.y_y = Linear(dy, dy, device=device, dtype=dtype)
        self.x_y = Xtoy(dx, dy)
        self.e_y = Etoy(de, dy)

        # Output layers
        self.x_out = Linear(dx, dx, device=device, dtype=dtype)
        self.e_out = Linear(dx, de, device=device, dtype=dtype)
        self.y_out = nn.Sequential(
            nn.Linear(dy, dy, device=device, dtype=dtype),
            nn.ReLU(),
            nn.Linear(dy, dy, device=device, dtype=dtype),
        )

    def forward(
        self,
        X: Tensor,
        E: Tensor,
        y: Tensor,
        node_mask: Tensor,
        structure: GraphStructure,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute attention update for node and edge features.

        Parameters
        ----------
        X
            Node features (bs, n, d).
        E
            Edge features (bs, n, n, d).
        y
            Global features (bs, dz).
        node_mask
            Boolean mask (bs, n).
        structure
            Pre-computed structural context (adjacency, eigenvectors,
            eigenvalues) from the input graph.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            Updated (newX, newE, new_y) with the same shapes.
        """
        bs, n, _ = X.shape
        x_mask = node_mask.unsqueeze(-1)  # bs, n, 1
        e_mask1 = x_mask.unsqueeze(2)  # bs, n, 1, 1
        e_mask2 = x_mask.unsqueeze(1)  # bs, 1, n, 1

        A = structure.adjacency
        V = structure.eigenvectors
        Lambda = structure.eigenvalues

        if A is None and (self._use_gnn_q or self._use_gnn_k or self._use_gnn_v):
            raise ValueError(
                "GraphStructure.adjacency must be populated when using GNN "
                "projections. E[..., 0] is the no-edge channel, not the adjacency."
            )

        uses_spectral = (
            self._use_spectral_q or self._use_spectral_k or self._use_spectral_v
        )
        if uses_spectral and (V is None or Lambda is None):
            raise ValueError(
                "GraphStructure.eigenvectors and .eigenvalues must be populated "
                "when using spectral projections."
            )

        # 1. Map X to keys and queries
        # - GNN projections use adjacency matrix
        # - Spectral projections use eigenvectors
        # - Linear projections use node features directly
        if self._use_gnn_q:
            Q = self.q(A, X) * x_mask
        elif self._use_spectral_q:
            assert V is not None and Lambda is not None
            Q = self.q(V, Lambda) * x_mask
        else:
            Q = self.q(X) * x_mask  # (bs, n, dx)

        if self._use_gnn_k:
            K = self.k(A, X) * x_mask
        elif self._use_spectral_k:
            assert V is not None and Lambda is not None
            K = self.k(V, Lambda) * x_mask
        else:
            K = self.k(X) * x_mask  # (bs, n, dx)
        _assert_correctly_masked(Q, x_mask)
        # 2. Reshape to (bs, n, n_head, df) with dx = n_head * df

        Q = Q.reshape((Q.size(0), Q.size(1), self.n_head, self.df))
        K = K.reshape((K.size(0), K.size(1), self.n_head, self.df))

        Q = Q.unsqueeze(2)  # (bs, n, 1, n_head, df)
        K = K.unsqueeze(1)  # (bs, 1, n, n_head, df)

        # Compute unnormalized attentions. Y is (bs, n, n, n_head, df)
        Y = Q * K
        Y = Y / math.sqrt(Y.size(-1))
        _assert_correctly_masked(Y, (e_mask1 * e_mask2).unsqueeze(-1))

        E1 = self.e_mul(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E1 = E1.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        E2 = self.e_add(E) * e_mask1 * e_mask2  # bs, n, n, dx
        E2 = E2.reshape((E.size(0), E.size(1), E.size(2), self.n_head, self.df))

        # Incorporate edge features to the self attention scores.
        Y = Y * (E1 + 1) + E2  # (bs, n, n, n_head, df)

        # Incorporate y to E
        newE = Y.flatten(start_dim=3)  # bs, n, n, dx
        ye1 = self.y_e_add(y).unsqueeze(1).unsqueeze(1)  # bs, 1, 1, dx
        ye2 = self.y_e_mul(y).unsqueeze(1).unsqueeze(1)
        newE = ye1 + (ye2 + 1) * newE

        # Output E
        newE = self.e_out(newE) * e_mask1 * e_mask2  # bs, n, n, de
        _assert_correctly_masked(newE, e_mask1 * e_mask2)

        # Compute attentions. attn is still (bs, n, n, n_head, df)
        softmax_mask = e_mask2.expand(-1, n, -1, self.n_head)  # bs, n, n, n_head
        attn = masked_softmax(Y, softmax_mask, dim=2)  # bs, n, n, n_head

        # Compute values (using 'val' to avoid shadowing eigenvector parameter V)
        if self._use_gnn_v:
            val = self.v(A, X) * x_mask
        elif self._use_spectral_v:
            assert V is not None and Lambda is not None
            val = self.v(V, Lambda) * x_mask
        else:
            val = self.v(X) * x_mask  # bs, n, dx

        val = val.reshape((val.size(0), val.size(1), self.n_head, self.df))
        val = val.unsqueeze(1)  # (bs, 1, n, n_head, df)

        # Compute weighted values
        weighted_V = attn * val
        weighted_V = weighted_V.sum(dim=2)

        # Send output to input dim
        weighted_V = weighted_V.flatten(start_dim=2)  # bs, n, dx

        # Incorporate y to X
        yx1 = self.y_x_add(y).unsqueeze(1)
        yx2 = self.y_x_mul(y).unsqueeze(1)
        newX = yx1 + (yx2 + 1) * weighted_V

        # Output X
        newX = self.x_out(newX) * x_mask
        _assert_correctly_masked(newX, x_mask)

        # Process y based on X and E
        y = self.y_y(y)
        e_y = self.e_y(E)
        x_y = self.x_y(X)
        new_y = y + x_y + e_y
        new_y = self.y_out(new_y)  # bs, dy

        return newX, newE, new_y


class _GraphTransformer(nn.Module):
    """Core Graph Transformer for node/edge feature processing.

    Always expects pre-encoded features: node features ``X`` of shape
    ``(bs, n, dx)``, edge features ``E`` of shape ``(bs, n, n, de)``,
    global features ``y`` of shape ``(bs, dy)``, and a ``node_mask`` of
    shape ``(bs, n)``.  Callers are responsible for encoding raw inputs
    (e.g. adjacency matrices) into these feature tensors before calling.

    Parameters
    ----------
    n_layers
        Number of transformer layers.
    input_dims
        Input dimensions dict with keys "X" (node), "E" (edge), "y" (global).
    hidden_mlp_dims
        Hidden MLP dimensions dict with keys "X", "E", "y".
    hidden_dims
        Transformer hidden dimensions with keys "dx", "de", "dy", "n_head".
    output_dims
        Output dimensions dict with keys "X", "E", "y".
    act_fn_in
        Activation function for input MLPs. Defaults to ReLU.
    act_fn_out
        Activation function for output MLPs. Defaults to ReLU.
    """

    n_layers: int
    input_dims: dict[str, int]
    hidden_mlp_dims: dict[str, int]
    hidden_dims: dict[str, int]
    output_dims: dict[str, int]
    out_dim_X: int
    out_dim_E: int
    out_dim_y: int
    mlp_in_X: nn.Sequential
    mlp_in_E: nn.Sequential
    mlp_in_y: nn.Sequential
    tf_layers: nn.ModuleList
    mlp_out_X: nn.Sequential
    mlp_out_E: nn.Sequential
    mlp_out_y: nn.Sequential

    def __init__(
        self,
        n_layers: int,
        input_dims: dict[str, int],
        hidden_mlp_dims: dict[str, int],
        hidden_dims: dict[str, int],
        output_dims: dict[str, int],
        act_fn_in: nn.Module | None = None,
        act_fn_out: nn.Module | None = None,
        projection_config: dict[str, bool | int] | None = None,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.input_dims = input_dims
        self.hidden_mlp_dims = hidden_mlp_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        self.out_dim_X = output_dims["X"]
        self.out_dim_E = output_dims["E"]
        self.out_dim_y = output_dims["y"]

        if act_fn_in is None:
            act_fn_in = nn.ReLU()
        if act_fn_out is None:
            act_fn_out = nn.ReLU()

        self.mlp_in_X = nn.Sequential(
            nn.Linear(input_dims["X"], hidden_mlp_dims["X"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["X"], hidden_dims["dx"]),
            act_fn_in,
        )

        self.mlp_in_E = nn.Sequential(
            nn.Linear(input_dims["E"], hidden_mlp_dims["E"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["E"], hidden_dims["de"]),
            act_fn_in,
        )

        self.mlp_in_y = nn.Sequential(
            nn.Linear(input_dims["y"], hidden_mlp_dims["y"]),
            act_fn_in,
            nn.Linear(hidden_mlp_dims["y"], hidden_dims["dy"]),
            act_fn_in,
        )

        # Projection flags live in projection_config; for backwards compat,
        # also check hidden_dims (projection_config takes precedence).
        _PROJ_KEYS = {
            "use_gnn_q",
            "use_gnn_k",
            "use_gnn_v",
            "gnn_num_terms",
            "use_spectral_q",
            "use_spectral_k",
            "use_spectral_v",
            "spectral_k",
            "spectral_num_terms",
        }
        pc: dict[str, bool | int] = {
            k: v for k, v in hidden_dims.items() if k in _PROJ_KEYS
        }
        if projection_config:
            pc.update(projection_config)

        use_gnn_q = bool(pc.get("use_gnn_q", False))
        use_gnn_k = bool(pc.get("use_gnn_k", False))
        use_gnn_v = bool(pc.get("use_gnn_v", False))
        gnn_num_terms = int(pc.get("gnn_num_terms", 2))
        self._use_gnn_projections = use_gnn_q or use_gnn_k or use_gnn_v

        use_spectral_q = bool(pc.get("use_spectral_q", False))
        use_spectral_k = bool(pc.get("use_spectral_k", False))
        use_spectral_v = bool(pc.get("use_spectral_v", False))
        spectral_k = int(pc.get("spectral_k", 16))
        spectral_num_terms = int(pc.get("spectral_num_terms", 3))
        self._use_spectral_projections = (
            use_spectral_q or use_spectral_k or use_spectral_v
        )
        self._spectral_k = spectral_k

        # Create eigen layer if using spectral projections
        if self._use_spectral_projections:
            from tmgg.models.layers.topk_eigen import TopKEigenLayer

            self.eigen_layer: TopKEigenLayer | None = TopKEigenLayer(k=spectral_k)
        else:
            self.eigen_layer = None

        self.tf_layers = nn.ModuleList(
            [
                XEyTransformerLayer(
                    dx=hidden_dims["dx"],
                    de=hidden_dims["de"],
                    dy=hidden_dims["dy"],
                    n_head=hidden_dims["n_head"],
                    dim_ffX=hidden_dims.get("dim_ffX", 2048),
                    dim_ffE=hidden_dims.get("dim_ffE", 128),
                    dim_ffy=hidden_dims.get("dim_ffy", 2048),
                    use_gnn_q=use_gnn_q,
                    use_gnn_k=use_gnn_k,
                    use_gnn_v=use_gnn_v,
                    gnn_num_terms=gnn_num_terms,
                    use_spectral_q=use_spectral_q,
                    use_spectral_k=use_spectral_k,
                    use_spectral_v=use_spectral_v,
                    spectral_k=spectral_k,
                    spectral_num_terms=spectral_num_terms,
                )
                for _ in range(n_layers)
            ]
        )

        self.mlp_out_X = nn.Sequential(
            nn.Linear(hidden_dims["dx"], hidden_mlp_dims["X"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["X"], output_dims["X"]),
        )

        self.mlp_out_E = nn.Sequential(
            nn.Linear(hidden_dims["de"], hidden_mlp_dims["E"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["E"], output_dims["E"]),
        )

        self.mlp_out_y = nn.Sequential(
            nn.Linear(hidden_dims["dy"], hidden_mlp_dims["y"]),
            act_fn_out,
            nn.Linear(hidden_mlp_dims["y"], output_dims["y"]),
        )

    def forward(self, data: GraphData) -> GraphData:
        """Process categorical graph features through the transformer stack.

        Parameters
        ----------
        data
            Batched graph features with categorical edge encoding (channel 0
            is "no-edge", channel 1+ are edge classes).

        Returns
        -------
        GraphData
            Transformed features with residual connections.
        """
        X_cat, E_cat, y_cat, node_mask = data.X, data.E, data.y, data.node_mask
        bs, n = X_cat.shape[0], X_cat.shape[1]

        diag_mask = torch.eye(n, device=E_cat.device)
        diag_mask = ~diag_mask.bool()
        diag_mask = diag_mask.unsqueeze(0).unsqueeze(-1).expand(bs, -1, -1, -1)

        # Residual slices from categorical input
        X_to_out: torch.Tensor = X_cat[..., : self.out_dim_X]
        E_to_out: torch.Tensor = E_cat[..., : self.out_dim_E]
        y_to_out: torch.Tensor = y_cat[..., : self.out_dim_y]

        # --- Structural context (from categorical graph topology, before any learned transforms) ---
        needs_adjacency = self._use_gnn_projections or (
            self._use_spectral_projections and self.eigen_layer is not None
        )
        binary_adj = data.to_binary_adjacency() if needs_adjacency else None

        eigenvectors: torch.Tensor | None = None
        eigenvalues: torch.Tensor | None = None
        if (
            binary_adj is not None
            and self._use_spectral_projections
            and self.eigen_layer is not None
        ):
            V_raw, Lambda_raw = self.eigen_layer(binary_adj)
            actual_k = V_raw.shape[-1]
            if actual_k < self._spectral_k:
                pad_size = self._spectral_k - actual_k
                eigenvectors = F.pad(V_raw, (0, pad_size))
                eigenvalues = F.pad(Lambda_raw, (0, pad_size))
            else:
                eigenvectors = V_raw
                eigenvalues = Lambda_raw

        structure = GraphStructure(
            adjacency=binary_adj if self._use_gnn_projections else None,
            eigenvectors=eigenvectors,
            eigenvalues=eigenvalues,
        )

        # --- Categorical → hidden: E no longer has channel semantics after MLPs ---
        E_hidden = self.mlp_in_E(E_cat)
        E_hidden = (E_hidden + E_hidden.transpose(1, 2)) / 2
        hidden = GraphData(
            X=self.mlp_in_X(X_cat),
            E=E_hidden,
            y=self.mlp_in_y(y_cat),
            node_mask=node_mask,
        ).mask_zero_diag()
        X, E, y = hidden.X, hidden.E, hidden.y

        for layer in self.tf_layers:
            X, E, y = layer(X, E, y, node_mask, structure)

        X_out: torch.Tensor = self.mlp_out_X(X)
        E_out: torch.Tensor = self.mlp_out_E(E)
        y_out: torch.Tensor = self.mlp_out_y(y)

        X_final = X_out + X_to_out
        E_final = (E_out + E_to_out) * diag_mask
        y_final = y_out + y_to_out

        E_symmetric = 1 / 2 * (E_final + torch.transpose(E_final, 1, 2))

        return GraphData(
            X=X_final, E=E_symmetric, y=y_final, node_mask=node_mask
        ).mask_zero_diag()


class GraphTransformer(GraphModel):
    """Unified graph transformer accepting ``GraphData`` for both categorical
    diffusion and adjacency-based denoising.

    Wraps ``_GraphTransformer`` and optionally applies extra feature
    augmentation (e.g. eigenvector extraction, cycle counts) and/or
    timestep injection before the inner transformer.

    Parameters
    ----------
    n_layers
        Number of transformer layers.
    input_dims
        Base input dimensions dict with keys ``"X"``, ``"E"``, ``"y"``.
        These are the *raw* feature dimensions before any augmentation;
        ``extra_features.adjust_dims()`` and ``use_timestep`` adjust them
        automatically.
    hidden_mlp_dims
        Hidden MLP dimensions with keys ``"X"``, ``"E"``, ``"y"``.
    hidden_dims
        Transformer hidden dimensions with keys ``"dx"``, ``"de"``,
        ``"dy"``, ``"n_head"``.
    output_dims
        Output dimensions with keys ``"X"``, ``"E"``, ``"y"``.
    act_fn_in
        Activation function for input MLPs. Defaults to ReLU.
    act_fn_out
        Activation function for output MLPs. Defaults to ReLU.
    extra_features
        Callable with ``adjust_dims(input_dims)`` and
        ``__call__(X, E, y, node_mask)`` returning ``(extra_X, extra_E,
        extra_y)``. Pass ``None`` for no augmentation.
    use_timestep
        If True, append the normalised diffusion timestep ``t`` to ``y``
        before the inner transformer, adding one dimension to ``y``.
    """

    n_layers: int
    input_dims: dict[str, int]
    hidden_mlp_dims: dict[str, int]
    hidden_dims: dict[str, int]
    output_dims: dict[str, int]
    extra_features: ExtraFeaturesProvider | None
    transformer: _GraphTransformer

    def __init__(
        self,
        n_layers: int,
        input_dims: dict[str, int],
        hidden_mlp_dims: dict[str, int],
        hidden_dims: dict[str, int],
        output_dims: dict[str, int],
        act_fn_in: nn.Module | None = None,
        act_fn_out: nn.Module | None = None,
        extra_features: ExtraFeaturesProvider | None = None,
        use_timestep: bool = False,
        projection_config: dict[str, bool | int] | None = None,
    ) -> None:
        super().__init__()
        self.n_layers = n_layers
        self.input_dims = input_dims
        self.hidden_mlp_dims = hidden_mlp_dims
        self.hidden_dims = hidden_dims
        self.output_dims = output_dims

        self.extra_features = extra_features
        self._use_timestep = use_timestep

        adjusted_input_dims = dict(input_dims)
        if extra_features is not None:
            adjusted_input_dims = extra_features.adjust_dims(adjusted_input_dims)
        if use_timestep:
            adjusted_input_dims = {
                **adjusted_input_dims,
                "y": adjusted_input_dims["y"] + 1,
            }

        self.transformer = _GraphTransformer(
            n_layers=n_layers,
            input_dims=adjusted_input_dims,
            hidden_mlp_dims=hidden_mlp_dims,
            hidden_dims=hidden_dims,
            output_dims=output_dims,
            act_fn_in=act_fn_in,
            act_fn_out=act_fn_out,
            projection_config=projection_config,
        )

    @override
    def forward(self, data: GraphData, t: torch.Tensor | None = None) -> GraphData:
        """Forward pass through the graph transformer.

        When ``extra_features`` is set, calls it to produce additional
        feature tensors that are concatenated with X, E, y before the
        inner transformer. When ``use_timestep`` is True and ``t`` is
        provided, appends it to y.

        Parameters
        ----------
        data
            Batched graph features (X, E, y, node_mask).
        t
            Normalised diffusion timestep, shape ``(bs,)``. Appended to
            ``y`` when ``use_timestep=True``.

        Returns
        -------
        GraphData
            Predicted features with output dimensions.
        """
        X, E, y, node_mask = data.X, data.E, data.y, data.node_mask

        if self.extra_features is not None:
            extra_X, extra_E, extra_y = self.extra_features(X, E, y, node_mask)
            X = torch.cat([X, extra_X], dim=-1)
            E = torch.cat([E, extra_E], dim=-1)
            y = torch.cat([y, extra_y], dim=-1)

        if self._use_timestep and t is not None:
            y = torch.cat([y, t.unsqueeze(-1)], dim=-1)

        augmented = GraphData(X=X, E=E, y=y, node_mask=node_mask)
        return self.transformer(augmented)

    @override
    def get_config(self) -> dict[str, Any]:
        """Return model configuration for serialization and logging."""
        return {
            "n_layers": self.n_layers,
            "input_dims": self.input_dims,
            "hidden_mlp_dims": self.hidden_mlp_dims,
            "hidden_dims": self.hidden_dims,
            "output_dims": self.output_dims,
            "extra_features": type(self.extra_features).__name__
            if self.extra_features
            else None,
            "use_timestep": self._use_timestep,
        }
