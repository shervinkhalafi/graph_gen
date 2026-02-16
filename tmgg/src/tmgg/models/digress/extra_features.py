"""Extra feature computation for discrete diffusion models.

Provides structural graph features — cycle counts and Laplacian eigenfeatures —
that augment the transformer's input in DiGress-style discrete diffusion. Three
feature modes are available:

- ``"cycles"``: per-node cycle counts for k=3,4,5 and per-graph counts for k=3,4,5,6.
  Adds (X+3, E+0, y+5) dimensions.
- ``"eigenvalues"``: cycles plus Laplacian eigenvalue features (connected component
  count, first k=5 non-zero eigenvalues). Adds (X+3, E+0, y+11).
- ``"all"``: eigenvalues plus eigenvector features (LCC indicator, k=2 lowest
  eigenvectors). Adds (X+6, E+0, y+11).

The implementation follows the DiGress baseline (Vignac et al., 2023) with the
interface adapted from ``noisy_data`` dicts to explicit ``(X, E, y, node_mask)``
arguments consistent with the tmgg convention.
"""

from __future__ import annotations

import torch
from torch import Tensor

# ------------------------------------------------------------------
# Dimension helper
# ------------------------------------------------------------------


def extra_features_dims(features_type: str) -> tuple[int, int, int]:
    """Return the extra feature dimensions ``(extra_X, extra_E, extra_y)``.

    Callers use this to compute ``input_dims`` for the transformer:
    ``input_dims["X"] = base_dx + extra_X``, etc.

    Parameters
    ----------
    features_type
        One of ``"cycles"``, ``"eigenvalues"``, ``"all"``.
    """
    dims = {
        "cycles": (3, 0, 5),
        "eigenvalues": (3, 0, 11),
        "all": (6, 0, 11),
    }
    if features_type not in dims:
        raise ValueError(
            f"Unknown features_type {features_type!r}; expected one of {list(dims)}"
        )
    return dims[features_type]


# ------------------------------------------------------------------
# Dummy (zero-width) extra features
# ------------------------------------------------------------------


class DummyExtraFeatures:
    """Returns zero-width tensors for extra features.

    Placeholder for experiments that need no structural augmentation (e.g.
    synthetic SBM). Replace with ``ExtraFeatures`` for molecular datasets
    or when reproducing published DiGress results.
    """

    def __call__(
        self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute (zero-width) extra features.

        Parameters
        ----------
        X
            Node features, shape ``(bs, n, dx)``.
        E
            Edge features, shape ``(bs, n, n, de)``.
        y
            Global features, shape ``(bs, dy)``.
        node_mask
            Boolean mask for valid nodes, shape ``(bs, n)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(extra_X, extra_E, extra_y)`` all with zero-width last
            dimension: shapes ``(bs, n, 0)``, ``(bs, n, n, 0)``,
            ``(bs, 0)``.
        """
        bs, n, _ = X.shape
        return (
            torch.zeros(bs, n, 0, device=X.device, dtype=X.dtype),
            torch.zeros(bs, n, n, 0, device=X.device, dtype=X.dtype),
            torch.zeros(bs, 0, device=X.device, dtype=X.dtype),
        )


# ------------------------------------------------------------------
# Full extra features
# ------------------------------------------------------------------


class ExtraFeatures:
    """Structural graph features for the discrete diffusion transformer.

    Computes cycle counts and (optionally) Laplacian eigenfeatures from the
    noisy adjacency, then appends them to the node/edge/global feature
    vectors.

    Parameters
    ----------
    extra_features_type
        Feature mode: ``"cycles"``, ``"eigenvalues"``, or ``"all"``.
    max_n_nodes
        Maximum number of nodes across the dataset, used to normalise
        the node-count feature appended to y.
    """

    def __init__(self, extra_features_type: str, max_n_nodes: int) -> None:
        self.max_n_nodes = max_n_nodes
        self.ncycles = NodeCycleFeatures()
        self.features_type = extra_features_type
        if extra_features_type in ("eigenvalues", "all"):
            self.eigenfeatures = EigenFeatures(mode=extra_features_type)
        elif extra_features_type != "cycles":
            raise ValueError(
                f"Unknown extra_features_type {extra_features_type!r}; "
                "expected 'cycles', 'eigenvalues', or 'all'"
            )

    def __call__(
        self, X: Tensor, E: Tensor, y: Tensor, node_mask: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute extra features from one-hot node/edge tensors.

        Parameters
        ----------
        X
            One-hot node features ``(bs, n, dx)``.
        E
            One-hot edge features ``(bs, n, n, de)``.
        y
            Global features ``(bs, dy)``.
        node_mask
            Boolean node mask ``(bs, n)``.

        Returns
        -------
        tuple[Tensor, Tensor, Tensor]
            ``(extra_X, extra_E, extra_y)`` with widths determined by the
            feature mode (see ``extra_features_dims``).
        """
        # Normalised node count: n_valid / max_n_nodes, shape (bs, 1)
        n = node_mask.sum(dim=1).unsqueeze(1).float() / self.max_n_nodes

        x_cycles, y_cycles = self.ncycles(E, node_mask)
        extra_edge_attr = torch.zeros((*E.shape[:-1], 0), device=E.device).type_as(E)

        if self.features_type == "cycles":
            return x_cycles, extra_edge_attr, torch.hstack((n, y_cycles))

        elif self.features_type == "eigenvalues":
            eigenfeatures = self.eigenfeatures(E, node_mask)
            n_components, batched_eigenvalues = eigenfeatures[0], eigenfeatures[1]
            return (
                x_cycles,
                extra_edge_attr,
                torch.hstack((n, y_cycles, n_components, batched_eigenvalues)),
            )

        else:
            # "all" mode
            eigenfeatures = self.eigenfeatures(E, node_mask)
            assert (
                len(eigenfeatures) == 4
            ), f"Expected 4 eigenfeatures, got {len(eigenfeatures)}"
            n_components, batched_eigenvalues = eigenfeatures[0], eigenfeatures[1]
            nonlcc_indicator, k_lowest_eigvec = eigenfeatures[2], eigenfeatures[3]
            return (
                torch.cat((x_cycles, nonlcc_indicator, k_lowest_eigvec), dim=-1),
                extra_edge_attr,
                torch.hstack((n, y_cycles, n_components, batched_eigenvalues)),
            )


# ------------------------------------------------------------------
# Cycle features
# ------------------------------------------------------------------


class NodeCycleFeatures:
    """Wraps ``KNodeCycles``, scales and clips outputs."""

    def __init__(self) -> None:
        self.kcycles = KNodeCycles()

    def __call__(self, E: Tensor, node_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Compute cycle counts from one-hot edge features.

        Parameters
        ----------
        E
            One-hot edge features ``(bs, n, n, de)``. Class 0 is
            interpreted as "no edge"; classes 1.. are edge types.
        node_mask
            Boolean node mask ``(bs, n)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(x_cycles, y_cycles)`` with shapes ``(bs, n, 3)`` and
            ``(bs, 4)`` respectively.
        """
        adj_matrix = E[..., 1:].sum(dim=-1).float()

        x_cycles, y_cycles = self.kcycles.k_cycles(adj_matrix=adj_matrix)
        x_cycles = x_cycles.type_as(adj_matrix) * node_mask.unsqueeze(-1)

        # Scale and clip to avoid large values on dense graphs
        x_cycles = x_cycles / 10
        y_cycles = y_cycles / 10
        x_cycles = x_cycles.clamp(max=1.0)
        y_cycles = y_cycles.clamp(max=1.0)
        return x_cycles, y_cycles


class KNodeCycles:
    """Per-node and per-graph cycle counts for k=3,4,5,6 via matrix powers."""

    # Set in k_cycles() / calculate_kpowers() before any method reads them
    adj_matrix: Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    d: Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    k1_matrix: Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    k2_matrix: Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    k3_matrix: Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    k4_matrix: Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    k5_matrix: Tensor  # pyright: ignore[reportUninitializedInstanceVariable]
    k6_matrix: Tensor  # pyright: ignore[reportUninitializedInstanceVariable]

    def calculate_kpowers(self) -> None:
        self.k1_matrix = self.adj_matrix.float()
        self.d = self.adj_matrix.sum(dim=-1)
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self) -> tuple[Tensor, Tensor]:
        """3-cycle count: diag(A^3) / 2 per node, tr(A^3) / 6 per graph."""
        c3 = batch_diagonal(self.k3_matrix)
        return (c3 / 2).unsqueeze(-1).float(), (c3.sum(dim=-1) / 6).unsqueeze(
            -1
        ).float()

    def k4_cycle(self) -> tuple[Tensor, Tensor]:
        """4-cycle count per node and per graph."""
        diag_a4 = batch_diagonal(self.k4_matrix)
        c4 = (
            diag_a4
            - self.d * (self.d - 1)
            - (self.adj_matrix @ self.d.unsqueeze(-1)).sum(dim=-1)
        )
        return (c4 / 2).unsqueeze(-1).float(), (c4.sum(dim=-1) / 8).unsqueeze(
            -1
        ).float()

    def k5_cycle(self) -> tuple[Tensor, Tensor]:
        """5-cycle count per node and per graph."""
        diag_a5 = batch_diagonal(self.k5_matrix)
        triangles = batch_diagonal(self.k3_matrix)
        c5 = (
            diag_a5
            - 2 * triangles * self.d
            - (self.adj_matrix @ triangles.unsqueeze(-1)).sum(dim=-1)
            + triangles
        )
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(
            -1
        ).float()

    def k6_cycle(self) -> tuple[None, Tensor]:
        """6-cycle count (graph-level only, no per-node decomposition)."""
        term_1_t = batch_trace(self.k6_matrix)
        term_2_t = batch_trace(self.k3_matrix**2)
        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4_matrix)
        term_6_t = batch_trace(self.k3_matrix)
        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)
        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)
        term10_t = batch_trace(self.k2_matrix)

        c6_t = (
            term_1_t
            - 3 * term_2_t
            + 9 * term3_t
            - 6 * term_4_t
            + 6 * term_5_t
            - 4 * term_6_t
            + 4 * term_7_t
            + 3 * term8_t
            - 12 * term9_t
            + 4 * term10_t
        )
        return None, (c6_t / 12).unsqueeze(-1).float()

    def k_cycles(self, adj_matrix: Tensor) -> tuple[Tensor, Tensor]:
        """Compute per-node (k=3,4,5) and per-graph (k=3,4,5,6) cycle counts.

        Parameters
        ----------
        adj_matrix
            Binary adjacency matrix ``(bs, n, n)``.

        Returns
        -------
        tuple[Tensor, Tensor]
            ``(kcyclesx, kcyclesy)`` with shapes ``(bs, n, 3)`` and
            ``(bs, 4)`` respectively.
        """
        self.adj_matrix = adj_matrix
        self.calculate_kpowers()

        k3x, k3y = self.k3_cycle()
        assert (k3x >= -0.1).all()

        k4x, k4y = self.k4_cycle()
        assert (k4x >= -0.1).all()

        k5x, k5y = self.k5_cycle()
        assert (k5x >= -0.1).all(), k5x

        _, k6y = self.k6_cycle()
        assert (k6y >= -0.1).all()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)
        return kcyclesx, kcyclesy


# ------------------------------------------------------------------
# Eigen features
# ------------------------------------------------------------------


class EigenFeatures:
    """Laplacian eigendecomposition features for graph structure.

    Adapted from DGN (https://github.com/Saro00/DGN).

    Parameters
    ----------
    mode
        ``"eigenvalues"`` returns ``(n_components, first_k_ev)`` of shapes
        ``(bs, 1)`` and ``(bs, 5)``. ``"all"`` additionally returns
        ``(nonlcc_indicator, k_lowest_eigvec)`` of shapes ``(bs, n, 1)``
        and ``(bs, n, 2)``.
    """

    def __init__(self, mode: str) -> None:
        assert mode in (
            "eigenvalues",
            "all",
        ), f"EigenFeatures mode must be 'eigenvalues' or 'all', got {mode!r}"
        self.mode = mode

    def __call__(
        self, E: Tensor, node_mask: Tensor
    ) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor]:
        """Compute eigenfeatures from one-hot edge tensor.

        Parameters
        ----------
        E
            One-hot edge features ``(bs, n, n, de)``.
        node_mask
            Boolean node mask ``(bs, n)``.

        Returns
        -------
        tuple
            For ``"eigenvalues"``: ``(n_components, first_k_ev)``.
            For ``"all"``: ``(n_components, first_k_ev, nonlcc_indicator, k_lowest_eigvec)``.
        """
        A = E[..., 1:].sum(dim=-1).float() * mask_2d(node_mask)
        L = compute_laplacian(A, normalize=False)

        # Pad masked nodes with large diagonal values so their eigenvalues
        # are pushed far above zero and don't interfere with the real spectrum.
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~node_mask.unsqueeze(1)) * (~node_mask.unsqueeze(2))
        L = L * mask_2d(node_mask) + mask_diag  # pyright: ignore[reportConstantRedefinition]  # math notation

        if self.mode == "eigenvalues":
            eigvals = torch.linalg.eigvalsh(L)
            eigvals = eigvals.type_as(A) / torch.sum(node_mask, dim=1, keepdim=True)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigvals)
            return n_connected_comp.type_as(A), batch_eigenvalues.type_as(A)

        else:
            # "all" mode
            eigvals, eigvectors = torch.linalg.eigh(L)
            eigvals = eigvals.type_as(A) / torch.sum(node_mask, dim=1, keepdim=True)
            eigvectors = eigvectors * mask_2d(node_mask)

            n_connected_comp, batch_eigenvalues = get_eigenvalues_features(eigvals)

            nonlcc_indicator, k_lowest_eigenvector = get_eigenvectors_features(
                vectors=eigvectors,
                node_mask=node_mask,
                n_connected=n_connected_comp,
            )
            return (
                n_connected_comp,
                batch_eigenvalues,
                nonlcc_indicator,
                k_lowest_eigenvector,
            )


# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------


def mask_2d(node_mask: Tensor) -> Tensor:
    """Expand a ``(bs, n)`` node mask to ``(bs, n, n)`` edge mask."""
    return node_mask.unsqueeze(1) * node_mask.unsqueeze(2)


def compute_laplacian(adjacency: Tensor, normalize: bool) -> Tensor:
    """Compute the graph Laplacian from a batched adjacency matrix.

    Parameters
    ----------
    adjacency
        Batched adjacency matrix ``(bs, n, n)``.
    normalize
        If False, returns the combinatorial Laplacian L = D - A.
        If True, returns the symmetric normalised Laplacian
        L_sym = I - D^{-1/2} A D^{-1/2}.

    Returns
    -------
    Tensor
        Symmetrised Laplacian ``(bs, n, n)``.
    """
    diag = torch.sum(adjacency, dim=-1)  # (bs, n)
    n = diag.shape[-1]
    D = torch.diag_embed(diag)  # (bs, n, n)
    combinatorial = D - adjacency

    if not normalize:
        return (combinatorial + combinatorial.transpose(1, 2)) / 2

    diag0 = diag.clone()
    diag[diag == 0] = 1e-12

    diag_norm = 1 / torch.sqrt(diag)
    D_norm = torch.diag_embed(diag_norm)
    L = (
        torch.eye(n, device=adjacency.device, dtype=adjacency.dtype).unsqueeze(0)
        - D_norm @ adjacency @ D_norm
    )
    L[diag0 == 0] = 0
    return (L + L.transpose(1, 2)) / 2


def get_eigenvalues_features(eigenvalues: Tensor, k: int = 5) -> tuple[Tensor, Tensor]:
    """Extract the first k non-zero eigenvalues from sorted eigenvalues.

    Parameters
    ----------
    eigenvalues
        Sorted eigenvalues ``(bs, n)`` from ``eigvalsh``.
    k
        Number of non-zero eigenvalues to keep.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(n_connected_components, first_k_ev)`` with shapes ``(bs, 1)``
        and ``(bs, k)``.
    """
    ev = eigenvalues
    bs, n = ev.shape
    n_connected_components = (ev < 1e-5).sum(dim=-1)
    assert (n_connected_components > 0).all(), (n_connected_components, ev)

    to_extend = int(n_connected_components.max().item()) + k - n
    if to_extend > 0:
        eigenvalues = torch.hstack(
            (eigenvalues, 2 * torch.ones(bs, to_extend).type_as(eigenvalues))
        )
    indices = torch.arange(k).type_as(eigenvalues).long().unsqueeze(
        0
    ) + n_connected_components.unsqueeze(1)
    first_k_ev = torch.gather(eigenvalues, dim=1, index=indices)
    return n_connected_components.unsqueeze(-1), first_k_ev


def get_eigenvectors_features(
    vectors: Tensor, node_mask: Tensor, n_connected: Tensor, k: int = 2
) -> tuple[Tensor, Tensor]:
    """Extract LCC indicator and k lowest eigenvectors.

    Parameters
    ----------
    vectors
        Eigenvectors ``(bs, n, n)`` from ``eigh`` (columns are eigenvectors).
    node_mask
        Boolean node mask ``(bs, n)``.
    n_connected
        Connected component count ``(bs, 1)`` from ``get_eigenvalues_features``.
    k
        Number of lowest eigenvectors to extract.

    Returns
    -------
    tuple[Tensor, Tensor]
        ``(not_lcc_indicator, k_lowest_eigvec)`` with shapes ``(bs, n, 1)``
        and ``(bs, n, k)``.
    """
    bs, n = vectors.size(0), vectors.size(1)

    # Indicator for nodes outside the largest connected component (LCC).
    # The first eigenvector (constant on connected components) identifies
    # which component each node belongs to. The mode gives the LCC.
    first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask
    # Random noise on masked positions prevents 0 from becoming the mode
    random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)
    first_ev = first_ev + random
    most_common = torch.mode(first_ev, dim=1).values
    mask = ~(first_ev == most_common.unsqueeze(1))
    not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

    # Eigenvectors for the first nonzero eigenvalues (Fiedler vector etc.)
    to_extend = int(n_connected.max().item()) + k - n
    if to_extend > 0:
        vectors = torch.cat(
            (vectors, torch.zeros(bs, n, to_extend).type_as(vectors)), dim=2
        )
    indices = torch.arange(k).type_as(vectors).long().unsqueeze(0).unsqueeze(
        0
    ) + n_connected.unsqueeze(2)  # (bs, 1, k)
    indices = indices.expand(-1, n, -1)  # (bs, n, k)
    first_k_ev = torch.gather(vectors, dim=2, index=indices)
    first_k_ev = first_k_ev * node_mask.unsqueeze(2)

    return not_lcc_indicator, first_k_ev


def batch_trace(X: Tensor) -> Tensor:
    """Batched matrix trace: ``(bs, n, n) -> (bs,)``."""
    return torch.diagonal(X, dim1=-2, dim2=-1).sum(dim=-1)


def batch_diagonal(X: Tensor) -> Tensor:
    """Batched matrix diagonal: ``(bs, n, n) -> (bs, n)``."""
    return torch.diagonal(X, dim1=-2, dim2=-1)
