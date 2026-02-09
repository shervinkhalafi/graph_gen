"""Batch containers for graph data.

Two container types cover the full pipeline:
- ``AdjacencyBatch`` wraps raw adjacency matrices for denoising models.
- ``CategoricalBatch`` wraps one-hot categorical node/edge features for
  discrete diffusion (DiGress-style).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True, slots=True)
class AdjacencyBatch:
    """Batch of graphs in adjacency-matrix representation.

    Used by denoising models that operate on raw adjacency matrices:
    spectral denoisers, GNNs, hybrids, baselines, and the DiGress
    GraphTransformer wrapper.

    Parameters
    ----------
    A : Tensor
        Adjacency matrices. Shape ``(n, n)`` for a single graph or
        ``(batch, n, n)`` for a batch.
    t : Tensor or None
        Diffusion timestep per sample. Shape ``(batch,)``. None means
        no timestep conditioning (standard denoising).

    Examples
    --------
    >>> A = torch.eye(5).unsqueeze(0).expand(4, -1, -1)
    >>> batch = AdjacencyBatch(A=A)
    >>> batch.num_nodes
    5
    >>> batch.batch_size
    4
    """

    A: Tensor
    t: Tensor | None = None

    def __post_init__(self) -> None:
        if self.A.ndim == 2:
            if self.A.shape[0] != self.A.shape[1]:
                raise ValueError(
                    f"Adjacency matrix must be square, got shape {self.A.shape}"
                )
        elif self.A.ndim == 3:
            if self.A.shape[1] != self.A.shape[2]:
                raise ValueError(
                    f"Batched adjacency matrices must be square in last two dims, "
                    f"got shape {self.A.shape}"
                )
        else:
            raise ValueError(
                f"Adjacency matrix must be 2D (n, n) or 3D (batch, n, n), "
                f"got {self.A.ndim}D tensor with shape {self.A.shape}"
            )

        if self.t is not None and self.t.ndim != 1:
            raise ValueError(
                f"Timestep tensor must be 1D (batch,), got {self.t.ndim}D "
                f"with shape {self.t.shape}"
            )

    @property
    def batch_size(self) -> int:
        """Number of graphs in the batch. Returns 1 for unbatched (2D) input."""
        if self.A.ndim == 2:
            return 1
        return self.A.shape[0]

    @property
    def num_nodes(self) -> int:
        """Number of nodes per graph."""
        return self.A.shape[-1]

    @property
    def device(self) -> torch.device:
        """Device of the underlying tensors."""
        return self.A.device

    def to(self, device: torch.device | str) -> AdjacencyBatch:
        """Move all tensors to the given device."""
        return AdjacencyBatch(
            A=self.A.to(device),
            t=self.t.to(device) if self.t is not None else None,
        )

    def ensure_batched(self) -> AdjacencyBatch:
        """Return a batch with A guaranteed to be 3D ``(batch, n, n)``.

        If A is 2D, unsqueezes dim 0. If already 3D, returns self unchanged.
        """
        if self.A.ndim == 2:
            return AdjacencyBatch(
                A=self.A.unsqueeze(0),
                t=self.t,
            )
        return self

    @classmethod
    def from_tensor(cls, A: Tensor, t: Tensor | None = None) -> AdjacencyBatch:
        """Construct from a raw adjacency tensor.

        This is the primary factory method. Equivalent to the constructor
        but reads more clearly at call sites.

        Parameters
        ----------
        A : Tensor
            Adjacency matrix or batch thereof.
        t : Tensor or None
            Optional timestep tensor.
        """
        return cls(A=A, t=t)


@dataclass(frozen=True, slots=True)
class CategoricalBatch:
    """Batch of graphs in one-hot categorical representation.

    Used by discrete diffusion models that operate on categorical node/edge
    features (DiGress-style). Node types are one-hot in X, edge types are
    one-hot in E, and variable-size graphs are handled via node_mask.

    Parameters
    ----------
    X : Tensor
        Node features, shape ``(batch, n, dx)``. One-hot encoded node types.
    E : Tensor
        Edge features, shape ``(batch, n, n, de)``. One-hot encoded edge types.
    y : Tensor
        Global graph features, shape ``(batch, dy)``. May be empty (``dy=0``).
    node_mask : Tensor
        Boolean mask indicating real nodes, shape ``(batch, n)``. Padded
        positions are False.
    t : Tensor or None
        Diffusion timestep per sample, shape ``(batch,)``. None means no
        timestep conditioning.

    Notes
    -----
    For synthetic binary graphs: ``dx=2`` (node present / absent), ``de=2``
    (edge present / absent). For molecular graphs: ``dx = num_atom_types``,
    ``de = num_bond_types``.
    """

    X: Tensor
    E: Tensor
    y: Tensor
    node_mask: Tensor
    t: Tensor | None = None

    def __post_init__(self) -> None:
        # Individual field validation
        if self.X.ndim != 3:
            raise ValueError(
                f"Node features X must be 3D (batch, n, dx), got {self.X.ndim}D "
                f"with shape {self.X.shape}"
            )
        if self.E.ndim != 4:
            raise ValueError(
                f"Edge features E must be 4D (batch, n, n, de), got {self.E.ndim}D "
                f"with shape {self.E.shape}"
            )
        if self.E.shape[1] != self.E.shape[2]:
            raise ValueError(
                f"Edge features must be square in spatial dims, "
                f"got shape {self.E.shape}"
            )
        if self.node_mask.ndim != 2:
            raise ValueError(
                f"node_mask must be 2D (batch, n), got {self.node_mask.ndim}D "
                f"with shape {self.node_mask.shape}"
            )

        # Cross-validate tensor shapes
        bs_x, n_x, _ = self.X.shape
        bs_e, n_e1, n_e2, _ = self.E.shape
        bs_m, n_m = self.node_mask.shape

        if not (bs_x == bs_e == bs_m):
            raise ValueError(
                f"Batch size mismatch: X has {bs_x}, E has {bs_e}, "
                f"node_mask has {bs_m}"
            )
        if not (n_x == n_e1 == n_e2 == n_m):
            raise ValueError(
                f"Node count mismatch: X has {n_x}, E has ({n_e1}, {n_e2}), "
                f"node_mask has {n_m}"
            )
        if self.y.ndim != 2 or self.y.shape[0] != bs_x:
            raise ValueError(
                f"y must be 2D with batch_size={bs_x}, got shape {self.y.shape}"
            )
        if self.t is not None:
            if self.t.ndim != 1:
                raise ValueError(
                    f"Timestep tensor must be 1D (batch,), got {self.t.ndim}D "
                    f"with shape {self.t.shape}"
                )
            if self.t.shape[0] != bs_x:
                raise ValueError(
                    f"Timestep batch size {self.t.shape[0]} != data batch size {bs_x}"
                )

    @property
    def batch_size(self) -> int:
        """Number of graphs in the batch."""
        return self.X.shape[0]

    @property
    def num_nodes(self) -> int:
        """Maximum number of nodes (including padding)."""
        return self.X.shape[1]

    @property
    def dx(self) -> int:
        """Number of node feature channels (node type classes)."""
        return self.X.shape[2]

    @property
    def de(self) -> int:
        """Number of edge feature channels (edge type classes)."""
        return self.E.shape[3]

    @property
    def dy(self) -> int:
        """Dimension of global features."""
        return self.y.shape[1]

    @property
    def device(self) -> torch.device:
        """Device of the underlying tensors."""
        return self.X.device

    def to(self, device: torch.device | str) -> CategoricalBatch:
        """Move all tensors to the given device."""
        return CategoricalBatch(
            X=self.X.to(device),
            E=self.E.to(device),
            y=self.y.to(device),
            node_mask=self.node_mask.to(device),
            t=self.t.to(device) if self.t is not None else None,
        )

    def mask(self) -> CategoricalBatch:
        """Zero out features at masked (padded) node positions.

        Returns a new CategoricalBatch with X zeroed where node_mask is
        False, and E zeroed where either endpoint is masked.
        """
        x_mask = self.node_mask.unsqueeze(-1)  # (bs, n, 1)
        e_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(1)  # (bs, n, n, 1)

        return CategoricalBatch(
            X=self.X * x_mask,
            E=self.E * e_mask,
            y=self.y,
            node_mask=self.node_mask,
            t=self.t,
        )

    def to_adjacency(self) -> Tensor:
        """Extract binary adjacency matrix from edge features.

        Assumes edge class 0 is "no edge" and any other class means "edge".
        Returns the argmax of E, thresholded at > 0.

        Returns
        -------
        Tensor
            Binary adjacency matrices of shape ``(batch, n, n)``.
        """
        edge_class = self.E.argmax(dim=-1)  # (batch, n, n)
        return (edge_class > 0).float()

    @classmethod
    def from_adjacency(
        cls,
        A: Tensor,
        node_mask: Tensor | None = None,
        t: Tensor | None = None,
    ) -> CategoricalBatch:
        """Construct a CategoricalBatch from binary adjacency matrices.

        Converts binary adjacency to a 2-class categorical representation
        (``dx=2``, ``de=2``) suitable for discrete diffusion on synthetic graphs.

        Parameters
        ----------
        A : Tensor
            Binary adjacency matrices, shape ``(batch, n, n)``.
        node_mask : Tensor or None
            Node mask, shape ``(batch, n)``. If None, assumes all nodes are real.
        t : Tensor or None
            Timestep tensor.

        Returns
        -------
        CategoricalBatch
            With ``dx=2`` (node present, node absent) and ``de=2`` (no edge, edge).
        """
        if A.ndim != 3:
            raise ValueError(f"A must be 3D (batch, n, n), got shape {A.shape}")

        bs, n, _ = A.shape

        if node_mask is None:
            node_mask = torch.ones(bs, n, device=A.device, dtype=torch.bool)

        # Node features: real nodes are class 1, padding is class 0
        X = torch.zeros(bs, n, 2, device=A.device, dtype=A.dtype)
        X[..., 0] = (~node_mask).float()  # class 0 = padding
        X[..., 1] = node_mask.float()  # class 1 = real node

        # Edge features: class 0 = no edge, class 1 = edge
        E = torch.zeros(bs, n, n, 2, device=A.device, dtype=A.dtype)
        E[..., 0] = 1.0 - A  # no edge
        E[..., 1] = A  # edge

        # Global features: empty
        y = torch.zeros(bs, 0, device=A.device, dtype=A.dtype)

        return cls(X=X, E=E, y=y, node_mask=node_mask, t=t)

    @classmethod
    def from_placeholder(
        cls,
        X: Tensor,
        E: Tensor,
        y: Tensor,
        node_mask: Tensor,
        t: Tensor | None = None,
    ) -> CategoricalBatch:
        """Construct from individual tensors (PlaceHolder replacement).

        This factory replaces the mutable PlaceHolder class from
        diffusion_utils.py with an immutable dataclass container.

        Parameters
        ----------
        X, E, y, node_mask, t
            Same semantics as the class attributes.
        """
        return cls(X=X, E=E, y=y, node_mask=node_mask, t=t)
