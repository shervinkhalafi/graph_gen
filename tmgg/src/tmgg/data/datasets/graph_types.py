"""Typed containers for categorical graph data and structural context.

``GraphData`` is the universal batch type for all experiments. It holds
batched node/edge features alongside a ``node_mask`` that tracks which
node positions are real versus padding. The four split feature fields
(``X_class`` / ``X_feat`` / ``E_class`` / ``E_feat``) are all optional;
at least one of the edge fields MUST be populated.

``GraphStructure`` bundles pre-computed topological features (adjacency,
eigenvectors, eigenvalues) derived from a ``GraphData`` instance before
any learned transformations. These remain constant across transformer
layer iterations.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace as _dc_replace
from typing import TYPE_CHECKING, Literal

import torch
from torch import Tensor

if TYPE_CHECKING:
    from torch_geometric.data import Batch, Data


@dataclass(frozen=True, slots=True)
class GraphData:
    """Batched graph features with a node validity mask and split feature fields.

    The dataclass owns only the unified-spec fields from
    ``docs/specs/2026-04-15-unified-graph-features-spec.md §5``: two
    required fields (``y``, ``node_mask``) and four optional split
    feature fields (``X_class`` / ``X_feat`` / ``E_class`` / ``E_feat``).
    At construction time at least one of the edge fields MUST be
    non-``None``; every other field is free to be empty.

    Parameters
    ----------
    y : Tensor
        Global features. Batched: ``(bs, dy)``; single: ``(dy,)``.
    node_mask : Tensor
        Boolean or float mask indicating real (vs padded) nodes.
        Batched: ``(bs, n)``; single: ``(n,)``.
    X_class : Tensor, optional
        Categorical node features (one-hot / PMF), shape
        ``(bs, n, dx_class)`` or ``(n, dx_class)``. ``None`` for
        structure-only graphs.
    X_feat : Tensor, optional
        Continuous node features, shape ``(bs, n, dx_feat)`` or
        ``(n, dx_feat)``.
    E_class : Tensor, optional
        Categorical edge features (one-hot / PMF), shape
        ``(bs, n, n, de_class)`` or ``(n, n, de_class)``. Channel 0
        conventionally encodes "no edge".
    E_feat : Tensor, optional
        Continuous edge features, shape ``(bs, n, n, de_feat)`` or
        ``(n, n, de_feat)``. Single-channel adjacency weights use
        ``de_feat == 1``.
    """

    y: Tensor
    node_mask: Tensor
    X_class: Tensor | None = None
    X_feat: Tensor | None = None
    E_class: Tensor | None = None
    E_feat: Tensor | None = None

    def __post_init__(self) -> None:
        """Validate shapes and the "at least one E_*" invariant.

        Raises
        ------
        ValueError
            When ``node_mask`` is missing, has the wrong rank, neither
            split edge field is populated, or a non-``None`` split
            field's leading dimensions disagree with ``node_mask``.
            Messages reference
            ``docs/specs/2026-04-15-unified-graph-features-spec.md §5``.
        """
        spec_ref = "docs/specs/2026-04-15-unified-graph-features-spec.md §5"

        # (1) node_mask present, 1D or 2D.
        nm = self.node_mask
        if nm is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError(
                f"GraphData requires a non-None node_mask (see {spec_ref})."
            )
        if nm.dim() not in (1, 2):
            raise ValueError(
                "GraphData.node_mask must be 1D (n,) or 2D (bs, n); "
                f"got shape {tuple(nm.shape)} (see {spec_ref})."
            )

        # (2) At least one edge field present.
        if self.E_class is None and self.E_feat is None:
            raise ValueError(
                "GraphData requires at least one of E_class or E_feat "
                f"to be populated (see {spec_ref})."
            )

        # (3) Leading-dim agreement with node_mask for any split field.
        if nm.dim() == 1:
            expected_n_dims: tuple[int, ...] = (int(nm.shape[0]),)
            expected_e_dims: tuple[int, ...] = (
                int(nm.shape[0]),
                int(nm.shape[0]),
            )
        else:
            expected_n_dims = (int(nm.shape[0]), int(nm.shape[1]))
            expected_e_dims = (
                int(nm.shape[0]),
                int(nm.shape[1]),
                int(nm.shape[1]),
            )

        def _check_leading(name: str, t: Tensor, expected: tuple[int, ...]) -> None:
            got = tuple(int(s) for s in t.shape[: len(expected)])
            if got != expected:
                raise ValueError(
                    f"GraphData.{name} leading dims {got} must match "
                    f"node_mask-derived {expected} (see {spec_ref})."
                )

        if self.X_class is not None:
            _check_leading("X_class", self.X_class, expected_n_dims)
        if self.X_feat is not None:
            _check_leading("X_feat", self.X_feat, expected_n_dims)
        if self.E_class is not None:
            _check_leading("E_class", self.E_class, expected_e_dims)
        if self.E_feat is not None:
            _check_leading("E_feat", self.E_feat, expected_e_dims)

    def replace(self, **kwargs: object) -> GraphData:
        """Return a copy with selected fields overridden.

        Thin typed wrapper over :func:`dataclasses.replace`. Accepts any
        subset of the dataclass fields; the returned instance re-runs
        ``__post_init__`` and therefore the same validation invariants.
        """
        return _dc_replace(self, **kwargs)

    def mask(self, collapse: bool = False) -> GraphData:
        """Zero out features at masked positions, optionally collapsing to indices.

        Operates on every populated split field; skips ``None`` fields.
        Mirrors upstream DiGress's ``PlaceHolder.mask(node_mask, collapse)``
        in ``digress-upstream-readonly/src/utils.py:103-131`` so that
        callers can use the upstream form directly. When ``collapse=False``
        (the default) the behaviour is unchanged: featured tensors are
        zeroed at padded positions and the categorical edge tensors are
        asserted symmetric. When ``collapse=True`` the categorical fields
        are argmaxed to integer indices with ``-1`` as a sentinel at
        masked positions (continuous ``X_feat`` / ``E_feat`` are still
        zeroed); the symmetry assertion is skipped because integer indices
        do not satisfy the same float-equality predicate.

        Parameters
        ----------
        collapse
            If ``False`` (default), zero features at masked positions and
            keep the one-hot / continuous representation. If ``True``,
            argmax categorical fields to integer class indices and use
            ``-1`` as the sentinel at masked positions.

        Returns
        -------
        GraphData
            New instance with masked values zeroed (or, when
            ``collapse=True``, with categorical fields argmaxed to
            integer indices).

        Raises
        ------
        AssertionError
            With ``collapse=False``: if any populated edge-feature tensor
            is not symmetric after masking.
        """
        x_mask = self.node_mask.unsqueeze(-1)  # (bs, n, 1)
        e_mask1 = x_mask.unsqueeze(2)  # (bs, n, 1, 1)
        e_mask2 = x_mask.unsqueeze(1)  # (bs, 1, n, 1)

        X_class_masked: Tensor | None = None
        E_class_masked: Tensor | None = None

        if collapse:
            edge_pair_mask = (e_mask1 * e_mask2).squeeze(-1)  # (bs, n, n)
            if self.X_class is not None:
                X_class_masked = torch.argmax(self.X_class, dim=-1)
                X_class_masked[self.node_mask == 0] = -1
            if self.E_class is not None:
                E_class_masked = torch.argmax(self.E_class, dim=-1)
                E_class_masked[edge_pair_mask == 0] = -1
        else:
            if self.X_class is not None:
                X_class_masked = self.X_class * x_mask
            if self.E_class is not None:
                E_class_masked = self.E_class * e_mask1 * e_mask2
                # Symmetry sanity check. ``mask()`` is hit ≥3× per training
                # step (forward noise, transformer output, loss target);
                # ``bool(allclose(...))`` syncs the GPU stream every call.
                # Guarded by ``__debug__`` so production runs (Python -O /
                # PYTHONOPTIMIZE=1) skip it. Symmetry is a math identity
                # of the producer (multiplying a symmetric input by
                # ``e_mask1 * e_mask2`` preserves symmetry).
                if __debug__:
                    assert torch.allclose(
                        E_class_masked, torch.transpose(E_class_masked, -3, -2)
                    ), (
                        "Edge features E_class must be symmetric after masking. "
                        f"Max asymmetry: {(E_class_masked - torch.transpose(E_class_masked, -3, -2)).abs().max().item():.2e}"
                    )

        # Continuous fields are always zeroed at masked positions; collapse
        # only affects the categorical (one-hot) split.
        X_feat_masked = self.X_feat * x_mask if self.X_feat is not None else None
        E_feat_masked: Tensor | None = None
        if self.E_feat is not None:
            E_feat_masked = self.E_feat * e_mask1 * e_mask2
            if not collapse:  # noqa: SIM102 - nested ``if __debug__:`` must stay nested
                if __debug__:
                    assert torch.allclose(
                        E_feat_masked, torch.transpose(E_feat_masked, -3, -2)
                    ), (
                        "Edge features E_feat must be symmetric after masking. "
                        f"Max asymmetry: {(E_feat_masked - torch.transpose(E_feat_masked, -3, -2)).abs().max().item():.2e}"
                    )

        return GraphData(
            y=self.y,
            node_mask=self.node_mask,
            X_class=X_class_masked,
            X_feat=X_feat_masked,
            E_class=E_class_masked,
            E_feat=E_feat_masked,
        )

    def mask_zero_diag(self) -> GraphData:
        """Zero masked positions and the diagonal of every edge tensor.

        Used inside the transformer where self-loops are excluded from
        the attention mechanism. Operates on every populated split
        field.
        """
        n = self.node_mask.size(-1)
        mask_diag = self.node_mask.unsqueeze(-1) * self.node_mask.unsqueeze(-2)
        eye = torch.eye(n, device=self.node_mask.device, dtype=torch.bool)
        # Broadcast the eye across leading dims
        while eye.dim() < mask_diag.dim():
            eye = eye.unsqueeze(0)
        mask_diag = mask_diag * (~eye)

        x_mask = self.node_mask.unsqueeze(-1)
        e_mask = mask_diag.unsqueeze(-1)

        X_class_masked = self.X_class * x_mask if self.X_class is not None else None
        X_feat_masked = self.X_feat * x_mask if self.X_feat is not None else None
        E_class_masked = self.E_class * e_mask if self.E_class is not None else None
        E_feat_masked = self.E_feat * e_mask if self.E_feat is not None else None

        return GraphData(
            y=self.y,
            node_mask=self.node_mask,
            X_class=X_class_masked,
            X_feat=X_feat_masked,
            E_class=E_class_masked,
            E_feat=E_feat_masked,
        )

    def type_as(self, x: Tensor) -> GraphData:
        """Return a new instance with feature tensors cast to match ``x``.

        The ``node_mask`` is left unchanged (it is boolean, not a feature).
        Split ``X_class`` / ``X_feat`` / ``E_class`` / ``E_feat`` fields
        are cast when present.
        """
        return GraphData(
            y=self.y.type_as(x),
            node_mask=self.node_mask,
            X_class=self.X_class.type_as(x) if self.X_class is not None else None,
            X_feat=self.X_feat.type_as(x) if self.X_feat is not None else None,
            E_class=self.E_class.type_as(x) if self.E_class is not None else None,
            E_feat=self.E_feat.type_as(x) if self.E_feat is not None else None,
        )

    def to(self, device: torch.device | str) -> GraphData:
        """Move all tensors to ``device``, returning a new instance.

        Needed for Lightning's ``transfer_batch_to_device`` since frozen
        dataclasses are not supported by ``apply_to_collection``.
        """
        return GraphData(
            y=self.y.to(device),
            node_mask=self.node_mask.to(device),
            X_class=self.X_class.to(device) if self.X_class is not None else None,
            X_feat=self.X_feat.to(device) if self.X_feat is not None else None,
            E_class=self.E_class.to(device) if self.E_class is not None else None,
            E_feat=self.E_feat.to(device) if self.E_feat is not None else None,
        )

    # ---- Conversion classmethods / methods ----------------------------

    @classmethod
    def from_structure_only(cls, node_mask: Tensor, edge_scalar: Tensor) -> GraphData:
        """Construct a structure-only graph with only ``E_feat`` populated.

        Wraps a dense scalar adjacency as a single-channel ``E_feat`` tensor
        of shape ``(bs, n, n, 1)`` (or ``(n, n, 1)`` for single graphs),
        leaving every other split field empty.

        Parameters
        ----------
        node_mask
            Boolean or float mask, ``(n,)`` or ``(bs, n)``.
        edge_scalar
            Dense scalar edge tensor, ``(n, n)`` or ``(bs, n, n)``.

        Returns
        -------
        GraphData
            Instance with ``E_feat`` set and every other feature field
            ``None``.
        """
        single = edge_scalar.dim() == 2
        if single:
            edge_scalar = edge_scalar.unsqueeze(0)
            if node_mask.dim() == 1:
                node_mask = node_mask.unsqueeze(0)

        if edge_scalar.dim() != 3:
            raise ValueError(
                "from_structure_only() expects a 2D or 3D edge_scalar tensor, "
                f"got shape {tuple(edge_scalar.shape)}"
            )
        if node_mask.dim() != 2:
            raise ValueError(
                "from_structure_only() expects a 1D or 2D node_mask, "
                f"got shape {tuple(node_mask.shape)}"
            )

        bs, n, _ = edge_scalar.shape
        if tuple(node_mask.shape) != (bs, n):
            raise ValueError(
                "from_structure_only(): node_mask shape "
                f"{tuple(node_mask.shape)} incompatible with edge_scalar "
                f"shape {tuple(edge_scalar.shape)}"
            )

        e_feat = edge_scalar.float().unsqueeze(-1)
        y = torch.zeros(bs, 0, device=edge_scalar.device, dtype=e_feat.dtype)

        if single:
            return cls(
                y=y.squeeze(0),
                node_mask=node_mask.squeeze(0),
                E_feat=e_feat.squeeze(0),
            )
        return cls(y=y, node_mask=node_mask, E_feat=e_feat)

    @classmethod
    def from_edge_scalar(
        cls,
        edge_scalar: Tensor,
        *,
        node_mask: Tensor,
        target: Literal["E_class", "E_feat"],
    ) -> GraphData:
        """Construct a graph populating exactly one split edge field from a scalar.

        Parameters
        ----------
        edge_scalar
            Dense scalar edges, ``(n, n)`` or ``(bs, n, n)``. For
            ``target="E_class"`` values are treated as 0/1 indicators of
            edge presence (no hard threshold applied; callers should pass
            already-binary tensors).
        node_mask
            Node-validity mask, ``(n,)`` or ``(bs, n)``.
        target
            Which split field to populate:
            ``"E_feat"`` (single-channel continuous adjacency) or
            ``"E_class"`` (two-channel one-hot, channel 0 = no-edge).

        Returns
        -------
        GraphData
            Instance with the selected split edge field populated and
            the other split fields ``None``.
        """
        if target == "E_feat":
            return cls.from_structure_only(node_mask, edge_scalar)

        single = edge_scalar.dim() == 2
        if single:
            edge_scalar = edge_scalar.unsqueeze(0)
            if node_mask.dim() == 1:
                node_mask = node_mask.unsqueeze(0)

        if edge_scalar.dim() != 3:
            raise ValueError(
                "from_edge_scalar() expects a 2D or 3D edge_scalar tensor, "
                f"got shape {tuple(edge_scalar.shape)}"
            )
        if node_mask.dim() != 2:
            raise ValueError(
                "from_edge_scalar() expects a 1D or 2D node_mask, "
                f"got shape {tuple(node_mask.shape)}"
            )

        bs, n, _ = edge_scalar.shape
        if tuple(node_mask.shape) != (bs, n):
            raise ValueError(
                "from_edge_scalar(): node_mask shape "
                f"{tuple(node_mask.shape)} incompatible with edge_scalar "
                f"shape {tuple(edge_scalar.shape)}"
            )

        adj = edge_scalar.float()
        e_class = torch.zeros(bs, n, n, 2, device=adj.device, dtype=adj.dtype)
        e_class[..., 0] = 1.0 - adj
        e_class[..., 1] = adj

        y = torch.zeros(bs, 0, device=adj.device, dtype=adj.dtype)

        if single:
            return cls(
                y=y.squeeze(0),
                node_mask=node_mask.squeeze(0),
                E_class=e_class.squeeze(0),
            )
        return cls(y=y, node_mask=node_mask, E_class=e_class)

    def to_edge_scalar(self, *, source: Literal["class", "feat"]) -> Tensor:
        """Return a dense scalar adjacency from the requested split edge field.

        Both paths mask the returned tensor by the outer product of
        ``node_mask`` so padded positions are zero.

        Parameters
        ----------
        source
            ``"feat"`` reads ``E_feat`` directly (squeezing a trailing
            single-channel axis if present). ``"class"`` returns the
            edge-probability ``1 − E_class[..., 0]`` for multi-channel
            categorical edges.

        Returns
        -------
        Tensor
            Dense scalar adjacency, shape ``(n, n)`` or ``(bs, n, n)``.

        Raises
        ------
        ValueError
            If the requested source field is ``None``.
        """
        if source == "feat":
            if self.E_feat is None:
                raise ValueError(
                    "to_edge_scalar(source='feat') requires E_feat to be "
                    "populated; got None."
                )
            e = self.E_feat
            edge_scalar = e.squeeze(-1) if e.shape[-1] == 1 else e[..., 0]
        else:  # source == "class"
            if self.E_class is None:
                raise ValueError(
                    "to_edge_scalar(source='class') requires E_class to be "
                    "populated; got None."
                )
            if self.E_class.shape[-1] > 1:
                edge_scalar = 1.0 - self.E_class[..., 0]
            else:
                edge_scalar = self.E_class[..., 0]

        mask2d = self.node_mask.unsqueeze(-1) * self.node_mask.unsqueeze(-2)
        return edge_scalar * mask2d.to(edge_scalar.dtype)

    def binarised_adjacency(self) -> Tensor:
        """Return a hard 0/1 adjacency view from whichever edge field is populated.

        When ``E_class`` is present we take ``argmax > 0`` (two-channel
        DiGress layout treats channel 0 as "no edge"); otherwise we
        threshold ``E_feat`` at 0.5. The result is masked by the outer
        product of ``node_mask`` so padded positions are zero.

        Raises
        ------
        ValueError
            If neither edge field is populated.
        """
        if self.E_class is not None:
            if self.E_class.shape[-1] > 1:
                adj = (self.E_class.argmax(dim=-1) > 0).float()
            else:
                adj = self.E_class[..., 0].clone()
        elif self.E_feat is not None:
            e = self.E_feat
            scalar = e.squeeze(-1) if e.shape[-1] == 1 else e[..., 0]
            adj = (scalar > 0.5).float()
        else:  # pragma: no cover - __post_init__ enforces at least one.
            raise ValueError(
                "binarised_adjacency() requires E_class or E_feat to be populated."
            )

        mask_2d = self.node_mask.unsqueeze(-1) * self.node_mask.unsqueeze(-2)
        return adj * mask_2d.float()

    @classmethod
    def synth_structure_only_x_class(cls, node_mask: Tensor, c_x: int) -> Tensor:
        """Derive a structure-only X_class tensor from node_mask alone.

        Synthesis convention (spec 2026-04-27-x-class-synth-unification):
        - C_x = 1: ones at valid nodes, zeros at padding. Canonical
          structure-only encoding; noise process is identity on X.
        - C_x = 2: [1 - node_ind, node_ind] one-hot (legacy [no-node,
          node]). Kept for backward compat with existing C_x=2 model
          presets.
        - C_x >= 3: raise — real categorical X must be populated by
          the dataset.

        E has no symmetric helper: edges are an adjacency property
        orthogonal to node_mask, so synthesis from node_mask alone is
        undefined for any C_e. ``_read_categorical_e`` raises when
        E_class is None.

        Parameters
        ----------
        node_mask
            Boolean mask of valid nodes, shape ``(B, N)``.
        c_x
            Categorical class width including any structural-filler
            slot. See spec §3 for the regime table.

        Returns
        -------
        Tensor
            Shape ``(B, N, c_x)``, dtype ``float32``, on the same
            device as ``node_mask``.
        """
        node_ind = node_mask.float()
        if c_x == 1:
            return node_ind.unsqueeze(-1)
        if c_x == 2:
            synth = torch.stack([1.0 - node_ind, node_ind], dim=-1)
            # Zero padding rows so loss predicates exclude them
            return synth * node_ind.unsqueeze(-1)
        raise ValueError(
            "GraphData.synth_structure_only_x_class: synthesis is only "
            f"defined for C_x in {{1, 2}}; got C_x={c_x}. C_x>=3 implies "
            "real categorical content; the dataset MUST populate "
            "X_class. See 2026-04-27-x-class-synth-unification-spec §3."
        )

    @classmethod
    def from_pyg_batch(
        cls,
        batch: Batch,
        *,
        n_max_static: int | None = None,
        num_atom_types_x: int | None = None,
        num_bond_types_e: int | None = None,
    ) -> GraphData:
        """Convert a PyG Batch to a dense GraphData.

        Parameters
        ----------
        batch
            PyG Batch from ``Batch.from_data_list()``.
        n_max_static
            Optional static node-count ceiling. When set, the dense
            adjacency and downstream tensors are padded to this width
            so every batch emerges at the same shape (unlocking
            ``torch.compile`` and ``cuda.graph`` capture downstream).
            When ``None`` (default) ``n_max`` is the largest graph in
            the current batch, preserving the legacy variable-shape
            behaviour. ``node_mask`` zeros padded positions either way,
            so numerics on real positions are bit-identical. See
            ``docs/reports/2026-04-28-sync-review/99-synthesis.md`` §6
            for the design rationale.
        num_atom_types_x
            Optional explicit width for the densified ``X_class`` one-hot
            when the input ``batch`` carries a per-graph ``x`` attribute
            of integer atom-class indices (shape ``(sum_n,)``). When
            ``None`` and ``x`` is present, the width is inferred from
            ``int(x.max()) + 1`` — adequate for tests but underspecified
            for production datasets that may not see every class in a
            given batch (callers MUST pass the codec's
            ``vocab.num_atom_types`` to guarantee consistent widths
            across batches; see the molecular collator wiring in
            ``data_modules/molecular/base.py``). When ``x`` is absent
            and this kwarg is ``None``, ``X_class`` is left ``None`` to
            preserve the structure-only path used by SPECTRE-SBM /
            SPECTRE-Planar.
        num_bond_types_e
            Optional explicit width for the densified ``E_class`` one-hot
            when the input ``batch`` carries a per-edge ``edge_attr``
            attribute of integer bond-class indices (shape
            ``(num_edges,)``). When ``None`` and ``edge_attr`` is
            present, the width is inferred from ``int(edge_attr.max()) +
            1`` — adequate for tests but underspecified for production
            datasets that may not see every class in a given batch
            (callers MUST pass the codec's ``vocab.num_bond_types`` to
            guarantee consistent widths across batches; see the
            molecular collator wiring in
            ``data_modules/molecular/base.py``). When ``edge_attr`` is
            absent, the legacy 2-class ``[no-edge, edge]`` densification
            is used (preserves SPECTRE-SBM / SPECTRE-Planar paths).

        Returns
        -------
        GraphData
            Dense batched representation with ``E_class`` always
            populated. When ``edge_attr`` is present in ``batch``, the
            width matches ``num_bond_types_e`` (or the inferred maximum)
            and channel 0 encodes "no edge"; otherwise the legacy
            2-class ``[no-edge, edge]`` layout is used. ``X_class`` is
            populated when the input batch carries a per-graph ``x``
            attribute of integer atom-class indices; otherwise ``None``
            (the spec forbids datasets emitting a degenerate
            "node-present / node-absent" one-hot that merely re-encodes
            ``node_mask`` — see
            ``docs/specs/2026-04-15-unified-graph-features-spec.md
            §"Removed fields"``). ``node_mask`` reflects actual node
            counts per graph; padded positions are marked ``False``.
        """
        import torch.nn.functional as F
        from torch_geometric.utils import remove_self_loops, to_dense_adj

        bs = int(batch.num_graphs)
        edge_index: Tensor = getattr(batch, "edge_index")  # noqa: B009
        batch_vec: Tensor = getattr(batch, "batch")  # noqa: B009
        edge_attr_in: Tensor | None = getattr(batch, "edge_attr", None)
        # Upstream parity (DiGress utils.py:53-62): drop self-loops on the
        # sparse ``edge_index`` BEFORE densification so any per-edge
        # attributes a future dataset carries on diagonal entries are
        # discarded with the edges, rather than zeroed only after a lossy
        # ``to_dense_adj`` accumulation. Pass ``edge_attr`` through the
        # same call so it stays index-aligned with ``edge_index`` after
        # the strip.
        edge_index, edge_attr_in = remove_self_loops(edge_index, edge_attr_in)
        adj = to_dense_adj(edge_index, batch_vec)
        n_packed = adj.shape[1]

        # Static-pad opt-in: when ``n_max_static`` exceeds the current
        # batch's packed n, pad ``adj`` with zeros so downstream
        # tensors have a fixed shape. ``node_mask`` zeros the padded
        # rows/cols so attention/loss/etc. exclude them; real-position
        # numerics are unchanged. Falls back to ``n_packed`` when the
        # ceiling is None or smaller (preserves variable-shape default).
        if n_max_static is not None and n_max_static > n_packed:
            pad = n_max_static - n_packed
            adj = F.pad(adj, (0, pad, 0, pad), value=0.0)
            n_max = n_max_static
        else:
            n_max = n_packed

        # Node mask from batch vector
        node_counts = torch.bincount(batch_vec, minlength=bs)
        arange = torch.arange(n_max, device=adj.device).unsqueeze(0).expand(bs, -1)
        node_mask = arange < node_counts.unsqueeze(1)

        # Symmetrise. Self-loops were already removed pre-densification.
        diag = torch.arange(n_max, device=adj.device)
        adj = (adj + adj.transpose(1, 2)).clamp(max=1.0)

        # Symmetry sanity assert. Runs on CPU worker tensors during
        # collation, so it does NOT block the GPU stream — listed for
        # completeness. Guarded by ``__debug__`` so production runs
        # (Python -O / PYTHONOPTIMIZE=1) skip the O(N²) ``allclose`` per
        # batch on workers. Symmetry is mathematically forced two lines
        # above (``adj + adj.transpose``).
        if __debug__:  # noqa: SIM102 - nested ``if __debug__:`` must stay nested
            if not torch.allclose(adj, adj.transpose(-2, -1)):
                max_asym = (adj - adj.transpose(-2, -1)).abs().max().item()
                raise AssertionError(
                    f"from_pyg_batch produced asymmetric adjacency: "
                    f"shape={tuple(adj.shape)}, max|adj - adj.T|={max_asym:.3e}"
                )

        # One-hot edge features. Two regimes:
        # - ``edge_attr`` absent (SPECTRE-SBM / SPECTRE-Planar): emit the
        #   legacy 2-class ``[no-edge, edge]`` one-hot.
        # - ``edge_attr`` present (molecular path): densify the per-edge
        #   bond class indices into a wider ``(bs, n, n, num_bond_types)``
        #   tensor that mirrors the atom-class ``X_class`` densification.
        #   Channel 0 ("no-edge" / "NONE") is set everywhere by default,
        #   then cleared at positions where a real bond exists before the
        #   bond's class is set. This preserves bond multiplicity through
        #   the dataset → collator boundary instead of collapsing it to
        #   "edge present / absent" — see the diagnosis in
        #   ``docs/reports/2026-04-29-dataset-shims-and-hacks/README.md``
        #   item #3.3.
        if edge_attr_in is None:
            E_class = torch.stack([1.0 - adj, adj], dim=-1)
        else:
            if edge_attr_in.dim() != 1:
                raise ValueError(
                    "from_pyg_batch: batch.edge_attr must be 1D integer "
                    f"bond-class indices of shape (num_edges,); got shape "
                    f"{tuple(edge_attr_in.shape)}."
                )
            if torch.is_floating_point(edge_attr_in):
                raise ValueError(
                    "from_pyg_batch: batch.edge_attr must be an integer "
                    f"dtype (bond-class indices); got dtype "
                    f"{edge_attr_in.dtype}."
                )
            edge_attr_long = edge_attr_in.long()
            if num_bond_types_e is None:
                num_bond_types = int(edge_attr_long.max().item()) + 1
            else:
                num_bond_types = int(num_bond_types_e)
                observed_max = int(edge_attr_long.max().item())
                if observed_max >= num_bond_types:
                    raise ValueError(
                        "from_pyg_batch: observed bond-class index "
                        f"{observed_max} exceeds num_bond_types_e="
                        f"{num_bond_types}."
                    )
            E_class = torch.zeros(
                (bs, n_max, n_max, num_bond_types),
                dtype=adj.dtype,
                device=adj.device,
            )
            # Default to NONE everywhere so positions without a real bond
            # carry the "no-edge" one-hot (parity with the legacy 2-class
            # branch above where ``[1, 0]`` is the no-edge encoding).
            E_class[..., 0] = 1.0
            # Map every retained sparse edge to its (graph, src, dst)
            # triple. ``edge_index`` is global (sums across graphs) so we
            # convert the source row id to a per-graph row id via the
            # cumulative node count, matching the X_class densification
            # below.
            node_counts_for_edge = torch.bincount(batch_vec, minlength=bs)
            cum_counts_e = torch.zeros(bs + 1, dtype=torch.long, device=adj.device)
            cum_counts_e[1:] = torch.cumsum(node_counts_for_edge, dim=0)
            graph_ids = batch_vec[edge_index[0]]
            src_in_graph = edge_index[0] - cum_counts_e[graph_ids]
            dst_in_graph = edge_index[1] - cum_counts_e[graph_ids]
            # Clear the NONE channel where a real bond exists, then set
            # the bond's class. PyG edge_index entries appear in both
            # directions for an undirected edge, so the symmetric
            # assignment is implicit; we additionally enforce symmetry
            # below to defend against directed inputs.
            E_class[graph_ids, src_in_graph, dst_in_graph, 0] = 0.0
            E_class[graph_ids, src_in_graph, dst_in_graph, edge_attr_long] = 1.0
            # Defensive symmetrise: take the elementwise OR (max) with
            # the transposed view so any single-direction entry from a
            # directed input becomes symmetric. The NONE channel is
            # restored at positions that are no-edge in *both* directions
            # because zeros stay zero under max.
            E_class = torch.maximum(E_class, E_class.transpose(1, 2))

        # Upstream parity: zero the diagonal of the target tensor so the
        # ``(true != 0).any(-1)`` row predicate inside the masked CE
        # helpers excludes self-loops automatically. Mirrors upstream
        # DiGress's ``utils.encode_no_edge`` at
        # ``digress-upstream-readonly/src/utils.py:73-74`` (``E[diag] = 0``),
        # which is the single data-layer site where upstream enforces the
        # same invariant. Without this, the diagonal would emit
        # ``[1, 0]`` one-hot "no-edge" targets that survive the predicate
        # and inflate the CE denominator (see
        # ``analysis/digress-loss-check/BUG_REPORT.md`` on 2026-04-21).
        E_class[:, diag, diag, :] = 0.0

        y = torch.zeros(bs, 0, device=adj.device)

        # Optional per-graph node-class densification. Datasets that
        # carry integer atom-class indices on each PyG ``Data.x`` (e.g.
        # the molecular path) get a dense one-hot ``X_class`` here;
        # purely structural datasets (SPECTRE-SBM / SPECTRE-Planar) leave
        # ``batch.x`` unset and we fall through with ``X_class=None``.
        x_attr: Tensor | None = getattr(batch, "x", None)
        X_class: Tensor | None = None
        if x_attr is not None:
            if x_attr.dim() != 1:
                raise ValueError(
                    "from_pyg_batch: batch.x must be 1D integer atom-class "
                    f"indices of shape (sum_n,); got shape {tuple(x_attr.shape)}."
                )
            if not torch.is_floating_point(x_attr):
                idx_long = x_attr.long()
            else:
                raise ValueError(
                    "from_pyg_batch: batch.x must be an integer dtype "
                    f"(atom-class indices); got dtype {x_attr.dtype}. Pass a "
                    "one-hot via a separate field if you need continuous "
                    "node features."
                )
            if num_atom_types_x is None:
                num_atom_types = int(idx_long.max().item()) + 1
            else:
                num_atom_types = int(num_atom_types_x)
                observed_max = int(idx_long.max().item())
                if observed_max >= num_atom_types:
                    raise ValueError(
                        "from_pyg_batch: observed atom-class index "
                        f"{observed_max} exceeds num_atom_types_x="
                        f"{num_atom_types}."
                    )
            # Per-row index within the row's graph: subtract the
            # cumulative node count of all preceding graphs from the
            # global row id. Equivalent to upstream's
            # ``to_dense_batch`` row-position calculation but spelled
            # out to avoid pulling in another PyG helper.
            cum_counts = torch.zeros(bs + 1, dtype=torch.long, device=adj.device)
            cum_counts[1:] = torch.cumsum(node_counts, dim=0)
            pos_in_graph = (
                torch.arange(idx_long.shape[0], device=adj.device)
                - cum_counts[batch_vec]
            )
            X_class = torch.zeros(
                (bs, n_max, num_atom_types),
                dtype=torch.float32,
                device=adj.device,
            )
            X_class[batch_vec, pos_in_graph, idx_long] = 1.0

        return cls(y=y, node_mask=node_mask, E_class=E_class, X_class=X_class)

    def to_pyg(self) -> Data:
        """Convert this (unbatched) GraphData to a PyG Data object.

        Returns
        -------
        torch_geometric.data.Data
            Sparse COO representation with ``edge_index`` and ``num_nodes``.

        Raises
        ------
        ValueError
            If the GraphData has batch size > 1.
        """
        from torch_geometric.data import Data
        from torch_geometric.utils import dense_to_sparse

        adj = self.binarised_adjacency()
        if adj.ndim == 3:
            if adj.shape[0] != 1:
                raise ValueError(
                    f"to_pyg() requires unbatched or batch-size-1 GraphData, "
                    f"got batch size {adj.shape[0]}"
                )
            adj = adj.squeeze(0)

        n = (
            int(self.node_mask.sum().item())
            if self.node_mask.ndim == 1
            else adj.shape[0]
        )
        adj_trimmed = adj[:n, :n]
        edge_index, _ = dense_to_sparse(adj_trimmed)
        return Data(edge_index=edge_index, num_nodes=n)

    @staticmethod
    def collate(graphs: list[GraphData]) -> GraphData:
        """Collate variable-size graphs into a padded batch.

        Pads every populated split field to the maximum node count.
        Padded node positions receive "no-node" (one-hot index 0) in
        categorical fields, padded edge positions receive "no-edge"
        (one-hot index 0), and ``node_mask`` is ``False`` for padded
        slots.

        Parameters
        ----------
        graphs
            Individual (unbatched) ``GraphData`` instances. All graphs
            MUST share the same set of populated split fields.

        Returns
        -------
        GraphData
            Batched instance with shapes ``(bs, n_max, ...)``.
        """
        if not graphs:
            raise ValueError("GraphData.collate() requires a non-empty list.")

        bs = len(graphs)
        ns = [g.node_mask.shape[0] for g in graphs]
        n_max = max(ns)
        dy = graphs[0].y.shape[0]

        mask_batch = torch.zeros(bs, n_max, dtype=torch.bool)
        y_batch = torch.zeros(bs, dy)

        def _pad_field(
            field_name: Literal["X_class", "X_feat", "E_class", "E_feat"],
        ) -> Tensor | None:
            """Pad a single split field from every graph, or return None."""
            first = getattr(graphs[0], field_name)
            if first is None:
                for g in graphs[1:]:
                    if getattr(g, field_name) is not None:
                        raise ValueError(
                            f"GraphData.collate(): {field_name} populated "
                            "inconsistently across graphs."
                        )
                return None

            is_edge = field_name.startswith("E_")
            d = int(first.shape[-1])
            if is_edge:
                batch = torch.zeros(bs, n_max, n_max, d, dtype=first.dtype)
            else:
                batch = torch.zeros(bs, n_max, d, dtype=first.dtype)

            for i, g in enumerate(graphs):
                tensor = getattr(g, field_name)
                if tensor is None:
                    raise ValueError(
                        f"GraphData.collate(): {field_name} missing on graph {i} "
                        "but present on graph 0."
                    )
                ni = ns[i]
                if is_edge:
                    batch[i, :ni, :ni] = tensor
                    # Padded edge positions -> "no edge" (class 0) when
                    # categorical; continuous fields stay at zero.
                    if field_name == "E_class":
                        batch[i, :ni, ni:, 0] = 1.0
                        batch[i, ni:, :, 0] = 1.0
                else:
                    batch[i, :ni] = tensor
                    if field_name == "X_class":
                        batch[i, ni:, 0] = 1.0
            return batch

        for i, (g, ni) in enumerate(zip(graphs, ns, strict=False)):
            if dy > 0:
                y_batch[i] = g.y
            mask_batch[i, :ni] = True

        X_class = _pad_field("X_class")
        X_feat = _pad_field("X_feat")
        E_class = _pad_field("E_class")
        E_feat = _pad_field("E_feat")

        return GraphData(
            y=y_batch,
            node_mask=mask_batch,
            X_class=X_class,
            X_feat=X_feat,
            E_class=E_class,
            E_feat=E_feat,
        )


def collapse_to_indices(data: GraphData) -> tuple[Tensor, Tensor | None]:
    """Argmax ``E_class`` (and optionally ``X_class``) to class indices.

    Thin tuple-returning alias for ``data.mask(collapse=True)``, kept so
    external callers that pre-date the merge keep working. The canonical
    form is ``GraphData.mask(collapse=True)`` (mirrors upstream's
    ``PlaceHolder.mask(node_mask, collapse=True)``).

    Parameters
    ----------
    data
        One-hot encoded graph features with ``node_mask``. ``E_class``
        MUST be populated.

    Returns
    -------
    (E_idx, X_idx)
        ``E_idx`` has shape ``(bs, n, n)``; ``X_idx`` has shape
        ``(bs, n)`` or is ``None`` when ``data.X_class`` is ``None``.

    Raises
    ------
    ValueError
        If ``data.E_class`` is ``None``.
    """
    if data.E_class is None:
        raise ValueError("collapse_to_indices() requires data.E_class to be populated.")

    collapsed = data.mask(collapse=True)
    assert collapsed.E_class is not None  # E_class non-None already guarded above
    return collapsed.E_class, collapsed.X_class


@dataclass(frozen=True, slots=True)
class GraphStructure:
    """Pre-computed structural features of a graph's topology.

    Derived from the categorical edge features of a ``GraphData`` instance
    before any learned transformations. These tensors remain constant
    across transformer layer iterations — they describe the input graph's
    structure, not the evolving hidden state.

    Parameters
    ----------
    adjacency
        Binary adjacency matrix, shape ``(bs, n, n)``. Populated when
        GNN projections need it; ``None`` otherwise.
    eigenvectors
        Top-k eigenvectors of the adjacency, shape ``(bs, n, k)``.
        Populated for spectral projections; ``None`` otherwise.
    eigenvalues
        Corresponding eigenvalues, shape ``(bs, k)``.
    """

    adjacency: Tensor | None = None
    eigenvectors: Tensor | None = None
    eigenvalues: Tensor | None = None
