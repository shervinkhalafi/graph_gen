"""Unified graph generation evaluator combining MMD and structural metrics.

This module provides a single ``GraphEvaluator`` that computes all graph
generation evaluation metrics in one pass:

- **Distributional MMD** (degree, clustering, spectral) via the existing
  ``compute_mmd_metrics`` infrastructure.
- **Orbit MMD** via ORCA graphlet counting (requires compiled ``orca`` binary).
- **SBM accuracy** via graph-tool community detection (requires ``graph-tool``).
- **Planarity accuracy** via networkx planarity check.
- **Uniqueness** as the fraction of distinct isomorphism classes among
  generated graphs.
- **Novelty** as the fraction of generated graphs not isomorphic to any
  training graph (requires training graphs via the constructor).

External dependencies (graph-tool, orca) are checked at import time.
When unavailable, the corresponding metric fields are set to ``None``; a
warning is emitted once at import for each missing dependency. When available,
failures propagate without suppression.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import logging
import multiprocessing
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

import networkx as nx
import numpy as np
import torch
from scipy.stats import chi2
from torch import Tensor

from tmgg.evaluation.mmd_metrics import (
    compute_mmd,
    compute_mmd_metrics,
)
from tmgg.evaluation.orca import (
    is_available as _orca_is_available,
)
from tmgg.evaluation.orca import run_orca

if TYPE_CHECKING:
    from tmgg.data.datasets.graph_types import GraphData

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency availability
# ---------------------------------------------------------------------------
_GRAPH_TOOL_IMPORT_ERROR: ImportError | None = None
gt: Any | None = None
if importlib.util.find_spec("graph_tool") is not None:
    try:
        import graph_tool.all as gt  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
    except ImportError as exc:
        _GRAPH_TOOL_IMPORT_ERROR = exc
        _GRAPH_TOOL_AVAILABLE = False
        warnings.warn(
            "graph-tool import failed during evaluator import; "
            "sbm_accuracy will be None in GraphEvaluator results.\n"
            f"Import error: {exc}",
            stacklevel=2,
        )
    else:
        _GRAPH_TOOL_AVAILABLE = True
else:
    _GRAPH_TOOL_AVAILABLE = False
_ORCA_AVAILABLE = _orca_is_available()

if not _GRAPH_TOOL_AVAILABLE and _GRAPH_TOOL_IMPORT_ERROR is None:
    warnings.warn(
        "graph-tool not installed; sbm_accuracy will be None in GraphEvaluator results.",
        stacklevel=2,
    )
if not _ORCA_AVAILABLE:
    warnings.warn(
        "orca binary not found; orbit_mmd will be None in GraphEvaluator results. "
        "Compile orca.cpp and set ORCA_PATH or place the binary "
        "in the expected location.",
        stacklevel=2,
    )


# ---------------------------------------------------------------------------
# EvaluationResults
# ---------------------------------------------------------------------------
@dataclass
class EvaluationResults:
    """Flat container for all graph generation evaluation metrics.

    Attributes
    ----------
    degree_mmd
        MMD on degree distributions (always computed).
    clustering_mmd
        MMD on clustering coefficient distributions (always computed).
    spectral_mmd
        MMD on normalized Laplacian eigenvalue distributions (always computed).
    orbit_mmd
        MMD on ORCA orbit counts. None if orca binary is unavailable.
    sbm_accuracy
        Fraction of generated graphs consistent with an SBM. None if
        graph-tool is unavailable.
    planarity_accuracy
        Fraction of generated graphs that are connected and planar.
        None if skipped via ``skip_metrics``.
    uniqueness
        Fraction of generated graphs in distinct isomorphism classes.
        None if skipped via ``skip_metrics``.
    novelty
        Fraction of generated graphs not isomorphic to any training graph.
        None if train_graphs were not provided in the constructor or skipped.
    """

    degree_mmd: float
    clustering_mmd: float
    spectral_mmd: float
    orbit_mmd: float | None
    sbm_accuracy: float | None
    planarity_accuracy: float | None
    uniqueness: float | None
    novelty: float | None

    def to_dict(self) -> dict[str, float | None]:
        """Return a flat dictionary of all metrics, suitable for logging."""
        return {
            "degree_mmd": self.degree_mmd,
            "clustering_mmd": self.clustering_mmd,
            "spectral_mmd": self.spectral_mmd,
            "orbit_mmd": self.orbit_mmd,
            "sbm_accuracy": self.sbm_accuracy,
            "planarity_accuracy": self.planarity_accuracy,
            "uniqueness": self.uniqueness,
            "novelty": self.novelty,
        }


# ---------------------------------------------------------------------------
# Structural metric helpers (ported from DiGress spectre_utils)
# ---------------------------------------------------------------------------


def _orca_normalized_counts(graph: nx.Graph[Any]) -> np.ndarray:
    """Compute normalized ORCA orbit counts for a graph.

    Removes self-loops, runs ORCA, and returns the mean orbit count vector
    (averaged over nodes). Graphs with fewer than 2 nodes are skipped by
    callers before reaching this function.
    """
    g_clean = graph.copy()
    g_clean.remove_edges_from(nx.selfloop_edges(g_clean))
    counts = run_orca(g_clean)
    return np.sum(counts, axis=0) / g_clean.number_of_nodes()


def compute_orbit_mmd(
    ref_graphs: list[nx.Graph[Any]],
    gen_graphs: list[nx.Graph[Any]],
    kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    sigma: float = 30.0,
) -> float:
    """Compute MMD on normalized ORCA orbit count vectors.

    Parameters
    ----------
    ref_graphs
        Reference graphs.
    gen_graphs
        Generated graphs.
    kernel
        Kernel for MMD computation.
    sigma
        Kernel bandwidth (default 30.0, matching DiGress).

    Returns
    -------
    float
        Orbit MMD value.
    """
    ref_counts: list[np.ndarray] = [
        _orca_normalized_counts(g) for g in ref_graphs if g.number_of_nodes() >= 2
    ]
    gen_counts: list[np.ndarray] = [
        _orca_normalized_counts(g) for g in gen_graphs if g.number_of_nodes() >= 2
    ]

    if not ref_counts or not gen_counts:
        return 0.0

    # Orbit counts are raw graphlet counts, not histograms — magnitude is the
    # signal. Upstream DiGress's ``orbit_stats_all`` calls compute_mmd with
    # is_hist=False, sigma=30.0 (digress-upstream-readonly/src/analysis/
    # spectre_utils.py:490). Matching that here.
    return compute_mmd(
        ref_counts, gen_counts, kernel=kernel, sigma=sigma, is_hist=False
    )


# Ported from DiGress (Vignac et al., ICLR 2023) / SPECTRE (Martinkus et al.,
# NeurIPS 2022).  Upstream: cvignac/DiGress, src/analysis/spectre_utils.py,
# functions ``eval_acc_sbm_graph`` and ``is_sbm_graph``.


def _is_sbm_graph(
    g_nx: nx.Graph[Any],
    p_intra: float = 0.3,
    p_inter: float = 0.005,
    strict: bool = True,
    refinement_steps: int = 1000,
) -> float:
    """Test whether a single graph is consistent with an SBM.

    Fits a stochastic block model via graph-tool's ``minimize_blockmodel_dl``,
    refines via merge-split MCMC, then applies a Wald test comparing recovered
    intra-/inter-block edge probabilities against expected values.

    Parameters
    ----------
    g_nx
        Graph to evaluate.
    p_intra
        Expected within-community edge probability.
    p_inter
        Expected between-community edge probability.
    strict
        If True, reject graphs with extreme block counts or sizes and
        return a binary pass/fail. If False, return the raw mean p-value.
    refinement_steps
        Number of MCMC refinement sweeps.

    Returns
    -------
    float
        ``1.0`` / ``0.0`` when *strict*, raw mean p-value otherwise.
    """
    if gt is None:
        raise RuntimeError("graph-tool is unavailable; cannot compute SBM accuracy.")

    adj = nx.adjacency_matrix(g_nx).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))

    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        # Graph too degenerate for blockmodel inference; scores 0.
        return 0.0 if not strict else 0.0

    for _ in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]

    if strict and (
        (node_counts > 40).sum() > 0
        or (node_counts < 20).sum() > 0
        or n_blocks > 5
        or n_blocks < 2
    ):
        return 0.0

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p_val = 1 - chi2.cdf(abs(np.asarray(W)), 1)
    mean_p = float(p_val.mean())

    if strict:
        return float(mean_p > 0.9)
    return mean_p


def compute_sbm_accuracy(
    graphs: list[nx.Graph[Any]],
    p_intra: float = 0.3,
    p_inter: float = 0.005,
    strict: bool = True,
    refinement_steps: int = 1000,
    is_parallel: bool = True,
) -> float:
    """Fraction of graphs consistent with a stochastic block model.

    Evaluates each graph via ``_is_sbm_graph`` (blockmodel inference +
    Wald test), optionally in parallel. When ``is_parallel=True`` the
    per-graph calls run inside a ``ProcessPoolExecutor`` that uses the
    ``spawn`` start method. A thread-pool backend was tried initially
    but graph-tool's blockmodel routines are not safe to call
    concurrently from Python threads: the workers share graph-tool's
    internal OpenMP/C++ state and abort the process with heap-
    corruption signals (``vector::_M_fill_insert``, ``malloc()
    unaligned tcache chunk``). Process isolation avoids the hazard at
    the cost of interpreter-spawn overhead. See
    ``docs/reports/2026-04-15-bug-modal-sigabrt.md``.

    Parameters
    ----------
    graphs
        Generated graphs to evaluate.
    p_intra
        Expected within-community edge probability.
    p_inter
        Expected between-community edge probability.
    strict
        If True, reject graphs with extreme block counts or sizes.
    refinement_steps
        Number of MCMC refinement sweeps.
    is_parallel
        If True, evaluate graphs in a spawn-based
        ``ProcessPoolExecutor``. If False, evaluate sequentially in
        the caller's process.

    Returns
    -------
    float
        Fraction of graphs passing the SBM test (strict) or mean p-value.

    Raises
    ------
    ImportError
        If graph-tool is not installed.
    """
    if not graphs:
        return 0.0

    if not is_parallel:
        count = sum(
            _is_sbm_graph(g, p_intra, p_inter, strict, refinement_steps) for g in graphs
        )
        return count / len(graphs)

    mp_context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(mp_context=mp_context) as executor:
        scores = list(
            executor.map(
                _is_sbm_graph,
                graphs,
                [p_intra] * len(graphs),
                [p_inter] * len(graphs),
                [strict] * len(graphs),
                [refinement_steps] * len(graphs),
            )
        )
    count = sum(scores)

    return count / len(graphs)


def compute_planarity_accuracy(graphs: list[nx.Graph[Any]]) -> float:
    """Fraction of generated graphs that are both connected and planar.

    Parameters
    ----------
    graphs
        Generated graphs.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if not graphs:
        return 0.0
    count = sum(
        1
        for g in graphs
        if g.number_of_nodes() > 0 and nx.is_connected(g) and nx.check_planarity(g)[0]
    )
    return count / len(graphs)


def compute_uniqueness(graphs: list[nx.Graph[Any]]) -> float:
    """Fraction of generated graphs in distinct isomorphism classes.

    Uses ``nx.faster_could_be_isomorphic`` as a fast filter followed by
    ``nx.is_isomorphic`` for confirmation, matching DiGress's approach.

    Parameters
    ----------
    graphs
        Generated graphs.

    Returns
    -------
    float
        Fraction in [0, 1].
    """
    if not graphs:
        return 0.0

    count_non_unique = 0
    evaluated: list[nx.Graph[Any]] = []

    for g in graphs:
        if g.number_of_nodes() == 0:
            continue
        is_unique = True
        for g_old in evaluated:
            if nx.faster_could_be_isomorphic(g, g_old) and nx.is_isomorphic(g, g_old):
                count_non_unique += 1
                is_unique = False
                break
        if is_unique:
            evaluated.append(g)

    return (len(graphs) - count_non_unique) / len(graphs)


def compute_novelty(
    generated: list[nx.Graph[Any]], train_graphs: list[nx.Graph[Any]]
) -> float:
    """Fraction of generated graphs not isomorphic to any training graph.

    Parameters
    ----------
    generated
        Generated graphs.
    train_graphs
        Training set graphs.

    Returns
    -------
    float
        Fraction in [0, 1]. 1.0 means all generated graphs are novel.
    """
    if not generated:
        return 0.0

    count_isomorphic = 0
    for gen_g in generated:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(gen_g, train_g) and nx.is_isomorphic(
                gen_g, train_g
            ):
                count_isomorphic += 1
                break

    return 1.0 - count_isomorphic / len(generated)


# ---------------------------------------------------------------------------
# GraphEvaluator
# ---------------------------------------------------------------------------
class GraphEvaluator:
    """Unified evaluator for graph generation quality.

    Accepts reference and generated graphs directly in ``evaluate()``, and
    optionally training graphs in the constructor for novelty computation.
    Computes all standard metrics: degree/clustering/spectral MMD, orbit
    MMD, SBM accuracy, planarity, uniqueness, and novelty.

    Parameters
    ----------
    eval_num_samples
        Maximum number of reference and generated graphs to evaluate.
        Inputs are truncated to this length inside ``evaluate()``.
    kernel
        Kernel for MMD computation (``"gaussian_tv"`` or ``"gaussian"``).
    sigma
        Fallback bandwidth for the MMD kernel when no per-metric override
        is supplied.
    p_intra
        Expected within-community edge probability for SBM accuracy.
    p_inter
        Expected between-community edge probability for SBM accuracy.
    skip_metrics
        Metric names to skip (``"orbit"``, ``"sbm"``, ``"planarity"``,
        ``"uniqueness"``, ``"novelty"``). Skipped metrics return ``None``
        in :class:`EvaluationResults`.
    train_graphs
        Training set graphs used for novelty evaluation. If None, novelty
        is not computed.
    degree_sigma, clustering_sigma, spectral_sigma
        Per-metric kernel bandwidths. Upstream DiGress ships
        ``clustering_sigma=0.1`` with the others at 1.0 for
        ``gaussian_tv``; parity runs should use those values. Each
        defaults to ``None``, which falls back to ``sigma``.
    binarise_threshold
        Threshold applied to ``E_feat[..., 0]`` when deriving a binary
        adjacency from a continuous-edge ``GraphData``. Defaults to
        ``0.5``. Ignored for the categorical path, which derives edges
        from ``argmax(E_class, dim=-1) != 0``. Per
        ``docs/specs/2026-04-15-unified-graph-features-spec.md``
        §"Evaluator contract".
    disagreement_warn_threshold
        Mean-disagreement rate above which a single warning per
        evaluation pass is emitted when both ``E_class`` and ``E_feat``
        are populated. Defaults to ``0.05``. See the spec section
        referenced above.
    sbm_refinement_steps
        Number of multiflip MCMC sweeps used to refine the block-model
        fit inside :func:`compute_sbm_accuracy`. Default 100 matches
        upstream DiGress's live ``SpectreSamplingMetrics`` value
        (``src/analysis/spectre_utils.py:830``). The underlying function
        defaults to 1000, but upstream never invokes it at that value —
        they always override to 100. Setting higher gives tighter
        block-model fits but is roughly 10× slower.

    Examples
    --------
    >>> evaluator = GraphEvaluator(eval_num_samples=100, train_graphs=train_set)
    >>> results = evaluator.evaluate(refs=validation_graphs, generated=generated_graphs)
    >>> print(results.to_dict())
    """

    def __init__(
        self,
        eval_num_samples: int,
        kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
        sigma: float = 1.0,
        p_intra: float = 0.3,
        p_inter: float = 0.005,
        skip_metrics: set[str] | frozenset[str] | list[str] | None = None,
        train_graphs: list[nx.Graph[Any]] | None = None,
        degree_sigma: float | None = None,
        clustering_sigma: float | None = None,
        spectral_sigma: float | None = None,
        binarise_threshold: float = 0.5,
        disagreement_warn_threshold: float = 0.05,
        sbm_refinement_steps: int = 100,
    ) -> None:
        self.eval_num_samples = eval_num_samples
        self.kernel: Literal["gaussian", "gaussian_tv"] = kernel
        self.sigma = sigma
        self.p_intra = p_intra
        self.p_inter = p_inter
        self.skip_metrics: frozenset[str] = frozenset(skip_metrics or ())
        self.train_graphs = list(train_graphs) if train_graphs is not None else []
        self._train_graphs_set = train_graphs is not None
        self.degree_sigma: float | None = degree_sigma
        self.clustering_sigma: float | None = clustering_sigma
        self.spectral_sigma: float | None = spectral_sigma
        self.binarise_threshold: float = binarise_threshold
        self.disagreement_warn_threshold: float = disagreement_warn_threshold
        self.sbm_refinement_steps: int = sbm_refinement_steps
        # Reset at the start of every ``evaluate()`` / ``to_networkx_graphs``
        # call so each validation pass emits at most one warning.
        self._disagreement_warned_this_pass: bool = False

    # ------------------------------------------------------------------
    # GraphData → binary adjacency helpers
    # ------------------------------------------------------------------
    def _graphdata_to_binary_adj(self, data: GraphData) -> Tensor:
        """Derive a binary adjacency tensor from a :class:`GraphData`.

        Follows the priority rule in
        ``docs/specs/2026-04-15-unified-graph-features-spec.md``
        §"Evaluator contract": ``E_class`` wins over ``E_feat`` when
        both are populated. ``E_class`` is argmax'd over the class
        channel, mapping any non-zero class to ``1``; ``E_feat`` is
        thresholded at :attr:`binarise_threshold`. Padded positions
        (per ``data.node_mask``) are always zeroed so the result is
        invariant to padding artefacts.

        Parameters
        ----------
        data
            Batched or single-graph ``GraphData``. At least one of
            ``E_class`` / ``E_feat`` must be non-``None``.

        Returns
        -------
        torch.Tensor
            Binary adjacency with the same shape as the source edge
            field sans the channel axis (``(bs, n, n)`` or ``(n, n)``).
            Dtype is :class:`torch.float32`.

        Raises
        ------
        ValueError
            When both ``E_class`` and ``E_feat`` are ``None``. Wave 9
            tightens the constructor invariant so this branch becomes
            unreachable; until then the message points callers at the
            spec.
        """
        dtype = torch.float32
        if data.E_class is not None:
            adj = (data.E_class.argmax(dim=-1) != 0).to(dtype)
        elif data.E_feat is not None:
            e_feat = data.E_feat
            if e_feat.dim() == data.node_mask.dim() + 1:
                # Shape ``(..., n, n)`` without a trailing channel axis.
                scalar = e_feat
            else:
                scalar = e_feat[..., 0]
            adj = (scalar > self.binarise_threshold).to(dtype)
        else:
            raise ValueError(
                "GraphEvaluator._graphdata_to_binary_adj requires at least "
                "one of data.E_class or data.E_feat to be non-None; got both "
                "None. See docs/specs/2026-04-15-unified-graph-features-spec.md "
                '§"Evaluator contract".'
            )

        node_mask = data.node_mask.to(dtype)
        if node_mask.dim() == 1:
            mask_2d = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        else:
            mask_2d = node_mask.unsqueeze(-1) * node_mask.unsqueeze(-2)
        return adj * mask_2d

    def _check_field_disagreement(self, data: GraphData) -> float | None:
        """Return the mean per-entry disagreement between ``E_class`` and ``E_feat``.

        When either field is ``None`` there is nothing to compare and
        the method returns ``None``. Otherwise it computes, restricted
        to the real (non-padded) entries defined by ``data.node_mask``,

        .. math::

            \\text{disagreement} = \\operatorname{mean}\\bigl[
                (\\operatorname*{argmax}_{c} E_{\\text{class}} \\neq 0)
                \\neq (E_{\\text{feat},0} > \\tau)
            \\bigr]

        where :math:`\\tau` is :attr:`binarise_threshold`. The return
        value is the scalar average over all real edge positions
        across the batch (not a per-graph mean) and lives in
        ``[0, 1]``.

        Parameters
        ----------
        data
            ``GraphData`` to inspect.

        Returns
        -------
        float or None
            ``None`` when at least one of the two fields is absent;
            otherwise a Python float in ``[0, 1]``.
        """
        if data.E_class is None or data.E_feat is None:
            return None

        class_edges = data.E_class.argmax(dim=-1) != 0
        e_feat = data.E_feat
        if e_feat.dim() == data.node_mask.dim() + 1:
            feat_scalar = e_feat
        else:
            feat_scalar = e_feat[..., 0]
        feat_edges = feat_scalar > self.binarise_threshold

        node_mask = data.node_mask.bool()
        mask = node_mask.unsqueeze(-1) & node_mask.unsqueeze(-2)

        denom = mask.sum().clamp(min=1).float()
        disagreement = ((class_edges != feat_edges) & mask).sum().float() / denom
        return float(disagreement.item())

    def to_networkx_graphs(
        self, graph_data_list: list[GraphData]
    ) -> list[nx.Graph[Any]]:
        """Convert a list of :class:`GraphData` to NetworkX graphs.

        Resets the per-pass disagreement warning flag, then loops over
        the supplied ``GraphData`` instances and feeds each through
        :meth:`_graphdata_to_binary_adj`. While iterating the method
        also accumulates the mean field-disagreement via
        :meth:`_check_field_disagreement` and emits exactly one
        ``logging.warning`` when the batch-averaged disagreement
        exceeds :attr:`disagreement_warn_threshold`. The warning fires
        only if both ``E_class`` and ``E_feat`` are populated on at
        least one sample in ``graph_data_list``.

        Parameters
        ----------
        graph_data_list
            Generated graphs in the unified split-field format.

        Returns
        -------
        list[networkx.Graph]
            One simple graph per input ``GraphData``.
        """
        self._disagreement_warned_this_pass = False

        nx_graphs: list[nx.Graph[Any]] = []
        disagreements: list[float] = []
        for data in graph_data_list:
            adj = self._graphdata_to_binary_adj(data)
            if adj.ndim == 3:
                adj = adj[0]
            a_np = adj.detach().cpu().numpy()
            nx_graphs.append(nx.from_numpy_array(a_np))

            rate = self._check_field_disagreement(data)
            if rate is not None:
                disagreements.append(rate)

        if disagreements:
            mean_rate = float(np.mean(disagreements))
            if (
                mean_rate > self.disagreement_warn_threshold
                and not self._disagreement_warned_this_pass
            ):
                logger.warning(
                    "GraphEvaluator: E_class and E_feat disagree on edge "
                    "presence at mean rate %.4f (> threshold %.4f). Using "
                    "E_class per docs/specs/2026-04-15-unified-graph-features-spec.md "
                    '§"Evaluator contract".',
                    mean_rate,
                    self.disagreement_warn_threshold,
                )
                self._disagreement_warned_this_pass = True

        return nx_graphs

    def evaluate(
        self, refs: list[nx.Graph[Any]], generated: list[nx.Graph[Any]]
    ) -> EvaluationResults | None:
        """Compute all evaluation metrics.

        Returns ``None`` if fewer than 2 reference graphs or fewer than 2
        generated graphs remain after truncation (MMD needs at least 2
        samples per distribution).

        Both ``refs`` and ``generated`` are truncated to ``eval_num_samples``
        before metric computation, so callers may pass all available graphs
        without pre-filtering.

        Parameters
        ----------
        refs
            Reference graphs (e.g. validation set).
        generated
            Generated NetworkX graphs.

        Returns
        -------
        EvaluationResults or None
            All metrics, or None if insufficient graphs after truncation.
        """
        self._disagreement_warned_this_pass = False

        refs = refs[: self.eval_num_samples]
        generated = generated[: self.eval_num_samples]

        if len(refs) < 2:
            return None

        if len(generated) < 2:
            return None

        # --- Core MMD metrics (always available) ---
        mmd_results = compute_mmd_metrics(
            refs,
            generated,
            kernel=self.kernel,
            sigma=self.sigma,
            degree_sigma=self.degree_sigma,
            clustering_sigma=self.clustering_sigma,
            spectral_sigma=self.spectral_sigma,
        )

        # --- Orbit MMD (requires orca, skippable) ---
        orbit_mmd: float | None = None
        if "orbit" not in self.skip_metrics and _ORCA_AVAILABLE:
            orbit_mmd = compute_orbit_mmd(refs, generated, kernel=self.kernel)

        # --- SBM accuracy (requires graph-tool, skippable) ---
        sbm_accuracy: float | None = None
        if "sbm" not in self.skip_metrics and _GRAPH_TOOL_AVAILABLE:
            sbm_accuracy = compute_sbm_accuracy(
                generated,
                p_intra=self.p_intra,
                p_inter=self.p_inter,
                refinement_steps=self.sbm_refinement_steps,
            )

        # --- Planarity (skippable) ---
        planarity_accuracy: float | None = None
        if "planarity" not in self.skip_metrics:
            planarity_accuracy = compute_planarity_accuracy(generated)

        # --- Uniqueness (skippable) ---
        uniqueness: float | None = None
        if "uniqueness" not in self.skip_metrics:
            uniqueness = compute_uniqueness(generated)

        # --- Novelty (needs train_graphs from constructor, skippable) ---
        novelty: float | None = None
        if "novelty" not in self.skip_metrics and self._train_graphs_set:
            novelty = compute_novelty(generated, self.train_graphs)

        return EvaluationResults(
            degree_mmd=mmd_results.degree_mmd,
            clustering_mmd=mmd_results.clustering_mmd,
            spectral_mmd=mmd_results.spectral_mmd,
            orbit_mmd=orbit_mmd,
            sbm_accuracy=sbm_accuracy,
            planarity_accuracy=planarity_accuracy,
            uniqueness=uniqueness,
            novelty=novelty,
        )
