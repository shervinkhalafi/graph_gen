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
  training graph (requires training graphs via ``setup()``).

External dependencies (graph-tool, orca) are checked at construction time.
When unavailable, the corresponding metric fields are set to ``None``; a
warning is emitted once per session for each missing dependency. When available,
failures propagate without suppression.
"""

from __future__ import annotations

import importlib.util
import logging
import warnings
from dataclasses import dataclass
from typing import Any, Literal

import networkx as nx
import numpy as np

from tmgg.experiments._shared_utils.evaluation_metrics.mmd_metrics import (
    compute_mmd,
    compute_mmd_metrics,
)
from tmgg.experiments._shared_utils.evaluation_metrics.orca import (
    is_available as _orca_is_available,
)
from tmgg.experiments._shared_utils.evaluation_metrics.orca import run_orca

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dependency availability
# ---------------------------------------------------------------------------
_GRAPH_TOOL_AVAILABLE = importlib.util.find_spec("graph_tool") is not None


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
        None if train_graphs were not provided via ``setup()`` or skipped.
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

    return compute_mmd(ref_counts, gen_counts, kernel=kernel, sigma=sigma)


def compute_sbm_accuracy(
    graphs: list[nx.Graph[Any]],
    p_intra: float = 0.3,
    p_inter: float = 0.005,
    strict: bool = True,
    refinement_steps: int = 100,
) -> float:
    """Evaluate fraction of graphs consistent with a stochastic block model.

    Uses graph-tool's minimize_blockmodel_dl to fit an SBM, then applies
    a Wald test to compare recovered parameters against the expected values.

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

    Returns
    -------
    float
        Fraction of graphs passing the SBM test.

    Raises
    ------
    ImportError
        If graph-tool is not installed.
    """
    import graph_tool.all as gt  # type: ignore[import-untyped]  # pyright: ignore[reportMissingImports]
    from scipy.stats import chi2

    count = 0.0
    for g_nx in graphs:
        adj = nx.adjacency_matrix(g_nx).toarray()
        idx = adj.nonzero()
        g = gt.Graph()
        g.add_edge_list(np.transpose(idx))
        try:
            state = gt.minimize_blockmodel_dl(g)
        except ValueError:
            if strict:
                continue
            else:
                continue  # skip graph

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
            continue

        max_intra_edges = node_counts * (node_counts - 1)
        est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

        max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
        np.fill_diagonal(edge_counts, 0)
        est_p_inter_vals = edge_counts / (max_inter_edges + 1e-6)

        W_p_intra = (est_p_intra - p_intra) ** 2 / (
            est_p_intra * (1 - est_p_intra) + 1e-6
        )
        W_p_inter = (est_p_inter_vals - p_inter) ** 2 / (
            est_p_inter_vals * (1 - est_p_inter_vals) + 1e-6
        )

        W = W_p_inter.copy()
        np.fill_diagonal(W, W_p_intra)
        p_val = 1 - chi2.cdf(abs(np.asarray(W)), 1)
        mean_p = float(p_val.mean())

        if strict:
            count += float(mean_p > 0.9)
        else:
            count += mean_p

    return count / max(len(graphs), 1)


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

    Combines the accumulate-then-evaluate lifecycle (for reference graphs) with
    computation of all standard metrics: degree/clustering/spectral MMD, orbit
    MMD, SBM accuracy, planarity, uniqueness, and novelty.

    Parameters
    ----------
    eval_num_samples
        Maximum number of reference graphs to accumulate.
    kernel
        Kernel for MMD computation (``"gaussian_tv"`` or ``"gaussian"``).
    sigma
        Bandwidth for the MMD kernel.
    p_intra
        Expected within-community edge probability for SBM accuracy.
    p_inter
        Expected between-community edge probability for SBM accuracy.
    skip_metrics
        Metric names to skip (``"orbit"``, ``"sbm"``, ``"planarity"``,
        ``"uniqueness"``, ``"novelty"``). Skipped metrics return ``None``
        in :class:`EvaluationResults`.

    Examples
    --------
    >>> evaluator = GraphEvaluator(eval_num_samples=100)
    >>> evaluator.setup(train_graphs=train_set)
    >>> for g in validation_graphs:
    ...     evaluator.accumulate(g)
    >>> results = evaluator.evaluate(generated_graphs)
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
    ) -> None:
        self.eval_num_samples = eval_num_samples
        self.kernel: Literal["gaussian", "gaussian_tv"] = kernel
        self.sigma = sigma
        self.p_intra = p_intra
        self.p_inter = p_inter
        self.skip_metrics: frozenset[str] = frozenset(skip_metrics or ())

        self.ref_graphs: list[nx.Graph[Any]] = []
        self.train_graphs: list[nx.Graph[Any]] = []
        self._train_graphs_set = False

        # Warn once about missing optional dependencies
        if not _GRAPH_TOOL_AVAILABLE and "sbm" not in self.skip_metrics:
            warnings.warn(
                "graph-tool not installed; sbm_accuracy will be None in results.",
                stacklevel=2,
            )
        if not _orca_is_available() and "orbit" not in self.skip_metrics:
            warnings.warn(
                "orca binary not found; orbit_mmd will be None in results. "
                "Compile orca.cpp and set ORCA_PATH or place the binary "
                "in the expected location.",
                stacklevel=2,
            )

    def setup(self, train_graphs: list[nx.Graph[Any]]) -> None:
        """Store training graphs for novelty computation.

        Parameters
        ----------
        train_graphs
            The training set graphs used for novelty evaluation.
        """
        self.train_graphs = list(train_graphs)
        self._train_graphs_set = True

    def accumulate(self, graph: nx.Graph[Any]) -> None:
        """Add a reference graph (up to ``eval_num_samples``).

        Parameters
        ----------
        graph
            A reference graph to use for MMD computation.
        """
        if len(self.ref_graphs) < self.eval_num_samples:
            self.ref_graphs.append(graph)

    def clear(self) -> None:
        """Reset accumulated reference graphs.

        Training graphs (from ``setup()``) are preserved.
        """
        self.ref_graphs = []

    def evaluate(self, generated: list[nx.Graph[Any]]) -> EvaluationResults | None:
        """Compute all evaluation metrics.

        Returns ``None`` if fewer than 2 reference graphs have been
        accumulated (MMD needs at least 2 samples per distribution).

        Parameters
        ----------
        generated
            List of generated NetworkX graphs.

        Returns
        -------
        EvaluationResults or None
            All metrics, or None if insufficient reference graphs.
        """
        if len(self.ref_graphs) < 2:
            return None

        if len(generated) < 2:
            return None

        # --- Core MMD metrics (always available) ---
        mmd_results = compute_mmd_metrics(
            self.ref_graphs, generated, kernel=self.kernel, sigma=self.sigma
        )

        # --- Orbit MMD (requires orca, skippable) ---
        orbit_mmd: float | None = None
        if "orbit" not in self.skip_metrics and _orca_is_available():
            orbit_mmd = compute_orbit_mmd(
                self.ref_graphs, generated, kernel=self.kernel
            )

        # --- SBM accuracy (requires graph-tool, skippable) ---
        sbm_accuracy: float | None = None
        if "sbm" not in self.skip_metrics and _GRAPH_TOOL_AVAILABLE:
            sbm_accuracy = compute_sbm_accuracy(
                generated,
                p_intra=self.p_intra,
                p_inter=self.p_inter,
            )

        # --- Planarity (skippable) ---
        planarity_accuracy: float | None = None
        if "planarity" not in self.skip_metrics:
            planarity_accuracy = compute_planarity_accuracy(generated)

        # --- Uniqueness (skippable) ---
        uniqueness: float | None = None
        if "uniqueness" not in self.skip_metrics:
            uniqueness = compute_uniqueness(generated)

        # --- Novelty (needs train_graphs from setup(), skippable) ---
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
