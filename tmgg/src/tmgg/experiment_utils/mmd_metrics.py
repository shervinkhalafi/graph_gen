"""Maximum Mean Discrepancy (MMD) metrics for graph generation evaluation.

This module computes distributional distance between reference and generated
graphs using MMD on graph-theoretic properties. Implementation is aligned
with DiGress (Vignac et al., 2023) for direct comparability:

- Spectral statistic uses normalized Laplacian eigenvalues (200 bins, fixed range)
- Clustering histogram uses 100 bins
- Default kernel is gaussian_tv: k(x,y) = exp(-TV(x,y)² / 2σ²)
- MMD uses unbiased estimator (unique pairs, no self-pairs)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import combinations
from typing import TYPE_CHECKING, Any, Literal, cast

import networkx as nx
import numpy as np
import torch

if TYPE_CHECKING:
    pass


@dataclass
class MMDResults:
    """Results from MMD computation (DiGress-compatible).

    Attributes
    ----------
    degree_mmd
        MMD on degree distributions.
    clustering_mmd
        MMD on clustering coefficient distributions.
    spectral_mmd
        MMD on normalized Laplacian eigenvalue distributions (DiGress "spectre").
    """

    degree_mmd: float
    clustering_mmd: float
    spectral_mmd: float

    def to_dict(self) -> dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "degree_mmd": self.degree_mmd,
            "clustering_mmd": self.clustering_mmd,
            "spectral_mmd": self.spectral_mmd,
        }


def adjacency_to_networkx(adjacency: torch.Tensor | np.ndarray) -> nx.Graph[Any]:
    """Convert adjacency matrix to NetworkX graph.

    Parameters
    ----------
    adjacency
        Adjacency matrix of shape (n, n).

    Returns
    -------
    nx.Graph
        Undirected graph.
    """
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.detach().cpu().numpy()

    # Threshold to binary for probabilistic adjacency
    A = (adjacency > 0.5).astype(float)
    return nx.from_numpy_array(A)


def compute_degree_histogram(
    G: nx.Graph[Any], max_degree: int | None = None
) -> np.ndarray:
    """Compute normalized degree histogram.

    Parameters
    ----------
    G
        NetworkX graph.
    max_degree
        Maximum degree to include. If None, uses max observed degree.

    Returns
    -------
    np.ndarray
        Normalized degree histogram.
    """
    # Extract degrees from DegreeView - cast required due to incomplete networkx stubs
    degree_view = G.degree()
    degrees: list[int] = [int(d) for _, d in cast(Any, degree_view)]
    if len(degrees) == 0:
        return np.array([1.0])

    max_d = max_degree if max_degree is not None else max(degrees)
    hist, _ = np.histogram(degrees, bins=np.arange(0, max_d + 2), density=True)
    return hist


def compute_clustering_histogram(G: nx.Graph[Any], num_bins: int = 100) -> np.ndarray:
    """Compute histogram of clustering coefficients (DiGress style).

    Parameters
    ----------
    G
        NetworkX graph.
    num_bins
        Number of bins for histogram. DiGress uses 100.

    Returns
    -------
    np.ndarray
        Normalized clustering coefficient histogram.
    """
    if G.number_of_nodes() == 0:
        return np.zeros(num_bins)

    # nx.clustering returns dict[node, float] - cast required due to incomplete stubs
    clustering_dict = cast(dict[Any, float], nx.clustering(G))
    clustering = list(clustering_dict.values())
    if len(clustering) == 0:
        return np.zeros(num_bins)

    hist, _ = np.histogram(
        clustering, bins=np.linspace(0, 1, num_bins + 1), density=True
    )
    return hist


def compute_spectral_histogram(G: nx.Graph[Any], num_bins: int = 200) -> np.ndarray:
    """Compute histogram of normalized Laplacian eigenvalues (DiGress spectre style).

    Uses fixed range [-1e-5, 2] and normalizes to PMF, matching DiGress's "spectre"
    metric exactly.

    Parameters
    ----------
    G
        NetworkX graph.
    num_bins
        Number of bins for histogram. DiGress uses 200.

    Returns
    -------
    np.ndarray
        PMF over eigenvalue histogram bins.
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros(num_bins)

    L = nx.normalized_laplacian_matrix(G).toarray()
    eigenvalues = np.linalg.eigvalsh(L)

    # Fixed range matching DiGress
    hist, _ = np.histogram(eigenvalues, bins=num_bins, range=(-1e-5, 2), density=False)

    # Normalize to PMF (DiGress style, not density)
    hist_sum = hist.sum()
    if hist_sum > 0:
        hist = hist / hist_sum
    return hist.astype(np.float64)


def compute_adjacency_spectral_histogram(
    G: nx.Graph[Any], num_bins: int = 20, normalized: bool = True
) -> np.ndarray:
    """Compute histogram of adjacency matrix eigenvalues (legacy tmgg style).

    This is the original tmgg spectral histogram, kept for compatibility. For
    DiGress-compatible evaluation, use compute_spectral_histogram instead.

    Parameters
    ----------
    G
        NetworkX graph.
    num_bins
        Number of bins for histogram.
    normalized
        If True, normalize eigenvalues by sqrt(n).

    Returns
    -------
    np.ndarray
        Normalized eigenvalue histogram.
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros(num_bins)

    A = nx.to_numpy_array(G)
    eigenvalues = np.linalg.eigvalsh(A)

    if normalized and n > 0:
        eigenvalues = eigenvalues / np.sqrt(n)

    # Use data-driven bin range centered on 0
    max_abs = max(abs(eigenvalues.min()), abs(eigenvalues.max()), 0.1)
    hist, _ = np.histogram(
        eigenvalues, bins=np.linspace(-max_abs, max_abs, num_bins + 1), density=True
    )
    return hist


def compute_laplacian_histogram(
    G: nx.Graph[Any], num_bins: int = 20, normalized: bool = True
) -> np.ndarray:
    """Compute normalized histogram of Laplacian eigenvalues.

    Parameters
    ----------
    G
        NetworkX graph.
    num_bins
        Number of bins for histogram.
    normalized
        If True, use normalized Laplacian.

    Returns
    -------
    np.ndarray
        Normalized Laplacian eigenvalue histogram.
    """
    n = G.number_of_nodes()
    if n == 0:
        return np.zeros(num_bins)

    if normalized:
        L = nx.normalized_laplacian_matrix(G).toarray()
    else:
        L = nx.laplacian_matrix(G).toarray()

    eigenvalues = np.linalg.eigvalsh(L)

    # Laplacian eigenvalues are non-negative
    max_val = max(eigenvalues.max(), 0.1)
    hist, _ = np.histogram(
        eigenvalues, bins=np.linspace(0, max_val, num_bins + 1), density=True
    )
    return hist


@dataclass
class GraphStatistics:
    """Statistics computed from a collection of graphs (DiGress-compatible).

    Attributes
    ----------
    degree
        List of degree histograms.
    clustering
        List of clustering coefficient histograms.
    spectral
        List of normalized Laplacian eigenvalue histograms (DiGress "spectre").
    """

    degree: list[np.ndarray]
    clustering: list[np.ndarray]
    spectral: list[np.ndarray]


def compute_graph_statistics(
    graphs: list[nx.Graph[Any]],
    max_workers: int | None = None,
) -> GraphStatistics:
    """Compute histograms of graph properties for a collection of graphs.

    Parameters
    ----------
    graphs
        List of NetworkX graphs.
    max_workers
        Maximum number of parallel workers.

    Returns
    -------
    GraphStatistics
        Statistics for all graphs.
    """

    def compute_single(
        G: nx.Graph[Any],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            compute_degree_histogram(G),
            compute_clustering_histogram(G),
            compute_spectral_histogram(G),
        )

    if max_workers == 1 or len(graphs) < 4:
        # Single-threaded for small batches
        results = [compute_single(g) for g in graphs]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(compute_single, graphs))

    degree = [r[0] for r in results]
    clustering = [r[1] for r in results]
    spectral = [r[2] for r in results]

    return GraphStatistics(
        degree=degree,
        clustering=clustering,
        spectral=spectral,
    )


def gaussian_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Compute Gaussian kernel between two histograms.

    Parameters
    ----------
    x
        First histogram.
    y
        Second histogram.
    sigma
        Kernel bandwidth.

    Returns
    -------
    float
        Kernel value.
    """
    # Pad to same length
    max_len = max(len(x), len(y))
    x_padded = np.zeros(max_len)
    y_padded = np.zeros(max_len)
    x_padded[: len(x)] = x
    y_padded[: len(y)] = y

    # Normalize to valid probability distributions
    x_sum = x_padded.sum()
    y_sum = y_padded.sum()
    if x_sum > 0:
        x_padded = x_padded / x_sum
    if y_sum > 0:
        y_padded = y_padded / y_sum

    # Compute L2 distance and Gaussian kernel
    dist = np.linalg.norm(x_padded - y_padded)
    return float(np.exp(-(dist**2) / (2 * sigma**2)))


def gaussian_tv_kernel(x: np.ndarray, y: np.ndarray, sigma: float = 1.0) -> float:
    """Gaussian kernel with Total Variation distance (DiGress style).

    Computes k(x,y) = exp(-TV(x,y)² / 2σ²), matching DiGress's gaussian_tv kernel.

    Parameters
    ----------
    x
        First histogram.
    y
        Second histogram.
    sigma
        Kernel bandwidth.

    Returns
    -------
    float
        Kernel value in (0, 1].
    """
    # Pad to same length
    max_len = max(len(x), len(y))
    x_padded = np.zeros(max_len)
    y_padded = np.zeros(max_len)
    x_padded[: len(x)] = x
    y_padded[: len(y)] = y

    # Normalize to PMF
    x_sum = x_padded.sum()
    y_sum = y_padded.sum()
    if x_sum > 0:
        x_padded = x_padded / x_sum
    if y_sum > 0:
        y_padded = y_padded / y_sum

    # TV distance = 0.5 * L1 distance for probability distributions
    tv = 0.5 * np.sum(np.abs(x_padded - y_padded))
    return float(np.exp(-(tv**2) / (2 * sigma**2)))


def compute_mmd(
    samples1: list[np.ndarray],
    samples2: list[np.ndarray],
    kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    sigma: float = 1.0,
) -> float:
    """Compute unbiased MMD estimator between two sets of histograms.

    Uses the unbiased estimator: E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    where x, x' are drawn from samples1, and y, y' from samples2.

    Parameters
    ----------
    samples1
        First set of histogram samples.
    samples2
        Second set of histogram samples.
    kernel
        Kernel type: "gaussian" (L2-based) or "gaussian_tv" (TV-based, DiGress style).
    sigma
        Bandwidth for kernel (both use same sigma parameter).

    Returns
    -------
    float
        MMD value (non-negative).
    """
    if len(samples1) < 2 or len(samples2) < 2:
        return float("inf")

    def kernel_fn(x: np.ndarray, y: np.ndarray) -> float:
        if kernel == "gaussian":
            return gaussian_kernel(x, y, sigma)
        return gaussian_tv_kernel(x, y, sigma)

    # Compute kernel expectations
    # k11: E[k(x, x')] for x, x' in samples1
    k11_values = [kernel_fn(x, y) for x, y in combinations(samples1, 2)]
    k11 = float(np.mean(k11_values)) if k11_values else 0.0

    # k22: E[k(y, y')] for y, y' in samples2
    k22_values = [kernel_fn(x, y) for x, y in combinations(samples2, 2)]
    k22 = float(np.mean(k22_values)) if k22_values else 0.0

    # k12: E[k(x, y)] for x in samples1, y in samples2
    k12_values = [kernel_fn(x, y) for x in samples1 for y in samples2]
    k12 = float(np.mean(k12_values)) if k12_values else 0.0

    return max(0.0, k11 + k22 - 2 * k12)


def compute_mmd_metrics(
    ref_graphs: list[nx.Graph[Any]],
    gen_graphs: list[nx.Graph[Any]],
    kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    sigma: float = 1.0,
    max_workers: int | None = None,
) -> MMDResults:
    """Compute MMD metrics between reference and generated graph distributions.

    Parameters
    ----------
    ref_graphs
        List of reference NetworkX graphs.
    gen_graphs
        List of generated NetworkX graphs.
    kernel
        Kernel type: "gaussian" (L2-based) or "gaussian_tv" (TV-based, DiGress style).
    sigma
        Bandwidth for kernel.
    max_workers
        Maximum number of parallel workers for statistics computation.

    Returns
    -------
    MMDResults
        MMD values for each graph statistic.
    """
    # Compute statistics for both sets
    ref_stats = compute_graph_statistics(ref_graphs, max_workers)
    gen_stats = compute_graph_statistics(gen_graphs, max_workers)

    # Compute MMD for each statistic
    degree_mmd = compute_mmd(ref_stats.degree, gen_stats.degree, kernel, sigma)
    clustering_mmd = compute_mmd(
        ref_stats.clustering, gen_stats.clustering, kernel, sigma
    )
    spectral_mmd = compute_mmd(ref_stats.spectral, gen_stats.spectral, kernel, sigma)

    return MMDResults(
        degree_mmd=degree_mmd,
        clustering_mmd=clustering_mmd,
        spectral_mmd=spectral_mmd,
    )


def compute_mmd_from_adjacencies(
    ref_adjacencies: torch.Tensor | np.ndarray | list,
    gen_adjacencies: torch.Tensor | np.ndarray | list,
    kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    sigma: float = 1.0,
    max_workers: int | None = None,
) -> MMDResults:
    """Convenience function to compute MMD from adjacency matrices.

    Parameters
    ----------
    ref_adjacencies
        Reference adjacency matrices (batch or list).
    gen_adjacencies
        Generated adjacency matrices (batch or list).
    kernel
        Kernel type: "gaussian" (L2-based) or "gaussian_tv" (TV-based, DiGress style).
    sigma
        Bandwidth for kernel.
    max_workers
        Maximum number of parallel workers.

    Returns
    -------
    MMDResults
        MMD values for each graph statistic.
    """
    # Convert to list of numpy arrays
    if (
        isinstance(ref_adjacencies, torch.Tensor)
        or isinstance(ref_adjacencies, np.ndarray)
        and ref_adjacencies.ndim == 3
    ):
        ref_list = [ref_adjacencies[i] for i in range(ref_adjacencies.shape[0])]
    else:
        ref_list = list(ref_adjacencies)

    if (
        isinstance(gen_adjacencies, torch.Tensor)
        or isinstance(gen_adjacencies, np.ndarray)
        and gen_adjacencies.ndim == 3
    ):
        gen_list = [gen_adjacencies[i] for i in range(gen_adjacencies.shape[0])]
    else:
        gen_list = list(gen_adjacencies)

    # Convert to NetworkX graphs
    ref_graphs = [adjacency_to_networkx(A) for A in ref_list]
    gen_graphs = [adjacency_to_networkx(A) for A in gen_list]

    return compute_mmd_metrics(ref_graphs, gen_graphs, kernel, sigma, max_workers)
