"""Synthetic graph generators for denoising experiments.

This module provides wrappers around NetworkX graph generators for creating
datasets of d-regular graphs, LFR community graphs, and random trees.
"""

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import torch


def generate_regular_graphs(
    n: int,
    d: int,
    num_graphs: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random d-regular graphs.

    A d-regular graph is a graph where every node has exactly degree d.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    d : int
        Degree of each node. Must satisfy d < n and n*d must be even.
    num_graphs : int
        Number of graphs to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).

    Raises
    ------
    ValueError
        If d >= n or n*d is odd (no d-regular graph exists).
    """
    import random

    if d >= n:
        raise ValueError(f"Degree d={d} must be less than n={n}")
    if (n * d) % 2 != 0:
        raise ValueError(f"n*d must be even, got n={n}, d={d}")

    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        graph_seed = base_seed + i
        G = nx.random_regular_graph(d, n, seed=graph_seed)
        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_tree_graphs(
    n: int,
    num_graphs: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random tree graphs using Prüfer sequences.

    A tree is a connected acyclic graph with n-1 edges. Trees are generated
    by sampling random Prüfer sequences and converting them to trees.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).
    """
    import random

    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        # Set random seed for this graph
        random.seed(base_seed + i)
        np.random.seed(base_seed + i)

        # Generate random Prüfer sequence and convert to tree
        # Prüfer sequence has length n-2, with elements in [0, n-1]
        prufer_seq = [random.randint(0, n - 1) for _ in range(n - 2)]
        G = nx.from_prufer_sequence(prufer_seq)
        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_lfr_graphs(
    n: int,
    num_graphs: int,
    tau1: float = 3.0,
    tau2: float = 1.5,
    mu: float = 0.1,
    average_degree: Optional[int] = None,
    min_community: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate LFR (Lancichinetti-Fortunato-Radicchi) benchmark graphs.

    LFR graphs are synthetic networks with planted community structure,
    commonly used for testing community detection algorithms.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    tau1 : float, optional
        Power law exponent for degree distribution. Default 3.0.
    tau2 : float, optional
        Power law exponent for community size distribution. Default 1.5.
    mu : float, optional
        Mixing parameter (fraction of inter-community edges). Lower values
        mean stronger community structure. Default 0.1.
    average_degree : int, optional
        Target average degree. Default is n // 10.
    min_community : int, optional
        Minimum community size. Default is n // 10.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).

    Notes
    -----
    The LFR generator can fail for certain parameter combinations. This
    function will retry with slightly different parameters if generation
    fails.
    """
    if average_degree is None:
        average_degree = max(3, n // 10)
    if min_community is None:
        min_community = max(5, n // 10)

    base_seed = seed if seed is not None else 42
    adjacencies = []

    max_degree = n - 1
    max_community = n // 2

    for i in range(num_graphs):
        graph_seed = base_seed + i

        # LFR can be finicky; try a few times with slight variations
        G = None
        for attempt in range(5):
            try:
                G = nx.LFR_benchmark_graph(
                    n=n,
                    tau1=tau1,
                    tau2=tau2,
                    mu=mu,
                    average_degree=average_degree,
                    max_degree=max_degree,
                    min_community=min_community,
                    max_community=max_community,
                    seed=graph_seed + attempt * 1000,
                )
                break
            except nx.ExceededMaxIterations:
                # Adjust parameters slightly and retry
                average_degree = max(3, average_degree - 1)
                continue

        if G is None:
            # Fall back to SBM-like structure if LFR fails
            # This shouldn't happen often with reasonable parameters
            G = nx.planted_partition_graph(
                l=max(2, n // min_community),
                k=min_community,
                p_in=1 - mu,
                p_out=mu,
                seed=graph_seed,
            )

        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_erdos_renyi_graphs(
    n: int,
    p: float,
    num_graphs: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate Erdős-Rényi random graphs.

    Each edge exists independently with probability p.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    p : float
        Edge probability (between 0 and 1).
    num_graphs : int
        Number of graphs to generate.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).
    """
    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        graph_seed = base_seed + i
        G = nx.fast_gnp_random_graph(n, p, seed=graph_seed)
        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_watts_strogatz_graphs(
    n: int,
    num_graphs: int,
    k: int = 4,
    p: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate Watts-Strogatz small-world graphs.

    Small-world graphs have high clustering and short path lengths,
    mimicking properties of social networks and neural networks.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    k : int, optional
        Each node is connected to k nearest neighbors in ring topology.
        Must be even. Default is 4.
    p : float, optional
        Probability of rewiring each edge. Default is 0.3.
        p=0 gives a ring lattice, p=1 gives random graph.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).

    Raises
    ------
    ValueError
        If k >= n or k is odd.
    """
    if k >= n:
        raise ValueError(f"k={k} must be less than n={n}")
    if k % 2 != 0:
        raise ValueError(f"k must be even, got {k}")

    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        graph_seed = base_seed + i
        G = nx.watts_strogatz_graph(n, k, p, seed=graph_seed)
        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_random_geometric_graphs(
    n: int,
    num_graphs: int,
    radius: float = 0.3,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random geometric graphs.

    Nodes are placed uniformly at random in a unit square, and edges
    connect nodes whose Euclidean distance is at most `radius`.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    radius : float, optional
        Distance threshold for edge creation. Default is 0.3.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).
    """
    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        graph_seed = base_seed + i
        G = nx.random_geometric_graph(n, radius, seed=graph_seed)
        A = nx.to_numpy_array(G, dtype=np.float32)
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


def generate_configuration_model_graphs(
    n: int,
    num_graphs: int,
    degree_sequence: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Generate random graphs with a specified degree sequence.

    The configuration model generates random graphs with a given degree
    sequence. If no sequence is provided, a default power-law-like
    sequence is used.

    Parameters
    ----------
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    degree_sequence : list of int, optional
        Desired degree for each node. Must sum to even number.
        If None, uses a default sequence based on node count.
    seed : int, optional
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices of shape (num_graphs, n, n).

    Notes
    -----
    The configuration model may produce multigraphs or self-loops. This
    implementation removes them, so actual degrees may be slightly lower
    than specified.
    """
    import random

    base_seed = seed if seed is not None else 42
    adjacencies = []

    for i in range(num_graphs):
        graph_seed = base_seed + i
        random.seed(graph_seed)
        np.random.seed(graph_seed)

        if degree_sequence is None:
            # Generate a power-law-like degree sequence
            # Most nodes have low degree, few have high degree
            seq = np.random.zipf(2.0, n)
            seq = np.clip(seq, 1, n - 1).astype(int)
            # Ensure sum is even
            if seq.sum() % 2 != 0:
                seq[0] += 1
            deg_seq = seq.tolist()
        else:
            deg_seq = list(degree_sequence)

        # Create graph from degree sequence
        G = nx.configuration_model(deg_seq, seed=graph_seed)
        # Remove parallel edges and self-loops
        G = nx.Graph(G)
        G.remove_edges_from(nx.selfloop_edges(G))

        A = nx.to_numpy_array(G, dtype=np.float32, nodelist=range(n))
        adjacencies.append(A)

    return np.stack(adjacencies, axis=0)


class SyntheticGraphDataset:
    """Dataset wrapper for synthetic graphs.

    Provides a unified interface for generating and accessing synthetic
    graph datasets of various types.

    Parameters
    ----------
    graph_type : str
        Type of graph. Supported types:
        - "regular": d-regular graphs
        - "tree": random trees
        - "lfr": LFR benchmark graphs
        - "erdos_renyi" / "er": Erdős-Rényi random graphs
        - "watts_strogatz" / "ws": small-world graphs
        - "random_geometric" / "rg": geometric proximity graphs
        - "configuration_model" / "cm": graphs with specified degree sequence
    n : int
        Number of nodes per graph.
    num_graphs : int
        Number of graphs to generate.
    seed : int, optional
        Random seed for reproducibility.
    **kwargs
        Additional parameters passed to the graph generator:
        - For "regular": d (degree)
        - For "lfr": tau1, tau2, mu, average_degree, min_community
        - For "erdos_renyi": p (edge probability)
        - For "watts_strogatz": k (neighbors), p (rewiring probability)
        - For "random_geometric": radius (edge threshold)
        - For "configuration_model": degree_sequence (list of degrees)

    Attributes
    ----------
    adjacencies : np.ndarray
        Generated adjacency matrices of shape (num_graphs, n, n).
    graph_type : str
        Type of graphs in this dataset.
    n : int
        Number of nodes per graph.
    """

    VALID_TYPES = {
        "regular", "tree", "lfr", "erdos_renyi", "er",
        "watts_strogatz", "ws", "random_geometric", "rg",
        "configuration_model", "cm"
    }

    # Map aliases to canonical names
    TYPE_ALIASES = {
        "er": "erdos_renyi",
        "ws": "watts_strogatz",
        "rg": "random_geometric",
        "cm": "configuration_model",
    }

    def __init__(
        self,
        graph_type: str,
        n: int,
        num_graphs: int,
        seed: Optional[int] = None,
        **kwargs,
    ):
        if graph_type not in self.VALID_TYPES:
            raise ValueError(
                f"graph_type must be one of {self.VALID_TYPES}, got '{graph_type}'"
            )

        # Resolve aliases
        canonical_type = self.TYPE_ALIASES.get(graph_type, graph_type)

        self.graph_type = canonical_type
        self.n = n
        self.num_graphs = num_graphs
        self.seed = seed
        self.kwargs = kwargs

        # Generate graphs
        if canonical_type == "regular":
            d = kwargs.get("d", 3)
            self.adjacencies = generate_regular_graphs(n, d, num_graphs, seed)
        elif canonical_type == "tree":
            self.adjacencies = generate_tree_graphs(n, num_graphs, seed)
        elif canonical_type == "lfr":
            self.adjacencies = generate_lfr_graphs(
                n, num_graphs, seed=seed, **kwargs
            )
        elif canonical_type == "erdos_renyi":
            p = kwargs.get("p", 0.1)
            self.adjacencies = generate_erdos_renyi_graphs(n, p, num_graphs, seed)
        elif canonical_type == "watts_strogatz":
            k = kwargs.get("k", 4)
            p = kwargs.get("p", 0.3)
            self.adjacencies = generate_watts_strogatz_graphs(n, num_graphs, k, p, seed)
        elif canonical_type == "random_geometric":
            radius = kwargs.get("radius", 0.3)
            self.adjacencies = generate_random_geometric_graphs(n, num_graphs, radius, seed)
        elif canonical_type == "configuration_model":
            degree_sequence = kwargs.get("degree_sequence", None)
            self.adjacencies = generate_configuration_model_graphs(
                n, num_graphs, degree_sequence, seed
            )

    def __len__(self) -> int:
        return self.num_graphs

    def __getitem__(self, idx: int) -> np.ndarray:
        return self.adjacencies[idx]

    def get_adjacency_matrices(self) -> np.ndarray:
        """Return all adjacency matrices.

        Returns
        -------
        np.ndarray
            Adjacency matrices of shape (num_graphs, n, n).
        """
        return self.adjacencies

    def to_torch(self) -> torch.Tensor:
        """Convert adjacency matrices to PyTorch tensor.

        Returns
        -------
        torch.Tensor
            Adjacency matrices as float32 tensor.
        """
        return torch.from_numpy(self.adjacencies).float()

    def train_val_test_split(
        self,
        train_ratio: float = 0.7,
        val_ratio: float = 0.1,
        seed: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split dataset into train/validation/test sets.

        Parameters
        ----------
        train_ratio : float
            Fraction of data for training. Default 0.7.
        val_ratio : float
            Fraction of data for validation. Default 0.1.
        seed : int, optional
            Random seed for shuffling.

        Returns
        -------
        train : np.ndarray
            Training adjacency matrices.
        val : np.ndarray
            Validation adjacency matrices.
        test : np.ndarray
            Test adjacency matrices.
        """
        rng = np.random.default_rng(seed)
        indices = rng.permutation(self.num_graphs)

        n_train = int(train_ratio * self.num_graphs)
        n_val = int(val_ratio * self.num_graphs)

        train_idx = indices[:n_train]
        val_idx = indices[n_train:n_train + n_val]
        test_idx = indices[n_train + n_val:]

        return (
            self.adjacencies[train_idx],
            self.adjacencies[val_idx],
            self.adjacencies[test_idx],
        )
