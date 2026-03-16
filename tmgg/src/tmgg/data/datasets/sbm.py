"""Data generation and manipulation utilities for graph denoising experiments."""

import numpy as np


def generate_sbm_adjacency(
    block_sizes: list[int],
    p_intra: float,
    p_inter: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Generate an adjacency matrix for a stochastic block model.

    Parameters
    ----------
    block_sizes
        Number of nodes in each community.
    p_intra
        Probability of edges within the same community.
    p_inter
        Probability of edges between different communities.
    rng
        NumPy random generator. Uses a fresh default if ``None``.

    Returns
    -------
    np.ndarray
        Symmetric binary adjacency matrix with zero diagonal.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_blocks = len(block_sizes)
    n = sum(block_sizes)

    # Initialize the adjacency matrix with zeros
    adj_matrix = np.zeros((n, n))

    # Calculate the starting index of each block
    block_starts = [0]
    for i in range(n_blocks - 1):
        block_starts.append(block_starts[-1] + block_sizes[i])

    for i in range(n_blocks):
        for j in range(i, n_blocks):
            density = p_intra if i == j else p_inter
            block_start_i = block_starts[i]
            block_end_i = block_start_i + block_sizes[i]
            block_start_j = block_starts[j]
            block_end_j = block_start_j + block_sizes[j]

            # Generate random edges within or between blocks
            block_i_size = block_sizes[i]
            block_j_size = block_sizes[j]
            block = (rng.random((block_i_size, block_j_size)) < density).astype(int)

            if i == j:
                # Intra-block: symmetrize by keeping upper triangle and mirroring
                block = np.triu(block) + np.triu(block, 1).T
            adj_matrix[block_start_i:block_end_i, block_start_j:block_end_j] = block

            # Mirror inter-block edges for symmetry (undirected graph)
            if i != j:
                adj_matrix[block_start_j:block_end_j, block_start_i:block_end_i] = (
                    block.T
                )

    # Simple graphs have no self-loops
    np.fill_diagonal(adj_matrix, 0)

    return adj_matrix


def generate_sbm_batch(
    num_graphs: int,
    num_nodes: int,
    num_blocks: int = 2,
    p_intra: float = 0.7,
    p_inter: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Generate a batch of SBM adjacency matrices.

    Computes approximately equal block sizes for ``num_nodes`` across
    ``num_blocks`` blocks, then calls ``generate_sbm_adjacency`` for each
    graph.  That function already produces symmetric matrices with zero
    diagonal, so no post-processing is needed.

    Parameters
    ----------
    num_graphs
        Number of graphs to generate.
    num_nodes
        Number of nodes per graph.
    num_blocks
        Number of communities in the SBM.
    p_intra
        Intra-block edge probability.
    p_inter
        Inter-block edge probability.
    seed
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Adjacency matrices, shape ``(num_graphs, num_nodes, num_nodes)``,
        dtype ``float32``.
    """
    rng = np.random.default_rng(seed)

    # Equal block sizes, distributing remainder to first blocks
    block_size = num_nodes // num_blocks
    remainder = num_nodes % num_blocks
    block_sizes = [block_size] * num_blocks
    for i in range(remainder):
        block_sizes[i] += 1

    adjacencies: list[np.ndarray] = []
    for _ in range(num_graphs):
        adj = generate_sbm_adjacency(
            block_sizes=block_sizes, p_intra=p_intra, p_inter=p_inter, rng=rng
        )
        adjacencies.append(adj.astype(np.float32))

    return np.stack(adjacencies, axis=0)


def generate_block_sizes(
    n: int,
    min_blocks: int = 2,
    max_blocks: int = 4,
    min_size: int = 2,
    max_size: int = 15,
) -> list[list[int]]:
    """Generate all valid block size partitions for n nodes.

    Finds all ways to partition n nodes into blocks where the number of blocks
    is in ``[min_blocks, max_blocks]``, each block size is in
    ``[min_size, max_size]``, and all sizes sum to n.  The algorithm uses
    recursive backtracking, pruning branches where the remaining nodes cannot
    satisfy the min/max size constraints for the remaining blocks.

    Parameters
    ----------
    n : int
        Total number of nodes to partition.
    min_blocks : int
        Minimum number of blocks in a partition.
    max_blocks : int
        Maximum number of blocks in a partition.
    min_size : int
        Minimum size for any single block.
    max_size : int
        Maximum size for any single block.

    Returns
    -------
    list[list[int]]
        All valid block size partitions; each partition is a list of integers
        summing to n.

    Examples
    --------
    >>> generate_block_sizes(6, min_blocks=2, max_blocks=3, min_size=2, max_size=4)
    [[2, 4], [3, 3], [4, 2], [2, 2, 2]]
    """
    valid_partitions = []

    def generate_partitions(remaining, blocks_left, current_partition):
        """Recursively generate partitions for remaining nodes.

        Ensures that after choosing a size for the current block, the
        remaining nodes can still be validly partitioned into the remaining
        blocks while respecting min_size/max_size constraints.

        Parameters
        ----------
        remaining
            Number of nodes left to partition.
        blocks_left
            Number of blocks left to create (including current).
        current_partition
            Current partial partition being built (mutated in place).
        """
        if blocks_left == 0:
            if remaining == 0:
                valid_partitions.append(current_partition[:])
            return

        # Lower bound: current block must be large enough that remaining
        # blocks can fit within max_size.
        start = max(min_size, remaining - (blocks_left - 1) * max_size)

        # Upper bound: current block must be small enough that remaining
        # blocks can satisfy min_size.
        end = min(max_size, remaining - (blocks_left - 1) * min_size) + 1

        for size in range(start, end):
            if size <= remaining:
                current_partition.append(size)
                generate_partitions(
                    remaining - size, blocks_left - 1, current_partition
                )
                current_partition.pop()

    # Try different numbers of blocks
    for num_blocks in range(min_blocks, max_blocks + 1):
        generate_partitions(n, num_blocks, [])

    return valid_partitions
