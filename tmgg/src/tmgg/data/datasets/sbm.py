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
    """
    Generate all valid block size partitions for n nodes.

    This function finds all ways to partition n nodes into blocks where:
    - The number of blocks is between min_blocks and max_blocks (inclusive)
    - Each block has size between min_size and max_size (inclusive)
    - The sum of all block sizes equals n

    The algorithm uses recursive backtracking to explore all valid partitions.
    For each position, it calculates the valid range of sizes based on:
    - Minimum size: Must be at least min_size, but also large enough that
      remaining blocks can fit within max_size constraint
    - Maximum size: Must be at most max_size, but also small enough that
      remaining blocks can satisfy min_size constraint

    Args:
        n: Total number of nodes to partition
        min_blocks: Minimum number of blocks in a partition
        max_blocks: Maximum number of blocks in a partition
        min_size: Minimum size for any block
        max_size: Maximum size for any block

    Returns:
        List of valid block size partitions, where each partition is a list
        of integers summing to n

    Example:
        >>> generate_block_sizes(6, min_blocks=2, max_blocks=3, min_size=2, max_size=4)
        [[2, 4], [3, 3], [4, 2], [2, 2, 2]]
    """
    valid_partitions = []

    # Try different numbers of blocks
    for num_blocks in range(min_blocks, max_blocks + 1):

        def generate_partitions(remaining, blocks_left, current_partition):
            """
            Recursively generate partitions for remaining nodes.

            The key insight: we must ensure that after choosing a size for the
            current block, the remaining nodes can still be validly partitioned
            into the remaining blocks while respecting all constraints.

            Args:
                remaining: Number of nodes left to partition
                blocks_left: Number of blocks left to create (including current)
                current_partition: Current partial partition being built
            """
            # Base case: no more blocks to create
            if blocks_left == 0:
                # Valid only if we've used exactly all nodes
                if remaining == 0:
                    valid_partitions.append(current_partition[:])
                return

            # Calculate valid size range for current block

            # LOWER BOUND rationale:
            # - Obviously need at least min_size nodes
            # - But also: if we take too few nodes now, the remaining blocks might
            #   be forced to exceed max_size to use up all remaining nodes
            # - Worst case: all other blocks take max_size. Then current block
            #   must take at least (remaining - (blocks_left-1)*max_size)
            # - Example: 20 nodes, 2 blocks left, max_size=15
            #   Current must be at least 20-1*15=5, or the last block would need >15
            start = max(min_size, remaining - (blocks_left - 1) * max_size)

            # UPPER BOUND rationale:
            # - Obviously can't exceed max_size
            # - But also: if we take too many nodes now, the remaining blocks might
            #   not have enough nodes to satisfy their min_size requirements
            # - Best case: all other blocks take min_size. Then current block
            #   can take at most (remaining - (blocks_left-1)*min_size)
            # - Example: 20 nodes, 2 blocks left, min_size=2
            #   Current can be at most 20-1*2=18, or the last block would have <2
            end = min(max_size, remaining - (blocks_left - 1) * min_size) + 1

            for size in range(start, end):
                if size <= remaining:
                    current_partition.append(size)
                    generate_partitions(
                        remaining - size, blocks_left - 1, current_partition
                    )
                    current_partition.pop()

        generate_partitions(n, num_blocks, [])

    return valid_partitions
