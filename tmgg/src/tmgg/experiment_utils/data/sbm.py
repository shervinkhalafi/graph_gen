"""Data generation and manipulation utilities for graph denoising experiments."""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.linalg import expm
from typing import List, Tuple, Optional, Union
import random


def generate_sbm_adjacency(
    block_sizes: List[int],
    p: float,
    q: float,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate an adjacency matrix for a stochastic block model with variable block sizes.

    Args:
        block_sizes: List of sizes for each block
        p: Probability of intra-block edges
        q: Probability of inter-block edges
        rng: Random number generator (optional)

    Returns:
        Adjacency matrix as a numpy array
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
            density = p if i == j else q
            block_start_i = block_starts[i]
            block_end_i = block_start_i + block_sizes[i]
            block_start_j = block_starts[j]
            block_end_j = block_start_j + block_sizes[j]

            # Generate random edges within or between blocks
            block_i_size = block_sizes[i]
            block_j_size = block_sizes[j]
            adj_matrix[block_start_i:block_end_i, block_start_j:block_end_j] = (
                rng.random((block_i_size, block_j_size)) < density
            ).astype(int)

            # Make the matrix symmetric (for undirected graphs)
            if i != j:
                adj_matrix[block_start_j:block_end_j, block_start_i:block_end_i] = (
                    adj_matrix[block_start_i:block_end_i, block_start_j:block_end_j].T
                )

    return adj_matrix


def generate_block_sizes(
    n: int,
    min_blocks: int = 2,
    max_blocks: int = 4,
    min_size: int = 2,
    max_size: int = 15,
) -> List[List[int]]:
    """
    Generate all valid block size partitions for n nodes.

    Args:
        n: Total number of nodes
        min_blocks: Minimum number of blocks
        max_blocks: Maximum number of blocks
        min_size: Minimum block size
        max_size: Maximum block size

    Returns:
        List of valid block size partitions
    """
    valid_partitions = []

    # Try different numbers of blocks
    for num_blocks in range(min_blocks, max_blocks + 1):

        def generate_partitions(remaining, blocks_left, current_partition):
            # Base cases
            if blocks_left == 0:
                if remaining == 0:
                    valid_partitions.append(current_partition[:])
                return

            # Try different sizes for current block
            start = max(min_size, remaining - (blocks_left - 1) * max_size)
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
