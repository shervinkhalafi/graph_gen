import numpy as np
from numpy.typing import NDArray
from scipy.linalg import expm

import torch
from torch.utils.data import Dataset
from typing import override


def generate_sbm_adjacency(
    block_sizes: list[int],
    p: float,
    q: float,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float64]:
    """
    Generate an adjacency matrix for a stochastic block model with variable block sizes.

    Parameters:
    - block_sizes: List of sizes for each block.
    - p: Probability of intra-block edges.
    - q: Probability of inter-block edges.
    - rng: Random number generator (optional).

    Returns:
    - Adjacency matrix as a numpy array.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_blocks = len(block_sizes)
    n = sum(block_sizes)

    # Initialize the adjacency matrix with zeros

    adj_matrix: NDArray[np.float64] = np.zeros((n, n))

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


# create a random n*n skew-symmetric matrix
def random_skew_symmetric_matrix(n: int) -> NDArray[np.float64]:
    A: NDArray[np.float64] = np.random.rand(n, n)
    return (A - A.T) / 2


def add_rotation_noise(
    A: torch.Tensor | NDArray[np.float64],
    eps: float,
    skew: NDArray[np.float64],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A_tensor: torch.Tensor = torch.tensor(A, dtype=torch.float32)
    eigenvalues, V = torch.linalg.eigh(A_tensor)
    R: NDArray[np.float64] = expm(eps * skew)
    R_tensor = torch.tensor(R, dtype=torch.float32).to(V.device)
    V_rot = V @ R_tensor

    # Initialize tensor of appropriate size
    eig_diag = torch.zeros(
        eigenvalues.shape[0],
        eigenvalues.shape[1],
        eigenvalues.shape[1],
        device=eigenvalues.device,
        dtype=eigenvalues.dtype,
    )

    # Fill diagonal elements for each matrix in the batch
    batch_indices = torch.arange(eigenvalues.shape[0])
    diag_indices = torch.arange(eigenvalues.shape[1])
    eig_diag[batch_indices[:, None], diag_indices, diag_indices] = eigenvalues

    result = torch.matmul(torch.matmul(V_rot, eig_diag), torch.transpose(V_rot, 1, 2))
    return torch.real(result), torch.real(V_rot), torch.real(eigenvalues)


def add_gaussian_noise(
    A: torch.Tensor | NDArray[np.float64], eps: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A_tensor: torch.Tensor = torch.tensor(A, dtype=torch.float32)

    A_noisy = A_tensor + eps * torch.randn_like(A_tensor)
    eigenvalues, V = torch.linalg.eigh(A_noisy)
    return (
        torch.tensor(A_noisy, dtype=torch.float32),
        torch.tensor(V, dtype=torch.float32),
        torch.tensor(eigenvalues, dtype=torch.float32),
    )


def add_digress_noise(
    A: torch.Tensor | NDArray[np.float64],
    p: float,
    rng: np.random.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Add noise to an adjacency matrix by flipping edges with probability p.

    Parameters:
    - adj_matrix: A 2D numpy array or tensor representing an adjacency matrix (0s and 1s)
    - p: Probability of flipping each element (0 to 1, 1 becomes 0 and 0 becomes 1)
    - rng: Random number generator (optional)

    Returns:
    - Noisy adjacency matrix with some edges flipped
    """
    if rng is None:
        rng = np.random.default_rng()

    # Generate random values for each element
    A_tensor = torch.tensor(A)
    random_values = torch.rand_like(A_tensor)

    # Create a mask for elements to flip (where random value < p)
    flip_mask = random_values < p

    # Flip the elements where the mask is True (using XOR operation)
    # XOR with 1 flips 0→1 and 1→0
    A_noisy: torch.Tensor = torch.where(flip_mask, 1 - A_tensor, A_tensor)

    eigenvalues, V = torch.linalg.eigh(A_noisy)

    return (
        torch.tensor(A_noisy, dtype=torch.float32),
        torch.tensor(V, dtype=torch.float32),
        torch.tensor(eigenvalues, dtype=torch.float32),
    )


class AdjacencyMatrixDataset(Dataset[torch.Tensor]):
    adj_matrix: NDArray[np.float64]
    num_samples_per_epoch: int

    def __init__(
        self, adj_matrix: NDArray[np.float64], num_samples_per_epoch: int
    ) -> None:
        self.adj_matrix = adj_matrix
        self.num_samples_per_epoch = num_samples_per_epoch

    @override
    def __len__(self) -> int:
        return self.num_samples_per_epoch

    @override
    def __getitem__(self, idx: int) -> torch.Tensor:
        # Generate a random permutation of the adjacency matrix
        permuted_matrix = self.permute_matrix(self.adj_matrix)
        return permuted_matrix

    def permute_matrix(self, matrix: NDArray[np.float64]) -> torch.Tensor:
        # Generate a random permutation of indices
        indices = np.random.permutation(matrix.shape[0])

        # Apply the permutation to both rows and columns
        permuted_matrix = matrix[indices, :][:, indices]

        # Convert to PyTorch tensor
        return torch.tensor(permuted_matrix, dtype=torch.float32)
