"""Spectral analysis of collected eigenstructure data.

Provides analysis functions for studying band gaps, eigenvalue distributions,
eigenvector coherence, and subspace properties of graph collections.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from .storage import iter_batches, load_decomposition_batch, load_manifest


@dataclass
class SpectralAnalysisResult:
    """Container for spectral analysis results."""

    dataset_name: str
    num_graphs: int

    # Spectral gap statistics (lambda_max - lambda_{max-1} for adjacency)
    spectral_gap_mean: float
    spectral_gap_std: float
    spectral_gap_min: float
    spectral_gap_max: float

    # Algebraic connectivity / band gap (lambda_2 for Laplacian)
    algebraic_connectivity_mean: float
    algebraic_connectivity_std: float
    algebraic_connectivity_min: float
    algebraic_connectivity_max: float

    # Eigenvalue distribution statistics
    eigenvalue_entropy_adj: float
    eigenvalue_entropy_lap: float

    # Eigenvector coherence (localization measure)
    coherence_mean: float
    coherence_std: float

    # Effective rank (participation ratio of eigenvalues)
    effective_rank_adj_mean: float
    effective_rank_lap_mean: float


def compute_spectral_gap(eigenvalues: torch.Tensor) -> torch.Tensor:
    """
    Compute spectral gap: difference between largest two eigenvalues.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        Sorted eigenvalues of shape (batch, n), ascending order.

    Returns
    -------
    torch.Tensor
        Spectral gaps of shape (batch,).

    Notes
    -----
    For adjacency matrices, the spectral gap relates to mixing time
    and expansion properties of the graph.
    """
    return eigenvalues[:, -1] - eigenvalues[:, -2]


def compute_algebraic_connectivity(lap_eigenvalues: torch.Tensor) -> torch.Tensor:
    """
    Compute algebraic connectivity (Fiedler value): second smallest Laplacian eigenvalue.

    Parameters
    ----------
    lap_eigenvalues : torch.Tensor
        Sorted Laplacian eigenvalues of shape (batch, n), ascending order.

    Returns
    -------
    torch.Tensor
        Algebraic connectivity of shape (batch,).

    Notes
    -----
    For connected graphs, lambda_1 = 0 and lambda_2 > 0. The algebraic
    connectivity measures robustness to edge removal and bounds
    isoperimetric properties.
    """
    return lap_eigenvalues[:, 1]


def compute_eigenvector_coherence(eigenvectors: torch.Tensor) -> torch.Tensor:
    """
    Compute eigenvector coherence: max squared component magnitude.

    Parameters
    ----------
    eigenvectors : torch.Tensor
        Eigenvectors of shape (batch, n, n).

    Returns
    -------
    torch.Tensor
        Coherence values of shape (batch,).

    Notes
    -----
    Low coherence indicates delocalized eigenvectors spread across all nodes.
    High coherence indicates localized eigenvectors concentrated on few nodes.
    For random graphs, coherence scales as O(log(n)/n).
    """
    squared = eigenvectors**2
    max_per_vec = squared.max(dim=1).values  # (batch, n)
    return max_per_vec.max(dim=1).values  # (batch,)


def compute_eigenvalue_entropy(eigenvalues: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute entropy of normalized eigenvalue distribution.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        Eigenvalues of shape (batch, n).
    eps : float
        Small value to avoid log(0).

    Returns
    -------
    float
        Mean entropy across all graphs.

    Notes
    -----
    High entropy indicates spread-out eigenvalues; low entropy indicates
    clustering. Normalized by log(n) to be scale-invariant.
    """
    abs_eigs = eigenvalues.abs()
    # Normalize to probability distribution per graph
    probs = abs_eigs / (abs_eigs.sum(dim=-1, keepdim=True) + eps)
    entropy = -(probs * (probs + eps).log()).sum(dim=-1)
    return entropy.mean().item()


def compute_effective_rank(eigenvalues: torch.Tensor, eps: float = 1e-10) -> float:
    """
    Compute effective rank via participation ratio of eigenvalues.

    Parameters
    ----------
    eigenvalues : torch.Tensor
        Eigenvalues of shape (batch, n).
    eps : float
        Small value to avoid division by zero.

    Returns
    -------
    float
        Mean effective rank across all graphs.

    Notes
    -----
    The participation ratio is (sum |lambda_i|)^2 / sum |lambda_i|^2,
    which equals n for uniform eigenvalues and 1 for a single nonzero eigenvalue.
    """
    abs_eigs = eigenvalues.abs()
    sum_squared = (abs_eigs**2).sum(dim=-1)
    sum_abs_squared = (abs_eigs.sum(dim=-1)) ** 2
    pr = sum_abs_squared / (sum_squared + eps)
    return pr.mean().item()


def compute_eigengap_delta(
    eigenvalues_clean: torch.Tensor,
    eigenvalues_other: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute change in spectral gap between clean and other (noisy/denoised).

    Parameters
    ----------
    eigenvalues_clean : torch.Tensor
        Clean graph eigenvalues, shape (batch, n), ascending order.
    eigenvalues_other : torch.Tensor
        Noisy or denoised eigenvalues, same shape.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with 'absolute' and 'relative' delta tensors, shape (batch,).
        Positive relative delta means the gap increased in the other graph.
    """
    gap_clean = eigenvalues_clean[:, -1] - eigenvalues_clean[:, -2]
    gap_other = eigenvalues_other[:, -1] - eigenvalues_other[:, -2]
    delta_absolute = gap_other - gap_clean
    delta_relative = delta_absolute / (gap_clean.abs() + 1e-10)
    return {"absolute": delta_absolute, "relative": delta_relative}


def compute_algebraic_connectivity_delta(
    lap_eigenvalues_clean: torch.Tensor,
    lap_eigenvalues_other: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """
    Compute change in Fiedler value (lambda_2 of Laplacian).

    Parameters
    ----------
    lap_eigenvalues_clean : torch.Tensor
        Clean graph Laplacian eigenvalues, shape (batch, n), ascending order.
    lap_eigenvalues_other : torch.Tensor
        Noisy or denoised Laplacian eigenvalues, same shape.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary with 'absolute' and 'relative' delta tensors, shape (batch,).
        Positive relative delta means connectivity increased in the other graph.
    """
    alg_conn_clean = lap_eigenvalues_clean[:, 1]
    alg_conn_other = lap_eigenvalues_other[:, 1]
    delta_absolute = alg_conn_other - alg_conn_clean
    delta_relative = delta_absolute / (alg_conn_clean.abs() + 1e-10)
    return {"absolute": delta_absolute, "relative": delta_relative}


def compute_eigenvalue_drift(
    eigenvalues_clean: torch.Tensor,
    eigenvalues_other: torch.Tensor,
) -> torch.Tensor:
    """
    Compute relative L2 drift of eigenvalues.

    Parameters
    ----------
    eigenvalues_clean : torch.Tensor
        Clean graph eigenvalues, shape (batch, n).
    eigenvalues_other : torch.Tensor
        Noisy or denoised eigenvalues, same shape.

    Returns
    -------
    torch.Tensor
        Relative drift ||lambda_other - lambda_clean||_2 / ||lambda_clean||_2,
        shape (batch,). Values near 0 indicate similar spectra.
    """
    diff = torch.norm(eigenvalues_other - eigenvalues_clean, dim=-1)
    norm = torch.norm(eigenvalues_clean, dim=-1) + 1e-10
    return diff / norm


def compute_subspace_distance(
    eigenvectors_clean: torch.Tensor,
    eigenvectors_other: torch.Tensor,
    k: int,
) -> torch.Tensor:
    """
    Compute subspace distance via projection Frobenius norm.

    Parameters
    ----------
    eigenvectors_clean : torch.Tensor
        Clean graph eigenvectors, shape (batch, n, n).
    eigenvectors_other : torch.Tensor
        Noisy or denoised eigenvectors, same shape.
    k : int
        Number of top eigenvectors for subspace comparison.

    Returns
    -------
    torch.Tensor
        Frobenius norm of projection difference, shape (batch,).
        Values near 0 indicate aligned subspaces.

    Notes
    -----
    This computes ||P_clean - P_other||_F where P = V_k @ V_k^T is the
    projection onto the top-k eigenvector subspace.
    """
    V_clean_k = eigenvectors_clean[:, :, -k:]  # Top-k eigenvectors (batch, n, k)
    V_other_k = eigenvectors_other[:, :, -k:]
    P_clean = V_clean_k @ V_clean_k.transpose(-2, -1)  # (batch, n, n)
    P_other = V_other_k @ V_other_k.transpose(-2, -1)
    return torch.norm(P_clean - P_other, p="fro", dim=(-2, -1))


def compute_principal_angles(
    V1: torch.Tensor, V2: torch.Tensor, k: int
) -> torch.Tensor:
    """
    Compute principal angles between subspaces spanned by top-k eigenvectors.

    Parameters
    ----------
    V1, V2 : torch.Tensor
        Eigenvector matrices of shape (n, n).
    k : int
        Number of top eigenvectors to consider.

    Returns
    -------
    torch.Tensor
        Principal angles in radians, shape (k,).

    Notes
    -----
    Principal angles measure the alignment between two subspaces. The first
    angle is 0 if the subspaces share a common direction, and pi/2 if they
    are orthogonal.
    """
    # Top-k eigenvectors by largest eigenvalue (last k columns, ascending order)
    U1 = V1[:, -k:]
    U2 = V2[:, -k:]

    # SVD of cross-correlation gives cosines of principal angles
    _, sigmas, _ = torch.linalg.svd(U1.T @ U2)
    sigmas = torch.clamp(sigmas, -1.0, 1.0)
    return torch.acos(sigmas)


def compute_procrustes_rotation(
    V1: torch.Tensor, V2: torch.Tensor, k: int
) -> dict[str, torch.Tensor]:
    """
    Compute optimal orthogonal Procrustes rotation aligning V2 to V1.

    Finds the rotation R that minimizes ‖U1 - U2 @ R‖_F where U1, U2 are
    the top-k eigenvector subspaces. The rotation angle quantifies how much
    the eigenvector basis rotated between the two matrices.

    Parameters
    ----------
    V1, V2 : torch.Tensor
        Eigenvector matrices of shape (n, n), columns sorted by ascending eigenvalue.
    k : int
        Number of top eigenvectors to consider.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing:
        - 'rotation': Optimal rotation matrix R of shape (k, k)
        - 'angle': Rotation angle in radians (scalar), computed from trace(R)
        - 'residual': Frobenius norm ‖U1 - U2 @ R‖_F after alignment (scalar)

    Notes
    -----
    The rotation angle is derived from trace(R) = 1 + 2*cos(θ) for 3D rotations.
    For higher dimensions, this gives an aggregate measure of rotation magnitude.
    A small angle indicates the eigenvector bases are nearly aligned; a large
    angle indicates significant rotation occurred.
    """
    # Top-k eigenvectors by largest eigenvalue (last k columns, ascending order)
    U1 = V1[:, -k:]
    U2 = V2[:, -k:]

    # Orthogonal Procrustes: find R minimizing ‖U1 - U2 @ R‖_F
    # Solution: R = V @ U^T where M = U1^T @ U2 = U @ S @ V^T
    M = U1.T @ U2
    U, _, Vt = torch.linalg.svd(M)
    R = Vt.T @ U.T  # Optimal orthogonal rotation

    # Rotation angle from trace: for orthogonal matrix, trace relates to rotation
    # In k dimensions: trace(R) = sum of cosines of rotation angles
    # We compute an aggregate angle measure
    trace_R = torch.trace(R)
    # Clamp to handle numerical issues: trace should be in [-k, k]
    cos_theta = (trace_R - 1) / max(k - 1, 1)
    cos_theta = cos_theta.clamp(-1.0, 1.0)
    angle = torch.acos(cos_theta)

    # Residual after optimal alignment
    residual = torch.norm(U1 - U2 @ R, p="fro")

    return {"rotation": R, "angle": angle, "residual": residual}


def compute_procrustes_rotation_batch(
    V1_batch: torch.Tensor, V2_batch: torch.Tensor, k: int
) -> dict[str, torch.Tensor]:
    """
    Compute Procrustes rotation for a batch of eigenvector pairs.

    Parameters
    ----------
    V1_batch, V2_batch : torch.Tensor
        Batched eigenvector matrices of shape (batch, n, n).
    k : int
        Number of top eigenvectors to consider.

    Returns
    -------
    dict[str, torch.Tensor]
        Dictionary containing:
        - 'angles': Rotation angles, shape (batch,)
        - 'residuals': Frobenius residuals, shape (batch,)
    """
    batch_size = V1_batch.shape[0]
    angles = []
    residuals = []

    for i in range(batch_size):
        result = compute_procrustes_rotation(V1_batch[i], V2_batch[i], k)
        angles.append(result["angle"])
        residuals.append(result["residual"])

    return {
        "angles": torch.stack(angles),
        "residuals": torch.stack(residuals),
    }


class SpectralAnalyzer:
    """
    Analyze collected eigenstructure data.

    Parameters
    ----------
    input_dir : Path
        Directory containing batch_*.safetensors files and manifest.json.
    """

    def __init__(self, input_dir: Path):
        self.input_dir = Path(input_dir)
        self.manifest = load_manifest(self.input_dir)

    def analyze(self) -> SpectralAnalysisResult:
        """
        Run full spectral analysis on collected data.

        Returns
        -------
        SpectralAnalysisResult
            Dataclass with all computed statistics.
        """
        all_spectral_gaps: list[torch.Tensor] = []
        all_algebraic_conn: list[torch.Tensor] = []
        all_coherences: list[torch.Tensor] = []
        all_adj_eigenvalues: list[torch.Tensor] = []
        all_lap_eigenvalues: list[torch.Tensor] = []

        batch_paths = iter_batches(self.input_dir)
        logger.info(f"Analyzing {len(batch_paths)} batches")

        for batch_path in batch_paths:
            tensors, _ = load_decomposition_batch(batch_path)

            eig_adj = tensors["eigenvalues_adj"]
            eig_lap = tensors["eigenvalues_lap"]
            vec_adj = tensors["eigenvectors_adj"]

            all_spectral_gaps.append(compute_spectral_gap(eig_adj))
            all_algebraic_conn.append(compute_algebraic_connectivity(eig_lap))
            all_coherences.append(compute_eigenvector_coherence(vec_adj))
            all_adj_eigenvalues.append(eig_adj)
            all_lap_eigenvalues.append(eig_lap)

        # Concatenate across batches
        spectral_gaps = torch.cat(all_spectral_gaps)
        algebraic_conn = torch.cat(all_algebraic_conn)
        coherences = torch.cat(all_coherences)
        adj_eigenvalues = torch.cat(all_adj_eigenvalues)
        lap_eigenvalues = torch.cat(all_lap_eigenvalues)

        return SpectralAnalysisResult(
            dataset_name=self.manifest["dataset_name"],
            num_graphs=self.manifest["num_graphs"],
            spectral_gap_mean=spectral_gaps.mean().item(),
            spectral_gap_std=spectral_gaps.std().item(),
            spectral_gap_min=spectral_gaps.min().item(),
            spectral_gap_max=spectral_gaps.max().item(),
            algebraic_connectivity_mean=algebraic_conn.mean().item(),
            algebraic_connectivity_std=algebraic_conn.std().item(),
            algebraic_connectivity_min=algebraic_conn.min().item(),
            algebraic_connectivity_max=algebraic_conn.max().item(),
            eigenvalue_entropy_adj=compute_eigenvalue_entropy(adj_eigenvalues),
            eigenvalue_entropy_lap=compute_eigenvalue_entropy(lap_eigenvalues),
            coherence_mean=coherences.mean().item(),
            coherence_std=coherences.std().item(),
            effective_rank_adj_mean=compute_effective_rank(adj_eigenvalues),
            effective_rank_lap_mean=compute_effective_rank(lap_eigenvalues),
        )

    def save_results(self, result: SpectralAnalysisResult, output_dir: Path) -> Path:
        """
        Save analysis results to JSON.

        Parameters
        ----------
        result : SpectralAnalysisResult
            Analysis results to save.
        output_dir : Path
            Directory to write analysis.json.

        Returns
        -------
        Path
            Path to saved JSON file.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path = output_dir / "analysis.json"
        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        return output_path

    def compute_eigenvalue_histogram(
        self, num_bins: int = 50, matrix_type: str = "adjacency"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute histogram of eigenvalues across all graphs.

        Parameters
        ----------
        num_bins : int
            Number of histogram bins.
        matrix_type : str
            Either "adjacency" or "laplacian".

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            (bin_edges, counts) for the histogram.
        """
        all_eigenvalues: list[torch.Tensor] = []
        key = "eigenvalues_adj" if matrix_type == "adjacency" else "eigenvalues_lap"

        for batch_path in iter_batches(self.input_dir):
            tensors, _ = load_decomposition_batch(batch_path)
            all_eigenvalues.append(tensors[key].flatten())

        eigenvalues = torch.cat(all_eigenvalues)
        return torch.histogram(eigenvalues, bins=num_bins)

    def compute_subspace_distances(
        self, k: int = 10, sample_size: int = 100
    ) -> dict[str, Any]:
        """
        Compute pairwise subspace distances between random pairs of graphs.

        Parameters
        ----------
        k : int
            Number of top eigenvectors to use for subspace comparison.
        sample_size : int
            Number of random pairs to sample.

        Returns
        -------
        dict
            Statistics on principal angles between graph subspaces.
        """
        all_eigenvectors: list[torch.Tensor] = []

        for batch_path in iter_batches(self.input_dir):
            tensors, _ = load_decomposition_batch(batch_path)
            all_eigenvectors.append(tensors["eigenvectors_adj"])

        eigenvectors = torch.cat(all_eigenvectors, dim=0)
        num_graphs = eigenvectors.shape[0]

        if num_graphs < 2:
            return {"error": "Need at least 2 graphs for subspace comparison"}

        # Sample random pairs
        rng = torch.Generator().manual_seed(42)
        idx1 = torch.randint(0, num_graphs, (sample_size,), generator=rng)
        idx2 = torch.randint(0, num_graphs, (sample_size,), generator=rng)

        first_angles: list[float] = []
        mean_angles: list[float] = []

        for i, j in zip(idx1.tolist(), idx2.tolist(), strict=False):
            if i == j:
                continue
            V1, V2 = eigenvectors[i], eigenvectors[j]
            angles = compute_principal_angles(V1, V2, k)
            first_angles.append(angles[0].item())
            mean_angles.append(angles.mean().item())

        first_angles_t = torch.tensor(first_angles)
        mean_angles_t = torch.tensor(mean_angles)

        return {
            "k": k,
            "num_pairs": len(first_angles),
            "first_principal_angle_mean": first_angles_t.mean().item(),
            "first_principal_angle_std": first_angles_t.std().item(),
            "mean_principal_angle_mean": mean_angles_t.mean().item(),
            "mean_principal_angle_std": mean_angles_t.std().item(),
        }
