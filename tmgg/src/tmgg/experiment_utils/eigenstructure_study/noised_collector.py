"""Noised eigenstructure collector.

Collects eigendecompositions of noised graphs at configurable noise levels,
reusing original adjacency matrices from Phase 1 collection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from loguru import logger

from tmgg.experiment_utils.data.eigendecomposition import compute_eigendecomposition
from tmgg.experiment_utils.data.noise_generators import create_noise_generator

from .laplacian import compute_laplacian
from .storage import (
    iter_batches,
    load_decomposition_batch,
    load_manifest,
    save_dataset_manifest,
    save_decomposition_batch,
)


class NoisedEigenstructureCollector:
    """
    Collect eigendecompositions for noised versions of graphs.

    Reads original adjacency matrices from Phase 1 output, applies noise
    at multiple levels, computes decompositions, and stores in separate
    subdirectories per noise level.

    Parameters
    ----------
    input_dir : Path
        Directory containing Phase 1 output (batch_*.safetensors and manifest.json).
    output_dir : Path
        Base directory for noised decompositions. Subdirectories eps_X.XXXX/
        will be created for each noise level.
    noise_type : str
        Type of noise: "gaussian", "digress", or "rotation".
    noise_levels : list[float]
        Noise levels (epsilon values) to apply.
    rotation_k : int, optional
        Dimension for rotation noise skew matrix. Required if noise_type="rotation".
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_dir: Path,
        output_dir: Path,
        noise_type: str,
        noise_levels: list[float],
        rotation_k: int | None = None,
        seed: int = 42,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.noise_type = noise_type.lower()
        self.noise_levels = noise_levels
        self.rotation_k = rotation_k
        self.seed = seed

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.manifest = load_manifest(self.input_dir)
        self._create_noise_generator()

    def _create_noise_generator(self) -> None:
        """Initialize the noise generator."""
        if self.noise_type == "rotation" and self.rotation_k is None:
            raise ValueError("rotation_k is required for rotation noise")

        self.noise_gen = create_noise_generator(
            self.noise_type,
            rotation_k=self.rotation_k,
            seed=self.seed,
        )

    def collect(self) -> None:
        """Run noised collection for all noise levels."""
        for eps in self.noise_levels:
            eps_dir = self.output_dir / f"eps_{eps:.4f}"
            eps_dir.mkdir(exist_ok=True)

            logger.info(f"Processing noise level eps={eps}")
            self._collect_at_noise_level(eps, eps_dir)

            # Save manifest for this noise level
            save_dataset_manifest(
                eps_dir,
                f"{self.manifest['dataset_name']}_noised_{self.noise_type}",
                {
                    **self.manifest["dataset_config"],
                    "noise_type": self.noise_type,
                    "noise_level": eps,
                    "original_dataset": self.manifest["dataset_name"],
                },
                self.manifest["num_graphs"],
                self.manifest["num_batches"],
                self.seed,
            )
            logger.info(f"Completed eps={eps}, saved to {eps_dir}")

    def _collect_at_noise_level(self, eps: float, output_dir: Path) -> None:
        """Collect decompositions for a single noise level."""
        for batch_path in iter_batches(self.input_dir):
            tensors, metadata = load_decomposition_batch(batch_path)
            A_batch = tensors["adjacency"]

            # Apply noise
            A_noised = self.noise_gen.add_noise(A_batch, eps)

            # Process each graph in the batch
            eig_adj_vals_list: list[torch.Tensor] = []
            eig_adj_vecs_list: list[torch.Tensor] = []
            eig_lap_vals_list: list[torch.Tensor] = []
            eig_lap_vecs_list: list[torch.Tensor] = []

            for i in range(A_noised.shape[0]):
                A = A_noised[i]
                L = compute_laplacian(A)

                vals_a, vecs_a = compute_eigendecomposition(A)
                vals_l, vecs_l = compute_eigendecomposition(L)

                eig_adj_vals_list.append(vals_a)
                eig_adj_vecs_list.append(vecs_a)
                eig_lap_vals_list.append(vals_l)
                eig_lap_vecs_list.append(vecs_l)

            # Extract batch index from filename
            batch_idx = int(batch_path.stem.split("_")[1])

            save_decomposition_batch(
                output_dir,
                batch_idx,
                torch.stack(eig_adj_vals_list),
                torch.stack(eig_adj_vecs_list),
                torch.stack(eig_lap_vals_list),
                torch.stack(eig_lap_vecs_list),
                A_noised,
                metadata,
            )


class NoisedAnalysisComparator:
    """
    Compare eigenstructure between original and noised graphs.

    Parameters
    ----------
    original_dir : Path
        Directory with original (Phase 1) decompositions.
    noised_base_dir : Path
        Base directory with noised decompositions (contains eps_*/ subdirs).
    """

    def __init__(self, original_dir: Path, noised_base_dir: Path):
        self.original_dir = Path(original_dir)
        self.noised_base_dir = Path(noised_base_dir)

    def get_noise_levels(self) -> list[float]:
        """Get available noise levels from directory structure."""
        levels = []
        for subdir in self.noised_base_dir.iterdir():
            if subdir.is_dir() and subdir.name.startswith("eps_"):
                try:
                    eps = float(subdir.name.replace("eps_", ""))
                    levels.append(eps)
                except ValueError:
                    continue
        return sorted(levels)

    def compute_eigenvalue_drift(self, eps: float) -> dict[str, Any]:
        """
        Compute eigenvalue drift between original and noised graphs.

        Parameters
        ----------
        eps : float
            Noise level to compare.

        Returns
        -------
        dict
            Statistics on eigenvalue drift for both adjacency and Laplacian.
        """
        noised_dir = self.noised_base_dir / f"eps_{eps:.4f}"

        adj_drifts: list[torch.Tensor] = []
        lap_drifts: list[torch.Tensor] = []

        original_batches = iter_batches(self.original_dir)
        noised_batches = iter_batches(noised_dir)

        for orig_path, noised_path in zip(
            original_batches, noised_batches, strict=False
        ):
            orig_tensors, _ = load_decomposition_batch(orig_path)
            noised_tensors, _ = load_decomposition_batch(noised_path)

            orig_adj = orig_tensors["eigenvalues_adj"]
            noised_adj = noised_tensors["eigenvalues_adj"]
            orig_lap = orig_tensors["eigenvalues_lap"]
            noised_lap = noised_tensors["eigenvalues_lap"]

            # Relative L2 drift per graph
            adj_diff = torch.norm(noised_adj - orig_adj, dim=-1)
            adj_norm = torch.norm(orig_adj, dim=-1) + 1e-10
            adj_drifts.append(adj_diff / adj_norm)

            lap_diff = torch.norm(noised_lap - orig_lap, dim=-1)
            lap_norm = torch.norm(orig_lap, dim=-1) + 1e-10
            lap_drifts.append(lap_diff / lap_norm)

        adj_drift = torch.cat(adj_drifts)
        lap_drift = torch.cat(lap_drifts)

        return {
            "noise_level": eps,
            "adjacency_drift_mean": adj_drift.mean().item(),
            "adjacency_drift_std": adj_drift.std().item(),
            "laplacian_drift_mean": lap_drift.mean().item(),
            "laplacian_drift_std": lap_drift.std().item(),
        }

    def compute_eigengap_delta(self, eps: float) -> dict[str, Any]:
        """
        Compute eigengap delta between original and noised graphs.

        Parameters
        ----------
        eps : float
            Noise level to compare.

        Returns
        -------
        dict
            Statistics on spectral gap change for adjacency matrices.
        """
        from .analyzer import compute_eigengap_delta

        noised_dir = self.noised_base_dir / f"eps_{eps:.4f}"

        abs_deltas: list[torch.Tensor] = []
        rel_deltas: list[torch.Tensor] = []

        original_batches = iter_batches(self.original_dir)
        noised_batches = iter_batches(noised_dir)

        for orig_path, noised_path in zip(
            original_batches, noised_batches, strict=False
        ):
            orig_tensors, _ = load_decomposition_batch(orig_path)
            noised_tensors, _ = load_decomposition_batch(noised_path)

            orig_adj = orig_tensors["eigenvalues_adj"]
            noised_adj = noised_tensors["eigenvalues_adj"]

            delta = compute_eigengap_delta(orig_adj, noised_adj)
            abs_deltas.append(delta["absolute"])
            rel_deltas.append(delta["relative"])

        abs_delta = torch.cat(abs_deltas)
        rel_delta = torch.cat(rel_deltas)

        return {
            "noise_level": eps,
            "eigengap_delta_abs_mean": abs_delta.mean().item(),
            "eigengap_delta_abs_std": abs_delta.std().item(),
            "eigengap_delta_rel_mean": rel_delta.mean().item(),
            "eigengap_delta_rel_std": rel_delta.std().item(),
        }

    def compute_algebraic_connectivity_delta(self, eps: float) -> dict[str, Any]:
        """
        Compute algebraic connectivity delta between original and noised graphs.

        Parameters
        ----------
        eps : float
            Noise level to compare.

        Returns
        -------
        dict
            Statistics on Fiedler value change for Laplacian matrices.
        """
        from .analyzer import compute_algebraic_connectivity_delta

        noised_dir = self.noised_base_dir / f"eps_{eps:.4f}"

        abs_deltas: list[torch.Tensor] = []
        rel_deltas: list[torch.Tensor] = []

        original_batches = iter_batches(self.original_dir)
        noised_batches = iter_batches(noised_dir)

        for orig_path, noised_path in zip(
            original_batches, noised_batches, strict=False
        ):
            orig_tensors, _ = load_decomposition_batch(orig_path)
            noised_tensors, _ = load_decomposition_batch(noised_path)

            orig_lap = orig_tensors["eigenvalues_lap"]
            noised_lap = noised_tensors["eigenvalues_lap"]

            delta = compute_algebraic_connectivity_delta(orig_lap, noised_lap)
            abs_deltas.append(delta["absolute"])
            rel_deltas.append(delta["relative"])

        abs_delta = torch.cat(abs_deltas)
        rel_delta = torch.cat(rel_deltas)

        return {
            "noise_level": eps,
            "alg_conn_delta_abs_mean": abs_delta.mean().item(),
            "alg_conn_delta_abs_std": abs_delta.std().item(),
            "alg_conn_delta_rel_mean": rel_delta.mean().item(),
            "alg_conn_delta_rel_std": rel_delta.std().item(),
        }

    def compute_subspace_distance(self, eps: float, k: int = 10) -> dict[str, Any]:
        """
        Compute subspace distance between original and noised eigenvectors.

        Parameters
        ----------
        eps : float
            Noise level to compare.
        k : int
            Number of top eigenvectors to compare.

        Returns
        -------
        dict
            Statistics on projection Frobenius distance.
        """
        from .analyzer import compute_subspace_distance

        noised_dir = self.noised_base_dir / f"eps_{eps:.4f}"

        distances: list[torch.Tensor] = []

        original_batches = iter_batches(self.original_dir)
        noised_batches = iter_batches(noised_dir)

        for orig_path, noised_path in zip(
            original_batches, noised_batches, strict=False
        ):
            orig_tensors, _ = load_decomposition_batch(orig_path)
            noised_tensors, _ = load_decomposition_batch(noised_path)

            orig_vecs = orig_tensors["eigenvectors_adj"]
            noised_vecs = noised_tensors["eigenvectors_adj"]

            dist = compute_subspace_distance(orig_vecs, noised_vecs, k)
            distances.append(dist)

        all_distances = torch.cat(distances)

        return {
            "noise_level": eps,
            "k": k,
            "subspace_distance_mean": all_distances.mean().item(),
            "subspace_distance_std": all_distances.std().item(),
        }

    def compute_subspace_deviation(self, eps: float, k: int = 10) -> dict[str, Any]:
        """
        Compute subspace deviation via principal angles (legacy method).

        Parameters
        ----------
        eps : float
            Noise level to compare.
        k : int
            Number of top eigenvectors to compare.

        Returns
        -------
        dict
            Statistics on principal angles between original and noised subspaces.

        Notes
        -----
        For batched computation, prefer compute_subspace_distance which uses
        projection Frobenius norm and processes entire batches efficiently.
        """
        from .analyzer import compute_principal_angles

        noised_dir = self.noised_base_dir / f"eps_{eps:.4f}"

        first_angles: list[float] = []
        mean_angles: list[float] = []

        original_batches = iter_batches(self.original_dir)
        noised_batches = iter_batches(noised_dir)

        for orig_path, noised_path in zip(
            original_batches, noised_batches, strict=False
        ):
            orig_tensors, _ = load_decomposition_batch(orig_path)
            noised_tensors, _ = load_decomposition_batch(noised_path)

            orig_vecs = orig_tensors["eigenvectors_adj"]
            noised_vecs = noised_tensors["eigenvectors_adj"]

            for i in range(orig_vecs.shape[0]):
                angles = compute_principal_angles(orig_vecs[i], noised_vecs[i], k)
                first_angles.append(angles[0].item())
                mean_angles.append(angles.mean().item())

        first_angles_t = torch.tensor(first_angles)
        mean_angles_t = torch.tensor(mean_angles)

        return {
            "noise_level": eps,
            "k": k,
            "first_principal_angle_mean": first_angles_t.mean().item(),
            "first_principal_angle_std": first_angles_t.std().item(),
            "mean_principal_angle_mean": mean_angles_t.mean().item(),
            "mean_principal_angle_std": mean_angles_t.std().item(),
        }

    def compute_procrustes_rotation(
        self, eps: float, k_values: list[int] | None = None
    ) -> dict[str, Any]:
        """
        Compute Procrustes rotation metrics between original and noised eigenvectors.

        Parameters
        ----------
        eps : float
            Noise level to compare.
        k_values : list[int], optional
            List of k values for which to compute Procrustes rotation.
            Defaults to [1, 2, 4, 8, 16].

        Returns
        -------
        dict
            Statistics on Procrustes rotation angles and residuals for each k.
        """
        from .analyzer import compute_procrustes_rotation

        if k_values is None:
            k_values = [1, 2, 4, 8, 16]

        noised_dir = self.noised_base_dir / f"eps_{eps:.4f}"

        # Collect angles and residuals for each k
        angles_by_k: dict[int, list[float]] = {k: [] for k in k_values}
        residuals_by_k: dict[int, list[float]] = {k: [] for k in k_values}

        original_batches = iter_batches(self.original_dir)
        noised_batches = iter_batches(noised_dir)

        for orig_path, noised_path in zip(
            original_batches, noised_batches, strict=False
        ):
            orig_tensors, _ = load_decomposition_batch(orig_path)
            noised_tensors, _ = load_decomposition_batch(noised_path)

            orig_vecs = orig_tensors["eigenvectors_adj"]
            noised_vecs = noised_tensors["eigenvectors_adj"]

            for i in range(orig_vecs.shape[0]):
                for k in k_values:
                    # Skip if k is larger than matrix dimension
                    if k > orig_vecs.shape[1]:
                        continue
                    result = compute_procrustes_rotation(
                        orig_vecs[i], noised_vecs[i], k
                    )
                    angles_by_k[k].append(result["angle"].item())
                    residuals_by_k[k].append(result["residual"].item())

        # Compute statistics for each k
        procrustes_stats: dict[str, Any] = {"noise_level": eps}
        for k in k_values:
            if angles_by_k[k]:
                angles_t = torch.tensor(angles_by_k[k])
                residuals_t = torch.tensor(residuals_by_k[k])
                procrustes_stats[f"procrustes_angle_k{k}_mean"] = angles_t.mean().item()
                procrustes_stats[f"procrustes_angle_k{k}_std"] = angles_t.std().item()
                procrustes_stats[f"procrustes_residual_k{k}_mean"] = (
                    residuals_t.mean().item()
                )
                procrustes_stats[f"procrustes_residual_k{k}_std"] = (
                    residuals_t.std().item()
                )

        return procrustes_stats

    def compute_full_comparison(
        self, k: int = 10, procrustes_k_values: list[int] | None = None
    ) -> list[dict[str, Any]]:
        """
        Compute full comparison across all noise levels.

        Computes all delta metrics:
        - Eigengap delta (spectral gap change)
        - Algebraic connectivity delta (Fiedler value change)
        - Eigenvalue drift (relative L2 distance)
        - Subspace distance (projection Frobenius norm)
        - Procrustes rotation (optimal rotation angle and residual)

        Parameters
        ----------
        k : int
            Number of eigenvectors for subspace comparison.
        procrustes_k_values : list[int], optional
            List of k values for Procrustes rotation analysis.
            Defaults to [1, 2, 4, 8, 16].

        Returns
        -------
        list[dict]
            List of comparison results, one per noise level. Each dict contains
            statistics for all metrics.
        """
        if procrustes_k_values is None:
            procrustes_k_values = [1, 2, 4, 8, 16]

        results = []
        for eps in self.get_noise_levels():
            logger.info(f"Comparing at eps={eps}")
            eigengap = self.compute_eigengap_delta(eps)
            alg_conn = self.compute_algebraic_connectivity_delta(eps)
            drift = self.compute_eigenvalue_drift(eps)
            subspace = self.compute_subspace_distance(eps, k)
            procrustes = self.compute_procrustes_rotation(eps, procrustes_k_values)

            combined = {
                "noise_level": eps,
                "k": k,
                # Eigengap delta
                "eigengap_delta_rel_mean": eigengap["eigengap_delta_rel_mean"],
                "eigengap_delta_rel_std": eigengap["eigengap_delta_rel_std"],
                # Algebraic connectivity delta
                "alg_conn_delta_rel_mean": alg_conn["alg_conn_delta_rel_mean"],
                "alg_conn_delta_rel_std": alg_conn["alg_conn_delta_rel_std"],
                # Eigenvalue drift
                "eigenvalue_drift_adj_mean": drift["adjacency_drift_mean"],
                "eigenvalue_drift_adj_std": drift["adjacency_drift_std"],
                "eigenvalue_drift_lap_mean": drift["laplacian_drift_mean"],
                "eigenvalue_drift_lap_std": drift["laplacian_drift_std"],
                # Subspace distance
                "subspace_distance_mean": subspace["subspace_distance_mean"],
                "subspace_distance_std": subspace["subspace_distance_std"],
            }
            # Add Procrustes metrics
            for key, value in procrustes.items():
                if key != "noise_level":  # Skip noise_level as it's already included
                    combined[key] = value

            results.append(combined)

        return results

    def compute_covariance_evolution(
        self, matrix_type: str = "adjacency"
    ) -> dict[str, Any]:
        """
        Compute eigenvalue covariance at each noise level.

        Tracks how the covariance structure evolves as noise is added:
        higher noise typically decorrelates eigenvalues, reducing off-diagonal
        terms relative to the original.

        Parameters
        ----------
        matrix_type : str
            Either "adjacency" or "laplacian".

        Returns
        -------
        dict
            Dictionary containing:
            - original: CovarianceResult for original (clean) graphs
            - per_noise_level: list of dicts with noise_level, CovarianceResult,
              and delta from original (Frobenius norm change, correlation change)
        """
        from .analyzer import (
            CovarianceResult,
            compute_covariance_summary,
            compute_eigenvalue_covariance,
        )

        key = "eigenvalues_adj" if matrix_type == "adjacency" else "eigenvalues_lap"

        # Load original eigenvalues
        orig_eigenvalues: list[torch.Tensor] = []
        for batch_path in iter_batches(self.original_dir):
            tensors, _ = load_decomposition_batch(batch_path)
            orig_eigenvalues.append(tensors[key])

        orig_eig = torch.cat(orig_eigenvalues, dim=0)
        orig_cov = compute_eigenvalue_covariance(orig_eig)
        orig_summary = compute_covariance_summary(orig_cov)

        original_result = CovarianceResult(
            matrix_type=matrix_type,
            num_graphs=orig_eig.shape[0],
            num_eigenvalues=orig_eig.shape[1],
            covariance_matrix=orig_cov.tolist(),
            **orig_summary,
        )

        # Compute per noise level
        per_level_results: list[dict[str, Any]] = []

        for eps in self.get_noise_levels():
            noised_dir = self.noised_base_dir / f"eps_{eps:.4f}"

            noised_eigenvalues: list[torch.Tensor] = []
            for batch_path in iter_batches(noised_dir):
                tensors, _ = load_decomposition_batch(batch_path)
                noised_eigenvalues.append(tensors[key])

            noised_eig = torch.cat(noised_eigenvalues, dim=0)
            noised_cov = compute_eigenvalue_covariance(noised_eig)
            noised_summary = compute_covariance_summary(noised_cov)

            noised_result = CovarianceResult(
                matrix_type=matrix_type,
                num_graphs=noised_eig.shape[0],
                num_eigenvalues=noised_eig.shape[1],
                covariance_matrix=noised_cov.tolist(),
                **noised_summary,
            )

            # Compute delta metrics
            frob_delta = (
                torch.norm(noised_cov - orig_cov, p="fro").item()
                / orig_summary["frobenius_norm"]
            )
            trace_delta = (noised_summary["trace"] - orig_summary["trace"]) / (
                orig_summary["trace"] + 1e-10
            )
            off_diag_delta = (
                noised_summary["off_diagonal_sum"] - orig_summary["off_diagonal_sum"]
            ) / (abs(orig_summary["off_diagonal_sum"]) + 1e-10)

            per_level_results.append(
                {
                    "noise_level": eps,
                    "covariance": noised_result,
                    "frobenius_delta_relative": frob_delta,
                    "trace_delta_relative": trace_delta,
                    "off_diagonal_delta_relative": off_diag_delta,
                }
            )

            logger.info(
                f"eps={eps}: frob_delta={frob_delta:.4f}, "
                f"off_diag_delta={off_diag_delta:.4f}"
            )

        return {
            "matrix_type": matrix_type,
            "original": original_result,
            "per_noise_level": per_level_results,
        }
