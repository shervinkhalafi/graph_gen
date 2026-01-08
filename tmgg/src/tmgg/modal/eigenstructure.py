"""Modal functions for eigenstructure study execution.

Provides Modal functions to run eigenstructure collection, analysis,
and comparison on Modal with volume storage for results.

These functions run on CPU (no GPU needed for eigendecomposition).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from tmgg.modal.app import MEMORY_CONFIGS, app
from tmgg.modal.image import create_tmgg_image
from tmgg.modal.volumes import (
    EIGENSTRUCTURE_MOUNT,
    eigenstructure_volume,
    get_eigenstructure_volume_mounts,
)

# Create image for eigenstructure study (CPU only)
try:
    from tmgg.modal.paths import discover_tmgg_path

    _tmgg_path = discover_tmgg_path()
    eigenstructure_image = create_tmgg_image(_tmgg_path)
except Exception:
    eigenstructure_image = None


@app.function(
    name="eigenstructure_collect",
    image=eigenstructure_image,
    memory=MEMORY_CONFIGS["medium"],
    timeout=3600,  # 1 hour
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_collect(
    dataset_name: str,
    dataset_config: dict[str, Any],
    output_path: str,
    batch_size: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Collect eigendecompositions for a dataset on Modal.

    Parameters
    ----------
    dataset_name
        Dataset type (sbm, er, regular, etc.).
    dataset_config
        JSON-serializable dataset configuration.
    output_path
        Output path relative to eigenstructure volume mount.
    batch_size
        Number of graphs per batch.
    seed
        Random seed.

    Returns
    -------
    dict
        Result dict with status, output_dir, and manifest.
    """
    from tmgg.experiment_utils.eigenstructure_study import EigenstructureCollector
    from tmgg.experiment_utils.eigenstructure_study.storage import load_manifest

    full_output_path = Path(EIGENSTRUCTURE_MOUNT) / output_path
    full_output_path.mkdir(parents=True, exist_ok=True)

    collector = EigenstructureCollector(
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        output_dir=full_output_path,
        batch_size=batch_size,
        seed=seed,
    )

    collector.collect()

    # Read num_graphs from saved manifest
    manifest = load_manifest(full_output_path)

    # Commit changes to volume
    eigenstructure_volume.commit()

    return {
        "status": "completed",
        "output_dir": str(full_output_path),
        "dataset_name": dataset_name,
        "num_graphs": manifest["num_graphs"],
    }


@app.function(
    name="eigenstructure_analyze",
    image=eigenstructure_image,
    memory=MEMORY_CONFIGS["medium"],
    timeout=1800,  # 30 minutes
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_analyze(
    input_path: str,
    output_path: str,
    subspace_k: int = 10,
    compute_covariance: bool = True,
    matrix_type: str = "adjacency",
) -> dict[str, Any]:
    """Analyze collected eigenstructure data on Modal.

    Parameters
    ----------
    input_path
        Input path relative to eigenstructure volume mount.
    output_path
        Output path relative to eigenstructure volume mount.
    subspace_k
        Number of eigenvectors for subspace analysis.
    compute_covariance
        Whether to compute eigenvalue covariance.
    matrix_type
        Matrix type for covariance (adjacency or laplacian).

    Returns
    -------
    dict
        Result dict with status and analysis summary.
    """
    from tmgg.experiment_utils.eigenstructure_study import SpectralAnalyzer

    full_input_path = Path(EIGENSTRUCTURE_MOUNT) / input_path
    full_output_path = Path(EIGENSTRUCTURE_MOUNT) / output_path
    full_output_path.mkdir(parents=True, exist_ok=True)

    analyzer = SpectralAnalyzer(full_input_path)

    # Run main analysis
    result = analyzer.analyze()
    with open(full_output_path / "analysis.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    # Run subspace analysis
    if subspace_k > 0:
        subspace_results = analyzer.compute_subspace_distances(k=subspace_k)
        with open(full_output_path / "subspace_analysis.json", "w") as f:
            json.dump(subspace_results, f, indent=2)

    # Compute covariance if requested
    if compute_covariance:
        cov_result = analyzer.compute_eigenvalue_covariance(matrix_type)
        with open(full_output_path / "covariance.json", "w") as f:
            json.dump(asdict(cov_result), f, indent=2)

    eigenstructure_volume.commit()

    return {
        "status": "completed",
        "output_dir": str(full_output_path),
        "spectral_gap_mean": result.spectral_gap_mean,
        "algebraic_connectivity_mean": result.algebraic_connectivity_mean,
    }


@app.function(
    name="eigenstructure_noised",
    image=eigenstructure_image,
    memory=MEMORY_CONFIGS["large"],  # More memory for noised processing
    timeout=7200,  # 2 hours
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_noised(
    input_path: str,
    output_path: str,
    noise_type: str,
    noise_levels: list[float],
    rotation_k: int | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Collect eigendecompositions for noised graphs on Modal.

    Parameters
    ----------
    input_path
        Input path (original data) relative to eigenstructure volume mount.
    output_path
        Output path relative to eigenstructure volume mount.
    noise_type
        Type of noise (gaussian, digress, rotation).
    noise_levels
        List of noise levels (epsilon values).
    rotation_k
        Dimension for rotation noise (required for rotation type).
    seed
        Random seed.

    Returns
    -------
    dict
        Result dict with status and noise levels processed.
    """
    from tmgg.experiment_utils.eigenstructure_study import NoisedEigenstructureCollector

    full_input_path = Path(EIGENSTRUCTURE_MOUNT) / input_path
    full_output_path = Path(EIGENSTRUCTURE_MOUNT) / output_path
    full_output_path.mkdir(parents=True, exist_ok=True)

    collector = NoisedEigenstructureCollector(
        input_dir=full_input_path,
        output_dir=full_output_path,
        noise_type=noise_type,
        noise_levels=noise_levels,
        rotation_k=rotation_k,
        seed=seed,
    )

    collector.collect()
    eigenstructure_volume.commit()

    return {
        "status": "completed",
        "output_dir": str(full_output_path),
        "noise_type": noise_type,
        "noise_levels": noise_levels,
    }


@app.function(
    name="eigenstructure_compare",
    image=eigenstructure_image,
    memory=MEMORY_CONFIGS["large"],
    timeout=3600,  # 1 hour
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_compare(
    original_path: str,
    noised_path: str,
    output_path: str,
    subspace_k: int = 10,
    procrustes_k_values: list[int] | None = None,
    compute_covariance_evolution: bool = True,
    matrix_type: str = "adjacency",
) -> dict[str, Any]:
    """Compare original and noised eigenstructure on Modal.

    Parameters
    ----------
    original_path
        Original data path relative to eigenstructure volume mount.
    noised_path
        Noised data path relative to eigenstructure volume mount.
    output_path
        Output path relative to eigenstructure volume mount.
    subspace_k
        Number of eigenvectors for subspace comparison.
    procrustes_k_values
        List of k values for Procrustes analysis.
    compute_covariance_evolution
        Whether to compute covariance evolution.
    matrix_type
        Matrix type for covariance evolution.

    Returns
    -------
    dict
        Result dict with status and comparison summary.
    """
    from tmgg.experiment_utils.eigenstructure_study import NoisedAnalysisComparator

    full_original_path = Path(EIGENSTRUCTURE_MOUNT) / original_path
    full_noised_path = Path(EIGENSTRUCTURE_MOUNT) / noised_path
    full_output_path = Path(EIGENSTRUCTURE_MOUNT) / output_path
    full_output_path.mkdir(parents=True, exist_ok=True)

    if procrustes_k_values is None:
        procrustes_k_values = [1, 2, 4, 8, 16]

    comparator = NoisedAnalysisComparator(full_original_path, full_noised_path)

    # Run full comparison
    results = comparator.compute_full_comparison(
        k=subspace_k, procrustes_k_values=procrustes_k_values
    )
    with open(full_output_path / "comparison.json", "w") as f:
        json.dump(results, f, indent=2)

    # Compute covariance evolution if requested
    if compute_covariance_evolution:
        evolution = comparator.compute_covariance_evolution(matrix_type)

        # Convert dataclasses to dicts for JSON serialization
        output_data = {
            "matrix_type": evolution["matrix_type"],
            "original": asdict(evolution["original"]),
            "per_noise_level": [
                {
                    "noise_level": item["noise_level"],
                    "covariance": asdict(item["covariance"]),
                    "frobenius_delta_relative": item["frobenius_delta_relative"],
                    "trace_delta_relative": item["trace_delta_relative"],
                    "off_diagonal_delta_relative": item["off_diagonal_delta_relative"],
                }
                for item in evolution["per_noise_level"]
            ],
        }

        with open(full_output_path / "covariance_evolution.json", "w") as f:
            json.dump(output_data, f, indent=2)

    eigenstructure_volume.commit()

    return {
        "status": "completed",
        "output_dir": str(full_output_path),
        "noise_levels": comparator.get_noise_levels(),
        "num_comparisons": len(results),
    }


@app.function(
    name="eigenstructure_list",
    image=eigenstructure_image,
    memory=MEMORY_CONFIGS["small"],
    timeout=60,
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_list() -> list[dict[str, Any]]:
    """List all eigenstructure studies in the volume.

    Returns
    -------
    list[dict]
        List of study directories with name and contents summary.
    """
    studies = []
    eigen_path = Path(EIGENSTRUCTURE_MOUNT)

    if not eigen_path.exists():
        return []

    for study_dir in eigen_path.iterdir():
        if not study_dir.is_dir():
            continue

        # Check for manifest to identify collected data
        manifest_path = study_dir / "manifest.json"
        has_manifest = manifest_path.exists()

        # Count batches if manifest exists
        batch_count = 0
        if has_manifest:
            batch_count = len(list(study_dir.glob("batch_*.safetensors")))

        # Check for noised subdirectories
        noised_dirs = [
            d.name
            for d in study_dir.iterdir()
            if d.is_dir() and d.name.startswith("eps_")
        ]

        studies.append(
            {
                "name": study_dir.name,
                "path": str(study_dir),
                "has_manifest": has_manifest,
                "batch_count": batch_count,
                "noised_levels": noised_dirs,
            }
        )

    return studies


@app.function(
    name="eigenstructure_list_files",
    image=eigenstructure_image,
    memory=MEMORY_CONFIGS["small"],
    timeout=120,
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_list_files(study_path: str) -> list[dict[str, Any]]:
    """List all files in a study directory.

    Parameters
    ----------
    study_path
        Path relative to eigenstructure volume mount.

    Returns
    -------
    list[dict]
        List of file info dicts with path, rel_path, and size.
    """
    base = Path(EIGENSTRUCTURE_MOUNT) / study_path
    if not base.exists():
        return []

    result = []
    for f in base.rglob("*"):
        if f.is_file():
            result.append(
                {
                    "path": str(f),
                    "rel_path": str(f.relative_to(base)),
                    "size": f.stat().st_size,
                }
            )
    return result


@app.function(
    name="eigenstructure_read_file",
    image=eigenstructure_image,
    memory=MEMORY_CONFIGS["medium"],
    timeout=300,
    volumes=get_eigenstructure_volume_mounts(),  # pyright: ignore[reportArgumentType]
)
def eigenstructure_read_file(file_path: str) -> bytes:
    """Read a file from the eigenstructure volume.

    Parameters
    ----------
    file_path
        Absolute path to file within the volume.

    Returns
    -------
    bytes
        File contents.
    """
    with open(file_path, "rb") as f:
        return f.read()
