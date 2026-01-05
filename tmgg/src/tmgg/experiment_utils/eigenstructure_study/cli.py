"""CLI interface for eigenstructure study tools.

Provides three commands:
- collect: Compute and store eigendecompositions for a dataset
- analyze: Run spectral analysis on collected data
- noised: Collect eigendecompositions for noised graphs
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import click
from loguru import logger


def setup_logging(verbose: bool) -> None:
    """Configure loguru for CLI output."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        level=level,
        format="<level>{level: <8}</level> | {message}",
    )


@click.group()
def main() -> None:
    """Graph eigenstructure study tools.

    Three-phase pipeline for analyzing graph spectral properties:

    \b
    1. collect - Compute eigendecompositions for a dataset
    2. analyze - Compute spectral statistics on collected data
    3. noised  - Collect decompositions for noised graphs
    """
    pass


@main.command()
@click.option(
    "--dataset",
    "-d",
    required=True,
    help="Dataset type (sbm, anu, classical, nx, regular, tree, lfr, er, ws, rg, cm, qm9, enzymes, proteins)",
)
@click.option(
    "--dataset-config",
    "-c",
    required=True,
    help="JSON config string or path to JSON file with dataset parameters",
)
@click.option(
    "--output-dir",
    "-o",
    default="results/eigenstructure_study",
    type=click.Path(path_type=Path),
    help="Output directory for safetensors files (default: results/eigenstructure_study)",
)
@click.option(
    "--batch-size",
    "-b",
    default=64,
    help="Number of graphs per batch",
)
@click.option(
    "--seed",
    "-s",
    default=42,
    help="Random seed for reproducibility",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def collect(
    dataset: str,
    dataset_config: str,
    output_dir: Path,
    batch_size: int,
    seed: int,
    verbose: bool,
) -> None:
    """Phase 1: Collect eigendecompositions for a dataset.

    Computes both adjacency and Laplacian decompositions for all graphs
    in the specified dataset, storing results in safetensors format.

    \b
    Examples:
        # SBM dataset with fixed partition
        tmgg-eigenstructure collect -d sbm \\
            -c '{"num_nodes": 50, "p_intra": 0.8, "q_inter": 0.1, "num_partitions": 100}' \\
            -o ./eigen_data/sbm

        # Erdos-Renyi graphs
        tmgg-eigenstructure collect -d er \\
            -c '{"num_nodes": 30, "num_graphs": 200, "p": 0.15}' \\
            -o ./eigen_data/er

        # PyG benchmark
        tmgg-eigenstructure collect -d enzymes \\
            -c '{"max_graphs": 500}' \\
            -o ./eigen_data/enzymes
    """
    setup_logging(verbose)

    # Parse config: either JSON string or file path
    if dataset_config.startswith("{"):
        config = json.loads(dataset_config)
    else:
        config_path = Path(dataset_config)
        if not config_path.exists():
            raise click.ClickException(f"Config file not found: {config_path}")
        with open(config_path) as f:
            config = json.load(f)

    from .collector import EigenstructureCollector

    collector = EigenstructureCollector(
        dataset_name=dataset,
        dataset_config=config,
        output_dir=output_dir,
        batch_size=batch_size,
        seed=seed,
    )

    try:
        collector.collect()
        click.echo(f"Collection complete. Output: {output_dir}")
    except Exception as e:
        logger.exception("Collection failed")
        raise click.ClickException(str(e)) from e


@main.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory with collected eigendecompositions",
)
@click.option(
    "--output-dir",
    "-o",
    default="results/eigenstructure_study",
    type=click.Path(path_type=Path),
    help="Output directory for analysis results (default: results/eigenstructure_study)",
)
@click.option(
    "--subspace-k",
    "-k",
    default=10,
    help="Number of eigenvectors for subspace analysis",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def analyze(
    input_dir: Path,
    output_dir: Path,
    subspace_k: int,
    verbose: bool,
) -> None:
    """Phase 2: Analyze collected eigenstructure data.

    Computes spectral statistics including:
    - Spectral gap (adjacency)
    - Algebraic connectivity (Laplacian)
    - Eigenvalue entropy
    - Eigenvector coherence
    - Effective rank
    - Subspace distances (optional)

    \b
    Examples:
        tmgg-eigenstructure analyze -i ./eigen_data/sbm -o ./analysis/sbm
    """
    setup_logging(verbose)

    from .analyzer import SpectralAnalyzer

    analyzer = SpectralAnalyzer(input_dir)

    try:
        result = analyzer.analyze()

        # Save main results
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "analysis.json"
        with open(output_path, "w") as f:
            json.dump(asdict(result), f, indent=2)

        # Compute and save subspace distances if requested
        if subspace_k > 0:
            subspace_results = analyzer.compute_subspace_distances(k=subspace_k)
            subspace_path = output_dir / "subspace_analysis.json"
            with open(subspace_path, "w") as f:
                json.dump(subspace_results, f, indent=2)
            click.echo(f"Subspace analysis saved to: {subspace_path}")

        click.echo(f"Analysis complete. Results: {output_path}")
        click.echo()
        click.echo("Summary:")
        click.echo(f"  Dataset: {result.dataset_name}")
        click.echo(f"  Graphs: {result.num_graphs}")
        click.echo(
            f"  Spectral gap: {result.spectral_gap_mean:.4f} +/- {result.spectral_gap_std:.4f}"
        )
        click.echo(
            f"  Algebraic connectivity: {result.algebraic_connectivity_mean:.4f} +/- {result.algebraic_connectivity_std:.4f}"
        )
        click.echo(
            f"  Coherence: {result.coherence_mean:.4f} +/- {result.coherence_std:.4f}"
        )
        click.echo(f"  Effective rank (adj): {result.effective_rank_adj_mean:.2f}")

    except Exception as e:
        logger.exception("Analysis failed")
        raise click.ClickException(str(e)) from e


@main.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory with original (Phase 1) decompositions",
)
@click.option(
    "--output-dir",
    "-o",
    default="results/eigenstructure_study/noised",
    type=click.Path(path_type=Path),
    help="Output directory for noised decompositions (default: results/eigenstructure_study/noised)",
)
@click.option(
    "--noise-type",
    "-t",
    required=True,
    type=click.Choice(["gaussian", "digress", "rotation"]),
    help="Type of noise to apply",
)
@click.option(
    "--noise-levels",
    "-n",
    required=True,
    help="Comma-separated noise levels (e.g., '0.01,0.05,0.1,0.2')",
)
@click.option(
    "--rotation-k",
    default=None,
    type=int,
    help="Dimension for rotation noise skew matrix (required for rotation noise)",
)
@click.option(
    "--seed",
    "-s",
    default=42,
    help="Random seed",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def noised(
    input_dir: Path,
    output_dir: Path,
    noise_type: str,
    noise_levels: str,
    rotation_k: int | None,
    seed: int,
    verbose: bool,
) -> None:
    """Phase 3: Collect eigendecompositions for noised graphs.

    Reads original adjacency matrices from Phase 1 output, applies noise
    at the specified levels, and computes new decompositions.

    Creates subdirectories eps_X.XXXX/ for each noise level.

    \b
    Examples:
        # Gaussian noise at multiple levels
        tmgg-eigenstructure noised -i ./eigen_data/sbm -o ./noised_data/sbm \\
            -t gaussian -n 0.01,0.05,0.1,0.2,0.3

        # Digress (edge flip) noise
        tmgg-eigenstructure noised -i ./eigen_data/er -o ./noised_data/er \\
            -t digress -n 0.05,0.1,0.2

        # Rotation noise
        tmgg-eigenstructure noised -i ./eigen_data/sbm -o ./noised_data/sbm_rot \\
            -t rotation -n 0.1,0.3,0.5 --rotation-k 20
    """
    setup_logging(verbose)

    # Parse noise levels
    levels = [float(x.strip()) for x in noise_levels.split(",")]

    from .noised_collector import NoisedEigenstructureCollector

    try:
        collector = NoisedEigenstructureCollector(
            input_dir=input_dir,
            output_dir=output_dir,
            noise_type=noise_type,
            noise_levels=levels,
            rotation_k=rotation_k,
            seed=seed,
        )
        collector.collect()
        click.echo(f"Noised collection complete. Output: {output_dir}")
        click.echo(f"Noise levels: {levels}")

    except Exception as e:
        logger.exception("Noised collection failed")
        raise click.ClickException(str(e)) from e


@main.command()
@click.option(
    "--original-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory with original (Phase 1) decompositions",
)
@click.option(
    "--noised-dir",
    "-n",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Directory with noised decompositions (contains eps_*/ subdirs)",
)
@click.option(
    "--output-dir",
    "-o",
    default="results/eigenstructure_study",
    type=click.Path(path_type=Path),
    help="Output directory for comparison results (default: results/eigenstructure_study)",
)
@click.option(
    "--subspace-k",
    "-k",
    default=10,
    help="Number of eigenvectors for subspace comparison",
)
@click.option(
    "--procrustes-k",
    "-p",
    default="1,2,4,8,16",
    help="Comma-separated k values for Procrustes rotation analysis (default: 1,2,4,8,16)",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable debug logging",
)
def compare(
    original_dir: Path,
    noised_dir: Path,
    output_dir: Path,
    subspace_k: int,
    procrustes_k: str,
    verbose: bool,
) -> None:
    """Compare original and noised eigenstructure.

    Computes delta metrics between original and noised graphs:
    - Eigengap delta: Change in spectral gap (lambda_max - lambda_{max-1})
    - Algebraic connectivity delta: Change in Fiedler value (lambda_2 of Laplacian)
    - Eigenvalue drift: Relative L2 distance between spectra
    - Subspace distance: Frobenius norm of projection difference
    - Procrustes rotation: Optimal rotation angle and residual for eigenvector alignment

    \b
    Examples:
        tmgg-eigenstructure compare \\
            -i ./eigen_data/sbm \\
            -n ./noised_data/sbm \\
            -o ./comparison/sbm
    """
    setup_logging(verbose)

    # Parse Procrustes k values
    procrustes_k_values = [int(x.strip()) for x in procrustes_k.split(",")]

    from .noised_collector import NoisedAnalysisComparator

    try:
        comparator = NoisedAnalysisComparator(original_dir, noised_dir)

        results = comparator.compute_full_comparison(
            k=subspace_k, procrustes_k_values=procrustes_k_values
        )

        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "comparison.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        click.echo(f"Comparison complete. Results: {output_path}")
        click.echo()
        click.echo("Summary (delta metrics):")
        for r in results:
            eps = r["noise_level"]
            click.echo(f"  eps={eps:.4f}:")
            click.echo(
                f"    eigengap_delta={r['eigengap_delta_rel_mean']:+.4f} +/- {r['eigengap_delta_rel_std']:.4f}"
            )
            click.echo(
                f"    alg_conn_delta={r['alg_conn_delta_rel_mean']:+.4f} +/- {r['alg_conn_delta_rel_std']:.4f}"
            )
            click.echo(
                f"    eigenvalue_drift_adj={r['eigenvalue_drift_adj_mean']:.4f} +/- {r['eigenvalue_drift_adj_std']:.4f}"
            )
            click.echo(
                f"    subspace_distance={r['subspace_distance_mean']:.4f} +/- {r['subspace_distance_std']:.4f}"
            )
            # Output Procrustes rotation metrics
            click.echo("    Procrustes rotation:")
            for k in procrustes_k_values:
                angle_key = f"procrustes_angle_k{k}_mean"
                angle_std_key = f"procrustes_angle_k{k}_std"
                if angle_key in r:
                    click.echo(
                        f"      k={k}: angle={r[angle_key]:.4f}+/-{r[angle_std_key]:.4f} rad"
                    )

    except Exception as e:
        logger.exception("Comparison failed")
        raise click.ClickException(str(e)) from e


if __name__ == "__main__":
    main()
