"""Aggregate MMD evaluation results from Modal volume into Polars DataFrame.

Downloads mmd_evaluation_*.json files from the Modal volume, parses them, and
creates a flattened Parquet file with one row per (run_id, checkpoint, split).

Usage
-----
# Download and aggregate all results
doppler run -- uv run python -m tmgg.modal.cli.aggregate_mmd \
    --output results/mmd_results.parquet

# Filter by run_id pattern
doppler run -- uv run python -m tmgg.modal.cli.aggregate_mmd \
    --pattern "stage2_digress_transformer_sbm_*" \
    --output results/sbm_results.parquet

# Filter by dataset
doppler run -- uv run python -m tmgg.modal.cli.aggregate_mmd \
    --dataset sbm er \
    --output results/sbm_er_results.parquet

# Only specific checkpoints
doppler run -- uv run python -m tmgg.modal.cli.aggregate_mmd \
    --checkpoints last best \
    --output results/best_checkpoints.parquet

# Dry run (show what would be downloaded)
doppler run -- uv run python -m tmgg.modal.cli.aggregate_mmd \
    --dry-run

# Use local cache (skip download if files exist)
doppler run -- uv run python -m tmgg.modal.cli.aggregate_mmd \
    --cache-dir ./results/cache \
    --output results/mmd_results.parquet
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
from tqdm import tqdm

MODAL_VOLUME_NAME = "tmgg-outputs"


def parse_run_id(run_id: str) -> dict[str, str | int | None]:
    """Extract metadata from run_id string.

    Parses run_id strings with the expected pattern:
    stage2_{model}_{dataset}_{embedding}_lr{lr}_wd{wd}_k{k}_s{seed}

    Parameters
    ----------
    run_id
        Full run identifier, e.g. "stage2_digress_transformer_sbm_default_lr1e-3_wd1e-2_k16_s1"

    Returns
    -------
    dict
        Extracted metadata with keys: model_type, dataset, embedding, lr, wd, k, seed_run.
        Returns empty values if pattern doesn't match.
    """
    # Pattern: stage2_{model}_{dataset}_{embedding}_lr{lr}_wd{wd}_k{k}_s{seed}
    pattern = r"stage2_(digress_transformer)_(\w+)_(\w+)_lr([\d\w-]+)_wd([\d\w-]+)_k(\d+)_s(\d+)"
    match = re.match(pattern, run_id)
    if match:
        return {
            "model_type": match.group(1),
            "dataset": match.group(2),
            "embedding": match.group(3),
            "lr": match.group(4),
            "wd": match.group(5),
            "k": int(match.group(6)),
            "seed_run": int(match.group(7)),
        }
    # Return empty/null values if pattern doesn't match
    return {
        "model_type": None,
        "dataset": None,
        "embedding": None,
        "lr": None,
        "wd": None,
        "k": None,
        "seed_run": None,
    }


def list_evaluation_files(pattern: str | None = None) -> list[dict[str, str]]:
    """List mmd_evaluation_*.json files on Modal volume.

    Parameters
    ----------
    pattern
        Optional glob pattern to filter run_ids.

    Returns
    -------
    list[dict]
        List of dicts with 'run_id', 'checkpoint', 'remote_path' keys.
    """
    # Use modal volume ls to get file list recursively
    cmd = ["modal", "volume", "ls", MODAL_VOLUME_NAME, "--json"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)

    # Parse JSON output - modal ls --json returns array of entries with "Filename" key
    try:
        entries: list[dict[str, str]] = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Fall back to line-by-line parsing if not JSON
        entries = []
        for line in result.stdout.strip().split("\n"):
            if line:
                entries.append({"Filename": line.strip()})

    # Find all run directories first (Modal uses capital "Filename")
    run_dirs: list[str] = []
    for entry in entries:
        filename = entry.get("Filename", entry.get("filename", ""))
        if filename and filename.startswith("stage2_"):
            run_dirs.append(filename.rstrip("/"))

    # For each run directory, list contents to find mmd_evaluation files
    evaluation_files: list[dict[str, str]] = []

    for run_id in tqdm(run_dirs, desc="Scanning run directories", unit="dir"):
        # Apply pattern filter at run_id level
        if pattern and not fnmatch.fnmatch(run_id, pattern):
            continue

        try:
            cmd = ["modal", "volume", "ls", MODAL_VOLUME_NAME, run_id, "--json"]
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)

            try:
                run_entries: list[dict[str, str]] = json.loads(result.stdout)
            except json.JSONDecodeError:
                run_entries = []
                for line in result.stdout.strip().split("\n"):
                    if line:
                        run_entries.append({"Filename": line.strip()})

            for entry in run_entries:
                filename = entry.get("Filename", entry.get("filename", ""))
                if (
                    filename
                    and filename.startswith("mmd_evaluation_")
                    and filename.endswith(".json")
                ):
                    # Extract checkpoint name from filename
                    checkpoint = filename.replace("mmd_evaluation_", "").replace(
                        ".json", ""
                    )
                    evaluation_files.append(
                        {
                            "run_id": run_id,
                            "checkpoint": checkpoint,
                            "remote_path": f"{run_id}/{filename}",
                        }
                    )
        except subprocess.CalledProcessError:
            # Skip directories we can't list
            continue

    return evaluation_files


def download_file(remote_path: str, cache_dir: Path) -> Path:
    """Download a single file from Modal volume to local cache.

    Parameters
    ----------
    remote_path
        Path within the Modal volume.
    cache_dir
        Local directory to store cached files.

    Returns
    -------
    Path
        Local path to downloaded file.
    """
    local_path = cache_dir / remote_path
    local_path.parent.mkdir(parents=True, exist_ok=True)

    # Check if already cached
    if local_path.exists():
        return local_path

    # Download using modal volume get
    cmd = ["modal", "volume", "get", MODAL_VOLUME_NAME, remote_path, str(local_path)]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    return local_path


def download_files(
    files: list[dict[str, str]], cache_dir: Path, use_cache: bool = True
) -> list[Path]:
    """Download files from Modal volume to local cache.

    Parameters
    ----------
    files
        List of file info dicts from list_evaluation_files().
    cache_dir
        Local directory to store cached files.
    use_cache
        If True, skip download for files that already exist locally.

    Returns
    -------
    list[Path]
        List of local file paths.
    """
    local_paths: list[Path] = []

    for file_info in tqdm(files, desc="Downloading files", unit="file"):
        remote_path = file_info["remote_path"]
        local_path = cache_dir / remote_path

        if use_cache and local_path.exists():
            local_paths.append(local_path)
            continue

        try:
            downloaded = download_file(remote_path, cache_dir)
            local_paths.append(downloaded)
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to download {remote_path}: {e}", file=sys.stderr)
            continue

    return local_paths


def load_and_flatten(json_path: Path) -> list[dict[str, Any]]:
    """Load JSON evaluation file and flatten to one row per split.

    Parameters
    ----------
    json_path
        Path to mmd_evaluation_*.json file.

    Returns
    -------
    list[dict]
        List of row dicts, one per split (train/val/test).
    """
    with open(json_path) as f:
        data = json.load(f)

    rows = []
    metadata = parse_run_id(data["run_id"])

    for split, metrics in data.get("results", {}).items():
        row = {
            "run_id": data["run_id"],
            "checkpoint_name": data.get("checkpoint_name", "unknown"),
            "split": split,
            # MMD metrics
            "degree_mmd": metrics.get("degree_mmd"),
            "clustering_mmd": metrics.get("clustering_mmd"),
            "spectral_mmd": metrics.get("spectral_mmd"),
            # Metadata
            "timestamp": data.get("timestamp"),
            # Params
            "num_samples": data.get("params", {}).get("num_samples"),
            "num_steps": data.get("params", {}).get("num_steps"),
            "mmd_kernel": data.get("params", {}).get("mmd_kernel"),
            "mmd_sigma": data.get("params", {}).get("mmd_sigma"),
            "seed": data.get("params", {}).get("seed"),
            # Parsed run_id components
            **metadata,
        }
        rows.append(row)

    return rows


def create_dataframe(rows: list[dict[str, Any]]) -> pl.DataFrame:
    """Create Polars DataFrame with proper schema from row dicts.

    Parameters
    ----------
    rows
        List of flattened row dicts.

    Returns
    -------
    pl.DataFrame
        DataFrame with proper column types.
    """
    if not rows:
        # Return empty DataFrame with expected schema
        return pl.DataFrame(
            schema={
                "run_id": pl.Utf8,
                "checkpoint_name": pl.Utf8,
                "split": pl.Utf8,
                "degree_mmd": pl.Float64,
                "clustering_mmd": pl.Float64,
                "spectral_mmd": pl.Float64,
                "timestamp": pl.Utf8,
                "num_samples": pl.Int64,
                "num_steps": pl.Int64,
                "mmd_kernel": pl.Utf8,
                "mmd_sigma": pl.Float64,
                "seed": pl.Int64,
                "model_type": pl.Utf8,
                "dataset": pl.Utf8,
                "embedding": pl.Utf8,
                "lr": pl.Utf8,
                "wd": pl.Utf8,
                "k": pl.Int64,
                "seed_run": pl.Int64,
            }
        )

    df = pl.DataFrame(rows)

    # Cast timestamp to datetime if present
    if "timestamp" in df.columns:
        df = df.with_columns(
            pl.col("timestamp").str.to_datetime(strict=False).alias("timestamp")
        )

    return df


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate MMD evaluation results from Modal volume into Parquet.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download and aggregate all results
  doppler run -- uv run python -m tmgg.modal.cli.aggregate_mmd \\
      --output results/mmd_results.parquet

  # Filter by run_id pattern
  doppler run -- uv run python -m tmgg.modal.cli.aggregate_mmd \\
      --pattern "stage2_digress_transformer_sbm_*" \\
      --output results/sbm_results.parquet

  # Filter by dataset
  doppler run -- uv run python -m tmgg.modal.cli.aggregate_mmd \\
      --dataset sbm er \\
      --output results/sbm_er_results.parquet

  # Dry run to see what would be downloaded
  doppler run -- uv run python -m tmgg.modal.cli.aggregate_mmd \\
      --dry-run
""",
    )

    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output Parquet file path (required unless --dry-run)",
    )
    parser.add_argument(
        "--pattern",
        help="Filter run_ids by glob pattern (e.g., 'stage2_digress_transformer_sbm_*')",
    )
    parser.add_argument(
        "--dataset",
        nargs="+",
        help="Filter by dataset(s) (e.g., --dataset sbm er)",
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        help="Filter by checkpoint name(s) (e.g., --checkpoints last best)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        help="Local cache directory for downloaded files (uses temp dir if not specified)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force re-download even if files exist in cache",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without downloading",
    )

    args = parser.parse_args()

    # Validate args
    if not args.dry_run and not args.output:
        print("Error: --output is required unless using --dry-run", file=sys.stderr)
        return 1

    # List evaluation files
    print("Listing evaluation files on Modal volume...")
    try:
        files = list_evaluation_files(pattern=args.pattern)
    except subprocess.CalledProcessError as e:
        print(f"Error listing Modal volume: {e}", file=sys.stderr)
        print(
            "Make sure Modal is authenticated (doppler run -- modal setup)",
            file=sys.stderr,
        )
        return 1

    if not files:
        print("No evaluation files found matching criteria")
        return 0

    print(f"Found {len(files)} evaluation file(s)")

    # Apply checkpoint filter
    if args.checkpoints:
        files = [f for f in files if f["checkpoint"] in args.checkpoints]
        print(f"After checkpoint filter: {len(files)} file(s)")

    # Dry run - just show what would be downloaded
    if args.dry_run:
        print("\nWould download:")
        for f in files:
            print(f"  {f['remote_path']}")
        print(f"\nTotal: {len(files)} files")
        return 0

    # Set up cache directory
    if args.cache_dir:
        cache_dir = args.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        print(f"Using cache directory: {cache_dir}")
    else:
        temp_dir = tempfile.mkdtemp(prefix="mmd_eval_")
        cache_dir = Path(temp_dir)
        print(f"Using temporary cache: {cache_dir}")

    # Download files
    local_paths = download_files(files, cache_dir, use_cache=not args.no_cache)

    if not local_paths:
        print("No files were downloaded successfully")
        return 1

    print(f"Downloaded {len(local_paths)} file(s)")

    # Load and flatten all files
    print("Loading and flattening data...")
    all_rows: list[dict[str, Any]] = []
    for path in tqdm(local_paths, desc="Processing files", unit="file"):
        try:
            rows = load_and_flatten(path)
            all_rows.extend(rows)
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Failed to parse {path}: {e}", file=sys.stderr)
            continue

    if not all_rows:
        print("No valid data rows extracted")
        return 1

    print(f"Extracted {len(all_rows)} rows")

    # Create DataFrame
    df = create_dataframe(all_rows)

    # Apply dataset filter after loading (works on parsed metadata)
    if args.dataset:
        df = df.filter(pl.col("dataset").is_in(args.dataset))
        print(f"After dataset filter: {len(df)} rows")

    # Create output directory if needed
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)

        # Save to Parquet
        df.write_parquet(args.output)
        print(f"\nSaved to: {args.output}")
        print(f"Shape: {df.shape}")
        print("\nSchema:")
        for col, dtype in df.schema.items():
            print(f"  {col}: {dtype}")

        # Print summary stats
        print("\nSummary:")
        if "dataset" in df.columns:
            dataset_counts = df.group_by("dataset").len().sort("dataset")
            print("  By dataset:")
            for row in dataset_counts.iter_rows(named=True):
                print(f"    {row['dataset']}: {row['len']} rows")

        if "checkpoint_name" in df.columns:
            ckpt_counts = df.group_by("checkpoint_name").len().sort("checkpoint_name")
            print("  By checkpoint:")
            for row in ckpt_counts.iter_rows(named=True):
                print(f"    {row['checkpoint_name']}: {row['len']} rows")

    return 0


if __name__ == "__main__":
    sys.exit(main())
