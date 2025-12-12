#!/usr/bin/env python3
"""Unify exported W&B data across multiple projects into consolidated DataFrames.

Reads exported runs from wandb_export/ directory structure, adds project/run
identifiers, aligns columns across runs, and produces unified parquet files.

Output files:
    - unified_history.parquet: All training history with run/project IDs
    - unified_summary.parquet: Final metrics per run with configs
    - unified_metadata.parquet: Run metadata (tags, state, timestamps)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
import pandas as pd
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, MofNCompleteColumn


def load_json(path: Path) -> dict[str, Any]:
    """Load JSON file, return empty dict on error."""
    try:
        return json.loads(path.read_text())
    except Exception as e:
        logger.warning(f"Failed to load {path}: {e}")
        return {}


def extract_config_fields(config: dict[str, Any]) -> dict[str, Any]:
    """Extract key configuration fields for analysis.

    Flattens nested config structure into flat dict with prefixed keys.
    """
    flat = {}

    # Model config
    model = config.get("model", {})
    if isinstance(model, dict):
        flat["model_target"] = model.get("_target_", "unknown")
        flat["model_d_model"] = model.get("d_model")
        flat["model_num_layers"] = model.get("num_layers")
        flat["model_num_heads"] = model.get("num_heads")
        flat["model_dropout"] = model.get("dropout")
        flat["model_k"] = model.get("k")  # for spectral models
        flat["model_loss_type"] = model.get("loss_type")

    # Data config
    data = config.get("data", {})
    if isinstance(data, dict):
        # Old configs use dataset_name, new configs use graph_type
        dataset_name = data.get("dataset_name")
        graph_type = data.get("graph_type")
        # Unified field: prefer graph_type (newer), fall back to dataset_name (older)
        flat["data_graph_type"] = graph_type or dataset_name
        # Keep original fields for debugging
        flat["data_dataset_name"] = dataset_name
        flat["data_graph_type_raw"] = graph_type

        flat["data_noise_type"] = data.get("noise_type")
        flat["data_noise_levels"] = str(data.get("noise_levels", []))
        flat["data_batch_size"] = data.get("batch_size")
        # Number of nodes: SingleGraphDataModule uses 'n', GraphDataModule uses dataset_config.num_nodes
        flat["data_num_nodes"] = data.get("n")
        dataset_config = data.get("dataset_config", {})
        if isinstance(dataset_config, dict):
            flat["data_num_nodes"] = flat["data_num_nodes"] or dataset_config.get("num_nodes")

    # Top-level config
    flat["learning_rate"] = config.get("learning_rate") or config.get("model", {}).get("learning_rate")
    flat["seed"] = config.get("seed")
    flat["noise_level"] = config.get("noise_level")

    # Extract model type from target
    target = flat.get("model_target", "")
    if "attention" in target.lower():
        flat["model_type"] = "attention"
    elif "spectral" in target.lower() or "linear_pe" in target.lower():
        flat["model_type"] = "spectral"
    elif "digress" in target.lower():
        flat["model_type"] = "digress"
    elif "gnn" in target.lower():
        flat["model_type"] = "gnn"
    elif "hybrid" in target.lower():
        flat["model_type"] = "hybrid"
    else:
        flat["model_type"] = "unknown"

    return flat


def load_run_data(run_dir: Path, project_id: str) -> dict[str, Any] | None:
    """Load all data for a single run.

    Returns dict with keys: run_id, project_id, config, summary, metadata, history
    """
    run_id = run_dir.name

    # Check if export is complete
    if not (run_dir / "_export_complete.marker").exists():
        logger.warning(f"Skipping incomplete export: {run_dir}")
        return None

    # Load config
    config = load_json(run_dir / "config.json")
    config_flat = extract_config_fields(config)

    # Load summary
    summary = load_json(run_dir / "summary.json")
    # Filter out non-scalar values (images, etc.)
    summary_scalar = {
        k: v for k, v in summary.items()
        if isinstance(v, (int, float, str, bool, type(None)))
    }

    # Load metadata
    metadata = load_json(run_dir / "metadata.json")

    # Load history
    history_path = run_dir / "metrics" / "history.parquet"
    if history_path.exists():
        try:
            history_df = pd.read_parquet(history_path)
            # Add identifiers
            history_df["run_id"] = run_id
            history_df["project_id"] = project_id
        except Exception as e:
            logger.warning(f"Failed to load history for {run_id}: {e}")
            history_df = None
    else:
        history_df = None

    return {
        "run_id": run_id,
        "project_id": project_id,
        "config": config_flat,
        "summary": summary_scalar,
        "metadata": metadata,
        "history": history_df,
    }


def unify_exports(
    export_dir: Path,
    projects: list[str] | None = None,
    output_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load and unify all exported runs.

    Parameters
    ----------
    export_dir
        Base directory containing exported projects (e.g., wandb_export/).
    projects
        List of project names to include. If None, includes all.
    output_dir
        Directory to save unified parquet files. If None, doesn't save.

    Returns
    -------
    tuple of (history_df, summary_df, metadata_df)
    """
    console = Console()

    # Find projects
    if projects:
        project_dirs = [export_dir / p for p in projects if (export_dir / p).exists()]
    else:
        project_dirs = [d for d in export_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]

    if not project_dirs:
        logger.error(f"No projects found in {export_dir}")
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    logger.info(f"Found {len(project_dirs)} projects to process")

    all_history = []
    all_summary = []
    all_metadata = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        for project_dir in project_dirs:
            project_id = project_dir.name
            run_dirs = [d for d in project_dir.iterdir() if d.is_dir() and not d.name.startswith("_")]

            if not run_dirs:
                continue

            task = progress.add_task(f"Processing {project_id}", total=len(run_dirs))

            for run_dir in run_dirs:
                run_data = load_run_data(run_dir, project_id)

                if run_data is None:
                    progress.advance(task)
                    continue

                # Build summary row
                summary_row = {
                    "run_id": run_data["run_id"],
                    "project_id": project_id,
                    **run_data["config"],
                    **run_data["summary"],
                }
                all_summary.append(summary_row)

                # Build metadata row
                meta = run_data["metadata"]
                meta_row = {
                    "run_id": run_data["run_id"],
                    "project_id": project_id,
                    "name": meta.get("name"),
                    "state": meta.get("state"),
                    "tags": str(meta.get("tags", [])),
                    "created_at": meta.get("created_at"),
                    "url": meta.get("url"),
                }
                all_metadata.append(meta_row)

                # Append history
                if run_data["history"] is not None:
                    # Add config fields to history for filtering
                    for key in ["model_type", "data_noise_type", "data_dataset_name", "learning_rate", "noise_level"]:
                        if key in run_data["config"]:
                            run_data["history"][key] = run_data["config"][key]
                    all_history.append(run_data["history"])

                progress.advance(task)

    # Concatenate with column alignment
    logger.info("Concatenating DataFrames...")

    if all_history:
        # Drop all-NA columns from each DataFrame before concat to avoid FutureWarning
        cleaned = []
        for df in all_history:
            if not df.empty:
                df_clean = df.dropna(axis=1, how="all")
                if not df_clean.empty:
                    cleaned.append(df_clean)
        if cleaned:
            history_df = pd.concat(cleaned, ignore_index=True, sort=False)
            # Convert object columns that should be numeric
            for col in history_df.columns:
                if history_df[col].dtype == object:
                    try:
                        history_df[col] = pd.to_numeric(history_df[col])
                    except (ValueError, TypeError):
                        pass  # Keep as object if conversion fails
        else:
            history_df = pd.DataFrame()
    else:
        history_df = pd.DataFrame()

    summary_df = pd.DataFrame(all_summary)
    metadata_df = pd.DataFrame(all_metadata)

    logger.info(f"Unified history: {len(history_df)} rows, {len(history_df.columns)} columns")
    logger.info(f"Unified summary: {len(summary_df)} rows, {len(summary_df.columns)} columns")
    logger.info(f"Unified metadata: {len(metadata_df)} rows")

    # Save if output_dir specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)

        history_path = output_dir / "unified_history.parquet"
        summary_path = output_dir / "unified_summary.parquet"
        metadata_path = output_dir / "unified_metadata.parquet"

        if not history_df.empty:
            history_df.to_parquet(history_path, index=False)
            logger.info(f"Saved: {history_path}")

        summary_df.to_parquet(summary_path, index=False)
        logger.info(f"Saved: {summary_path}")

        metadata_df.to_parquet(metadata_path, index=False)
        logger.info(f"Saved: {metadata_path}")

    return history_df, summary_df, metadata_df


@click.command()
@click.option(
    "--export-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("wandb_export"),
    help="Directory containing exported W&B data",
)
@click.option(
    "--projects",
    multiple=True,
    help="Specific projects to include (default: all)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("analysis_output"),
    help="Directory to save unified parquet files",
)
def main(export_dir: Path, projects: tuple[str, ...], output_dir: Path) -> None:
    """Unify exported W&B data into consolidated parquet files."""
    import sys
    from loguru import logger

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    project_list = list(projects) if projects else None

    history_df, summary_df, metadata_df = unify_exports(
        export_dir=export_dir,
        projects=project_list,
        output_dir=output_dir,
    )

    click.echo(f"\nUnified {len(summary_df)} runs from {export_dir}")
    click.echo(f"Output saved to: {output_dir.absolute()}")


if __name__ == "__main__":
    main()
