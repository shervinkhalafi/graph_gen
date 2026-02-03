# /// script
# dependencies = [
#     "wandb>=0.15",
#     "pandas>=2.0",
#     "pyarrow",
#     "rich",
#     "python-dotenv",
# ]
# ///
"""Export W&B runs with full configs and metrics to parquet files.

Fetches all runs from specified entity/project combinations and exports
flattened configs (prefixed with `config_`) and metrics (prefixed with `metric_`)
to polars-native parquet format.

Credentials:
    The tool looks for W&B API key in this order:
    1. --api-key command line argument
    2. WANDB_API_KEY environment variable
    3. GRAPH_DENOISE_TEAM_SERVICE in .env file (for team access)
    4. Default wandb credentials (~/.netrc)

Usage:
    uv run wandb-tools/export_runs.py --entity graph_denoise_team --project spectral_denoising
    uv run wandb-tools/export_runs.py -e graph_denoise_team -p "*" -o wandb_export/
    uv run wandb-tools/export_runs.py -e igorkraw --since 7d --include-history
"""

import argparse
import json
import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import wandb
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
)
from rich.table import Table

console = Console()


def get_wandb_api(api_key: str | None = None) -> wandb.Api:
    """Get W&B API instance with appropriate credentials.

    Tries credentials in order:
    1. Explicit api_key parameter
    2. WANDB_API_KEY environment variable
    3. GRAPH_DENOISE_TEAM_SERVICE from .env (for team access)
    4. Default wandb credentials
    """
    # Load .env from current dir or parent dirs
    load_dotenv()

    # Try explicit key first
    if api_key:
        console.print("[dim]Using provided API key[/dim]")
        return wandb.Api(api_key=api_key)

    # Try environment variables
    env_key = os.environ.get("WANDB_API_KEY")
    if env_key:
        console.print("[dim]Using WANDB_API_KEY from environment[/dim]")
        return wandb.Api(api_key=env_key)

    # Try team service key from .env
    team_key = os.environ.get("GRAPH_DENOISE_TEAM_SERVICE")
    if team_key:
        console.print("[dim]Using GRAPH_DENOISE_TEAM_SERVICE from .env[/dim]")
        return wandb.Api(api_key=team_key)

    # Fall back to default credentials
    console.print("[dim]Using default wandb credentials[/dim]")
    return wandb.Api()


def flatten_dict(d: dict, prefix: str = "", sep: str = "_") -> dict:
    """Recursively flatten a nested dictionary.

    Parameters
    ----------
    d
        Dictionary to flatten
    prefix
        Key prefix (used for nested calls)
    sep
        Separator between nested keys

    Returns
    -------
    Flattened dictionary with concatenated keys
    """
    items: dict = {}
    for k, v in d.items():
        new_key = f"{prefix}{sep}{k}" if prefix else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep))
        elif isinstance(v, list | tuple):
            items[new_key] = str(v)
        elif isinstance(v, int | float | str | bool) or v is None:
            items[new_key] = v
        else:
            items[new_key] = str(v)
    return items


def sanitize_metric_key(key: str) -> str:
    """Sanitize metric keys by replacing slashes and dots."""
    return key.replace("/", "_").replace(".", "_").replace("-", "_")


def extract_all_metrics(summary: dict) -> dict:
    """Extract all numeric metrics from run summary with standardized keys.

    All metrics are prefixed with `metric_` and keys are sanitized.
    """
    metrics = {}
    for k, v in summary.items():
        if k.startswith("_"):
            continue
        if isinstance(v, int | float):
            clean_key = f"metric_{sanitize_metric_key(k)}"
            metrics[clean_key] = v
    return metrics


def extract_config(config: dict) -> dict:
    """Extract and flatten config with `config_` prefix."""
    flat = flatten_dict(config)
    return {f"config_{k}": v for k, v in flat.items()}


def parse_since_arg(since_str: str) -> datetime:
    """Parse --since argument into datetime.

    Supports:
    - Relative: "7d" (7 days), "24h" (24 hours), "2w" (2 weeks)
    - Absolute: "2024-01-15" (ISO format)
    """
    if since_str.endswith("d"):
        days = int(since_str[:-1])
        return datetime.now() - timedelta(days=days)
    elif since_str.endswith("h"):
        hours = int(since_str[:-1])
        return datetime.now() - timedelta(hours=hours)
    elif since_str.endswith("w"):
        weeks = int(since_str[:-1])
        return datetime.now() - timedelta(weeks=weeks)
    else:
        return datetime.fromisoformat(since_str)


def serialize_for_json(obj):
    """Recursively convert non-serializable objects to strings."""
    if isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list | tuple):
        return [serialize_for_json(v) for v in obj]
    elif isinstance(obj, int | float | str | bool) or obj is None:
        return obj
    else:
        return str(obj)


def fetch_project_runs(
    api: wandb.Api,
    entity: str,
    project: str,
    since_date: datetime | None = None,
    include_history: bool = False,
) -> tuple[list[dict], list[pd.DataFrame]]:
    """Fetch all runs from a single project.

    Parameters
    ----------
    api
        W&B API instance
    entity
        W&B entity (user or team)
    project
        Project name
    since_date
        Only include runs created after this date
    include_history
        Whether to fetch training history (slower)

    Returns
    -------
    Tuple of (runs_data, history_dfs)

    Notes
    -----
    Each run record includes both flattened columns (for querying) and raw JSON
    columns (`_config_json`, `_summary_json`) for lossless data preservation.
    """
    runs_data = []
    history_dfs = []

    filters = {}
    if since_date:
        filters = {"created_at": {"$gte": since_date.isoformat()}}

    try:
        runs = api.runs(
            f"{entity}/{project}",
            filters=filters if filters else None,
            per_page=1000,
        )

        for run_count, run in enumerate(runs, start=1):
            if run_count % 100 == 0:
                console.print(f"[dim]  ... fetched {run_count} runs[/dim]")
            summary = dict(run.summary) if run.summary else {}
            config = dict(run.config) if run.config else {}

            # Build run record with core metadata
            run_record = {
                "id": run.id,
                "name": run.name,
                "display_name": getattr(run, "display_name", run.name),
                "state": run.state,
                "entity": entity,
                "project": project,
                "created_at": run.created_at,
                "tags": str(list(run.tags) if run.tags else []),
                "url": run.url,
            }

            # Store raw JSON for lossless preservation (serialized to handle non-JSON types)
            run_record["_config_json"] = json.dumps(serialize_for_json(config))
            run_record["_summary_json"] = json.dumps(serialize_for_json(summary))

            # Add flattened config (for query convenience)
            run_record.update(extract_config(config))

            # Add all metrics (flattened)
            run_record.update(extract_all_metrics(summary))

            runs_data.append(run_record)

            # Fetch history if requested
            if include_history:
                try:
                    history = run.history(pandas=True)
                    if not history.empty:
                        history["run_id"] = run.id
                        history["entity"] = entity
                        history["project"] = project
                        history_dfs.append(history)
                except Exception as e:
                    console.print(
                        f"[yellow]Warning: Could not fetch history for {run.id}: {e}[/yellow]"
                    )

    except wandb.errors.CommError as e:
        console.print(f"[red]Error fetching {entity}/{project}: {e}[/red]")
    except Exception as e:
        console.print(f"[red]Unexpected error fetching {entity}/{project}: {e}[/red]")

    return runs_data, history_dfs


def list_projects(api: wandb.Api, entity: str, pattern: str | None = None) -> list[str]:
    """List projects for an entity, optionally filtered by pattern."""
    try:
        projects = list(api.projects(entity))
        project_names = [p.name for p in projects]

        if pattern and pattern != "*":
            # Support glob-like patterns
            regex = pattern.replace("*", ".*")
            project_names = [p for p in project_names if re.match(regex, p)]

        return project_names
    except Exception as e:
        console.print(f"[red]Error listing projects for {entity}: {e}[/red]")
        return []


def export_to_parquet(
    runs_data: list[dict],
    history_dfs: list[pd.DataFrame],
    output_dir: Path,
    entity: str,
    project: str | None,
) -> None:
    """Export runs data and history to parquet files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine filename prefix
    prefix = f"{entity}_{project}" if project and project != "*" else entity

    # Export runs
    if runs_data:
        runs_df = pd.DataFrame(runs_data)

        # Convert mixed-type columns to string to avoid parquet errors
        for col in runs_df.columns:
            if runs_df[col].apply(type).nunique() > 1:
                runs_df[col] = runs_df[col].astype(str)

        runs_path = output_dir / f"{prefix}_runs.parquet"
        runs_df.to_parquet(runs_path, index=False)
        console.print(f"[green]Saved {len(runs_df)} runs to {runs_path}[/green]")

    # Export history
    if history_dfs:
        history_df = pd.concat(history_dfs, ignore_index=True)
        history_path = output_dir / f"{prefix}_history.parquet"
        history_df.to_parquet(history_path, index=False)
        console.print(
            f"[green]Saved history ({len(history_df)} rows) to {history_path}[/green]"
        )

    # Export metadata
    metadata = {
        "exported_at": datetime.now().isoformat(),
        "entity": entity,
        "project": project,
        "total_runs": len(runs_data),
        "total_history_rows": sum(len(df) for df in history_dfs) if history_dfs else 0,
    }
    metadata_path = output_dir / f"{prefix}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def print_summary(runs_data: list[dict]) -> None:
    """Print summary table of exported runs."""
    if not runs_data:
        console.print("[yellow]No runs exported[/yellow]")
        return

    df = pd.DataFrame(runs_data)

    # Group by project
    table = Table(title="Export Summary")
    table.add_column("Project", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Finished", justify="right")
    table.add_column("Running", justify="right")
    table.add_column("Crashed", justify="right")

    for project in df["project"].unique():
        proj_df = df[df["project"] == project]
        table.add_row(
            project,
            str(len(proj_df)),
            str(len(proj_df[proj_df["state"] == "finished"])),
            str(len(proj_df[proj_df["state"] == "running"])),
            str(len(proj_df[proj_df["state"] == "crashed"])),
        )

    console.print(table)
    console.print(f"\n[bold]Total: {len(runs_data)} runs[/bold]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export W&B runs to parquet files")
    parser.add_argument(
        "--entity",
        "-e",
        action="append",
        default=[],
        help="W&B entity (can specify multiple). Default: igorkraw, graph_denoise_team",
    )
    parser.add_argument(
        "--project",
        "-p",
        default=None,
        help="Project name or pattern (* for all, supports globs like 'spectral*')",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("wandb_export"),
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--since",
        "-s",
        default=None,
        help="Only export runs created since (e.g., '7d', '24h', '2024-01-15')",
    )
    parser.add_argument(
        "--include-history",
        action="store_true",
        help="Also export training history (slower, larger files)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="W&B API key (or set WANDB_API_KEY env var, or GRAPH_DENOISE_TEAM_SERVICE in .env)",
    )
    args = parser.parse_args()

    # Default entities
    entities = args.entity if args.entity else ["igorkraw", "graph_denoise_team"]

    # Parse since date
    since_date = None
    if args.since:
        since_date = parse_since_arg(args.since)
        console.print(f"[dim]Filtering runs since {since_date.isoformat()}[/dim]\n")

    api = get_wandb_api(args.api_key)
    all_runs: list[dict] = []

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        console=console,
    ) as progress:
        for entity in entities:
            # Get project list
            if args.project == "*" or args.project is None:
                projects = list_projects(api, entity)
            elif "*" in args.project:
                projects = list_projects(api, entity, args.project)
            else:
                projects = [args.project]

            if not projects:
                console.print(f"[yellow]No projects found for {entity}[/yellow]")
                continue

            task = progress.add_task(f"[{entity}]", total=len(projects))

            for project in projects:
                progress.update(task, description=f"[{entity}] {project}")

                runs, history = fetch_project_runs(
                    api,
                    entity,
                    project,
                    since_date,
                    args.include_history,
                )

                # Write per-project parquet immediately (don't wait for all)
                if runs:
                    export_to_parquet(runs, history, args.output, entity, project)
                    all_runs.extend(runs)

                progress.advance(task)

    # Print final summary
    if all_runs:
        print_summary(all_runs)
    else:
        console.print("[yellow]No runs found to export[/yellow]")


if __name__ == "__main__":
    main()
