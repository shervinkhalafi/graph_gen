# /// script
# dependencies = [
#     "pandas>=2.0",
#     "pyarrow",
#     "rich",
# ]
# ///
"""Aggregate and postprocess exported W&B parquet files.

Reads multiple parquet exports, merges them into a unified dataframe with
consistent schema, and adds parsed feature columns for analysis.

Parsing functions live in the co-located ``wandb_parsing`` module.

Usage:
    uv run wandb-tools/aggregate_runs.py wandb_export/*.parquet -o analysis/unified.parquet
    uv run wandb-tools/aggregate_runs.py wandb_export/ -o analysis/unified.parquet --state finished
    uv run wandb-tools/aggregate_runs.py wandb_export/ --filter "stage2c" -o analysis/stage2c.parquet
"""

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table
from wandb_parsing import enrich_dataframe, parse_protocol

console = Console()


def load_parquet_files(paths: list[Path]) -> pd.DataFrame:
    """Load and concatenate multiple parquet files.

    Tracks source files and reports column schema differences between files
    to help identify potential data loss during aggregation.
    """
    dfs = []
    column_sources: dict[str, set[str]] = {}  # column -> set of source files

    for path in paths:
        if path.is_dir():
            # Load all parquet files in directory
            parquet_files = list(path.glob("*_runs.parquet"))
            if not parquet_files:
                parquet_files = list(path.glob("*.parquet"))
            for pq_file in parquet_files:
                if "_history" not in pq_file.name:
                    try:
                        df = pd.read_parquet(pq_file)
                        df["_source_file"] = pq_file.name
                        # Track which columns come from which files
                        for col in df.columns:
                            if col not in column_sources:
                                column_sources[col] = set()
                            column_sources[col].add(pq_file.name)
                        dfs.append(df)
                        console.print(
                            f"[dim]Loaded {len(df)} runs from {pq_file}[/dim]"
                        )
                    except Exception as e:
                        console.print(f"[red]Error loading {pq_file}: {e}[/red]")
        else:
            try:
                df = pd.read_parquet(path)
                df["_source_file"] = path.name
                for col in df.columns:
                    if col not in column_sources:
                        column_sources[col] = set()
                    column_sources[col].add(path.name)
                dfs.append(df)
                console.print(f"[dim]Loaded {len(df)} runs from {path}[/dim]")
            except Exception as e:
                console.print(f"[red]Error loading {path}: {e}[/red]")

    if not dfs:
        return pd.DataFrame()

    # Report schema differences
    if len(dfs) > 1:
        all_sources = set()
        for sources in column_sources.values():
            all_sources.update(sources)
        n_sources = len(all_sources)

        partial_cols = [
            col for col, sources in column_sources.items() if len(sources) < n_sources
        ]
        if partial_cols:
            console.print(
                f"\n[yellow]Warning: {len(partial_cols)} columns not present in all files[/yellow]"
            )
            console.print(
                f"[dim]  Partial columns: {', '.join(partial_cols[:10])}{'...' if len(partial_cols) > 10 else ''}[/dim]"
            )

    # Concatenate with column alignment
    unified = pd.concat(dfs, ignore_index=True, sort=False)
    return unified


def filter_dataframe(
    df: pd.DataFrame,
    state_filter: str | None = None,
    name_filter: str | None = None,
    project_filter: str | None = None,
    protocol_filter: str | None = None,
) -> pd.DataFrame:
    """Apply filters to dataframe."""
    if state_filter and state_filter != "all":
        df = df[df["state"] == state_filter]

    if name_filter:
        df = df[df["name"].str.contains(name_filter, case=False, na=False)]

    if project_filter:
        df = df[df["project"].str.contains(project_filter, case=False, na=False)]

    # Protocol filter (requires enrichment first, so we compute it here)
    if protocol_filter and protocol_filter != "all":
        protocol_col = df.apply(parse_protocol, axis=1)
        before = len(df)
        df = df[protocol_col == protocol_filter].copy()
        console.print(
            f"[dim]Protocol filter '{protocol_filter}': {before} -> {len(df)} runs[/dim]"
        )

    return df


def print_summary(df: pd.DataFrame) -> None:
    """Print aggregation summary."""
    table = Table(title="Aggregation Summary")
    table.add_column("Dimension", style="cyan")
    table.add_column("Values", justify="right")
    table.add_column("Sample", style="dim")

    dimensions = [
        ("Total runs", str(len(df)), ""),
        (
            "Projects",
            str(df["project"].nunique() if "project" in df.columns else 0),
            ", ".join(df["project"].unique()[:3]) if "project" in df.columns else "",
        ),
        (
            "Stages",
            str(df["stage"].nunique() if "stage" in df.columns else 0),
            ", ".join(df["stage"].unique()[:5]) if "stage" in df.columns else "",
        ),
        (
            "Architectures",
            str(df["arch"].nunique() if "arch" in df.columns else 0),
            ", ".join(df["arch"].unique()[:5]) if "arch" in df.columns else "",
        ),
        (
            "Model types",
            str(df["model_type"].nunique() if "model_type" in df.columns else 0),
            ", ".join(df["model_type"].unique()[:5])
            if "model_type" in df.columns
            else "",
        ),
    ]

    for name, count, sample in dimensions:
        table.add_row(name, count, sample[:50] + "..." if len(sample) > 50 else sample)

    console.print(table)

    # State distribution
    if "state" in df.columns:
        console.print("\n[bold]State distribution:[/bold]")
        for state, count in df["state"].value_counts().items():
            console.print(f"  {state}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate exported W&B parquet files")
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="Parquet files or directories to aggregate",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        required=True,
        help="Output parquet file path",
    )
    parser.add_argument(
        "--state",
        "-s",
        default=None,
        help="Filter by run state (finished, crashed, running, all)",
    )
    parser.add_argument(
        "--filter",
        "-f",
        default=None,
        help="Filter runs by name pattern",
    )
    parser.add_argument(
        "--project",
        "-p",
        default=None,
        help="Filter by project name pattern",
    )
    parser.add_argument(
        "--no-enrich",
        action="store_true",
        help="Skip adding parsed columns (stage, arch, etc.)",
    )
    parser.add_argument(
        "--protocol",
        default="distribution",
        choices=["distribution", "single_graph", "all"],
        help="Filter by training protocol (default: distribution)",
    )
    args = parser.parse_args()

    console.print("[bold]Loading parquet files...[/bold]\n")
    df = load_parquet_files(args.inputs)

    if df.empty:
        console.print("[red]No data loaded[/red]")
        return

    console.print(f"\n[bold]Loaded {len(df)} total runs[/bold]")

    # Apply filters
    df = filter_dataframe(df, args.state, args.filter, args.project, args.protocol)
    if df.empty:
        console.print("[yellow]No runs after filtering[/yellow]")
        return

    console.print(f"After filtering: {len(df)} runs")

    # Enrich with parsed columns
    if not args.no_enrich:
        console.print("\n[dim]Enriching with parsed columns...[/dim]")
        json_cols = [c for c in df.columns if c in ["_config_json", "_summary_json"]]
        if json_cols:
            console.print(
                f"[dim]Preserving lossless JSON columns: {', '.join(json_cols)}[/dim]"
            )
        df = enrich_dataframe(df)

    # Print summary
    console.print()
    print_summary(df)

    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)

    # Convert mixed-type columns to string for parquet compatibility
    for col in df.columns:
        if df[col].apply(type).nunique() > 1:
            df[col] = df[col].astype(str)

    df.to_parquet(args.output, index=False)
    console.print(f"\n[green]Saved unified data to {args.output}[/green]")
    console.print(f"[dim]Schema: {len(df.columns)} columns[/dim]")


if __name__ == "__main__":
    main()
