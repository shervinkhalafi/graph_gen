# /// script
# dependencies = [
#     "pandas>=2.0",
#     "pyarrow",
#     "rich",
# ]
# ///
"""Quick CLI analysis of aggregated W&B run data.

Load parquet data, group/aggregate by specified columns, print summary tables,
and export filtered results.

Usage:
    uv run wandb-tools/analyze_runs.py  # Uses latest export from wandb_export/
    uv run wandb-tools/analyze_runs.py analysis/unified.parquet --group-by stage
    uv run wandb-tools/analyze_runs.py --filter "stage3_pyg_dist" --group-by architecture,dataset
    uv run wandb-tools/analyze_runs.py --top 20 --metric test_accuracy --descending
"""

import argparse
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()

# Default export location
DEFAULT_EXPORT_DIR = Path("wandb_export")
DEFAULT_PROJECT = "graph_denoise_team_spectral_denoising_runs.parquet"


def find_latest_export() -> Path | None:
    """Find the most recent parquet export file."""
    if not DEFAULT_EXPORT_DIR.exists():
        return None

    # Look for spectral_denoising first, then any parquet
    default_path = DEFAULT_EXPORT_DIR / DEFAULT_PROJECT
    if default_path.exists():
        return default_path

    parquets = list(DEFAULT_EXPORT_DIR.glob("*.parquet"))
    if parquets:
        return max(parquets, key=lambda p: p.stat().st_mtime)
    return None


def parse_architecture(name: str) -> str:
    """Parse architecture from run display name."""
    if pd.isna(name):
        return "unknown"
    name = str(name)
    if "digress_transformer_gnn_qk" in name:
        return "digress_transformer_gnn_qk"
    elif "digress_transformer" in name:
        return "digress_transformer"
    elif "self_attention" in name:
        return "self_attention"
    elif "filter_bank" in name:
        return "filter_bank"
    elif "linear_pe" in name:
        return "linear_pe"
    elif "mlp" in name.lower():
        return "mlp"
    return "other"


def parse_dataset(name: str) -> str:
    """Parse dataset from run display name."""
    if pd.isna(name):
        return "unknown"
    name = str(name)
    if "pyg_enzymes" in name:
        return "pyg_enzymes"
    elif "pyg_proteins" in name:
        return "pyg_proteins"
    elif "ego" in name.lower():
        return "ego"
    elif "community" in name.lower():
        return "community"
    elif "grid" in name.lower():
        return "grid"
    elif "sbm" in name.lower():
        return "sbm"
    return "synthetic"


def enrich_with_parsed_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed architecture and dataset columns from display_name."""
    name_col = "display_name" if "display_name" in df.columns else "name"

    if name_col in df.columns:
        df["architecture"] = df[name_col].apply(parse_architecture)
        df["dataset"] = df[name_col].apply(parse_dataset)

    return df


def find_metric_columns(df: pd.DataFrame) -> list[str]:
    """Find all metric columns in the dataframe."""
    metrics = []
    metric_names = {"test_mse", "test_subspace", "val_loss", "train_loss"}
    for col in df.columns:
        is_metric_col = col.startswith("metric_") or col in metric_names
        is_numeric = df[col].dtype in [
            "float64",
            "int64",
        ] or pd.api.types.is_numeric_dtype(df[col])
        if is_metric_col and is_numeric:
            metrics.append(col)
    return metrics


def detect_target_metric(df: pd.DataFrame, preferred: str | None = None) -> str | None:
    """Detect the primary metric column to use for analysis.

    Looks for common metric names in order of preference.
    """
    if preferred and preferred in df.columns:
        return preferred

    # Common metric names in order of preference
    candidates = [
        "metric_test_mse",
        "test_mse",
        "metric_test_loss",
        "metric_val_mse",
        "metric_val_loss",
        "val_loss",
        "metric_test_subspace_distance",
        "test_subspace",
    ]

    for candidate in candidates:
        if candidate in df.columns and df[candidate].notna().any():
            return candidate

    # Fall back to first numeric metric column
    metric_cols = find_metric_columns(df)
    return metric_cols[0] if metric_cols else None


def aggregate_by_groups(
    df: pd.DataFrame,
    group_cols: list[str],
    metric_cols: list[str],
) -> pd.DataFrame:
    """Aggregate metrics by grouping columns."""
    valid_groups = [c for c in group_cols if c in df.columns]
    if not valid_groups:
        console.print(
            f"[yellow]No valid grouping columns found. Available: {list(df.columns)[:20]}...[/yellow]"
        )
        return pd.DataFrame()

    valid_metrics = [c for c in metric_cols if c in df.columns]
    if not valid_metrics:
        console.print("[yellow]No valid metric columns found[/yellow]")
        return pd.DataFrame()

    # Build aggregation dict
    agg_dict = {"name": "count"}
    for col in valid_metrics:
        agg_dict[col] = ["mean", "std", "min", "max"]

    try:
        grouped = df.groupby(valid_groups, dropna=False).agg(agg_dict)
        # Flatten multi-level columns
        grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]
        grouped = grouped.rename(columns={"name_count": "n_runs"})
        return grouped.reset_index()
    except Exception as e:
        console.print(f"[red]Error during aggregation: {e}[/red]")
        return pd.DataFrame()


def print_dataframe_table(
    df: pd.DataFrame,
    title: str,
    max_rows: int = 30,
    float_precision: int = 4,
) -> None:
    """Print DataFrame as a rich table."""
    if df.empty:
        console.print(f"[yellow]No data for '{title}'[/yellow]")
        return

    table = Table(title=title)

    for col in df.columns:
        justify = "right" if pd.api.types.is_numeric_dtype(df[col]) else "left"
        table.add_column(col, justify=justify)

    for _, row in df.head(max_rows).iterrows():
        values = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                values.append("-")
            elif isinstance(val, float):
                if abs(val) < 0.0001 or abs(val) > 10000:
                    values.append(f"{val:.2e}")
                else:
                    values.append(f"{val:.{float_precision}f}")
            else:
                values.append(str(val)[:50])  # Truncate long strings
        table.add_row(*values)

    console.print(table)

    if len(df) > max_rows:
        console.print(f"[dim]... and {len(df) - max_rows} more rows[/dim]")


def print_top_runs(
    df: pd.DataFrame,
    metric: str,
    n: int = 10,
    ascending: bool = True,
) -> None:
    """Print top N runs by a metric."""
    if metric not in df.columns:
        console.print(f"[yellow]Metric '{metric}' not found[/yellow]")
        return

    df_valid = df[df[metric].notna()].copy()
    if df_valid.empty:
        console.print(f"[yellow]No runs with valid '{metric}' values[/yellow]")
        return

    top_runs = (
        df_valid.nsmallest(n, metric) if ascending else df_valid.nlargest(n, metric)
    )

    # Select columns to display
    display_cols = [
        "name",
        "project",
        metric,
        "state",
        "architecture",
        "dataset",
        "stage",
        "arch",
        "model_type",
    ]
    available_cols = [c for c in display_cols if c in top_runs.columns]

    direction = "lowest" if ascending else "highest"
    print_dataframe_table(
        top_runs[available_cols], f"Top {n} runs by {direction} {metric}"
    )


def print_schema_info(df: pd.DataFrame) -> None:
    """Print schema information about the dataframe."""
    console.print(f"\n[bold]Schema: {len(df.columns)} columns[/bold]")

    # Categorize columns
    config_cols = [c for c in df.columns if c.startswith("config_")]
    metric_cols = [c for c in df.columns if c.startswith("metric_")]
    meta_cols = [c for c in df.columns if c not in config_cols + metric_cols]

    console.print(f"  Config columns: {len(config_cols)}")
    console.print(f"  Metric columns: {len(metric_cols)}")
    console.print(f"  Metadata columns: {len(meta_cols)}")

    # Show sample metric columns
    if metric_cols:
        console.print(
            f"\n[dim]Metric columns sample: {', '.join(metric_cols[:10])}...[/dim]"
        )


def print_pivot_table(
    df: pd.DataFrame,
    metric: str,
    row_col: str = "architecture",
    col_col: str = "dataset",
) -> None:
    """Print a pivot table of metric by row and column dimensions."""
    if metric not in df.columns:
        console.print(f"[yellow]Metric '{metric}' not found for pivot[/yellow]")
        return

    try:
        pivot = df.pivot_table(
            values=metric,
            index=row_col,
            columns=col_col,
            aggfunc=["mean", "std", "count"],
        )

        # Print mean values
        console.print(f"\n[bold]{metric} by {row_col} × {col_col}[/bold]")

        table = Table(title=f"Mean {metric}")
        table.add_column(row_col, style="bold")

        datasets = pivot["mean"].columns.tolist()
        for ds in datasets:
            table.add_column(ds, justify="right")

        for arch in pivot.index:
            row = [arch]
            for ds in datasets:
                mean_val = pivot.loc[arch, ("mean", ds)]
                count_val = pivot.loc[arch, ("count", ds)]
                if pd.notna(mean_val):
                    row.append(f"{mean_val:.4f} (n={int(count_val)})")
                else:
                    row.append("-")
            table.add_row(*row)

        console.print(table)
    except Exception as e:
        console.print(f"[red]Error creating pivot table: {e}[/red]")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze aggregated W&B run data")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=None,
        help="Input parquet file (default: latest from wandb_export/)",
    )
    parser.add_argument(
        "--group-by",
        "-g",
        default=None,
        help="Column(s) to group by (comma-separated). E.g., 'stage', 'architecture,dataset'",
    )
    parser.add_argument(
        "--metric",
        "-m",
        default=None,
        help="Primary metric column for analysis (auto-detected if not specified)",
    )
    parser.add_argument(
        "--filter",
        "-f",
        default=None,
        help="Filter runs by name pattern",
    )
    parser.add_argument(
        "--state",
        "-s",
        default="finished",
        help="Filter by run state (finished, crashed, running, all)",
    )
    parser.add_argument(
        "--top",
        "-t",
        type=int,
        default=10,
        help="Show top N runs by metric",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        default=True,
        help="Sort ascending (default for loss/error metrics)",
    )
    parser.add_argument(
        "--descending",
        action="store_true",
        help="Sort descending (for accuracy/score metrics)",
    )
    parser.add_argument(
        "--pivot",
        action="store_true",
        help="Show pivot table of metric by architecture × dataset",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Export results to CSV",
    )
    parser.add_argument(
        "--schema",
        action="store_true",
        help="Print schema information",
    )
    args = parser.parse_args()

    # Find input file
    input_path = args.input
    if input_path is None:
        input_path = find_latest_export()
        if input_path is None:
            console.print(
                "[red]No input file specified and no exports found in wandb_export/[/red]"
            )
            console.print(
                "Run: uv run wandb-tools/export_runs.py --entity <entity> --project <project>"
            )
            return

    # Load data
    console.print(f"[bold]Loading {input_path}[/bold]\n")
    df = pd.read_parquet(input_path)
    console.print(f"Loaded {len(df)} runs")

    # Enrich with parsed columns
    df = enrich_with_parsed_columns(df)

    # Print schema if requested
    if args.schema:
        print_schema_info(df)
        return

    # Apply filters
    if args.state and args.state != "all" and "state" in df.columns:
        df = df[df["state"] == args.state]
        console.print(f"Filtered to {len(df)} {args.state} runs")

    if args.filter:
        name_col = "display_name" if "display_name" in df.columns else "name"
        df = df[df[name_col].str.contains(args.filter, case=False, na=False)]
        console.print(f"Filtered to {len(df)} runs matching '{args.filter}'")

    if df.empty:
        console.print("[yellow]No runs after filtering[/yellow]")
        return

    # Detect metric
    ascending = not args.descending
    metric = detect_target_metric(df, args.metric)
    if metric:
        console.print(f"\n[dim]Using metric: {metric}[/dim]")
    else:
        console.print("[yellow]No suitable metric column found[/yellow]")

    # Show top runs
    if metric:
        console.print()
        print_top_runs(df, metric, args.top, ascending)

    # Show pivot table if requested
    if args.pivot and metric:
        print_pivot_table(df, metric)

    # Group and aggregate
    if args.group_by:
        group_cols = [c.strip() for c in args.group_by.split(",")]
        metric_cols = find_metric_columns(df)

        console.print()
        agg_df = aggregate_by_groups(df, group_cols, metric_cols)

        if not agg_df.empty:
            # Sort by mean of primary metric if available
            sort_col = (
                f"{metric}_mean"
                if metric and f"{metric}_mean" in agg_df.columns
                else None
            )
            if sort_col:
                agg_df = agg_df.sort_values(sort_col, ascending=ascending)

            print_dataframe_table(agg_df, f"Aggregated by {', '.join(group_cols)}")

            if args.output:
                agg_df.to_csv(args.output, index=False)
                console.print(
                    f"\n[green]Saved aggregated results to {args.output}[/green]"
                )


if __name__ == "__main__":
    main()
