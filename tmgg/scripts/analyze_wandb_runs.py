# /// script
# dependencies = [
#     "pandas>=2.0",
#     "rich",
# ]
# ///
"""
Analyze W&B experiment runs from exported JSON.

Usage:
    uv run scripts/analyze_wandb_runs.py wandb_runs_export.json
    uv run scripts/analyze_wandb_runs.py wandb_runs_export.json --group-by project
    uv run scripts/analyze_wandb_runs.py wandb_runs_export.json --filter "stage2c"
"""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def load_data(path: Path) -> pd.DataFrame:
    """Load exported W&B data into a DataFrame."""
    with open(path) as f:
        data = json.load(f)

    # Handle various formats: plain list, grouped dict, or our export format
    if isinstance(data, list):
        runs = data
    elif isinstance(data.get("data"), dict):
        runs = []
        for project_runs in data["data"].values():
            runs.extend(project_runs)
    elif isinstance(data.get("data"), list):
        runs = data["data"]
    else:
        runs = data

    return pd.DataFrame(runs)


def parse_run_name(name: str) -> dict:
    """Extract hyperparameters from run name."""
    parsed = {}

    # Stage pattern
    stage_match = re.search(r"stage(\d+[a-z]?)", name)
    if stage_match:
        parsed["stage"] = f"stage{stage_match.group(1)}"

    # k value
    k_match = re.search(r"_k(\d+)", name)
    if k_match:
        parsed["k"] = int(k_match.group(1))

    # Learning rate
    lr_match = re.search(r"_lr([\d.e-]+)", name)
    if lr_match:
        parsed["lr"] = lr_match.group(1)

    # Weight decay
    wd_match = re.search(r"_wd([\d.e-]+)", name)
    if wd_match:
        parsed["wd"] = wd_match.group(1)

    # Seed
    seed_match = re.search(r"_s(\d+)$", name)
    if seed_match:
        parsed["seed"] = int(seed_match.group(1))

    # Epsilon
    eps_match = re.search(r"_eps([\d.]+)", name)
    if eps_match:
        parsed["eps"] = float(eps_match.group(1))

    return parsed


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed columns from run names."""
    parsed = df["name"].apply(parse_run_name).apply(pd.Series)
    return pd.concat([df, parsed], axis=1)


def aggregate_stats(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Aggregate metrics by specified grouping columns."""
    metric_cols = [
        "test_mse",
        "test_subspace",
        "test_eigenval",
        "final_mse_01",
        "final_mse_03",
    ]
    available_metrics = [c for c in metric_cols if c in df.columns]

    agg_dict = {"name": "count"}
    for col in available_metrics:
        agg_dict[col] = ["mean", "min", "max", "std"]

    grouped = df.groupby(group_cols, dropna=False).agg(agg_dict)
    grouped.columns = ["_".join(col).strip("_") for col in grouped.columns]
    grouped = grouped.rename(columns={"name_count": "n_runs"})

    return grouped.reset_index()


def print_table(df: pd.DataFrame, title: str, max_rows: int = 30) -> None:
    """Print DataFrame as rich table."""
    table = Table(title=title)

    for col in df.columns:
        table.add_column(
            col, justify="right" if df[col].dtype in ["float64", "int64"] else "left"
        )

    for _, row in df.head(max_rows).iterrows():
        values = []
        for col in df.columns:
            val = row[col]
            if pd.isna(val):
                values.append("-")
            elif isinstance(val, float):
                values.append(f"{val:.4f}" if abs(val) < 100 else f"{val:.2f}")
            else:
                values.append(str(val))
        table.add_row(*values)

    console.print(table)
    if len(df) > max_rows:
        console.print(f"[dim]... and {len(df) - max_rows} more rows[/dim]")


def main():
    parser = argparse.ArgumentParser(description="Analyze W&B experiment data")
    parser.add_argument(
        "input", type=Path, help="Input JSON file from fetch_wandb_runs.py"
    )
    parser.add_argument(
        "--group-by",
        "-g",
        default="stage",
        help="Column(s) to group by (comma-separated). Options: project, stage, k, lr, wd",
    )
    parser.add_argument(
        "--filter", "-f", default=None, help="Filter runs by name pattern"
    )
    parser.add_argument(
        "--state",
        "-s",
        default="finished",
        help="Filter by run state (finished, crashed, running, all)",
    )
    parser.add_argument(
        "--top", "-t", type=int, default=10, help="Show top N runs by test_mse"
    )
    parser.add_argument("--csv", default=None, help="Export aggregated results to CSV")

    args = parser.parse_args()

    console.print(f"[bold]Loading {args.input}[/bold]\n")
    df = load_data(args.input)
    console.print(f"Loaded {len(df)} runs\n")

    # Filter by state
    if args.state != "all":
        df = df[df["state"] == args.state]
        console.print(f"Filtered to {len(df)} {args.state} runs\n")

    # Filter by name pattern
    if args.filter:
        df = df[df["name"].str.contains(args.filter, case=False, na=False)]
        console.print(f"Filtered to {len(df)} runs matching '{args.filter}'\n")

    if df.empty:
        console.print("[yellow]No runs match the filters[/yellow]")
        return

    # Enrich with parsed fields
    df = enrich_dataframe(df)

    # Show top runs by test_mse
    if "test_mse" in df.columns and df["test_mse"].notna().any():
        display_cols = [
            "name",
            "project",
            "test_mse",
            "test_subspace",
            "stage",
            "k",
            "lr",
        ]
        available_cols = [c for c in display_cols if c in df.columns]
        top_runs = df.nsmallest(args.top, "test_mse")[available_cols]
        print_table(top_runs, f"Top {args.top} Runs by test_mse")
        console.print()

    # Group and aggregate
    group_cols = [c.strip() for c in args.group_by.split(",")]
    valid_cols = [c for c in group_cols if c in df.columns]

    if valid_cols:
        agg_df = aggregate_stats(df, valid_cols)
        # Sort by mean test_mse if available
        if "test_mse_mean" in agg_df.columns:
            agg_df = agg_df.sort_values("test_mse_mean")
        print_table(agg_df, f"Aggregated by {', '.join(valid_cols)}")

        if args.csv:
            agg_df.to_csv(args.csv, index=False)
            console.print(f"\n[green]Saved to {args.csv}[/green]")
    else:
        console.print(
            f"[yellow]No valid grouping columns found. Available: {list(df.columns)}[/yellow]"
        )


if __name__ == "__main__":
    main()
