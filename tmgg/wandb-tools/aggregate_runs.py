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

Usage:
    uv run wandb-tools/aggregate_runs.py wandb_export/*.parquet -o analysis/unified.parquet
    uv run wandb-tools/aggregate_runs.py wandb_export/ -o analysis/unified.parquet --state finished
    uv run wandb-tools/aggregate_runs.py wandb_export/ --filter "stage2c" -o analysis/stage2c.parquet
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from rich.console import Console
from rich.table import Table

console = Console()


def parse_stage(name: str) -> str:
    """Extract stage identifier from run name."""
    if pd.isna(name):
        return "unknown"
    name = str(name)
    # Check for specific stage patterns (more specific first)
    stage_patterns = [
        (r"stage2c", "stage2c"),
        (r"stage2b", "stage2b"),
        (r"stage2a", "stage2a"),
        (r"stage1f", "stage1f"),
        (r"stage1e", "stage1e"),
        (r"stage1d", "stage1d"),
        (r"stage1c", "stage1c"),
        (r"stage1b", "stage1b"),
        (r"stage1a", "stage1a"),
        (r"stage2", "stage2"),
        (r"stage1", "stage1"),
        (r"stage3", "stage3"),
    ]
    for pattern, stage in stage_patterns:
        if re.search(pattern, name.lower()):
            return stage
    return "other"


def parse_architecture(name: str) -> str:
    """Extract architecture type from run name."""
    if pd.isna(name):
        return "unknown"
    name = str(name).lower()
    arch_patterns = [
        (r"gnn_all", "gnn_all"),
        (r"gnn_qk", "gnn_qk"),
        (r"gnn_v", "gnn_v"),
        (r"self_attention", "self_attention"),
        (r"digress_transformer", "digress_transformer"),
        (r"digress_default", "digress_default"),
        (r"asymmetric", "asymmetric"),
        (r"spectral_linear", "spectral_linear"),
        (r"spectral", "spectral"),
        (r"digress", "digress"),
    ]
    for pattern, arch in arch_patterns:
        if re.search(pattern, name):
            return arch
    return "other"


def parse_model_type(name: str, config_target: str | None = None) -> str:
    """Extract model type from run name or config target."""
    # Try config target first
    if config_target and not pd.isna(config_target):
        target_str = str(config_target).lower()
        if "spectral" in target_str:
            return "spectral"
        if "digress" in target_str:
            return "digress"
        if "gnn" in target_str:
            return "gnn"
        if "attention" in target_str:
            return "attention"
        if "hybrid" in target_str:
            return "hybrid"

    # Fall back to name parsing
    if pd.isna(name):
        return "unknown"
    name = str(name).lower()
    if "spectral" in name:
        return "spectral"
    if "digress" in name:
        return "digress"
    if "gnn" in name:
        return "gnn"
    return "unknown"


def parse_run_name_fields(name: str) -> dict:
    """Extract hyperparameters encoded in run name.

    Common patterns:
    - _k{value}: k value for spectral models
    - _lr{value}: learning rate
    - _wd{value}: weight decay
    - _s{value}: seed
    - _eps{value}: epsilon value
    """
    parsed = {}
    if pd.isna(name):
        return parsed
    name = str(name)

    # k value
    k_match = re.search(r"_k(\d+)", name)
    if k_match:
        parsed["k"] = int(k_match.group(1))

    # Learning rate
    lr_match = re.search(r"_lr([\d.e-]+)", name)
    if lr_match:
        parsed["lr_parsed"] = lr_match.group(1)

    # Weight decay
    wd_match = re.search(r"_wd([\d.e-]+)", name)
    if wd_match:
        parsed["wd_parsed"] = wd_match.group(1)

    # Seed
    seed_match = re.search(r"_s(\d+)(?:_|$)", name)
    if seed_match:
        parsed["seed_parsed"] = int(seed_match.group(1))

    # Epsilon
    eps_match = re.search(r"_eps([\d.]+)", name)
    if eps_match:
        parsed["eps_parsed"] = float(eps_match.group(1))

    # Asymmetric flag
    parsed["asymmetric_flag"] = "asymmetric" in name.lower()

    return parsed


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


def parse_protocol(row: pd.Series) -> str:
    """Determine if run used single-graph or distribution protocol.

    Single-graph: train/val/test use the SAME graph with varying noise.
    Distribution: train/val/test use DIFFERENT graphs from a distribution.
    """
    # Check config_data_same_graph_all_splits
    same_graph = row.get("config_data_same_graph_all_splits")
    if same_graph is True or same_graph == "True":
        return "single_graph"

    # Check data module target
    target = row.get("config_data__target_", "")
    if pd.notna(target) and "SingleGraph" in str(target):
        return "single_graph"

    return "distribution"


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed/derived columns for analysis.

    Preserves raw JSON columns (`_config_json`, `_summary_json`) if present.
    These columns contain lossless representations of the original W&B data.
    """
    df = df.copy()

    # Log presence of JSON columns for lossless data access
    json_cols = [c for c in df.columns if c in ["_config_json", "_summary_json"]]
    if json_cols:
        console.print(
            f"[dim]Preserving lossless JSON columns: {', '.join(json_cols)}[/dim]"
        )

    # Parse protocol (single-graph vs distribution)
    df["protocol"] = df.apply(parse_protocol, axis=1)

    # Parse stage from name
    df["stage"] = df["name"].apply(parse_stage)

    # Parse architecture
    df["arch"] = df["name"].apply(parse_architecture)

    # Parse model type (using config if available)
    config_target_col = None
    for col in ["config_model__target_", "config_model_target"]:
        if col in df.columns:
            config_target_col = col
            break

    if config_target_col:
        df["model_type"] = df.apply(
            lambda row: parse_model_type(row["name"], row.get(config_target_col)),
            axis=1,
        )
    else:
        df["model_type"] = df["name"].apply(lambda n: parse_model_type(n, None))

    # Parse name-encoded fields
    parsed_fields = df["name"].apply(parse_run_name_fields).apply(pd.Series)
    for col in parsed_fields.columns:
        if col not in df.columns:
            df[col] = parsed_fields[col]

    # Extract numeric values from config columns where possible
    numeric_candidates = [
        "config_learning_rate",
        "config_weight_decay",
        "config_model_k",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


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
