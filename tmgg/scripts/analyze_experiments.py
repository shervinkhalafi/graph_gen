# /// script
# dependencies = [
#     "wandb>=0.15",
#     "pandas>=2.0",
#     "pyarrow",
#     "scikit-learn>=1.3",
#     "rich",
#     "python-dotenv",
# ]
# ///
"""
Download and analyze W&B experiment data with hyperparameter importance analysis.

Usage:
    uv run scripts/analyze_experiments.py                    # Full analysis
    uv run scripts/analyze_experiments.py --skip-download    # Use cached data
    uv run scripts/analyze_experiments.py --importance-only  # Only importance analysis
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder

console = Console()

# Configuration
OUTPUT_DIR = Path("eigenstructure_results_full")
PARQUET_FILE = OUTPUT_DIR / "all_runs.parquet"

ENTITY = "graph_denoise_team"
PROJECTS = [
    "spectral_denoising",
    "tmgg-stage2_validation",
    "tmgg-stage1_5_crossdata",
    "00_initial_experiment_widening",
]


def get_api_key() -> str:
    """Get W&B API key from environment."""
    load_dotenv()
    key = os.getenv("GRAPH_DENOISE_TEAM_SERVICE")
    if not key:
        raise ValueError("GRAPH_DENOISE_TEAM_SERVICE not found in .env")
    return key


def download_runs() -> pd.DataFrame:
    """Download all runs from W&B and save to parquet."""
    import wandb

    api_key = get_api_key()
    api = wandb.Api(api_key=api_key)

    all_runs = []

    for project_name in PROJECTS:
        console.print(f"Fetching {project_name}...")
        runs = api.runs(f"{ENTITY}/{project_name}")

        for run in runs:
            summary = dict(run.summary) if run.summary else {}
            config = dict(run.config) if run.config else {}

            # Flatten config
            flat_config: dict[str, object] = {}

            def flatten(d: dict, prefix: str, out: dict[str, object]) -> None:
                for k, v in d.items():
                    key = f"{prefix}{k}" if prefix else k
                    if isinstance(v, dict):
                        flatten(v, f"{key}_", out)
                    elif isinstance(v, list | tuple):
                        out[f"config_{key}"] = str(v)
                    elif isinstance(v, int | float | str | bool) or v is None:
                        out[f"config_{key}"] = v
                    else:
                        out[f"config_{key}"] = str(v)

            flatten(config, "", flat_config)

            # Extract numeric metrics
            metrics = {}
            for k, v in summary.items():
                if isinstance(v, int | float):
                    clean_k = k.replace("/", "_").replace(".", "_")
                    metrics[f"metric_{clean_k}"] = v

            run_data = {
                "id": run.id,
                "name": run.name,
                "state": run.state,
                "project": project_name,
                "created_at": run.created_at,
                **flat_config,
                **metrics,
            }
            all_runs.append(run_data)

        console.print(
            f"  -> {len([r for r in all_runs if r['project'] == project_name])} runs"
        )

    df = pd.DataFrame(all_runs)

    # Convert mixed-type columns to string
    for col in df.columns:
        if df[col].apply(type).nunique() > 1:
            df[col] = df[col].astype(str)

    OUTPUT_DIR.mkdir(exist_ok=True)
    df.to_parquet(PARQUET_FILE, index=False)
    console.print(f"\n[green]Saved {len(df)} runs to {PARQUET_FILE}[/green]")

    return df


def load_data() -> pd.DataFrame:
    """Load data from parquet file."""
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(
            f"{PARQUET_FILE} not found. Run without --skip-download first."
        )
    return pd.read_parquet(PARQUET_FILE)


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create analysis-ready features from raw data."""
    df = df[df["state"] == "finished"].copy()

    # Parse stage from name
    def parse_stage(name):
        if pd.isna(name):
            return "unknown"
        name = str(name)
        if "stage2c" in name:
            return "stage2c"
        if "stage1f" in name:
            return "stage1f"
        if "stage1d" in name:
            return "stage1d"
        if "stage1c" in name:
            return "stage1c"
        if "stage2" in name:
            return "stage2"
        if "stage1" in name:
            return "stage1"
        return "other"

    # Parse architecture from name
    def parse_arch(name):
        if pd.isna(name):
            return "unknown"
        name = str(name)
        if "gnn_all" in name:
            return "gnn_all"
        if "gnn_qk" in name:
            return "gnn_qk"
        if "gnn_v" in name:
            return "gnn_v"
        if "self_attention" in name:
            return "self_attention"
        if "digress_transformer" in name or "digress_default" in name:
            return "digress_default"
        if "asymmetric" in name:
            return "asymmetric"
        if "spectral_linear" in name:
            return "spectral_linear"
        return "other"

    # Create feature columns
    df["stage"] = df["name"].apply(parse_stage)
    df["arch"] = df["name"].apply(parse_arch)
    df["model_type"] = df["config_model__target_"].apply(
        lambda x: "spectral"
        if "Spectral" in str(x)
        else "digress"
        if "Digress" in str(x)
        else "unknown"
    )
    df["k"] = pd.to_numeric(df["config_model_k"], errors="coerce")
    df["lr"] = pd.to_numeric(df["config_learning_rate"], errors="coerce")
    df["wd"] = pd.to_numeric(df["config_weight_decay"], errors="coerce")
    df["asymmetric"] = df["config_asymmetric"].apply(
        lambda x: 1 if x == "True" or x is True else 0
    )

    # Target metrics
    df["test_mse"] = pd.to_numeric(df["metric_test_mse"], errors="coerce")
    df["test_subspace"] = pd.to_numeric(
        df["metric_test_subspace_distance"], errors="coerce"
    )

    return df


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics by various groupings."""
    df_valid = df[df["test_mse"].notna() & (df["test_mse"] > 0)].copy()

    def agg_stats(group):
        return pd.Series(
            {
                "n": len(group),
                "mean": group["test_mse"].mean(),
                "std": group["test_mse"].std(),
                "min": group["test_mse"].min(),
                "subspace_mean": group["test_subspace"].mean(),
            }
        )

    sections = [
        ("PROJECT", "project"),
        ("STAGE", "stage"),
        ("ARCHITECTURE", "arch"),
        ("MODEL TYPE", "model_type"),
        ("K VALUE", "k"),
        ("LEARNING RATE", "lr"),
        ("WEIGHT DECAY", "wd"),
        ("ASYMMETRIC", "asymmetric"),
    ]

    for title, col in sections:
        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]ANALYSIS BY {title}[/bold]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")

        result = (
            df_valid.groupby(col)
            .apply(agg_stats, include_groups=False)
            .sort_values("mean")
        )

        table = Table()
        table.add_column(col, style="cyan")
        table.add_column("N", justify="right")
        table.add_column("Mean MSE", justify="right")
        table.add_column("Std", justify="right")
        table.add_column("Min MSE", justify="right")
        table.add_column("Subspace", justify="right")

        for idx, row in result.iterrows():
            table.add_row(
                str(idx),
                str(int(row["n"])),
                f"{row['mean']:.4f}",
                f"{row['std']:.4f}" if pd.notna(row["std"]) else "-",
                f"{row['min']:.4f}",
                f"{row['subspace_mean']:.4f}"
                if pd.notna(row["subspace_mean"])
                else "-",
            )

        console.print(table)


def compute_importance(df: pd.DataFrame, target: str = "test_mse") -> pd.DataFrame:
    """Compute hyperparameter importance using Random Forest."""
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold]HYPERPARAMETER IMPORTANCE FOR {target}[/bold]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    # Prepare features
    feature_cols = ["stage", "arch", "model_type", "k", "lr", "wd", "asymmetric"]
    df_valid = df[df[target].notna() & (df[target] > 0)].copy()

    # Encode categorical features
    encoders = {}
    X_data = {}

    for col in feature_cols:
        if col in ["k", "lr", "wd", "asymmetric"]:
            X_data[col] = df_valid[col].fillna(-1).values
        else:
            le = LabelEncoder()
            X_data[col] = le.fit_transform(df_valid[col].fillna("unknown").astype(str))
            encoders[col] = le

    X = pd.DataFrame(X_data)
    y = df_valid[target].values

    # Remove rows with NaN
    mask = ~np.isnan(X.values).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    console.print(f"Training on {len(X)} samples with {len(feature_cols)} features")

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    # Feature importance (impurity-based)
    importance_impurity = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_impurity": rf.feature_importances_,
        }
    ).sort_values("importance_impurity", ascending=False)

    # Permutation importance (more reliable)
    console.print("Computing permutation importance...")
    perm_importance = permutation_importance(
        rf, X, y, n_repeats=10, random_state=42, n_jobs=-1
    )

    importance_perm = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_perm": perm_importance.importances_mean,
            "importance_perm_std": perm_importance.importances_std,
        }
    ).sort_values("importance_perm", ascending=False)

    # Merge results
    importance = importance_impurity.merge(importance_perm, on="feature")
    importance = importance.sort_values("importance_perm", ascending=False)

    # Print results
    table = Table(title=f"Hyperparameter Importance for {target}")
    table.add_column("Feature", style="cyan")
    table.add_column("Impurity Importance", justify="right")
    table.add_column("Permutation Importance", justify="right")
    table.add_column("± Std", justify="right")

    for _, row in importance.iterrows():
        table.add_row(
            row["feature"],
            f"{row['importance_impurity']:.4f}",
            f"{row['importance_perm']:.4f}",
            f"±{row['importance_perm_std']:.4f}",
        )

    console.print(table)

    # Model performance
    from sklearn.model_selection import cross_val_score

    scores = cross_val_score(rf, X, y, cv=5, scoring="r2")
    console.print(
        f"\nRandom Forest R² (5-fold CV): {scores.mean():.3f} ± {scores.std():.3f}"
    )

    return importance


def main():
    parser = argparse.ArgumentParser(description="Analyze W&B experiments")
    parser.add_argument(
        "--skip-download", action="store_true", help="Use cached parquet data"
    )
    parser.add_argument(
        "--importance-only", action="store_true", help="Only compute importance"
    )
    parser.add_argument("--output", "-o", default=None, help="Save importance to CSV")
    args = parser.parse_args()

    # Load or download data
    if args.skip_download:
        console.print("[dim]Loading cached data...[/dim]")
        df = load_data()
    else:
        console.print("[bold]Downloading data from W&B...[/bold]")
        df = download_runs()

    console.print(f"Loaded {len(df)} total runs")

    # Prepare features
    df = prepare_features(df)
    df_valid = df[df["test_mse"].notna() & (df["test_mse"] > 0)]
    console.print(f"Valid runs with test_mse > 0: {len(df_valid)}")

    if not args.importance_only:
        print_summary_stats(df)

    # Compute importance for both targets
    importance_mse = compute_importance(df, "test_mse")
    importance_subspace = compute_importance(df, "test_subspace")

    if args.output:
        combined = importance_mse.merge(
            importance_subspace, on="feature", suffixes=("_mse", "_subspace")
        )
        combined.to_csv(args.output, index=False)
        console.print(f"\n[green]Saved importance to {args.output}[/green]")

    # Print top 10 runs
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print("[bold]TOP 10 RUNS BY TEST MSE[/bold]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")

    top10 = df_valid.nsmallest(10, "test_mse")[
        ["name", "project", "stage", "arch", "k", "lr", "test_mse", "test_subspace"]
    ]
    console.print(top10.to_string())


if __name__ == "__main__":
    main()
