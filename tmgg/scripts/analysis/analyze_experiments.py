#!/usr/bin/env python3
"""Analyze unified experiment data and generate reports/plots.

Reads unified parquet files from unify_exports.py and creates:
1. Best hyperparameter selection per (model, dataset, noise_level)
2. Training dynamics plots (loss/accuracy vs steps)
3. Denoising evolution visualizations

Usage:
    python analyze_experiments.py --input-dir analysis_output --output-dir reports
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def load_unified_data(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load unified history and summary DataFrames."""
    history_path = input_dir / "unified_history.parquet"
    summary_path = input_dir / "unified_summary.parquet"

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    summary_df = pd.read_parquet(summary_path)
    logger.info(f"Loaded summary: {len(summary_df)} runs")

    if history_path.exists():
        history_df = pd.read_parquet(history_path)
        logger.info(f"Loaded history: {len(history_df)} rows")
    else:
        logger.warning("No history file found, some analyses will be skipped")
        history_df = pd.DataFrame()

    return history_df, summary_df


def identify_best_runs(
    summary_df: pd.DataFrame,
    group_cols: Sequence[str],
    metric: str = "val/loss",
    minimize: bool = True,
) -> pd.DataFrame:
    """Identify best run per group based on metric.

    Parameters
    ----------
    summary_df
        Summary DataFrame with run configs and final metrics.
    group_cols
        Columns to group by (e.g., ["model_type", "data_dataset_name", "noise_level"]).
    metric
        Metric column to optimize.
    minimize
        If True, select minimum; if False, select maximum.

    Returns
    -------
    DataFrame with best runs per group.
    """
    # Filter to runs that have the metric
    df = summary_df.dropna(subset=[metric]).copy()

    if df.empty:
        _ = logger.warning(f"No runs have metric '{metric}'")
        return pd.DataFrame()

    # Handle missing group columns gracefully
    available_group_cols = [c for c in group_cols if c in df.columns]
    if not available_group_cols:
        _ = logger.warning(f"No group columns found: {group_cols}")
        return df

    # Find best per group
    idx: pd.Series[Any]
    if minimize:
        idx = df.groupby(available_group_cols, dropna=False)[metric].idxmin()
    else:
        idx = df.groupby(available_group_cols, dropna=False)[metric].idxmax()

    best_df = df.loc[idx.dropna()].copy()
    best_df["is_best"] = True

    _ = logger.info(f"Selected {len(best_df)} best runs from {len(df)} total")
    return best_df


def plot_training_dynamics(
    history_df: pd.DataFrame,
    best_runs: pd.DataFrame,
    output_dir: Path,
    max_runs_per_plot: int = 10,
) -> None:
    """Plot training dynamics (loss curves) for best runs.

    Creates plots showing:
    - Train loss vs steps
    - Validation loss vs steps
    - Combined comparison across models
    """
    if history_df.empty or best_runs.empty:
        _ = logger.warning("No data for training dynamics plots")
        return

    plots_dir = output_dir / "training_dynamics"
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Filter history to best runs
    best_run_ids = set(best_runs["run_id"].unique())
    history_best = history_df[history_df["run_id"].isin(best_run_ids)].copy()

    if history_best.empty:
        _ = logger.warning("No history data for best runs")
        return

    step_col = "_step" if "_step" in history_best.columns else "trainer/global_step"

    if step_col not in history_best.columns:
        _ = logger.warning("No step column found")
        return

    # Set style
    _ = sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 6)
    plt.rcParams["figure.dpi"] = 150

    # Plot 1: Train loss per model type
    if (
        "train_loss_step" in history_best.columns
        or "train_loss_epoch" in history_best.columns
    ):
        train_col = (
            "train_loss_step"
            if "train_loss_step" in history_best.columns
            else "train_loss_epoch"
        )

        fig: Figure
        ax: Axes
        fig, ax = plt.subplots()
        for run_id in list(best_run_ids)[:max_runs_per_plot]:
            run_data = history_best[history_best["run_id"] == run_id]
            run_data = run_data.dropna(subset=[train_col, step_col])
            if not run_data.empty:
                # Get model type for label
                model_type: Any = (
                    run_data["model_type"].iloc[0]
                    if "model_type" in run_data.columns
                    else run_id
                )
                _ = ax.plot(
                    run_data[step_col],
                    run_data[train_col],
                    label=f"{model_type} ({run_id[:8]})",
                    alpha=0.8,
                )

        _ = ax.set_xlabel("Step")
        _ = ax.set_ylabel("Train Loss")
        _ = ax.set_title("Training Loss Dynamics (Best Runs)")
        _ = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_yscale("log")
        _ = plt.tight_layout()
        _ = plt.savefig(plots_dir / "train_loss_dynamics.png", bbox_inches="tight")
        plt.close()
        _ = logger.info(f"Saved: {plots_dir / 'train_loss_dynamics.png'}")

    # Plot 2: Validation loss per model type
    if "val/loss" in history_best.columns:
        fig, ax = plt.subplots()
        for run_id in list(best_run_ids)[:max_runs_per_plot]:
            run_data = history_best[history_best["run_id"] == run_id]
            run_data = run_data.dropna(subset=["val/loss", step_col])
            if not run_data.empty:
                model_type = (
                    run_data["model_type"].iloc[0]
                    if "model_type" in run_data.columns
                    else run_id
                )
                _ = ax.plot(
                    run_data[step_col],
                    run_data["val/loss"],
                    label=f"{model_type} ({run_id[:8]})",
                    alpha=0.8,
                )

        _ = ax.set_xlabel("Step")
        _ = ax.set_ylabel("Validation Loss")
        _ = ax.set_title("Validation Loss Dynamics (Best Runs)")
        _ = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
        ax.set_yscale("log")
        _ = plt.tight_layout()
        _ = plt.savefig(plots_dir / "val_loss_dynamics.png", bbox_inches="tight")
        plt.close()
        _ = logger.info(f"Saved: {plots_dir / 'val_loss_dynamics.png'}")

    # Plot 3: Comparison heatmap of final metrics
    metric_cols = [
        "val/loss",
        "val/mse",
        "test/loss",
        "test/mse",
        "val/eigenvalue_error",
        "test/eigenvalue_error",
    ]
    available_metrics = [c for c in metric_cols if c in best_runs.columns]

    if available_metrics and "model_type" in best_runs.columns:
        pivot_data = best_runs.groupby("model_type")[available_metrics].mean()

        if not pivot_data.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            _ = sns.heatmap(pivot_data, annot=True, fmt=".4f", cmap="RdYlGn_r", ax=ax)
            _ = ax.set_title("Mean Final Metrics by Model Type (Best Runs)")
            _ = plt.tight_layout()
            _ = plt.savefig(plots_dir / "metrics_heatmap.png", bbox_inches="tight")
            plt.close()
            _ = logger.info(f"Saved: {plots_dir / 'metrics_heatmap.png'}")


def plot_metric_evolution(
    history_df: pd.DataFrame,
    best_runs: pd.DataFrame,
    output_dir: Path,
    metrics: Sequence[str] | None = None,
) -> None:
    """Plot evolution of various metrics over training.

    Creates per-metric plots showing how metrics evolve during training.
    """
    if history_df.empty or best_runs.empty:
        return

    plots_dir = output_dir / "metric_evolution"
    plots_dir.mkdir(parents=True, exist_ok=True)

    best_run_ids = set(best_runs["run_id"].unique())
    history_best = history_df[history_df["run_id"].isin(best_run_ids)].copy()

    if history_best.empty:
        return

    step_col = "_step" if "_step" in history_best.columns else "trainer/global_step"

    # Default metrics to plot
    metrics_to_use: Sequence[str]
    if metrics is None:
        metrics_to_use = [
            "val/mse",
            "val/loss",
            "val/eigenvalue_error",
            "val/subspace_distance",
            "test/mse",
            "test/loss",
            "test/eigenvalue_error",
        ]
    else:
        metrics_to_use = metrics

    available_metrics = [m for m in metrics_to_use if m in history_best.columns]

    _ = sns.set_style("whitegrid")

    for metric in available_metrics:
        fig, ax = plt.subplots(figsize=(10, 6))

        # Group by model type
        if "model_type" in history_best.columns:
            for model_type in history_best["model_type"].unique():
                model_data = history_best[history_best["model_type"] == model_type]
                model_data = model_data.dropna(subset=[metric, step_col])

                if model_data.empty:
                    continue

                # Average across runs of same model type
                grouped = model_data.groupby(step_col)[metric].agg(["mean", "std"])
                _ = ax.plot(grouped.index, grouped["mean"], label=model_type, alpha=0.9)
                _ = ax.fill_between(
                    grouped.index,
                    grouped["mean"] - grouped["std"],
                    grouped["mean"] + grouped["std"],
                    alpha=0.2,
                )
        else:
            # Just plot all runs
            for run_id in list(best_run_ids)[:10]:
                run_data = history_best[history_best["run_id"] == run_id]
                run_data = run_data.dropna(subset=[metric, step_col])
                if not run_data.empty:
                    _ = ax.plot(
                        run_data[step_col],
                        run_data[metric],
                        label=run_id[:8],
                        alpha=0.7,
                    )

        _ = ax.set_xlabel("Step")
        _ = ax.set_ylabel(metric)
        _ = ax.set_title(f"{metric} Evolution (Best Runs)")
        _ = ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        # Log scale for loss metrics
        if "loss" in metric or "mse" in metric or "error" in metric:
            with contextlib.suppress(Exception):
                ax.set_yscale("log")

        _ = plt.tight_layout()
        safe_name = metric.replace("/", "_").replace(".", "_")
        _ = plt.savefig(plots_dir / f"{safe_name}_evolution.png", bbox_inches="tight")
        plt.close()
        _ = logger.info(f"Saved: {plots_dir / f'{safe_name}_evolution.png'}")


def create_summary_report(
    summary_df: pd.DataFrame,
    best_runs: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create summary report with tables and statistics."""
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Best runs table
    best_runs_path = report_dir / "best_runs.csv"
    best_runs.to_csv(best_runs_path, index=False)
    _ = logger.info(f"Saved: {best_runs_path}")

    # Summary statistics by model type
    if "model_type" in summary_df.columns:
        metric_cols = [
            c
            for c in summary_df.columns
            if any(x in c for x in ["loss", "mse", "error", "accuracy", "distance"])
            and summary_df[c].dtype in [np.float64, np.float32, np.int64]
        ]

        if metric_cols:
            stats = summary_df.groupby("model_type")[metric_cols].agg(
                ["mean", "std", "min", "max"]
            )
            stats_path = report_dir / "model_type_statistics.csv"
            stats.to_csv(stats_path)
            _ = logger.info(f"Saved: {stats_path}")

    # Best hyperparameters per group
    config_cols = [
        c
        for c in best_runs.columns
        if c.startswith("model_") or c.startswith("data_") or c == "learning_rate"
    ]
    if config_cols:
        best_configs = best_runs[
            ["run_id", "project_id", "model_type"] + config_cols
        ].copy()
        best_configs_path = report_dir / "best_hyperparameters.csv"
        best_configs.to_csv(best_configs_path, index=False)
        _ = logger.info(f"Saved: {best_configs_path}")

    # Generate markdown report
    report_md = ["# Experiment Analysis Report\n"]
    report_md.append(f"Total runs analyzed: {len(summary_df)}\n")
    report_md.append(f"Best runs selected: {len(best_runs)}\n\n")

    if "model_type" in summary_df.columns:
        report_md.append("## Runs by Model Type\n")
        model_counts = summary_df["model_type"].value_counts()
        for model, count in model_counts.items():
            report_md.append(f"- {model}: {count} runs\n")
        report_md.append("\n")

    if "data_dataset_name" in summary_df.columns:
        report_md.append("## Runs by Dataset\n")
        dataset_counts = summary_df["data_dataset_name"].value_counts()
        for dataset, count in dataset_counts.items():
            report_md.append(f"- {dataset}: {count} runs\n")
        report_md.append("\n")

    report_md.append("## Best Runs Summary\n\n")
    report_md.append("| Run ID | Model | Dataset | Val Loss | Test Loss |\n")
    report_md.append("|--------|-------|---------|----------|----------|\n")

    for _, row in best_runs.head(20).iterrows():
        run_id = str(row.get("run_id", ""))[:8]
        model = str(row.get("model_type", ""))
        dataset = str(row.get("data_dataset_name", ""))
        val_loss: Any = row.get("val/loss", np.nan)
        test_loss: Any = row.get("test/loss", np.nan)
        val_str = f"{val_loss:.6f}" if pd.notna(val_loss) else "-"
        test_str = f"{test_loss:.6f}" if pd.notna(test_loss) else "-"
        report_md.append(
            f"| {run_id} | {model} | {dataset} | {val_str} | {test_str} |\n"
        )

    report_path = report_dir / "analysis_report.md"
    report_path.write_text("".join(report_md))
    _ = logger.info(f"Saved: {report_path}")


def plot_hyperparameter_comparison(
    summary_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Create plots comparing performance across hyperparameters."""
    plots_dir = output_dir / "hyperparameter_analysis"
    plots_dir.mkdir(parents=True, exist_ok=True)

    _ = sns.set_style("whitegrid")

    # Learning rate vs validation loss
    if "learning_rate" in summary_df.columns and "val/loss" in summary_df.columns:
        df = summary_df.dropna(subset=["learning_rate", "val/loss"])
        if not df.empty and "model_type" in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))

            for model_type in df["model_type"].unique():
                model_data = df[df["model_type"] == model_type]
                _ = ax.scatter(
                    model_data["learning_rate"],
                    model_data["val/loss"],
                    label=model_type,
                    alpha=0.7,
                )

            _ = ax.set_xlabel("Learning Rate")
            _ = ax.set_ylabel("Validation Loss")
            ax.set_xscale("log")
            ax.set_yscale("log")
            _ = ax.set_title("Learning Rate vs Validation Loss")
            _ = ax.legend()
            _ = plt.tight_layout()
            _ = plt.savefig(plots_dir / "lr_vs_val_loss.png", bbox_inches="tight")
            plt.close()
            _ = logger.info(f"Saved: {plots_dir / 'lr_vs_val_loss.png'}")

    # Box plot of final metrics by model type
    metric_cols = ["val/loss", "val/mse", "test/loss", "test/mse"]
    available = [c for c in metric_cols if c in summary_df.columns]

    if available and "model_type" in summary_df.columns:
        for metric in available:
            df = summary_df.dropna(subset=[metric])
            if df.empty:
                continue

            fig, ax = plt.subplots(figsize=(10, 6))
            _ = sns.boxplot(data=df, x="model_type", y=metric, ax=ax)
            _ = ax.set_title(f"{metric} by Model Type")
            _ = ax.set_ylabel(metric)
            _ = plt.xticks(rotation=45)
            _ = plt.tight_layout()

            safe_name = metric.replace("/", "_")
            _ = plt.savefig(plots_dir / f"{safe_name}_boxplot.png", bbox_inches="tight")
            plt.close()
            _ = logger.info(f"Saved: {plots_dir / f'{safe_name}_boxplot.png'}")


@click.command()
@click.option(
    "--input-dir",
    type=click.Path(exists=True, path_type=Path),
    default=Path("analysis_output"),
    help="Directory containing unified parquet files",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("reports"),
    help="Directory to save reports and plots",
)
@click.option(
    "--metric",
    default="val/loss",
    help="Metric to use for selecting best runs",
)
@click.option(
    "--group-by",
    multiple=True,
    default=["model_type", "data_graph_type", "data_noise_levels"],
    help="Columns to group by when selecting best runs",
)
def main(
    input_dir: Path, output_dir: Path, metric: str, group_by: tuple[str, ...]
) -> None:
    """Analyze experiments and generate reports/plots."""
    import sys

    _ = logger.remove()
    _ = logger.add(sys.stderr, level="INFO")

    # Load data
    history_df, summary_df = load_unified_data(input_dir)

    if summary_df.empty:
        _ = click.echo("No data to analyze")
        return

    # Select best runs
    best_runs = identify_best_runs(
        summary_df,
        group_cols=list(group_by),
        metric=metric,
        minimize=True,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Create reports
    create_summary_report(summary_df, best_runs, output_dir)

    # Create plots
    plot_training_dynamics(history_df, best_runs, output_dir)
    plot_metric_evolution(history_df, best_runs, output_dir)
    plot_hyperparameter_comparison(summary_df, output_dir)

    _ = click.echo(f"\nAnalysis complete. Results saved to: {output_dir.absolute()}")
    _ = click.echo(f"  - Reports: {output_dir / 'reports'}")
    _ = click.echo(f"  - Training dynamics: {output_dir / 'training_dynamics'}")
    _ = click.echo(f"  - Metric evolution: {output_dir / 'metric_evolution'}")
    _ = click.echo(
        f"  - Hyperparameter analysis: {output_dir / 'hyperparameter_analysis'}"
    )


if __name__ == "__main__":
    main()
