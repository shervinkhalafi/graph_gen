# /// script
# dependencies = [
#     "pandas>=2.0",
#     "pyarrow",
#     "scipy",
#     "rich",
#     "tabulate",
# ]
# ///
"""
Semantic grouping analysis for eigenstructure experiments.

Analyzes performance across semantic groupings:
- Model type (DiGress vs Spectral)
- Dataset
- Asymmetric attention
- Noise levels
- Input embeddings
- DiGress component ablations
- Spectral architecture variants
- Hyperparameters (k, lr, wd)

Averages only across seeds; picks best hyperparameters but reports distribution.
Includes statistical significance tests.

Usage:
    uv run scripts/semantic_analysis.py
    uv run scripts/semantic_analysis.py --output report.md
    uv run scripts/semantic_analysis.py --metric val_loss
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from scipy import stats
from tabulate import tabulate

console = Console()

PARQUET_FILE = Path("eigenstructure_results_full/all_runs.parquet")


@dataclass
class GroupStats:
    """Statistics for a semantic grouping."""

    name: str
    n: int
    best: float
    mean: float
    std: float
    min_val: float
    max_val: float


@dataclass
class ComparisonResult:
    """Result of statistical comparison between groups."""

    group1: str
    group2: str
    diff: float
    t_stat: float
    p_value: float
    cohens_d: float
    significant: bool
    effect_size: str  # negligible, small, medium, large


def load_data() -> pd.DataFrame:
    """Load and prepare experiment data."""
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(
            f"{PARQUET_FILE} not found. Run scripts/analyze_experiments.py first."
        )

    df = pd.read_parquet(PARQUET_FILE)
    df = df[df["state"] == "finished"].copy()
    return df


def parse_semantic_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Parse all semantic groupings from raw data."""

    def parse_dataset(row):
        # Priority: config columns over name parsing
        for col in [
            "config_data_dataset_name",
            "config_dataset_name",
            "config_data_graph_type",
            "config_graph_type",
        ]:
            val = row.get(col)
            if pd.notna(val) and str(val) != "nan":
                return str(val)
        # Fallback to name parsing for legacy runs
        name = str(row.get("name", "")).lower()
        if "sbm_small" in name:
            return "sbm_small"
        if "sbm" in name:
            return "sbm"
        return "planar_default"

    def parse_digress_ablation(name):
        if pd.isna(name):
            return None
        name = str(name)
        if "gnn_all" in name:
            return "gnn_all"
        if "gnn_qk" in name:
            return "gnn_qk"
        if "gnn_v" in name:
            return "gnn_v"
        if "digress_transformer" in name or "digress_default" in name:
            return "default"
        return None

    def parse_spectral_arch(name):
        if pd.isna(name):
            return None
        name = str(name)
        if "self_attention" in name:
            return "self_attention"
        if "linear_pe" in name or "spectral_linear" in name:
            return "linear_pe"
        if "filter_bank" in name:
            return "filter_bank"
        return None

    def parse_embedding(name):
        if pd.isna(name):
            return "default"
        name = str(name).lower()
        if "_pe_" in name or "spectral_pe" in name:
            return "spectral_pe"
        return "default"

    def parse_noise(x):
        if pd.isna(x) or x == "nan":
            return "mixed"
        s = str(x)
        if "0.01, 0.05" in s or "0.01," in s and "0.3" in s:
            return "multi_[0.01-0.3]"
        if s == "[0.01]":
            return "single_0.01"
        if s == "[0.05]":
            return "single_0.05"
        if s == "[0.1]":
            return "single_0.1"
        if s == "[0.2]":
            return "single_0.2"
        if s == "[0.3]":
            return "single_0.3"
        return "other"

    # Apply parsers
    df["dataset"] = df.apply(parse_dataset, axis=1)
    df["digress_ablation"] = df["name"].apply(parse_digress_ablation)
    df["spectral_arch"] = df["name"].apply(parse_spectral_arch)
    df["embedding"] = df["name"].apply(parse_embedding)
    df["noise_level"] = df["config_noise_levels"].apply(parse_noise)

    df["model_type"] = df["config_model__target_"].apply(
        lambda x: "spectral"
        if "Spectral" in str(x)
        else "digress"
        if "Digress" in str(x)
        else "unknown"
    )
    df["asymmetric"] = df["config_asymmetric"].apply(
        lambda x: "yes" if x == "True" or x is True else "no"
    )

    # Hyperparameters
    df["k"] = pd.to_numeric(df["config_model_k"], errors="coerce")
    df["lr"] = pd.to_numeric(df["config_learning_rate"], errors="coerce")
    df["wd"] = pd.to_numeric(df["config_weight_decay"], errors="coerce")

    # Metrics
    df["test_acc"] = pd.to_numeric(df["metric_test_accuracy"], errors="coerce")
    df["val_acc"] = pd.to_numeric(df["metric_val_accuracy"], errors="coerce")
    df["val_loss"] = pd.to_numeric(df["metric_val_loss"], errors="coerce")
    df["test_mse"] = pd.to_numeric(df["metric_test_mse"], errors="coerce")
    df["test_subspace"] = pd.to_numeric(
        df["metric_test_subspace_distance"], errors="coerce"
    )

    return df


def compute_cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (g1.mean() - g2.mean()) / pooled_std


def interpret_effect_size(d: float) -> str:
    """Interpret Cohen's d effect size."""
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def analyze_grouping(
    df: pd.DataFrame,
    group_col: str,
    metric: str = "test_acc",
    higher_is_better: bool = True,
    filter_func=None,
) -> tuple[list[GroupStats], list[ComparisonResult]]:
    """Analyze a semantic grouping and compute statistics + comparisons."""

    if filter_func:
        df = df[filter_func(df)]

    df = df[df[metric].notna()].copy()

    if len(df) == 0:
        return [], []

    # Compute stats per group
    group_stats = []
    groups = df[group_col].dropna().unique()

    for group in sorted(groups, key=str):
        data = df[df[group_col] == group][metric]
        if len(data) == 0:
            continue

        group_stats.append(
            GroupStats(
                name=str(group),
                n=len(data),
                best=data.max() if higher_is_better else data.min(),
                mean=data.mean(),
                std=data.std() if len(data) > 1 else 0.0,
                min_val=data.min(),
                max_val=data.max(),
            )
        )

    # Sort by best performance
    group_stats.sort(key=lambda x: x.best, reverse=higher_is_better)

    # Pairwise comparisons (each group vs next best)
    comparisons = []
    for i, gs in enumerate(group_stats):
        if i + 1 >= len(group_stats):
            break

        gs2 = group_stats[i + 1]
        g1_data = df[df[group_col] == gs.name][metric].values
        g2_data = df[df[group_col] == gs2.name][metric].values

        if len(g1_data) < 2 or len(g2_data) < 2:
            continue

        t_stat, p_val = stats.ttest_ind(g1_data, g2_data)
        d = compute_cohens_d(g1_data, g2_data)

        comparisons.append(
            ComparisonResult(
                group1=gs.name,
                group2=gs2.name,
                diff=gs.mean - gs2.mean,
                t_stat=t_stat,
                p_value=p_val,
                cohens_d=d,
                significant=p_val < 0.05,
                effect_size=interpret_effect_size(d),
            )
        )

    return group_stats, comparisons


def format_stats_table(
    title: str,
    group_stats: list[GroupStats],
    comparisons: list[ComparisonResult],
    metric: str,
    higher_is_better: bool = True,
) -> str:
    """Format statistics as a markdown table."""
    lines = [f"### {title}\n"]

    # Stats table
    headers = ["Group", "N", f"{metric} BEST", f"{metric} mean±std", "Range"]
    rows = []
    for gs in group_stats:
        std_str = f"±{gs.std:.4f}" if gs.std > 0 else ""
        rows.append(
            [
                gs.name,
                gs.n,
                f"{gs.best:.4f}",
                f"{gs.mean:.4f}{std_str}",
                f"[{gs.min_val:.4f}, {gs.max_val:.4f}]",
            ]
        )

    lines.append(tabulate(rows, headers=headers, tablefmt="github"))
    lines.append("")

    # Significance tests
    if comparisons:
        lines.append("\n**Statistical comparisons (vs next best):**\n")
        for comp in comparisons:
            sig_marker = "✓" if comp.significant else "✗"
            direction = ">" if (comp.diff > 0) == higher_is_better else "<"
            lines.append(
                f"- {comp.group1} {direction} {comp.group2}: "
                f"Δ={abs(comp.diff):.4f}, p={comp.p_value:.4f} {sig_marker}, "
                f"d={comp.cohens_d:.2f} ({comp.effect_size})"
            )

    lines.append("")
    return "\n".join(lines)


def create_summary_table(all_results: dict, metric: str) -> str:
    """Create overview summary table."""
    lines = ["## Summary Overview\n"]

    headers = [
        "Grouping",
        "Best Group",
        f"Best {metric}",
        "vs 2nd Best",
        "Significant?",
        "Effect Size",
    ]
    rows = []

    for grouping, (group_stats, comps) in all_results.items():
        if not group_stats:
            continue

        best = group_stats[0]
        if len(group_stats) > 1 and comps:
            comp = comps[0]
            sig = "YES" if comp.significant else "NO"
            vs_str = f"Δ={abs(comp.diff):.4f} (p={comp.p_value:.4f})"
            effect = comp.effect_size
        else:
            sig = "-"
            vs_str = "-"
            effect = "-"

        rows.append(
            [
                grouping,
                best.name,
                f"{best.best:.4f}",
                vs_str,
                sig,
                effect,
            ]
        )

    lines.append(tabulate(rows, headers=headers, tablefmt="github"))
    lines.append("")
    return "\n".join(lines)


def run_analysis(
    metric: str = "test_acc",
    output_file: str | None = None,
) -> str:
    """Run full semantic grouping analysis."""

    higher_is_better = metric in ["test_acc", "val_acc"]

    console.print(f"[bold]Loading data from {PARQUET_FILE}...[/bold]")
    df = load_data()
    df = parse_semantic_groups(df)

    df_valid = df[df[metric].notna()].copy()
    console.print(f"[bold]Valid runs with {metric}: {len(df_valid)}[/bold]\n")

    # Define all semantic groupings
    groupings = {
        "Model Type": ("model_type", None),
        "Dataset": ("dataset", None),
        "Asymmetric Attention": ("asymmetric", None),
        "Noise Level": ("noise_level", None),
        "Input Embeddings": ("embedding", None),
        "DiGress Ablations": (
            "digress_ablation",
            lambda d: d["model_type"] == "digress",
        ),
        "Spectral Architecture": (
            "spectral_arch",
            lambda d: d["model_type"] == "spectral",
        ),
        "K Value": ("k", None),
        "Learning Rate": ("lr", None),
        "Weight Decay": ("wd", None),
    }

    all_results = {}
    report_sections = []

    report_sections.append("# Semantic Grouping Analysis\n")
    report_sections.append(
        f"**Metric**: {metric} ({'higher' if higher_is_better else 'lower'} is better)\n"
    )
    report_sections.append(f"**Total valid runs**: {len(df_valid)}\n")

    for title, (col, filter_func) in groupings.items():
        console.print(f"[cyan]Analyzing {title}...[/cyan]")

        stats_list, comps = analyze_grouping(
            df_valid, col, metric, higher_is_better, filter_func
        )

        if stats_list:
            all_results[title] = (stats_list, comps)
            section = format_stats_table(
                title, stats_list, comps, metric, higher_is_better
            )
            report_sections.append(section)

    # Create summary table
    summary = create_summary_table(all_results, metric)
    report_sections.insert(3, summary)  # Insert after header

    # Add conclusions
    report_sections.append("## Conclusions\n")

    conclusions = []
    for title, (_group_stats, comps) in all_results.items():
        if not comps:
            continue

        comp = comps[0]
        if comp.significant and comp.effect_size in ["medium", "large"]:
            conclusions.append(
                f"- **{title}**: {comp.group1} significantly better than {comp.group2} "
                f"(d={comp.cohens_d:.2f}, {comp.effect_size} effect)"
            )
        elif not comp.significant or comp.effect_size == "negligible":
            conclusions.append(
                f"- **{title}**: No meaningful difference between groups "
                f"(p={comp.p_value:.3f}, d={comp.cohens_d:.2f})"
            )

    report_sections.append("\n".join(conclusions))

    report = "\n".join(report_sections)

    # Output
    if output_file:
        Path(output_file).write_text(report)
        console.print(f"\n[green]Report saved to {output_file}[/green]")
    else:
        console.print(report)

    return report


def main():
    parser = argparse.ArgumentParser(description="Semantic grouping analysis")
    parser.add_argument(
        "--metric",
        "-m",
        default="test_acc",
        choices=["test_acc", "val_acc", "val_loss", "test_mse", "test_subspace"],
        help="Metric to analyze (default: test_acc)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for markdown report",
    )
    args = parser.parse_args()

    run_analysis(metric=args.metric, output_file=args.output)


if __name__ == "__main__":
    main()
