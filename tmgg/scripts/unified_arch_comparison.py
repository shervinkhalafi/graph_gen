# /// script
# dependencies = [
#     "pandas>=2.0",
#     "pyarrow",
#     "numpy",
#     "rich",
#     "tabulate",
# ]
# ///
"""
Unified architecture comparison across all model types and datasets.

Combines Spectral (self_attention, linear_pe, filter_bank) and DiGress
(default, gnn_all, gnn_qk, gnn_v) architectures into a single comparison
per dataset.

Usage:
    uv run scripts/unified_arch_comparison.py
    uv run scripts/unified_arch_comparison.py --output report.md
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rich.console import Console
from tabulate import tabulate

console = Console()

PARQUET_FILE = Path("eigenstructure_results_full/all_runs.parquet")


def load_data() -> pd.DataFrame:
    """Load experiment data from parquet."""
    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"{PARQUET_FILE} not found.")
    df = pd.read_parquet(PARQUET_FILE)
    return df[df["state"] == "finished"].copy()


def parse_architecture(row: pd.Series) -> str | None:
    """Extract unified architecture label from run data."""
    model_target = str(row.get("config_model__target_", ""))
    name = str(row.get("name", "")).lower()

    # DiGress variants - check config first, then name
    if "Digress" in model_target:
        digress_arch = str(
            row.get("config_digress_arch", row.get("config_model_digress_arch", ""))
        )
        if digress_arch and digress_arch != "nan":
            if "gnn_all" in digress_arch:
                return "digress_gnn_all"
            if "gnn_qk" in digress_arch:
                return "digress_gnn_qk"
            if "gnn_v" in digress_arch:
                return "digress_gnn_v"
            if "default" in digress_arch:
                return "digress_default"
        # Fall back to name parsing
        if "gnn_all" in name:
            return "digress_gnn_all"
        if "gnn_qk" in name:
            return "digress_gnn_qk"
        if "gnn_v" in name:
            return "digress_gnn_v"
        return "digress_default"

    # Spectral variants - use config_model_model_type
    if "Spectral" in model_target:
        model_type = str(
            row.get("config_model_model_type", row.get("config_model_type", ""))
        )
        if model_type and model_type != "nan":
            if model_type == "self_attention":
                return "spectral_self_attn"
            if model_type == "linear_pe":
                return "spectral_linear_pe"
            if model_type == "filter_bank":
                return "spectral_filter_bank"
        # Fall back to name parsing
        if "self_attention" in name:
            return "spectral_self_attn"
        if "linear_pe" in name:
            return "spectral_linear_pe"
        if "filter_bank" in name:
            return "spectral_filter_bank"
        return "spectral_unknown"

    return None


def parse_dataset(row: pd.Series) -> str:
    """Extract dataset name from run data."""
    for col in [
        "config_data_dataset_name",
        "config_dataset_name",
        "config_data_graph_type",
        "config_graph_type",
    ]:
        val = row.get(col)
        if pd.notna(val) and str(val) != "nan":
            return str(val)

    name = str(row.get("name", "")).lower()
    if "sbm_small" in name:
        return "sbm_small"
    if "sbm" in name:
        return "sbm"
    return "unknown"


def generate_unified_comparison(df: pd.DataFrame, metric: str = "test_accuracy") -> str:
    """Generate unified architecture comparison tables."""
    df = df.copy()
    df["architecture"] = df.apply(parse_architecture, axis=1)
    df["dataset"] = df.apply(parse_dataset, axis=1)

    # Handle metric column naming
    metric_col = f"metric_{metric}"
    if metric_col not in df.columns:
        metric_col = metric
    df["_metric"] = pd.to_numeric(df[metric_col], errors="coerce")

    df = df[df["architecture"].notna() & df["_metric"].notna()].copy()

    lines = ["# Unified Architecture Comparison\n"]
    lines.append("Compares all architectures (Spectral + DiGress) per dataset.\n")
    lines.append(f"**Metric**: {metric_col} (higher is better)\n")

    datasets = sorted(df["dataset"].unique())
    all_archs = [
        "spectral_self_attn",
        "spectral_linear_pe",
        "spectral_filter_bank",
        "digress_default",
        "digress_gnn_all",
        "digress_gnn_qk",
        "digress_gnn_v",
    ]

    # Short names for table headers
    arch_short = {
        "spectral_self_attn": "S:self_attn",
        "spectral_linear_pe": "S:linear_pe",
        "spectral_filter_bank": "S:filter_bank",
        "digress_default": "D:default",
        "digress_gnn_all": "D:gnn_all",
        "digress_gnn_qk": "D:gnn_qk",
        "digress_gnn_v": "D:gnn_v",
    }

    # Collect stats for all datasets and architectures
    all_stats: dict[str, dict[str, dict]] = {}
    for dataset in datasets:
        ds_df = df[df["dataset"] == dataset]
        if len(ds_df) == 0:
            continue
        all_stats[dataset] = {}
        for arch in all_archs:
            arch_df = ds_df[ds_df["architecture"] == arch]
            if len(arch_df) == 0:
                continue
            vals = arch_df["_metric"].values
            q1, median, q3 = (
                float(np.percentile(vals, 25)),
                float(np.median(vals)),
                float(np.percentile(vals, 75)),
            )
            all_stats[dataset][arch] = {
                "n": len(vals),
                "max": float(vals.max()),
                "median": median,
                "iqr": (q1, q3),
                "mean": float(vals.mean()),
                "std": float(vals.std()) if len(vals) > 1 else 0.0,
            }

    # Summary data for later
    summary_data = []

    # === TABLE 1: Winner by Maximum ===
    lines.append("## Winners by Maximum (Best Run)\n")
    lines.append("Format: max (n). **Bold** = winner.\n")

    headers = ["Dataset"] + [arch_short[a] for a in all_archs] + ["Winner"]
    rows = []
    for dataset in datasets:
        if dataset not in all_stats:
            continue
        row = [dataset]
        best_arch, best_val = None, -1
        for arch in all_archs:
            if arch not in all_stats[dataset]:
                row.append("-")
                continue
            s = all_stats[dataset][arch]
            if s["max"] > best_val:
                best_val = s["max"]
                best_arch = arch
            row.append(f"{s['max']:.3f} ({s['n']})")
        if best_arch:
            idx = all_archs.index(best_arch) + 1
            row[idx] = f"**{row[idx]}**"
            row.append(arch_short[best_arch])
        else:
            row.append("-")
        rows.append(row)
        summary_data.append({"dataset": dataset, "max_winner": best_arch})
    lines.append(tabulate(rows, headers=headers, tablefmt="github"))
    lines.append("")

    # === TABLE 2: Winner by Median (with IQR) ===
    lines.append("## Winners by Median (with IQR)\n")
    lines.append("Format: median [Q1-Q3] (n). **Bold** = winner.\n")

    rows = []
    for dataset in datasets:
        if dataset not in all_stats:
            continue
        row = [dataset]
        best_arch, best_val = None, -1
        for arch in all_archs:
            if arch not in all_stats[dataset]:
                row.append("-")
                continue
            s = all_stats[dataset][arch]
            if s["median"] > best_val:
                best_val = s["median"]
                best_arch = arch
            row.append(
                f"{s['median']:.3f} [{s['iqr'][0]:.3f}-{s['iqr'][1]:.3f}] ({s['n']})"
            )
        if best_arch:
            idx = all_archs.index(best_arch) + 1
            row[idx] = f"**{row[idx]}**"
            row.append(arch_short[best_arch])
        else:
            row.append("-")
        rows.append(row)
        # Update summary
        for sd in summary_data:
            if sd["dataset"] == dataset:
                sd["median_winner"] = best_arch
                break
    lines.append(tabulate(rows, headers=headers, tablefmt="github"))
    lines.append("")

    # === TABLE 3: Winner by Mean (with Std) ===
    lines.append("## Winners by Mean (with Std)\n")
    lines.append("Format: mean±std (n). **Bold** = winner.\n")

    rows = []
    for dataset in datasets:
        if dataset not in all_stats:
            continue
        row = [dataset]
        best_arch, best_val = None, -1
        for arch in all_archs:
            if arch not in all_stats[dataset]:
                row.append("-")
                continue
            s = all_stats[dataset][arch]
            if s["mean"] > best_val:
                best_val = s["mean"]
                best_arch = arch
            std_str = f"±{s['std']:.3f}" if s["std"] > 0 else ""
            row.append(f"{s['mean']:.3f}{std_str} ({s['n']})")
        if best_arch:
            idx = all_archs.index(best_arch) + 1
            row[idx] = f"**{row[idx]}**"
            row.append(arch_short[best_arch])
        else:
            row.append("-")
        rows.append(row)
        # Update summary
        for sd in summary_data:
            if sd["dataset"] == dataset:
                sd["mean_winner"] = best_arch
                break
    lines.append(tabulate(rows, headers=headers, tablefmt="github"))
    lines.append("")

    # Summary statistics
    lines.append("\n## Summary\n")

    # Count wins by architecture for each aggregation method
    for agg_name, key in [
        ("Maximum", "max_winner"),
        ("Median", "median_winner"),
        ("Mean", "mean_winner"),
    ]:
        win_counts: dict[str, int] = {}
        for entry in summary_data:
            winner = entry.get(key)
            if winner:
                win_counts[winner] = win_counts.get(winner, 0) + 1

        lines.append(f"### Wins by Architecture ({agg_name})\n")
        win_rows = sorted(win_counts.items(), key=lambda x: -x[1])
        for arch, count in win_rows:
            lines.append(f"- **{arch_short[arch]}**: {count} datasets")
        lines.append("")

    # Wins by model type (using max as canonical)
    spectral_wins = sum(
        1 for e in summary_data if e.get("max_winner") and "spectral" in e["max_winner"]
    )
    digress_wins = sum(
        1 for e in summary_data if e.get("max_winner") and "digress" in e["max_winner"]
    )

    lines.append("### Wins by Model Type\n")
    lines.append(f"- **Spectral**: {spectral_wins} datasets")
    lines.append(f"- **DiGress**: {digress_wins} datasets")
    lines.append("")

    # Best architecture per model type
    lines.append("### Best Architecture per Model Type\n")
    spectral_archs = [
        "spectral_self_attn",
        "spectral_linear_pe",
        "spectral_filter_bank",
    ]
    digress_archs = [
        "digress_default",
        "digress_gnn_all",
        "digress_gnn_qk",
        "digress_gnn_v",
    ]

    for model_type, archs in [("Spectral", spectral_archs), ("DiGress", digress_archs)]:
        arch_means = {}
        for arch in archs:
            arch_df = df[df["architecture"] == arch]
            if len(arch_df) > 0:
                arch_means[arch] = arch_df["_metric"].mean()

        if arch_means:
            best = max(arch_means.items(), key=lambda x: x[1])
            lines.append(
                f"- **{model_type}**: {arch_short[best[0]]} (mean {best[1]:.4f})"
            )
    lines.append("")

    # Overall best
    overall_means = df.groupby("architecture")["_metric"].mean()
    if len(overall_means) > 0:
        best_overall = overall_means.idxmax()
        lines.append("### Overall Best Architecture\n")
        lines.append(
            f"**{arch_short.get(best_overall, best_overall)}** "
            f"(mean {overall_means[best_overall]:.4f})"
        )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Unified architecture comparison")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output file for markdown report",
    )
    parser.add_argument(
        "--metric",
        "-m",
        default="test_accuracy",
        help="Metric to compare (default: test_accuracy)",
    )
    args = parser.parse_args()

    console.print("[bold]Loading data...[/bold]")
    df = load_data()
    console.print(f"[bold]Loaded {len(df)} finished runs[/bold]\n")

    report = generate_unified_comparison(df, metric=args.metric)

    if args.output:
        Path(args.output).write_text(report)
        console.print(f"[green]Report saved to {args.output}[/green]")
    else:
        console.print(report)


if __name__ == "__main__":
    main()
