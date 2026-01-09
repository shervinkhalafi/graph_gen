# /// script
# dependencies = [
#     "pandas>=2.0",
#     "pyarrow",
#     "tabulate",
# ]
# ///
"""
Breakdown of experiment variations in the eigenstructure study.

Creates comprehensive tables showing experimental dimensions, configurations,
and architecture comparisons across datasets.

Usage:
    uv run scripts/experiment_breakdown.py                    # Full analysis (default)
    uv run scripts/experiment_breakdown.py -m summary         # Summary breakdown only
    uv run scripts/experiment_breakdown.py -m nested          # Full HP configurations
    uv run scripts/experiment_breakdown.py -m compact         # Aggregated over HP
    uv run scripts/experiment_breakdown.py -m comparison      # Architecture comparison
    uv run scripts/experiment_breakdown.py -o output_dir/     # Custom output directory
"""

import argparse
from pathlib import Path

import pandas as pd
from tabulate import tabulate

PARQUET_FILE = Path("eigenstructure_results_full/all_runs.parquet")


def load_and_parse(df: pd.DataFrame) -> pd.DataFrame:
    """Parse all experimental dimensions from raw data."""
    df = df[df["state"] == "finished"].copy()

    # Model type
    def get_model_type(target):
        if pd.isna(target):
            return "unknown"
        if "Spectral" in str(target):
            return "Spectral"
        if "Digress" in str(target):
            return "DiGress"
        return "unknown"

    df["model_type"] = df["config_model__target_"].apply(get_model_type)

    # Dataset - check config columns first, then fall back to name parsing
    def get_dataset(row):
        # Priority: config_data_dataset_name > config_dataset_name > config_data_graph_type > config_graph_type
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

    df["dataset"] = df.apply(get_dataset, axis=1)

    # Architecture - use config_model_model_type for spectral, parse name for digress
    def get_architecture(row):
        model_type = row["model_type"]
        name = str(row.get("name", ""))
        config_arch = row.get("config_model_model_type", None)

        if model_type == "Spectral":
            if pd.notna(config_arch):
                return str(config_arch)
            # Fallback to name parsing
            if "self_attention" in name:
                return "self_attention"
            if "linear_pe" in name or "spectral_linear" in name:
                return "linear_pe"
            if "filter_bank" in name:
                return "filter_bank"
            return "default"
        elif model_type == "DiGress":
            if "gnn_all" in name:
                return "gnn_all"
            if "gnn_qk" in name:
                return "gnn_qk"
            if "gnn_v" in name:
                return "gnn_v"
            return "default"
        return "unknown"

    df["architecture"] = df.apply(get_architecture, axis=1)

    # Asymmetric attention
    df["asymmetric"] = df["config_asymmetric"].apply(
        lambda x: "yes" if x == "True" or x is True else "no"
    )

    # Input embeddings
    def get_embedding(name):
        if pd.isna(name):
            return "default"
        name = str(name).lower()
        if "_pe_" in name or "spectral_pe" in name:
            return "spectral_pe"
        return "default"

    df["input_embedding"] = df["name"].apply(get_embedding)

    # Noise levels
    def get_noise(x):
        if pd.isna(x) or str(x) == "nan":
            return "unknown"
        s = str(x)
        if "0.01, 0.05" in s or ("0.01," in s and "0.3" in s):
            return "multi"
        if s == "[0.01]":
            return "0.01"
        if s == "[0.05]":
            return "0.05"
        if s == "[0.1]":
            return "0.1"
        if s == "[0.2]":
            return "0.2"
        if s == "[0.3]":
            return "0.3"
        return "other"

    df["noise_level"] = df["config_noise_levels"].apply(get_noise)

    # Hyperparameters
    df["k"] = pd.to_numeric(df["config_model_k"], errors="coerce")
    df["lr"] = pd.to_numeric(df["config_learning_rate"], errors="coerce")
    df["wd"] = pd.to_numeric(df["config_weight_decay"], errors="coerce")

    # Metrics
    df["test_acc"] = pd.to_numeric(df["metric_test_accuracy"], errors="coerce")

    return df


def create_breakdown_table(df: pd.DataFrame) -> str:
    """Create comprehensive breakdown table."""
    lines = ["# Experiment Breakdown\n"]
    lines.append(f"**Total finished runs**: {len(df)}\n")

    # 1. Model × Architecture breakdown
    lines.append("## Model Type × Architecture\n")
    model_arch = df.groupby(["model_type", "architecture"]).size().reset_index(name="N")
    model_arch = model_arch.sort_values(["model_type", "N"], ascending=[True, False])
    lines.append(
        tabulate(model_arch, headers="keys", tablefmt="github", showindex=False)
    )
    lines.append("")

    # 2. Dataset breakdown
    lines.append("\n## Dataset\n")
    dataset = df.groupby("dataset").size().reset_index(name="N")
    dataset = dataset.sort_values("N", ascending=False)
    lines.append(tabulate(dataset, headers="keys", tablefmt="github", showindex=False))
    lines.append("")

    # 3. Model × Dataset coverage
    lines.append("\n## Model Type × Dataset Coverage\n")
    model_ds = df.groupby(["model_type", "dataset"]).size().unstack(fill_value=0)
    lines.append(model_ds.to_markdown())
    lines.append("")

    # 4. Architecture × Dataset coverage
    lines.append("\n## Architecture × Dataset Coverage\n")
    arch_ds = (
        df.groupby(["model_type", "architecture", "dataset"])
        .size()
        .unstack(fill_value=0)
    )
    lines.append(arch_ds.to_markdown())
    lines.append("")

    # 5. Other experimental dimensions
    lines.append("\n## Other Experimental Dimensions\n")

    dims = [
        ("Asymmetric Attention", "asymmetric"),
        ("Input Embedding", "input_embedding"),
        ("Noise Level", "noise_level"),
    ]

    for title, col in dims:
        lines.append(f"\n### {title}\n")
        counts = df.groupby(col).size().reset_index(name="N")
        counts = counts.sort_values("N", ascending=False)
        lines.append(
            tabulate(counts, headers="keys", tablefmt="github", showindex=False)
        )
        lines.append("")

    # 6. Hyperparameters
    lines.append("\n## Hyperparameters\n")

    hp_dims = [
        ("K (eigenvectors)", "k"),
        ("Learning Rate", "lr"),
        ("Weight Decay", "wd"),
    ]

    for title, col in hp_dims:
        lines.append(f"\n### {title}\n")
        counts = df.groupby(col).size().reset_index(name="N")
        counts = counts.sort_values(col)
        lines.append(
            tabulate(counts, headers="keys", tablefmt="github", showindex=False)
        )
        lines.append("")

    # 7. Full cross-tabulation: Model × Arch × Dataset with mean accuracy
    lines.append("\n## Performance Summary: Model × Architecture × Dataset\n")
    lines.append("(mean test_acc ± std, N runs)\n")

    summary_rows = []
    for model in ["DiGress", "Spectral"]:
        model_df = df[df["model_type"] == model]
        for arch in sorted(model_df["architecture"].unique()):
            arch_df = model_df[model_df["architecture"] == arch]
            row = {"Model": model, "Architecture": arch}
            for ds in ["planar", "sbm", "sbm_small"]:
                ds_df = arch_df[arch_df["dataset"] == ds]
                acc = ds_df["test_acc"].dropna()
                if len(acc) > 0:
                    row[ds] = f"{acc.mean():.3f}±{acc.std():.3f} (n={len(acc)})"
                else:
                    row[ds] = "-"
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    lines.append(
        tabulate(summary_df, headers="keys", tablefmt="github", showindex=False)
    )
    lines.append("")

    return "\n".join(lines)


def create_nested_table(df: pd.DataFrame) -> str:
    """Create single nested table with all experimental dimensions."""
    lines = ["# Full Experiment Configuration Table\n"]
    lines.append(f"**Total finished runs**: {len(df)}\n")

    # Group by all dimensions
    dims = [
        "model_type",
        "architecture",
        "dataset",
        "asymmetric",
        "input_embedding",
        "noise_level",
        "k",
        "lr",
        "wd",
    ]

    # Get counts and mean accuracy per configuration
    grouped = (
        df.groupby(dims, dropna=False)
        .agg(
            N=("test_acc", "size"),
            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),
            best_acc=("test_acc", "max"),
        )
        .reset_index()
    )

    # Sort hierarchically
    grouped = grouped.sort_values(dims)

    # Format the table with hierarchical display
    lines.append("## All Configurations\n")
    lines.append(
        "| Model | Arch | Dataset | Asym | Embed | Noise | K | LR | WD | N | Acc (mean±std) | Best |"
    )
    lines.append(
        "|-------|------|---------|------|-------|-------|---|----|----|---|----------------|------|"
    )

    prev_values = [""] * len(dims)
    for _, row in grouped.iterrows():
        cells = []
        for i, dim in enumerate(dims):
            val = row[dim]
            # Format special values
            if dim == "lr" or dim == "wd":
                val_str = f"{val:.0e}" if pd.notna(val) else "-"
            elif dim == "k":
                val_str = str(int(val)) if pd.notna(val) else "-"
            else:
                val_str = str(val) if pd.notna(val) else "-"

            # Show value only if different from previous row (hierarchical display)
            if val_str == prev_values[i] and i < 3:  # Only collapse first 3 levels
                cells.append("")
            else:
                cells.append(val_str)
                prev_values[i] = val_str
                # Reset subsequent levels when this level changes
                for j in range(i + 1, len(dims)):
                    prev_values[j] = ""

        # Add metrics
        n = int(row["N"])
        mean_acc = row["mean_acc"]
        std_acc = row["std_acc"]
        best_acc = row["best_acc"]

        if pd.notna(mean_acc):
            acc_str = (
                f"{mean_acc:.3f}±{std_acc:.3f}"
                if pd.notna(std_acc)
                else f"{mean_acc:.3f}"
            )
            best_str = f"{best_acc:.3f}"
        else:
            acc_str = "-"
            best_str = "-"

        line = "| " + " | ".join(cells) + f" | {n} | {acc_str} | {best_str} |"
        lines.append(line)

    # Summary stats
    lines.append(f"\n**Total unique configurations**: {len(grouped)}")
    lines.append(
        f"**Configurations with test_acc**: {grouped['mean_acc'].notna().sum()}"
    )

    return "\n".join(lines)


def create_compact_nested_table(df: pd.DataFrame) -> str:
    """Create compact nested table aggregating over hyperparameters."""
    lines = ["# Compact Experiment Configuration Table\n"]
    lines.append(
        "(Aggregated over hyperparameter settings - shows best HP result per config)\n"
    )
    lines.append(f"**Total finished runs**: {len(df)}\n")

    # Group by semantic dimensions only (not HP)
    dims = [
        "model_type",
        "architecture",
        "dataset",
        "asymmetric",
        "input_embedding",
        "noise_level",
    ]

    # Get counts and metrics per semantic configuration
    grouped = (
        df.groupby(dims, dropna=False)
        .agg(
            N=("test_acc", "size"),
            mean_acc=("test_acc", "mean"),
            std_acc=("test_acc", "std"),
            best_acc=("test_acc", "max"),
            n_hp_configs=("test_acc", lambda x: len(x.dropna())),
        )
        .reset_index()
    )

    grouped = grouped.sort_values(dims)

    lines.append("## Configurations (aggregated over HP)\n")
    lines.append(
        "| Model | Arch | Dataset | Asym | Embed | Noise | N | Acc (mean±std) | Best |"
    )
    lines.append(
        "|-------|------|---------|------|-------|-------|---|----------------|------|"
    )

    prev_values = [""] * len(dims)
    for _, row in grouped.iterrows():
        cells = []
        for i, dim in enumerate(dims):
            val = row[dim]
            val_str = str(val) if pd.notna(val) else "-"

            if val_str == prev_values[i] and i < 2:  # Collapse first 2 levels
                cells.append("")
            else:
                cells.append(val_str)
                prev_values[i] = val_str
                for j in range(i + 1, len(dims)):
                    prev_values[j] = ""

        n = int(row["N"])
        mean_acc = row["mean_acc"]
        std_acc = row["std_acc"]
        best_acc = row["best_acc"]

        if pd.notna(mean_acc):
            acc_str = (
                f"{mean_acc:.3f}±{std_acc:.3f}"
                if pd.notna(std_acc)
                else f"{mean_acc:.3f}"
            )
            best_str = f"{best_acc:.3f}"
        else:
            acc_str = "-"
            best_str = "-"

        line = "| " + " | ".join(cells) + f" | {n} | {acc_str} | {best_str} |"
        lines.append(line)

    lines.append(f"\n**Total unique semantic configurations**: {len(grouped)}")

    return "\n".join(lines)


def create_architecture_comparison(df: pd.DataFrame) -> str:
    """Create architecture comparison table highlighting best per dataset."""
    df_valid = df[df["test_acc"].notna()].copy()

    lines = ["# Architecture Comparison by Dataset\n"]
    lines.append(
        "Highlights best architecture per dataset. **Bold** = winner, (n) = run count."
    )
    lines.append("Format: best/mean (n). B=best peak, M=best mean.\n")

    # Spectral architectures
    lines.append("## Spectral Architectures\n")

    spectral = df_valid[df_valid["model_type"] == "Spectral"]
    datasets = sorted(spectral["dataset"].unique())

    rows = []
    for ds in datasets:
        ds_data = spectral[spectral["dataset"] == ds]
        row = {"Dataset": ds}

        best_val, best_arch = 0, ""
        mean_val, mean_arch = 0, ""

        for arch in ["self_attention", "linear_pe", "filter_bank"]:
            arch_data = ds_data[ds_data["architecture"] == arch]["test_acc"]
            if len(arch_data) > 0:
                b, m, n = arch_data.max(), arch_data.mean(), len(arch_data)
                row[f"{arch}_best"] = b
                row[f"{arch}_mean"] = m
                row[f"{arch}_n"] = n
                if b > best_val:
                    best_val, best_arch = b, arch
                if m > mean_val:
                    mean_val, mean_arch = m, arch
            else:
                row[f"{arch}_best"] = None
                row[f"{arch}_mean"] = None
                row[f"{arch}_n"] = 0

        row["best_arch"] = best_arch
        row["mean_arch"] = mean_arch
        rows.append(row)

    header = "| Dataset | self_attn | linear_pe | filter_bank | Winner |"
    sep = "|---------|-----------|-----------|-------------|--------|"
    lines.append(header)
    lines.append(sep)

    for row in rows:
        cells = [row["Dataset"]]
        for arch in ["self_attention", "linear_pe", "filter_bank"]:
            b = row.get(f"{arch}_best")
            m = row.get(f"{arch}_mean")
            n = row.get(f"{arch}_n", 0)

            if b is not None:
                is_best = arch == row["best_arch"]
                is_mean = arch == row["mean_arch"]
                markers = []
                if is_best:
                    markers.append("B")
                if is_mean:
                    markers.append("M")
                marker = "".join(markers)
                if marker:
                    cells.append(f"**{b:.3f}/{m:.3f}** ({n}) {marker}")
                else:
                    cells.append(f"{b:.3f}/{m:.3f} ({n})")
            else:
                cells.append("-")

        winner = (
            row["best_arch"]
            .replace("self_attention", "self_attn")
            .replace("filter_bank", "FB")
            .replace("linear_pe", "lin_pe")
        )
        mean_w = (
            row["mean_arch"]
            .replace("self_attention", "self_attn")
            .replace("filter_bank", "FB")
            .replace("linear_pe", "lin_pe")
        )
        cells.append(f"{winner}/{mean_w}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "**Filter bank wins**: pyg_enzymes (B+M), ring_of_cliques (B+M), pyg_proteins (M only)"
    )
    lines.append("")

    # DiGress architectures
    lines.append("## DiGress Architectures\n")

    digress = df_valid[df_valid["model_type"] == "DiGress"]
    datasets = sorted(digress["dataset"].unique())

    rows = []
    for ds in datasets:
        ds_data = digress[digress["dataset"] == ds]
        row = {"Dataset": ds}

        best_val, best_arch = 0, ""
        mean_val, mean_arch = 0, ""

        for arch in ["default", "gnn_all", "gnn_qk", "gnn_v"]:
            arch_data = ds_data[ds_data["architecture"] == arch]["test_acc"]
            if len(arch_data) > 0:
                b, m, n = arch_data.max(), arch_data.mean(), len(arch_data)
                row[f"{arch}_best"] = b
                row[f"{arch}_mean"] = m
                row[f"{arch}_n"] = n
                if b > best_val:
                    best_val, best_arch = b, arch
                if m > mean_val:
                    mean_val, mean_arch = m, arch
            else:
                row[f"{arch}_best"] = None
                row[f"{arch}_mean"] = None
                row[f"{arch}_n"] = 0

        row["best_arch"] = best_arch
        row["mean_arch"] = mean_arch
        rows.append(row)

    header = "| Dataset | default | gnn_all | gnn_qk | gnn_v | Winner |"
    sep = "|---------|---------|---------|--------|-------|--------|"
    lines.append(header)
    lines.append(sep)

    for row in rows:
        cells = [row["Dataset"]]
        for arch in ["default", "gnn_all", "gnn_qk", "gnn_v"]:
            b = row.get(f"{arch}_best")
            m = row.get(f"{arch}_mean")
            n = row.get(f"{arch}_n", 0)

            if b is not None:
                is_best = arch == row["best_arch"]
                is_mean = arch == row["mean_arch"]
                markers = []
                if is_best:
                    markers.append("B")
                if is_mean:
                    markers.append("M")
                marker = "".join(markers)
                if marker:
                    cells.append(f"**{b:.3f}/{m:.3f}** ({n}) {marker}")
                else:
                    cells.append(f"{b:.3f}/{m:.3f} ({n})")
            else:
                cells.append("-")

        cells.append(f"{row['best_arch']}/{row['mean_arch']}")
        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")
    lines.append(
        "**Default wins or ties best** in all datasets. Mean differences are <0.3%."
    )

    return "\n".join(lines)


def run_full_analysis(df: pd.DataFrame, output_dir: Path) -> None:
    """Run all analyses and save to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    analyses = [
        ("experiment_breakdown.md", create_breakdown_table),
        ("full_nested_breakdown.md", create_nested_table),
        ("compact_nested_breakdown.md", create_compact_nested_table),
        ("architecture_comparison.md", create_architecture_comparison),
    ]

    for filename, func in analyses:
        report = func(df)
        path = output_dir / filename
        path.write_text(report)
        print(f"Saved {filename}")

    print(f"\nAll reports saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Experiment breakdown analysis")
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output markdown file (or directory for 'full' mode)",
    )
    parser.add_argument(
        "--mode",
        "-m",
        default="full",
        choices=["full", "summary", "nested", "compact", "comparison"],
        help="Output mode: full (default, all reports), summary, nested, compact, comparison",
    )
    args = parser.parse_args()

    if not PARQUET_FILE.exists():
        raise FileNotFoundError(f"{PARQUET_FILE} not found")

    df = pd.read_parquet(PARQUET_FILE)
    df = load_and_parse(df)

    if args.mode == "full":
        output_dir = Path(args.output) if args.output else PARQUET_FILE.parent
        run_full_analysis(df, output_dir)
    else:
        if args.mode == "nested":
            report = create_nested_table(df)
        elif args.mode == "compact":
            report = create_compact_nested_table(df)
        elif args.mode == "comparison":
            report = create_architecture_comparison(df)
        else:
            report = create_breakdown_table(df)

        if args.output:
            Path(args.output).write_text(report)
            print(f"Saved to {args.output}")
        else:
            print(report)


if __name__ == "__main__":
    main()
