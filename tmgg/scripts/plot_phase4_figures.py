# /// script
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.24",
#   "pandas>=2.0",
# ]
# ///
"""Render Phase 4 eigenvalue-study paper figures from the sweep CSV.

Reads ``docs/reports/2026-04-19-phase4-eigenvalue-study/phase4_sweep.csv`` and
emits F1/F2/F3 (and optionally FS1/FS2) into the sibling ``figures/`` directory
as paired .pdf + .png outputs.

See ``docs/reports/2026-04-19-phase4-eigenvalue-study/figures-spec.md`` for the
design spec and ``figures.md`` for the reader-facing commentary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import cm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

REPORT_DIR = Path("docs/reports/2026-04-19-phase4-eigenvalue-study")
CSV_PATH = REPORT_DIR / "phase4_sweep.csv"
DEFAULT_FIG_DIR = REPORT_DIR / "figures"

SEEDS = (42, 123, 2024, 7, 11)
K_VALUES = (4, 8, 16, 32)
NOISE_LEVELS = (0.01, 0.05, 0.1, 0.15, 0.2)

SYN_DATASETS = ("sbm_d0.00", "sbm_d0.33", "sbm_d0.67", "sbm_d1.00")
REAL_DATASETS = ("spectre_sbm", "enzymes", "proteins", "collab")
ALL_DATASETS = SYN_DATASETS + REAL_DATASETS

SYN_DIVERSITIES = {
    "sbm_d0.00": 0.00,
    "sbm_d0.33": 0.33,
    "sbm_d0.67": 0.67,
    "sbm_d1.00": 1.00,
}

HEADLINE_CELL = {
    "frame_mode": "frechet",
    "estimator_label": "knn_top_k",
    "noise_type": "gaussian",
}

PASS_THRESHOLD = 0.10
NULL_CEILING = 0.30


def set_style() -> None:
    """Apply the house style: small serif, no titles baked in."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "lines.linewidth": 1.1,
            "lines.markersize": 3.5,
        }
    )


def k_colour(k: int) -> tuple[float, float, float, float]:
    """Plasma-sequential colour keyed to k in {4, 8, 16, 32}."""
    idx = K_VALUES.index(k)
    return cm.plasma(0.15 + 0.7 * idx / (len(K_VALUES) - 1))


def load_sweep(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {
        "seed",
        "dataset",
        "noise_type",
        "noise_level",
        "frame_mode",
        "estimator_label",
        "permuted",
        "k",
        "g_hat",
        "trace_cov_B",
        "fve",
        "num_graphs",
        "n",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing expected columns: {sorted(missing)}")
    return df


def filter_headline(
    df: pd.DataFrame,
    *,
    noise_type: str = "gaussian",
    frame_mode: str = "frechet",
    estimator_label: str = "knn_top_k",
) -> pd.DataFrame:
    mask = (
        (df["noise_type"] == noise_type)
        & (df["frame_mode"] == frame_mode)
        & (df["estimator_label"] == estimator_label)
    )
    return df.loc[mask].copy()


def assert_seed_count(group: pd.DataFrame, context: str) -> None:
    n = group["seed"].nunique()
    if n != len(SEEDS):
        raise RuntimeError(
            f"Expected {len(SEEDS)} seeds for {context}, got {n} "
            f"(seeds={sorted(group['seed'].unique())})"
        )


def aggregate_fve(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Mean and SD of FVE across seeds, grouped by key columns."""
    agg = df.groupby(key_cols, as_index=False).agg(
        fve_mean=("fve", "mean"),
        fve_sd=("fve", "std"),
        n_seeds=("seed", "nunique"),
    )
    bad = agg[agg["n_seeds"] != len(SEEDS)]
    if not bad.empty:
        raise RuntimeError("Cells missing seeds:\n" + bad.to_string(index=False))
    return agg


def paired_margin(df: pd.DataFrame, key_cols: list[str]) -> pd.DataFrame:
    """Calibrated margin (real − null) per seed, then seed-stats.

    Pairs permuted=False and permuted=True rows on the full key + seed, takes
    the difference, then aggregates across seeds.
    """
    real = df[~df["permuted"]].set_index(key_cols + ["seed"])["fve"]
    null = df[df["permuted"]].set_index(key_cols + ["seed"])["fve"]
    aligned = pd.concat([real.rename("real"), null.rename("null")], axis=1)
    missing = aligned[aligned.isna().any(axis=1)]
    if not missing.empty:
        raise RuntimeError(
            f"Unpaired real/null rows after alignment:\n{missing.head()}"
        )
    aligned["margin"] = aligned["real"] - aligned["null"]
    grouped = aligned.groupby(key_cols).agg(
        margin_mean=("margin", "mean"),
        margin_sd=("margin", "std"),
        real_mean=("real", "mean"),
        real_sd=("real", "std"),
        null_mean=("null", "mean"),
        null_sd=("null", "std"),
        n_seeds=("margin", "size"),
    )
    bad = grouped[grouped["n_seeds"] != len(SEEDS)]
    if not bad.empty:
        raise RuntimeError("Margin cells missing seeds:\n" + bad.to_string())
    return grouped.reset_index()


# ----------------------------------------------------------------------------
# F1 — FVE vs ε per dataset (double-column, 2×4 grid)
# ----------------------------------------------------------------------------


def plot_f1(
    df_headline: pd.DataFrame,
    out_dir: Path,
    *,
    noise_type_label: str,
    filename_stem: str,
) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(6.75, 3.5), sharey="row")
    axes = axes.flatten()

    for panel_idx, dataset in enumerate(ALL_DATASETS):
        ax = axes[panel_idx]
        sub = df_headline[df_headline["dataset"] == dataset]
        if sub.empty:
            ax.set_visible(False)
            continue

        for k in K_VALUES:
            k_sub = sub[sub["k"] == k]
            real = k_sub[~k_sub["permuted"]]
            null = k_sub[k_sub["permuted"]]
            if real.empty or null.empty:
                continue

            agg_real = aggregate_fve(real, ["noise_level"]).sort_values("noise_level")
            agg_null = aggregate_fve(null, ["noise_level"]).sort_values("noise_level")

            colour = k_colour(k)
            ax.fill_between(
                agg_null["noise_level"].to_numpy(),
                (agg_null["fve_mean"] - agg_null["fve_sd"]).to_numpy(),
                (agg_null["fve_mean"] + agg_null["fve_sd"]).to_numpy(),
                color=colour,
                alpha=0.12,
                linewidth=0,
            )
            ax.errorbar(
                agg_real["noise_level"].to_numpy(),
                agg_real["fve_mean"].to_numpy(),
                yerr=agg_real["fve_sd"].to_numpy(),
                color=colour,
                marker="o",
                capsize=1.5,
                label=f"k={k}",
            )

        ax.axhline(
            PASS_THRESHOLD, color="black", linestyle="--", linewidth=0.6, alpha=0.6
        )
        ax.set_ylim(0.0, 0.8)
        ax.set_xticks(list(NOISE_LEVELS))
        ax.set_xticklabels([f"{v:g}" for v in NOISE_LEVELS])
        n_graphs = sub["num_graphs"].iloc[0]
        ax.set_title(f"{dataset} (N={n_graphs})")
        if panel_idx % 4 == 0:
            ax.set_ylabel("FVE")
        if panel_idx >= 4:
            ax.set_xlabel(r"$\varepsilon$")

    k_handles = [
        Line2D([0], [0], color=k_colour(k), marker="o", label=f"k={k}")
        for k in K_VALUES
    ]
    null_handle = Patch(
        facecolor="#808080", alpha=0.25, label="permutation null (±1 SD)"
    )
    threshold_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        linewidth=0.6,
        label=f"pass threshold ({PASS_THRESHOLD:g})",
    )
    fig.legend(
        handles=[*k_handles, null_handle, threshold_handle],
        loc="lower center",
        ncol=6,
        bbox_to_anchor=(0.5, -0.02),
        frameon=False,
    )
    fig.suptitle("")  # no top title
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 1.0])

    save_figure(fig, out_dir, filename_stem)
    n_panels = sum(
        1 for d in ALL_DATASETS if not df_headline[df_headline["dataset"] == d].empty
    )
    print(
        f"wrote F1 [{noise_type_label}] (panels={n_panels}, series={len(K_VALUES)}, "
        f"points={n_panels * len(K_VALUES) * len(NOISE_LEVELS)}) "
        f"→ figures/{filename_stem}.{{pdf,png}}"
    )


# ----------------------------------------------------------------------------
# F2 — calibrated margin vs diversity (synthetic SBM, single-column)
# ----------------------------------------------------------------------------


def plot_f2(df_headline: pd.DataFrame, out_dir: Path) -> None:
    sub = df_headline[
        (df_headline["dataset"].isin(SYN_DATASETS))
        & (df_headline["noise_level"] == 0.1)
    ].copy()
    sub["diversity"] = sub["dataset"].map(SYN_DIVERSITIES)

    fig, ax = plt.subplots(figsize=(3.25, 2.5))

    for k in K_VALUES:
        k_sub = sub[sub["k"] == k]
        margin = paired_margin(k_sub, ["dataset", "diversity"]).sort_values("diversity")
        ax.errorbar(
            margin["diversity"].to_numpy(),
            margin["margin_mean"].to_numpy(),
            yerr=margin["margin_sd"].to_numpy(),
            color=k_colour(k),
            marker="o",
            capsize=1.5,
            label=f"k={k}",
        )

    ax.axhline(PASS_THRESHOLD, color="black", linestyle="--", linewidth=0.6, alpha=0.6)
    ax.set_xticks(sorted(SYN_DIVERSITIES.values()))
    ax.set_xlabel("SBM diversity")
    ax.set_ylabel("calibrated margin (real − null)")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()

    save_figure(fig, out_dir, "F2_calibrated_margin_vs_diversity")
    print(
        f"wrote F2 (series={len(K_VALUES)}, points={len(SYN_DATASETS) * len(K_VALUES)}) "
        "→ figures/F2_calibrated_margin_vs_diversity.{pdf,png}"
    )


# ----------------------------------------------------------------------------
# F3 — dataset ranking (horizontal bars, double-column)
# ----------------------------------------------------------------------------


def plot_f3(
    df_headline: pd.DataFrame, out_dir: Path, *, frame_label: str, filename_stem: str
) -> None:
    sub = df_headline[(df_headline["k"] == 8) & (df_headline["noise_level"] == 0.1)]

    margins = paired_margin(sub, ["dataset"])
    margins = margins.sort_values("margin_mean", ascending=True).reset_index(drop=True)

    passed = (margins["margin_mean"] >= PASS_THRESHOLD) & (
        margins["null_mean"] < NULL_CEILING
    )
    colours = np.where(passed, "#3b7dd8", "#b0b0b0")

    se_diff = np.sqrt(
        margins["real_sd"].to_numpy() ** 2 / len(SEEDS)
        + margins["null_sd"].to_numpy() ** 2 / len(SEEDS)
    )

    fig, ax = plt.subplots(figsize=(6.75, 2.75))
    bars = ax.barh(
        margins["dataset"].to_numpy(),
        margins["margin_mean"].to_numpy(),
        xerr=se_diff,
        color=colours,
        edgecolor="black",
        linewidth=0.4,
        error_kw={"ecolor": "black", "capsize": 2.0, "elinewidth": 0.6},
    )
    ax.axvline(PASS_THRESHOLD, color="black", linestyle="--", linewidth=0.6, alpha=0.6)

    for bar, margin_val in zip(bars, margins["margin_mean"].to_numpy(), strict=False):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{margin_val:.2f}",
            va="center",
            fontsize=7,
        )

    ax.set_xlabel(f"calibrated margin (real − null), {frame_label} frame")
    ax.set_xlim(left=min(0.0, float(margins["margin_mean"].min()) - 0.05))
    legend_handles = [
        Patch(
            color="#3b7dd8",
            label=f"pass (margin ≥ {PASS_THRESHOLD:g}, null < {NULL_CEILING:g})",
        ),
        Patch(color="#b0b0b0", label="fail"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", frameon=False)
    fig.tight_layout()

    save_figure(fig, out_dir, filename_stem)
    print(
        f"wrote F3 [{frame_label}] (bars={len(margins)}) "
        f"→ figures/{filename_stem}.{{pdf,png}}"
    )


# ----------------------------------------------------------------------------
# save helper
# ----------------------------------------------------------------------------


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------------
# main
# ----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=CSV_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIG_DIR)
    parser.add_argument(
        "--supplementary", action="store_true", help="also render FS1 and FS2"
    )
    args = parser.parse_args()

    set_style()
    df = load_sweep(args.csv)

    df_head = filter_headline(df)
    plot_f1(
        df_head,
        args.output_dir,
        noise_type_label="gaussian",
        filename_stem="F1_fve_vs_epsilon",
    )
    plot_f2(df_head, args.output_dir)
    plot_f3(
        df_head,
        args.output_dir,
        frame_label="Fréchet",
        filename_stem="F3_dataset_ranking",
    )

    if args.supplementary:
        df_digress = filter_headline(df, noise_type="digress")
        plot_f1(
            df_digress,
            args.output_dir,
            noise_type_label="digress",
            filename_stem="FS1_fve_vs_epsilon_digress",
        )
        df_per_graph = filter_headline(df, frame_mode="per_graph")
        plot_f3(
            df_per_graph,
            args.output_dir,
            frame_label="per-graph",
            filename_stem="FS2_dataset_ranking_per_graph",
        )


if __name__ == "__main__":
    main()
