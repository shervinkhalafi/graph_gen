#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "wandb>=0.18",
#   "pandas",
#   "pyarrow",
#   "matplotlib",
#   "numpy",
#   "Pillow",
# ]
# ///
"""SBM-repro panel: idempotent W&B fetch, plots, timeline, Typst report.

Reads ``GRAPH_DENOISE_TEAM_SERVICE`` from ``.env`` for the W&B API key.

Outputs (under ``--out``, default
``wandb_export/sbm-repro-report-2026-05-05/``)::

    data/<variant>/history.parquet     full scan_history(), keyed by step
    data/<variant>/summary.json        last-known summary (state, latest metrics)
    media/<variant>/<kind>_step<N>.png most-recent + earliest media/* per kind
    figures/curves_<metric>.png        cross-variant overlay plots
    figures/timeline_<kind>.png        variant×step grid of generated samples
    report.typ                         Typst document
    report.pdf                         compiled (if ``typst`` on PATH)

Idempotent: parquet/json/png are skipped when present unless ``--refresh``.
Re-runs always regenerate figures and the Typst doc (cheap).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import wandb
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"
DEFAULT_OUT = REPO_ROOT / "wandb_export" / "sbm-repro-report-2026-05-05"

# Run roster — keep variant order = display order in figures/tables
RUNS: list[tuple[str, str, str]] = [
    (
        "vignac",
        "discrete-sbm-vignac-repro",
        "discrete_sbm_vignac_repro_DiffusionModule_dSpectreSBMDataModule_lr1e-3_wd1e-4_L8_s666_fresh_20260504T162100",
    ),
    (
        "pearl",
        "discrete-sbm-pearl-repro",
        "discrete_sbm_pearl_repro_DiffusionModule_dSpectreSBMDataModule_lr1e-3_wd1e-4_L8_s666_fresh_20260504T213604",
    ),
    (
        "pearl-spec",
        "discrete-sbm-pearl-spectral-repro",
        "discrete_sbm_pearl_spectral_repro_DiffusionModule_dSpectreSBMDataModule_lr1e-3_wd1e-4_L8_s666_fresh_20260504T214912",
    ),
    (
        "pearl-gnnconv-norm",
        "discrete-sbm-pearl-gnnconv-norm-repro",
        "discrete_sbm_pearl_gnnconv_norm_repro_DiffusionModule_dSpectreSBMDataModule_lr1e-3_wd1e-4_L8_s666_fresh_20260504T220515",
    ),
    (
        "pearl-gnnconv-raw",
        "discrete-sbm-pearl-gnnconv-raw-repro",
        "discrete_sbm_pearl_gnnconv_raw_repro_DiffusionModule_dSpectreSBMDataModule_lr1e-3_wd1e-4_L8_s666_fresh_20260504T220549",
    ),
]

TRAIN_KEYS = [
    "trainer/global_step",
    "train/loss_step",
    "train/loss_epoch",
    "train/lr",
    "impl-perf/train/step_time_s",
]
VAL_KEYS = ["trainer/global_step", "val/epoch_NLL", "val/loss"]
GEN_KEYS = [
    "trainer/global_step",
    "gen-val/clustering_mmd",
    "gen-val/degree_mmd",
    "gen-val/orbit_mmd",
    "gen-val/spectral_mmd",
    "gen-val/sbm_accuracy",
    "gen-val/uniqueness",
    "gen-val/modularity_q",
]

CURVE_PLOTS = [
    ("train_loss_step", "train/loss_step", "Train loss (step-level)", "log"),
    ("train_loss_epoch", "train/loss_epoch", "Train loss (epoch-level)", "log"),
    ("val_NLL", "val/epoch_NLL", "Validation NLL", "linear"),
    ("clustering_mmd", "gen-val/clustering_mmd", "Generated clustering MMD", "linear"),
    ("degree_mmd", "gen-val/degree_mmd", "Generated degree MMD", "linear"),
    ("orbit_mmd", "gen-val/orbit_mmd", "Generated orbit MMD", "linear"),
    ("spectral_mmd", "gen-val/spectral_mmd", "Generated spectral MMD", "linear"),
    ("sbm_accuracy", "gen-val/sbm_accuracy", "SBM block-recovery accuracy", "linear"),
    ("step_time_s", "impl-perf/train/step_time_s", "Train step time (s)", "linear"),
]

# Per-variant architectural deltas. Everything else (n_layers=8, dx=256,
# de=64, dy=64, n_head=8, ffN, AdamW, bf16-mixed, no EMA, …) is shared
# verbatim across the 5 runs.
# Three projection_{q,k,v} rows are always identical per variant, so we
# collapse them to a single ``projection`` row. Strings stay short so
# the table fits a portrait page without text overflow.
VARIANT_DELTAS: dict[str, dict[str, str]] = {
    "vignac": {
        "extra_features": "ExtraFeatures",
        "projection": "Linear",
        "eigh/step": "yes (extra_features)",
        "extra_X": "6",
        "extra_y": "11",
    },
    "pearl": {
        "extra_features": "PEARLExtraFeatures",
        "projection": "Linear",
        "eigh/step": "no",
        "extra_X": "19",
        "extra_y": "5",
    },
    "pearl-spec": {
        "extra_features": "PEARLExtraFeatures",
        "projection": "SpectralProjection (k=16, K=3)",
        "eigh/step": "yes (projection)",
        "extra_X": "19",
        "extra_y": "5",
    },
    "pearl-gnnconv-norm": {
        "extra_features": "PEARLExtraFeatures",
        "projection": "BareGraphConv A_norm (K=3)",
        "eigh/step": "no",
        "extra_X": "19",
        "extra_y": "5",
    },
    "pearl-gnnconv-raw": {
        "extra_features": "PEARLExtraFeatures",
        "projection": "BareGraphConv A raw (K=3)",
        "eigh/step": "no",
        "extra_X": "19",
        "extra_y": "5",
    },
}

SHARED_HYPERPARAMS = {
    "transformer layers": "8",
    "node embedding dx": "256",
    "edge embedding de": "64",
    "global y embedding dy": "64",
    "attention heads": "8",
    "feed-forward dx → dx": "256",
    "feed-forward de → de": "64",
    "feed-forward dy → dy": "2048",
    "optimizer": "AdamW (fused), lr=1e-3, wd=1e-4",
    "mixed precision": "bf16-mixed (TF32 enabled)",
    "torch.compile": "enabled (compile_model=true)",
    "EMA": "disabled (matches upstream)",
    "diffusion": "discrete D3PM, 500 steps",
    "data": "SPECTRE SBM (200 train graphs)",
    "batch size": "12 (drop_last_train=true)",
    "max nodes (static pad)": "200",
    "seed": "666",
}


@dataclass
class VariantData:
    label: str
    project: str
    run_id: str
    history: pd.DataFrame
    summary: dict
    state: str
    url: str
    images: dict[str, list[tuple[int, Path]]]  # kind -> [(step, path), ...]
    metadata: dict  # GPU, host, CPU count, started_at, …
    config: dict  # Hydra-resolved config fragments (model_config, etc.)


def load_api_key() -> str:
    if not ENV_PATH.exists():
        sys.exit(f"missing {ENV_PATH}; expected GRAPH_DENOISE_TEAM_SERVICE entry")
    for line in ENV_PATH.read_text().splitlines():
        m = re.match(r"^GRAPH_DENOISE_TEAM_SERVICE=(.*)$", line)
        if m:
            return m.group(1).strip()
    sys.exit(f"GRAPH_DENOISE_TEAM_SERVICE not found in {ENV_PATH}")


def fetch_history(
    api: wandb.Api, project: str, rid: str, keys: Iterable[str]
) -> pd.DataFrame:
    runs = list(
        api.runs(f"graph_denoise_team/{project}", filters={"display_name": rid})
    )
    if not runs:
        return pd.DataFrame()
    r = runs[0]
    rows: list[dict] = []
    for row in r.scan_history(keys=list(keys), page_size=1000):
        rows.append({k: row.get(k) for k in keys})
    return pd.DataFrame(rows)


def fetch_run_history_idempotent(
    api: wandb.Api, label: str, project: str, rid: str, out: Path, refresh: bool
) -> tuple[pd.DataFrame, dict, str, str]:
    """Fetch + cache history per metric group (train/val/gen-val), then merge.

    ``scan_history(keys=[...])`` returns only rows where ALL requested keys are
    present. Since train, val, and gen-val metrics are logged at different
    cadences, querying them together produces an empty intersection. We fetch
    each group separately and concat — preserving every step-keyed row.
    """
    var_dir = out / "data" / label
    var_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = var_dir / "history.parquet"
    summary_path = var_dir / "summary.json"

    runs = list(
        api.runs(f"graph_denoise_team/{project}", filters={"display_name": rid})
    )
    if not runs:
        print(f"[{label}] WARN: run not found", flush=True)
        return pd.DataFrame(), {}, "missing", ""
    r = runs[0]
    state, url = r.state, r.url

    # Per-metric fetch: scan_history(keys=[...]) requires every key present
    # in every row, so we fetch each non-step metric paired with global_step
    # and concat.
    metric_keys = sorted(
        set(
            k
            for ks in (TRAIN_KEYS, VAL_KEYS, GEN_KEYS)
            for k in ks
            if k != "trainer/global_step"
        )
    )

    cached_df: pd.DataFrame | None = None
    if parquet_path.exists() and not refresh:
        loaded = pd.read_parquet(parquet_path)
        assert isinstance(loaded, pd.DataFrame)
        cached_df = loaded
        cached_metrics = set(loaded.columns) - {"trainer/global_step"}
        missing = [m for m in metric_keys if m not in cached_metrics]
        if not missing:
            print(
                f"[{label}] cached history ({parquet_path.stat().st_size//1024} KB) — skip fetch",
                flush=True,
            )
            if summary_path.exists():
                return cached_df, json.loads(summary_path.read_text()), state, url
        else:
            print(
                f"[{label}] cached history present, fetching {len(missing)} new metric(s)…",
                flush=True,
            )
            metric_keys = missing  # only fetch the new ones
    else:
        print(f"[{label}] fetching history…", flush=True)

    frames: list[pd.DataFrame] = []
    if cached_df is not None and not cached_df.empty:
        frames.append(cached_df)
    for metric in metric_keys:
        sub = fetch_history(api, project, rid, ["trainer/global_step", metric])
        if sub.empty:
            continue
        sub = sub.dropna(subset=["trainer/global_step", metric]).copy()
        if sub.empty:
            continue
        sub["trainer/global_step"] = sub["trainer/global_step"].astype(int)
        frames.append(sub)
        print(f"  {metric}: {len(sub)} rows", flush=True)
    df = (
        pd.concat(frames, ignore_index=True)
        .sort_values("trainer/global_step")
        .reset_index(drop=True)
        if frames
        else pd.DataFrame()
    )
    if not df.empty:
        df.to_parquet(parquet_path, index=False)
    # ``r.summary`` is the legacy ``wandb.old.summary.HTTPSummary`` —
    # ``__iter__`` is not implemented and falls back to ``__getitem__(0)``,
    # which raises KeyError. Use ``.keys()`` explicitly. (ruff SIM118
    # complains; ignore it for this object.)
    summary_keys = list(r.summary.keys())  # noqa: SIM118
    summary = {
        k: r.summary.get(k)
        for k in summary_keys
        if any(t in k for t in ("trainer/", "val/", "gen-val/", "train/", "epoch"))
    }
    summary["_state"] = state
    summary["_url"] = url
    summary["_runtime"] = float(r.summary.get("_runtime", 0))
    # Persist metadata + select config alongside summary so the report
    # can describe compute and architecture without an extra W&B round-trip.
    summary["_metadata"] = dict(r.metadata or {})
    cfg_dict = dict(r.config)
    summary["_config"] = {
        k: cfg_dict[k]
        for k in (
            "compile_model",
            "model_class",
            "model_name",
            "model_config",
            "trainer",
            "data",
            "optimizer",
            "model",
        )
        if k in cfg_dict
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    return df, summary, state, url


def fetch_media_idempotent(
    api: wandb.Api,
    label: str,
    project: str,
    rid: str,
    out: Path,
    refresh: bool,
    n_per_kind: int = 4,
) -> dict[str, list[tuple[int, Path]]]:
    media_dir = out / "media" / label
    media_dir.mkdir(parents=True, exist_ok=True)

    runs = list(
        api.runs(f"graph_denoise_team/{project}", filters={"display_name": rid})
    )
    if not runs:
        return {}
    r = runs[0]

    kinds: dict[str, list[tuple[int, wandb.apis.public.File]]] = {}
    for f in r.files():
        if not f.name.endswith(".png"):
            continue
        if "graph_samples" in f.name:
            kind = "graph"
        elif "adjacency_samples" in f.name:
            kind = "adj"
        else:
            continue
        m = re.search(r"_samples_(\d+)_", f.name)
        step = int(m.group(1)) if m else 0
        kinds.setdefault(kind, []).append((step, f))

    out_paths: dict[str, list[tuple[int, Path]]] = {}
    for kind, lst in kinds.items():
        # Pick evenly spaced checkpoints across the available range
        lst_sorted = sorted(lst, key=lambda x: x[0])
        if len(lst_sorted) <= n_per_kind:
            picks = lst_sorted
        else:
            idxs = [
                int(i * (len(lst_sorted) - 1) / (n_per_kind - 1))
                for i in range(n_per_kind)
            ]
            picks = [lst_sorted[i] for i in sorted(set(idxs))]
        for step, f in picks:
            target = media_dir / f"{kind}_step{step:08d}.png"
            if target.exists() and not refresh:
                continue
            try:
                f.download(root=str(media_dir), replace=True)
                src = media_dir / f.name
                if src.exists() and src != target:
                    src.replace(target)
            except Exception as e:
                print(f"[{label}] image download failed for {f.name}: {e}", flush=True)
        # Re-collect resulting files on disk (idempotent across re-runs)
        collected: list[tuple[int, Path]] = []
        for p in media_dir.glob(f"{kind}_step*.png"):
            m = re.search(r"step(\d+)", p.name)
            if m is None:
                continue
            collected.append((int(m.group(1)), p))
        out_paths[kind] = sorted(collected)
    return out_paths


def plot_curve(
    variants: list[VariantData],
    metric_key: str,
    title: str,
    yscale: str,
    out_path: Path,
    y_clip_percentile: float | None = None,
) -> bool:
    """Per-variant overlay. Noisy series (>500 points) get a faint raw line
    plus a rolling-mean (~1% of length) emphasised line so trends survive
    visual saturation.

    ``y_clip_percentile`` (e.g. 99) caps the y-axis at that percentile across
    all variants so a few outliers don't crush the visible range — used for
    step-time plots where compile/checkpoint pauses produce 10-20× spikes
    that would otherwise hide the steady-state difference.
    """
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    cmap = plt.get_cmap("tab10")
    all_y: list[float] = []
    for i, v in enumerate(variants):
        if v.history.empty or metric_key not in v.history.columns:
            continue
        sub = v.history[["trainer/global_step", metric_key]].dropna()
        sub = sub.sort_values("trainer/global_step")
        if sub.empty:
            continue
        x = sub["trainer/global_step"].to_numpy()
        y = sub[metric_key].to_numpy()
        all_y.extend(y.tolist())
        color = cmap(i)
        if len(sub) > 500:
            ax.plot(x, y, color=color, linewidth=0.6, alpha=0.18)
            window = max(11, len(sub) // 100)
            smooth = (
                sub[metric_key]
                .rolling(window=window, min_periods=1, center=True)
                .mean()
                .to_numpy()
            )
            ax.plot(
                x, smooth, label=f"{v.label}", color=color, linewidth=1.6, alpha=0.95
            )
        else:
            ax.plot(
                x,
                y,
                label=v.label,
                color=color,
                linewidth=1.4,
                alpha=0.9,
                marker="o",
                markersize=2.5,
            )
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    if y_clip_percentile is not None and all_y:
        import numpy as np

        ymax = float(np.percentile(all_y, y_clip_percentile))
        ymin = max(0.0, float(np.percentile(all_y, 0.5)))
        ax.set_ylim(ymin * 0.95, ymax * 1.1)
        ax.text(
            0.02,
            0.97,
            f"(y clipped to {y_clip_percentile:.0f}th pctile)",
            transform=ax.transAxes,
            fontsize=8,
            alpha=0.6,
            va="top",
        )
    ax.set_xlabel("trainer/global_step")
    ax.set_ylabel(metric_key)
    ax.set_yscale(yscale)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def plot_timeline(
    variants: list[VariantData], kind: str, out_path: Path, max_cols: int = 4
) -> bool:
    """Grid of generated samples: rows = variants, cols = steps.

    Columns are derived from the *union* of steps across variants (rounded
    to a tolerance bucket since each variant logs at slightly different
    step numbers — vignac may log at step 31547, others at 31548). Cells
    where a variant has no image at a given step are blanked (axis turned
    off entirely, no border) so the grid does not show empty boxes.
    """
    rows_data: list[tuple[str, list[tuple[int, Path]]]] = []
    for v in variants:
        if kind in v.images and v.images[kind]:
            rows_data.append((v.label, v.images[kind]))
    if not rows_data:
        return False

    # Bucket steps by 5% tolerance so close-but-not-identical step numbers
    # share a column (variants log at different exact steps depending on
    # epoch-end vs cadence).
    all_steps = sorted({s for _, lst in rows_data for s, _ in lst})
    if not all_steps:
        return False
    tol = max(100, int(0.025 * max(all_steps)))
    bucket_centers: list[int] = []
    for s in all_steps:
        if not bucket_centers or abs(s - bucket_centers[-1]) > tol:
            bucket_centers.append(s)
    bucket_centers = bucket_centers[:max_cols]

    def lookup(imgs: list[tuple[int, Path]], target: int) -> Path | None:
        for s, p in imgs:
            if abs(s - target) <= tol:
                return p
        return None

    n_rows = len(rows_data)
    n_cols = len(bucket_centers)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[a] for a in axes]
    for r, (label, imgs) in enumerate(rows_data):
        for c, target in enumerate(bucket_centers):
            ax = axes[r][c]
            path = lookup(imgs, target)
            if path is None:
                # Blank cell: kill all visual furniture so empty slots
                # don't read as bordered placeholder boxes.
                ax.axis("off")
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.imshow(Image.open(path))
                if r == 0:
                    ax.set_title(f"≈ step {target:,}", fontsize=10)
            if c == 0 and path is not None:
                ax.set_ylabel(
                    label, fontsize=10, rotation=0, ha="right", va="center", labelpad=40
                )
            elif c == 0 and path is None:
                # Use text on a still-axis-off subplot to keep the row label.
                ax.text(
                    -0.05,
                    0.5,
                    label,
                    fontsize=10,
                    ha="right",
                    va="center",
                    transform=ax.transAxes,
                )
    fig.suptitle(f"Generated samples timeline — {kind}", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return True


def latest_at_common_steps(
    variants: list[VariantData], metric: str, n_picks: int = 6
) -> tuple[list[int], pd.DataFrame]:
    sets = []
    for v in variants:
        if v.history.empty or metric not in v.history.columns:
            sets.append(set())
            continue
        s = set(v.history.dropna(subset=[metric])["trainer/global_step"].astype(int))
        sets.append(s)
    sets = [s for s in sets if s]
    if not sets:
        return [], pd.DataFrame()
    common = sorted(set.intersection(*sets))
    if not common:
        return [], pd.DataFrame()
    if len(common) <= n_picks:
        picks = common
    else:
        picks = sorted(
            {common[int(i * (len(common) - 1) / (n_picks - 1))] for i in range(n_picks)}
        )
    rows = []
    for step in picks:
        row: dict = {"step": step}
        for v in variants:
            if v.history.empty or metric not in v.history.columns:
                row[v.label] = None
                continue
            sub = v.history[
                (v.history["trainer/global_step"] == step) & v.history[metric].notna()
            ]
            row[v.label] = float(sub[metric].iloc[0]) if not sub.empty else None
        rows.append(row)
    return picks, pd.DataFrame(rows)


def fmt_typst_table(title: str, df: pd.DataFrame, direction: str = "↓") -> str:
    """Render a Typst table at 9pt with a per-row best-cell highlight.

    Best cell per row (excluding the ``step`` column) is bolded; for
    metrics where higher is better (``↑``) the maximum wins, otherwise
    the minimum.
    """
    if df.empty:
        return f"_{title}: no data_\n"
    cols = list(df.columns)
    n = len(cols)
    align_str = "(" + ", ".join(["right"] * n) + ")"
    header = ", ".join(f"[*{c}*]" for c in cols)
    body_rows = []
    higher_better = direction == "↑"
    for _, r in df.iterrows():
        # Decide best cell index for non-step numeric values.
        numeric_idx = [
            i
            for i, c in enumerate(cols)
            if c != "step"
            and r[c] is not None
            and isinstance(r[c], int | float)
            and not (isinstance(r[c], float) and (r[c] != r[c]))
        ]
        best_i = None
        if numeric_idx:
            best_i = (max if higher_better else min)(
                numeric_idx, key=lambda i: float(r[cols[i]])
            )
        cells = []
        for i, c in enumerate(cols):
            v = r[c]
            if v is None or (isinstance(v, float) and (v != v)):
                cells.append("[—]")
            elif c == "step":
                cells.append(f"[{int(v):,}]")
            else:
                txt = f"{v:.4f}"
                cells.append(f"[*{txt}*]" if i == best_i else f"[{txt}]")
        body_rows.append(", ".join(cells))
    body = ",\n  ".join(body_rows)
    return (
        f"=== {title} ({direction} better)\n\n"
        f"#text(size: 9pt)[\n"
        f"#table(\n"
        f"  columns: {n}, align: {align_str},\n"
        f"  table.header({header}),\n"
        f"  {body}\n"
        f")\n]\n\n"
    )


def compute_table_rows(variants: list[VariantData]) -> pd.DataFrame:
    """Per-variant compute summary derived from W&B metadata + history."""
    rows = []
    for v in variants:
        md = v.metadata or {}
        runtime_s = float(v.summary.get("_runtime", 0) or 0)
        step = v.summary.get("trainer/global_step")
        # Median step time from history (post-compile, so excludes warmup)
        median_step_s = None
        if not v.history.empty and "impl-perf/train/step_time_s" in v.history.columns:
            ser = v.history["impl-perf/train/step_time_s"].dropna()
            if not ser.empty:
                # Drop the first 5% as compile/warmup
                cutoff = max(1, int(len(ser) * 0.05))
                median_step_s = float(ser.iloc[cutoff:].median())
        steps_per_min = 60.0 / median_step_s if median_step_s else None
        gpu_h = runtime_s / 3600.0
        rows.append(
            {
                "variant": v.label,
                "GPU": md.get("gpu", "—"),
                "host": md.get("host", "—"),
                "CPUs": md.get("cpu_count", "—"),
                "step": step if step is not None else "—",
                "wall (h)": round(gpu_h, 2),
                "med step (s)": round(median_step_s, 4) if median_step_s else "—",
                "steps/min": round(steps_per_min, 1) if steps_per_min else "—",
                # Trim ISO timestamp to "YYYY-MM-DD HH:MM" for table width.
                "started": ((md.get("startedAt") or "")[:16]).replace("T", " "),
            }
        )
    return pd.DataFrame(rows)


def plot_throughput_timeline(variants: list[VariantData], out_path: Path) -> bool:
    """Plot rolling-mean step time across training for each variant."""
    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.get_cmap("tab10")
    plotted = False
    for i, v in enumerate(variants):
        if v.history.empty or "impl-perf/train/step_time_s" not in v.history.columns:
            continue
        sub = v.history[["trainer/global_step", "impl-perf/train/step_time_s"]].dropna()
        if sub.empty:
            continue
        sub = sub.sort_values("trainer/global_step")
        x = sub["trainer/global_step"].to_numpy()
        y = sub["impl-perf/train/step_time_s"].to_numpy()
        color = cmap(i)
        ax.plot(x, y, color=color, linewidth=0.5, alpha=0.15)
        window = max(11, len(sub) // 100)
        smooth = (
            sub["impl-perf/train/step_time_s"]
            .rolling(window=window, min_periods=1, center=True)
            .mean()
            .to_numpy()
        )
        ax.plot(x, smooth, label=v.label, color=color, linewidth=1.6, alpha=0.95)
        plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel("trainer/global_step")
    ax.set_ylabel("step time (s)")
    ax.set_title("Train step time per variant (lower = faster, A100 SXM4 40GB)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return True


def write_d2_diagrams(out: Path, fig_dir: Path) -> dict[str, Path]:
    """Emit D2 source for the shared DiGress base + the swap-point overlay,
    then compile each to PNG. Returns ``{label: png_path}`` for whatever
    compiled successfully (silent skip if d2 missing)."""
    d2_dir = out / "diagrams"
    d2_dir.mkdir(exist_ok=True)
    diagrams: dict[str, str] = {}

    # 1) Shared DiGress backbone (base architecture, identical across all 5)
    diagrams["digress_base"] = """
direction: down
title: |md
  # Shared DiGress backbone
  All five variants instantiate this graph, differing only in the
  `extra_features` block and the Q/K/V projection class.
| {near: top-center}

inputs: {
  shape: rectangle
  X_t: "X_t (bs, n, |X|)\\nnoised node 1-hot"
  E_t: "E_t (bs, n, n, |E|)\\nnoised edge 1-hot"
  y_t: "y_t (bs, dy)\\ntimestep + globals"
}

extra_features: "extra_features(X, E, y, mask)\\n→ extra_X, extra_E, extra_y" {
  style.fill: "#fff3a0"
  style.stroke-width: 3
}

cat: "Concatenate\\nX ⊕ extra_X | E ⊕ extra_E | y ⊕ extra_y"

mlp_in: "mlp_in_X / E / y\\n(linear projections to dx=256, de=64, dy=64)"

block_loop: "× 8 transformer layers" {
  style.stroke-dash: 3
  attn: "Q/K/V projection\\n(swappable: Linear | Spectral | BareGraphConv)" {
    style.fill: "#a0e0ff"
    style.stroke-width: 3
  }
  flin: "FiLM(λ_E from E_t) on attn weights"
  ff_x: "feed-forward (dx → dx)"
  ff_e: "feed-forward (de → de)"
  ff_y: "feed-forward (dy → 2048 → dy)"
  attn -> flin -> ff_x
  attn -> flin -> ff_e
  ff_x -> ff_y
  ff_e -> ff_y
}

mlp_out: "mlp_out_X / E / y → output_dims"
heads: "p_θ(X_{t-1}, E_{t-1} | X_t, E_t, y_t)"
loss: "discrete D3PM cross-entropy"

inputs.X_t -> extra_features
inputs.E_t -> extra_features
inputs.y_t -> extra_features
inputs.X_t -> cat
inputs.E_t -> cat
inputs.y_t -> cat
extra_features -> cat
cat -> mlp_in -> block_loop -> mlp_out -> heads -> loss
"""

    # 2) The two swap points (extra_features and projection_q/k/v) shown
    #    side-by-side for each variant
    diagrams["variant_swaps"] = """
direction: right
title: |md
  # What each variant swaps in
  Two configurable slots in the backbone. Variants differ only in
  which class fills these slots.
| {near: top-center}

shared: "Shared backbone\\n(see digress_base diagram)" {
  style.fill: "#eeeeee"
}

ef_slot: "extra_features slot" {
  style.fill: "#fff3a0"
  vignac: "ExtraFeatures\\n(eigh per step)"
  pearl_class: "PEARLExtraFeatures\\n(R-PEARL GNN, no eigh)"
  vignac.style.fill: "#fde68a"
  pearl_class.style.fill: "#bbf7d0"
}

proj_slot: "Q/K/V projection slot" {
  style.fill: "#a0e0ff"
  linear: "Linear (default)"
  spectral: "SpectralProjectionLayer\\n(eigh, k=16, K=3)"
  gnn_norm: "BareGraphConvolution\\nA_norm, K=3"
  gnn_raw: "BareGraphConvolution\\nA raw, K=3"
  linear.style.fill: "#bae6fd"
  spectral.style.fill: "#fbcfe8"
  gnn_norm.style.fill: "#bbf7d0"
  gnn_raw.style.fill: "#fecaca"
}

variants: "Variant → (extra_features, projection)" {
  v_vignac: "vignac\\n= (ExtraFeatures, Linear)"
  v_pearl: "pearl\\n= (PEARLExtraFeatures, Linear)"
  v_pearl_spec: "pearl-spec\\n= (PEARLExtraFeatures, SpectralProjection)"
  v_pearl_norm: "pearl-gnnconv-norm\\n= (PEARLExtraFeatures, BareGraphConv A_norm)"
  v_pearl_raw: "pearl-gnnconv-raw\\n= (PEARLExtraFeatures, BareGraphConv A raw)"
}

shared -> ef_slot
shared -> proj_slot
ef_slot -> variants
proj_slot -> variants
"""

    out_paths: dict[str, Path] = {}
    for name, src in diagrams.items():
        d2_path = d2_dir / f"{name}.d2"
        d2_path.write_text(src.strip() + "\n")
        png_path = fig_dir / f"diagram_{name}.png"
        if shutil.which("d2") is None:
            continue
        # --layout=dagre, --pad=20 for tighter PNGs
        res = subprocess.run(
            [
                "d2",
                "--layout=dagre",
                "--theme=0",
                "--pad=20",
                str(d2_path),
                str(png_path),
            ],
            capture_output=True,
            text=True,
        )
        if res.returncode == 0:
            out_paths[name] = png_path
            print(f"  ✓ {png_path.name}", flush=True)
        else:
            print(f"  ✘ d2 compile failed for {name}:\n{res.stderr[:300]}", flush=True)
    return out_paths


def write_typst_report(variants: list[VariantData], out: Path, fig_dir: Path) -> Path:
    typ = []
    typ.append('#set page(paper: "a4", margin: 1.8cm)')
    typ.append('#set text(font: "DejaVu Sans", size: 10pt)')
    typ.append('#set heading(numbering: "1.")')
    typ.append("")
    typ.append("= SBM repro panel — DiGress + PEARL ablations\n")
    typ.append("Generated by `scripts/sbm_repro_report.py`. Variants:\n")
    typ.append("")
    typ.append("#table(")
    typ.append("  columns: 5, align: (left,) * 5,")
    typ.append("  [*variant*], [*state*], [*step*], [*runtime (h)*], [*W&B*],")
    rows = []
    for v in variants:
        step = v.summary.get("trainer/global_step", "—")
        runtime = v.summary.get("_runtime", 0)
        rows.append(
            f"  [{v.label}], [{v.state}], [{step}], "
            f'[{(runtime or 0)/3600:.1f}], [#link("{v.url}")[run]]'
        )
    typ.append(",\n".join(rows))
    typ.append(")\n")

    # ========== Architecture ==========
    typ.append("== Architecture: shared backbone and per-variant deltas\n")
    typ.append(
        "All five runs use the same DiGress graph-transformer backbone "
        "(8 layers, dx=256, de=64, dy=64, 8 attention heads, FiLM "
        "conditioning on the noise-level eigenvalues) trained on the "
        "SPECTRE SBM dataset under bf16-mixed with `torch.compile` "
        "enabled. They differ only in two slots: the `extra_features` "
        "block that produces the per-node positional encoding, and the "
        "Q/K/V projection class inside each transformer layer.\n"
    )
    base_png = fig_dir / "diagram_digress_base.png"
    if base_png.exists():
        # The backbone diagram is portrait; constrain by ``height`` (not
        # ``width``) so it fits on a single page instead of overflowing
        # to the next.
        typ.append("=== Backbone (shared)\n")
        typ.append(
            '#align(center)[#image("figures/diagram_digress_base.png", height: 80%)]\n'
        )
    swap_png = fig_dir / "diagram_variant_swaps.png"
    if swap_png.exists():
        typ.append("=== Swap points and variant assignments\n")
        typ.append('#image("figures/diagram_variant_swaps.png", width: 100%)\n')

    # Shared hyperparameters in a 2-up layout to halve vertical space.
    typ.append("=== Shared hyperparameters\n")
    items = list(SHARED_HYPERPARAMS.items())
    half = (len(items) + 1) // 2
    left, right = items[:half], items[half:]
    typ.append("#text(size: 9pt)[")
    typ.append(
        "#table(\n  columns: (auto, 1fr, auto, 1fr), align: (left, left, left, left),"
    )
    typ.append("  [*setting*], [*value*], [*setting*], [*value*],")
    pair_rows = []
    for i in range(half):
        l_key, l_val = left[i]
        r_key, r_val = right[i] if i < len(right) else ("", "")
        pair_rows.append(f"  [{l_key}], [{l_val}], [{r_key}], [{r_val}]")
    typ.append(",\n".join(pair_rows) + "\n)\n]\n\n")

    # Transposed delta table: variants are rows, settings are columns.
    # 6 columns (variant + 5 settings) at 9pt fits portrait A4 cleanly.
    typ.append("=== Per-variant deltas\n")
    delta_keys = list(next(iter(VARIANT_DELTAS.values())).keys())
    n_cols = 1 + len(delta_keys)
    typ.append("#text(size: 9pt)[")
    typ.append(f"#table(\n  columns: {n_cols}, align: (left,) * {n_cols},")
    typ.append("  [*variant*], " + ", ".join(f"[*{k}*]" for k in delta_keys) + ",")
    body = []
    for variant_label, deltas in VARIANT_DELTAS.items():
        cells = [f"[`{variant_label}`]"] + [
            f"[{deltas.get(k, '—')}]" for k in delta_keys
        ]
        body.append("  " + ", ".join(cells))
    typ.append(",\n".join(body) + "\n)\n]\n\n")

    # ========== Compute summary ==========
    typ.append("== Compute and runtime\n")
    compute_df = compute_table_rows(variants)
    # Lift uniform columns out of the table when every row carries the
    # same value (typically GPU=A100, host=modal, CPUs=24) — saves width.
    uniform_cols = [
        c
        for c in compute_df.columns
        if c not in ("variant",) and compute_df[c].nunique(dropna=False) == 1
    ]
    uniform_caption = ", ".join(f"{c}={compute_df[c].iloc[0]}" for c in uniform_cols)
    table_df = compute_df.drop(columns=uniform_cols)
    if uniform_caption:
        typ.append(f"_All runs share: {uniform_caption}._\n\n")
    typ.append("#text(size: 9pt)[")
    n_cols = len(table_df.columns)
    typ.append(f"#table(\n  columns: {n_cols}, align: (left,) * {n_cols},")
    typ.append("  " + ", ".join(f"[*{c}*]" for c in table_df.columns) + ",")
    rows_typ = []
    for _, r in table_df.iterrows():
        cells = [f"[{r[c]}]" for c in table_df.columns]
        rows_typ.append("  " + ", ".join(cells))
    typ.append(",\n".join(rows_typ) + "\n)\n]\n\n")

    perf_png = fig_dir / "curves_step_time_s.png"
    if perf_png.exists():
        typ.append("=== Step-time over training\n")
        typ.append('#image("figures/curves_step_time_s.png", width: 100%)\n')

    typ.append("== Training and validation curves\n")
    for slug, _, title, _ in CURVE_PLOTS:
        if slug == "step_time_s":  # already shown in compute section
            continue
        path = fig_dir / f"curves_{slug}.png"
        if path.exists():
            typ.append(f"=== {title}\n")
            typ.append(f'#image("figures/curves_{slug}.png", width: 100%)\n')

    typ.append("== Like-to-like comparison at common eval steps\n")
    for _slug, key, title, _ in CURVE_PLOTS:
        if not key.startswith(("val/", "gen-val/")):
            continue
        _, df = latest_at_common_steps(variants, key)
        direction = "↑" if "accuracy" in key or "uniqueness" in key else "↓"
        typ.append(fmt_typst_table(f"{title} (`{key}`)", df, direction))

    typ.append("== Generated samples timeline\n")
    for kind in ("graph", "adj"):
        path = fig_dir / f"timeline_{kind}.png"
        if path.exists():
            kind_pretty = "Graph layouts" if kind == "graph" else "Adjacency matrices"
            typ.append(f"=== {kind_pretty}\n")
            typ.append(f'#image("figures/timeline_{kind}.png", width: 100%)\n')

    typ_path = out / "report.typ"
    typ_path.write_text("\n".join(typ))
    return typ_path


def maybe_compile_typst(typ_path: Path) -> Path | None:
    if shutil.which("typst") is None:
        print("typst not on PATH — skipping PDF compile", flush=True)
        return None
    pdf = typ_path.with_suffix(".pdf")
    res = subprocess.run(
        ["typst", "compile", str(typ_path), str(pdf)],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        print(f"typst compile FAILED:\n{res.stderr}", flush=True)
        return None
    print(f"compiled {pdf}", flush=True)
    return pdf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--refresh", action="store_true", help="re-fetch history + media even if cached"
    )
    ap.add_argument("--n-images-per-kind", type=int, default=4)
    args = ap.parse_args()

    os.environ["WANDB_API_KEY"] = load_api_key()
    api = wandb.Api(timeout=60)

    args.out.mkdir(parents=True, exist_ok=True)
    fig_dir = args.out / "figures"
    fig_dir.mkdir(exist_ok=True)

    variants: list[VariantData] = []
    for label, proj, rid in RUNS:
        df, summary, state, url = fetch_run_history_idempotent(
            api, label, proj, rid, args.out, args.refresh
        )
        images = fetch_media_idempotent(
            api, label, proj, rid, args.out, args.refresh, args.n_images_per_kind
        )
        metadata = summary.get("_metadata", {}) or {}
        config = summary.get("_config", {}) or {}
        variants.append(
            VariantData(
                label, proj, rid, df, summary, state, url, images, metadata, config
            )
        )

    print("\nrendering curve plots…", flush=True)
    for slug, key, title, yscale in CURVE_PLOTS:
        out_path = fig_dir / f"curves_{slug}.png"
        # Step-time has compile/checkpoint outliers ~10× steady state; clip
        # the y-axis so the per-variant differences (0.15-0.20 s/step) are
        # visible.
        clip = 99.0 if slug == "step_time_s" else None
        ok = plot_curve(variants, key, title, yscale, out_path, y_clip_percentile=clip)
        print(f"  {'✓' if ok else '·'} {out_path.name}", flush=True)

    print("rendering timeline plots…", flush=True)
    for kind in ("graph", "adj"):
        out_path = fig_dir / f"timeline_{kind}.png"
        ok = plot_timeline(variants, kind, out_path)
        print(f"  {'✓' if ok else '·'} {out_path.name}", flush=True)

    print("rendering D2 architecture diagrams…", flush=True)
    write_d2_diagrams(args.out, fig_dir)

    print("writing typst report…", flush=True)
    typ_path = write_typst_report(variants, args.out, fig_dir)
    print(f"  ✓ {typ_path}", flush=True)
    maybe_compile_typst(typ_path)


if __name__ == "__main__":
    main()
