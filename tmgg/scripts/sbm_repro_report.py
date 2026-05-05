#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "wandb>=0.18",
#   "pandas",
#   "pyarrow",
#   "matplotlib",
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

TRAIN_KEYS = ["trainer/global_step", "train/loss_step", "train/loss_epoch", "train/lr"]
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
]


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

    if parquet_path.exists() and summary_path.exists() and not refresh:
        print(
            f"[{label}] cached history ({parquet_path.stat().st_size//1024} KB) — skip fetch",
            flush=True,
        )
        return (
            pd.read_parquet(parquet_path),
            json.loads(summary_path.read_text()),
            state,
            url,
        )

    print(f"[{label}] fetching history…", flush=True)
    # Per-metric fetch: scan_history(keys=[...]) requires every key present
    # in every row, so we fetch each non-step metric paired with global_step
    # and concat.
    frames: list[pd.DataFrame] = []
    metric_keys = sorted(
        set(
            k
            for ks in (TRAIN_KEYS, VAL_KEYS, GEN_KEYS)
            for k in ks
            if k != "trainer/global_step"
        )
    )
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
    summary = {
        k: r.summary.get(k)
        for k in r.summary
        if any(t in k for t in ("trainer/", "val/", "gen-val/", "train/", "epoch"))
    }
    summary["_state"] = state
    summary["_url"] = url
    summary["_runtime"] = float(r.summary.get("_runtime", 0))
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
) -> bool:
    """Per-variant overlay. Noisy series (>500 points) get a faint raw line
    plus a rolling-mean (~1% of length) emphasised line so trends survive
    visual saturation."""
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = False
    cmap = plt.get_cmap("tab10")
    for i, v in enumerate(variants):
        if v.history.empty or metric_key not in v.history.columns:
            continue
        sub = v.history[["trainer/global_step", metric_key]].dropna()
        sub = sub.sort_values("trainer/global_step")
        if sub.empty:
            continue
        x = sub["trainer/global_step"].to_numpy()
        y = sub[metric_key].to_numpy()
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
    """Grid: rows = variants, cols = checkpoint steps."""
    rows_data: list[tuple[str, list[tuple[int, Path]]]] = []
    for v in variants:
        if kind in v.images and v.images[kind]:
            rows_data.append((v.label, v.images[kind][:max_cols]))
    if not rows_data:
        return False
    n_rows = len(rows_data)
    n_cols = max(len(r[1]) for r in rows_data)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 2.6 * n_rows))
    if n_rows == 1:
        axes = [axes]
    if n_cols == 1:
        axes = [[a] for a in axes]
    for r, (label, imgs) in enumerate(rows_data):
        for c in range(n_cols):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c < len(imgs):
                step, path = imgs[c]
                try:
                    ax.imshow(Image.open(path))
                    ax.set_title(f"step {step}", fontsize=9)
                except Exception as e:
                    ax.text(
                        0.5, 0.5, f"err: {e}"[:40], ha="center", va="center", fontsize=7
                    )
            if c == 0:
                ax.set_ylabel(
                    label, fontsize=10, rotation=0, ha="right", va="center", labelpad=40
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
    if df.empty:
        return f"_{title}: no data_\n"
    cols = list(df.columns)
    n = len(cols)
    align_str = "(" + ", ".join(["right"] * n) + ")"
    header = " , ".join(f"[*{c}*]" for c in cols)
    body_rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if v is None or (isinstance(v, float) and (v != v)):
                cells.append("[—]")
            elif c == "step":
                cells.append(f"[{int(v):,}]")
            else:
                cells.append(f"[{v:.4f}]")
        body_rows.append(", ".join(cells))
    body = ",\n  ".join(body_rows)
    return (
        f"=== {title} ({direction} better)\n\n"
        f"#table(\n"
        f"  columns: {n}, align: {align_str},\n"
        f"  {header},\n"
        f"  {body}\n"
        f")\n\n"
    )


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

    typ.append("== Training and validation curves\n")
    for slug, _, title, _ in CURVE_PLOTS:
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
        variants.append(VariantData(label, proj, rid, df, summary, state, url, images))

    print("\nrendering curve plots…", flush=True)
    for slug, key, title, yscale in CURVE_PLOTS:
        out_path = fig_dir / f"curves_{slug}.png"
        ok = plot_curve(variants, key, title, yscale, out_path)
        print(f"  {'✓' if ok else '·'} {out_path.name}", flush=True)

    print("rendering timeline plots…", flush=True)
    for kind in ("graph", "adj"):
        out_path = fig_dir / f"timeline_{kind}.png"
        ok = plot_timeline(variants, kind, out_path)
        print(f"  {'✓' if ok else '·'} {out_path.name}", flush=True)

    print("writing typst report…", flush=True)
    typ_path = write_typst_report(variants, args.out, fig_dir)
    print(f"  ✓ {typ_path}", flush=True)
    maybe_compile_typst(typ_path)


if __name__ == "__main__":
    main()
