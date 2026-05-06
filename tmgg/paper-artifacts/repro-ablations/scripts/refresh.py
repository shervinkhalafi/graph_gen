#!/usr/bin/env -S uv run --with wandb --with pandas --with pyarrow --with pyyaml --with requests
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "wandb>=0.18",
#     "pandas>=2.2",
#     "pyarrow>=18",
#     "pyyaml>=6.0",
#     "requests>=2.32",
# ]
# ///
"""Refresh the paper-artifacts/repro-ablations/ bundle.

Reads data/runs_index.source.yaml, queries wandb for each run's history
+ metadata + media, and writes:
  - data/runs_index.csv
  - data/all_metrics_long.csv
  - data/per_run_history/<run_id>.parquet
  - media/per_run/<run_id>/*.png
  - snapshots/runlog-<date>.md, snapshots/ablations-measurement-<date>.md
  - CHANGELOG.md (append)

Usage: from the bundle root, `uv run scripts/refresh.py`.

See ../HOW-TO-UPDATE.md for the full update protocol.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import warnings
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import wandb
import yaml

VALID_NAMESPACES = (
    "mmd",
    "train",
    "val",
    "diagnostics",
    "system",
    "impl_perf",
    "lr",
    "progress",
    "other",
)


def classify_metric_namespace(metric_name: str) -> str:
    """Map a wandb metric key to a stable namespace label.

    Order matters: progress/ paths under diagnostics-{train,val}/ should
    be tagged 'progress' rather than 'diagnostics'. lr keys must be
    caught before train/* swallows train/lr.
    """
    if re.match(r"^diagnostics-(train|val)/progress/", metric_name):
        return "progress"
    if metric_name.startswith("lr-") or metric_name == "train/lr":
        return "lr"
    if metric_name.startswith("gen-val/"):
        return "mmd"
    if metric_name.startswith("train/"):
        return "train"
    if metric_name.startswith("val/"):
        return "val"
    if metric_name.startswith("diagnostics-train/") or metric_name.startswith(
        "diagnostics-val/"
    ):
        return "diagnostics"
    if metric_name.startswith("system/"):
        return "system"
    if metric_name.startswith("impl-perf/"):
        return "impl_perf"
    return "other"


def build_run_slug(dataset: str, variant: str, postfix: bool, run_id: str) -> str:
    """Produce a stable, lineage-friendly slug for a run.

    Format: <dataset>_<variant>_<phase>_<run_id> where phase ∈ {prefix,postfix}.
    """
    phase = "postfix" if postfix else "prefix"
    return f"{dataset}_{variant}_{phase}_{run_id}"


# ---------- paths ----------
SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = BUNDLE_ROOT.parent.parent
SOURCE_YAML = BUNDLE_ROOT / "data" / "runs_index.source.yaml"
INDEX_CSV = BUNDLE_ROOT / "data" / "runs_index.csv"
METRICS_LONG_CSV = BUNDLE_ROOT / "data" / "all_metrics_long.csv"
HISTORY_DIR = BUNDLE_ROOT / "data" / "per_run_history"
MEDIA_DIR = BUNDLE_ROOT / "media" / "per_run"
SNAPSHOTS_DIR = BUNDLE_ROOT / "snapshots"
CHANGELOG = BUNDLE_ROOT / "CHANGELOG.md"
RUNLOG_SOURCE = REPO_ROOT / "runlog.md"
DOCS_EVAL_DIR = REPO_ROOT / "docs" / "eval"

WANDB_ENTITY = "graph_denoise_team"


def _wandb_project_for(config_name: str) -> str:
    """Map a hydra experiment config name to its wandb project string.

    Convention: hyphens not underscores.
    """
    return config_name.replace("_", "-")


def fetch_run_history(
    api: wandb.Api, entity: str, project: str, run_id: str
) -> tuple[pd.DataFrame, dict, str | None, dict]:
    """Fetch wandb run history (wide DataFrame), summary, state, and metadata.

    Returns
    -------
    history_df : pd.DataFrame
        One row per logged step, columns are wandb keys. Image / non-scalar
        keys are dropped.
    summary : dict
        Run summary metrics (latest values per key).
    state : str | None
        wandb state.
    metadata : dict
        {display_name, created_at_iso, heartbeat_at_iso, final_step, display_url}.
    """
    run = api.run(f"{entity}/{project}/{run_id}")
    rows = list(run.scan_history())
    if not rows:
        history_df = pd.DataFrame()
    else:
        history_df = pd.DataFrame(rows)
        non_scalar = [
            c
            for c in history_df.columns
            if history_df[c].apply(lambda v: isinstance(v, dict)).any()
        ]
        if non_scalar:
            history_df = history_df.drop(columns=non_scalar)
    summary = dict(run.summary)
    state = run.state
    metadata = {
        "display_name": run.name,
        "created_at_iso": run.created_at,
        "heartbeat_at_iso": getattr(run, "heartbeatAt", None),
        "final_step": summary.get("trainer/global_step") or summary.get("_step") or 0,
        "display_url": run.url,
    }
    return history_df, summary, state, metadata


def fetch_run_media(
    api: wandb.Api, entity: str, project: str, run_id: str, target_dir: Path
) -> list[Path]:
    """Pull the latest gen-val/adjacency_samples and gen-val/graph_samples PNGs."""
    target_dir.mkdir(parents=True, exist_ok=True)
    run = api.run(f"{entity}/{project}/{run_id}")
    summary = dict(run.summary)

    saved: list[Path] = []
    for key in ("gen-val/adjacency_samples", "gen-val/graph_samples"):
        media = summary.get(key)
        if not isinstance(media, dict):
            continue
        rel_path = media.get("path")
        if not rel_path:
            continue
        try:
            f = run.file(rel_path)
            f.download(root=str(target_dir), replace=True)
            downloaded_path = target_dir / rel_path
            local = target_dir / Path(rel_path).name
            if downloaded_path.exists() and downloaded_path != local:
                downloaded_path.replace(local)
                # Clean up any empty parent dirs created by the wandb client
                for p in sorted(downloaded_path.parents, reverse=True):
                    if p == target_dir or not p.exists():
                        break
                    try:
                        p.rmdir()
                    except OSError:
                        break
            saved.append(local)
        except Exception as e:
            print(f"  warn: could not fetch {key} for {run_id}: {e}", file=sys.stderr)
    return saved


def melt_history(history_df: pd.DataFrame, run_meta: dict) -> pd.DataFrame:
    """Convert one run's wide history frame into long format."""
    if history_df.empty:
        return pd.DataFrame()

    keep_index_cols = [
        c
        for c in ("_step", "_timestamp", "trainer/global_step", "epoch")
        if c in history_df.columns
    ]
    metric_cols = [c for c in history_df.columns if c not in keep_index_cols]

    melted = history_df.melt(
        id_vars=keep_index_cols,
        value_vars=metric_cols,
        var_name="metric_name",
        value_name="metric_value",
    )
    melted = melted.dropna(subset=["metric_value"])
    melted["metric_value"] = pd.to_numeric(melted["metric_value"], errors="coerce")
    melted = melted.dropna(subset=["metric_value"])

    melted["metric_namespace"] = melted["metric_name"].map(classify_metric_namespace)

    if "_timestamp" in melted.columns:
        melted["timestamp_utc"] = pd.to_datetime(
            melted["_timestamp"], unit="s", utc=True
        ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        melted = melted.drop(columns=["_timestamp"])
    else:
        melted["timestamp_utc"] = pd.NaT

    if "trainer/global_step" in melted.columns:
        melted["trainer_global_step"] = melted["trainer/global_step"]
        melted = melted.drop(columns=["trainer/global_step"])
    else:
        melted["trainer_global_step"] = pd.NA

    if "epoch" not in melted.columns:
        melted["epoch"] = pd.NA

    for col in ("run_slug", "run_id", "postfix", "dataset", "variant"):
        melted[col] = run_meta[col]

    return melted[
        [
            "run_slug",
            "run_id",
            "postfix",
            "dataset",
            "variant",
            "_step",
            "trainer_global_step",
            "epoch",
            "timestamp_utc",
            "metric_namespace",
            "metric_name",
            "metric_value",
        ]
    ]


def snapshot_runlog_and_measurement(date_str: str) -> list[Path]:
    """Copy runlog.md + the latest ablations-measurement doc into snapshots/."""
    saved = []
    for src, prefix in (
        (RUNLOG_SOURCE, "runlog"),
        (None, "ablations-measurement"),
    ):
        if prefix == "ablations-measurement":
            measurement_files = sorted(DOCS_EVAL_DIR.glob("*-ablations_measurment.md"))
            if not measurement_files:
                warnings.warn(
                    "no docs/eval/<date>-ablations_measurment.md found; skipping snapshot",
                    stacklevel=2,
                )
                continue
            src = measurement_files[-1]
        dest = SNAPSHOTS_DIR / f"{prefix}-{date_str}.md"
        if dest.exists():
            print(
                f"  warn: snapshot {dest.name} already exists; not overwriting",
                file=sys.stderr,
            )
            saved.append(dest)
            continue
        shutil.copy2(src, dest)
        saved.append(dest)
    return saved


def append_changelog(date_str: str, runs_count: int, summary_line: str) -> None:
    """Append a refresh row to CHANGELOG.md, creating the file if needed."""
    if not CHANGELOG.exists():
        CHANGELOG.write_text("# Changelog\n\nAppend-only log of bundle refreshes.\n\n")
    timestamp_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    block = (
        f"## {date_str} — refresh @ {timestamp_utc}\n\n"
        f"- runs_indexed: {runs_count}\n"
        f"- summary: {summary_line}\n\n"
    )
    with CHANGELOG.open("a") as f:
        f.write(block)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--date",
        default=datetime.now(UTC).strftime("%Y-%m-%d"),
        help="Snapshot date suffix (default: today UTC).",
    )
    parser.add_argument(
        "--summary",
        default="scheduled refresh",
        help="One-line summary appended to CHANGELOG.",
    )
    parser.add_argument(
        "--skip-media",
        action="store_true",
        help="Skip wandb media downloads (faster for dry-runs).",
    )
    args = parser.parse_args(argv)

    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    MEDIA_DIR.mkdir(parents=True, exist_ok=True)
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[refresh] reading {SOURCE_YAML.relative_to(REPO_ROOT)}")
    src = yaml.safe_load(SOURCE_YAML.read_text())
    runs_in = src["runs"]
    print(f"[refresh] {len(runs_in)} runs in source YAML")

    api = wandb.Api()
    index_rows = []
    long_frames = []

    for spec in runs_in:
        run_id = spec["run_id"]
        project = _wandb_project_for(spec["config_name"])
        slug = build_run_slug(spec["dataset"], spec["variant"], spec["postfix"], run_id)
        print(f"[refresh] {slug}")

        try:
            history_df, summary, state, meta = fetch_run_history(
                api, WANDB_ENTITY, project, run_id
            )
        except Exception as e:
            print(f"  error: fetch failed: {e}", file=sys.stderr)
            continue

        if not history_df.empty:
            history_df.to_parquet(HISTORY_DIR / f"{run_id}.parquet", index=False)

        index_rows.append(
            {
                "run_slug": slug,
                "run_id": run_id,
                "display_name": meta["display_name"],
                "config_name": spec["config_name"],
                "wandb_project": project,
                "wandb_url": meta["display_url"],
                "dataset": spec["dataset"],
                "variant": spec["variant"],
                "postfix": spec["postfix"],
                "parent_run_id": spec.get("parent_run_id"),
                "parent_run_slug": None,
                "continuation_type": spec["continuation_type"],
                "launched_at_utc": meta["created_at_iso"],
                "ended_at_utc": meta["heartbeat_at_iso"]
                if state != "running"
                else None,
                "final_state": state,
                "final_step": int(meta["final_step"] or 0),
                "health": spec["health"],
                "notes": spec.get("notes", ""),
            }
        )

        run_meta = {
            "run_slug": slug,
            "run_id": run_id,
            "postfix": spec["postfix"],
            "dataset": spec["dataset"],
            "variant": spec["variant"],
        }
        long_df = melt_history(history_df, run_meta)
        if not long_df.empty:
            long_frames.append(long_df)

        if not args.skip_media:
            try:
                fetch_run_media(api, WANDB_ENTITY, project, run_id, MEDIA_DIR / run_id)
            except Exception as e:
                print(f"  warn: media fetch failed: {e}", file=sys.stderr)

    slug_by_id = {r["run_id"]: r["run_slug"] for r in index_rows}
    for r in index_rows:
        if r["parent_run_id"] is not None:
            r["parent_run_slug"] = slug_by_id.get(r["parent_run_id"])

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(INDEX_CSV, index=False)
    print(f"[refresh] wrote {INDEX_CSV.relative_to(REPO_ROOT)} ({len(index_df)} rows)")

    if long_frames:
        long_df = pd.concat(long_frames, ignore_index=True)
        long_df.to_csv(METRICS_LONG_CSV, index=False)
        print(
            f"[refresh] wrote {METRICS_LONG_CSV.relative_to(REPO_ROOT)} ({len(long_df)} rows)"
        )
    else:
        print("[refresh] no metric data fetched", file=sys.stderr)

    snapshots = snapshot_runlog_and_measurement(args.date)
    print(f"[refresh] snapshots: {[s.name for s in snapshots]}")

    append_changelog(args.date, len(index_rows), args.summary)
    print("[refresh] done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
