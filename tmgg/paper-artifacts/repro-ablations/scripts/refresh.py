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

import re

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
    # Progress paths first (they're under diagnostics-* but distinct)
    if re.match(r"^diagnostics-(train|val)/progress/", metric_name):
        return "progress"
    # lr keys (train/lr falls here, beating train/* below)
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
    Stable across refreshes as long as run_id is immutable (it is, in wandb).
    """
    phase = "postfix" if postfix else "prefix"
    return f"{dataset}_{variant}_{phase}_{run_id}"
