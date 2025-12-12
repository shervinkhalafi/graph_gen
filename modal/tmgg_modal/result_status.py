"""Result status tracking for experiment resume logic.

Provides finer-grained control over which experiments to skip/resume
based on the state of their results.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from tmgg_modal.storage import TigrisStorage


class ResultStatus(Enum):
    """Status of an experiment result.

    COMPLETE: Result exists with valid metrics
    PARTIAL: Result exists but missing required metrics
    STALE: Result exists but is older than threshold
    MISSING: No result found
    """

    COMPLETE = "complete"
    PARTIAL = "partial"
    STALE = "stale"
    MISSING = "missing"


def check_result_status(
    storage: "TigrisStorage",
    run_id: str,
    required_metrics: list[str] | None = None,
    max_age_hours: float | None = None,
) -> ResultStatus:
    """Check the status of an experiment result.

    Parameters
    ----------
    storage
        Storage backend to check.
    run_id
        Unique run identifier.
    required_metrics
        List of metric names that must be present for COMPLETE status.
        Defaults to ['best_val_loss'].
    max_age_hours
        If set, results older than this are considered STALE.

    Returns
    -------
    ResultStatus
        Status of the result.
    """
    if required_metrics is None:
        required_metrics = ["best_val_loss"]

    result_key = f"results/{run_id}.json"

    # Check if result exists
    if not storage.exists(result_key):
        return ResultStatus.MISSING

    # Download and parse result
    try:
        result = storage.download_metrics(f"results/{run_id}")
    except Exception:
        return ResultStatus.PARTIAL  # Exists but can't be read

    # Check for required metrics
    metrics = result.get("metrics", {})
    for metric_name in required_metrics:
        value = metrics.get(metric_name)
        if value is None or value == float("inf"):
            return ResultStatus.PARTIAL

    # Check age if threshold set
    if max_age_hours is not None:
        completed_at = result.get("completed_at")
        if completed_at:
            try:
                completed_time = datetime.fromisoformat(completed_at)
                age = datetime.now() - completed_time
                if age > timedelta(hours=max_age_hours):
                    return ResultStatus.STALE
            except (ValueError, TypeError):
                pass  # Can't parse timestamp, treat as not stale

    return ResultStatus.COMPLETE


def filter_configs_by_status(
    storage: Optional["TigrisStorage"],
    configs: list[dict[str, Any]],
    skip_statuses: set[ResultStatus] | None = None,
    required_metrics: list[str] | None = None,
    max_age_hours: float | None = None,
) -> tuple[list[dict[str, Any]], dict[str, ResultStatus]]:
    """Filter configs based on their result status.

    Parameters
    ----------
    storage
        Storage backend. If None, returns all configs with MISSING status.
    configs
        List of experiment configurations.
    skip_statuses
        Set of statuses to skip. Defaults to {COMPLETE}.
    required_metrics
        Metrics required for COMPLETE status.
    max_age_hours
        Age threshold for STALE detection.

    Returns
    -------
    tuple[list[dict], dict[str, ResultStatus]]
        Filtered configs and a mapping of run_id -> status for all configs.
    """
    if skip_statuses is None:
        skip_statuses = {ResultStatus.COMPLETE}

    status_map: dict[str, ResultStatus] = {}
    filtered_configs = []

    if storage is None:
        # No storage, all are missing
        for cfg in configs:
            run_id = cfg.get("run_id", "unknown")
            status_map[run_id] = ResultStatus.MISSING
            filtered_configs.append(cfg)
        return filtered_configs, status_map

    for cfg in configs:
        run_id = cfg.get("run_id", "unknown")
        status = check_result_status(
            storage,
            run_id,
            required_metrics=required_metrics,
            max_age_hours=max_age_hours,
        )
        status_map[run_id] = status

        if status not in skip_statuses:
            filtered_configs.append(cfg)

    return filtered_configs, status_map


def summarize_status_map(status_map: dict[str, ResultStatus]) -> str:
    """Create a human-readable summary of result statuses.

    Parameters
    ----------
    status_map
        Mapping of run_id -> ResultStatus.

    Returns
    -------
    str
        Summary string.
    """
    counts = {status: 0 for status in ResultStatus}
    for status in status_map.values():
        counts[status] += 1

    parts = []
    if counts[ResultStatus.COMPLETE] > 0:
        parts.append(f"{counts[ResultStatus.COMPLETE]} complete")
    if counts[ResultStatus.PARTIAL] > 0:
        parts.append(f"{counts[ResultStatus.PARTIAL]} partial")
    if counts[ResultStatus.STALE] > 0:
        parts.append(f"{counts[ResultStatus.STALE]} stale")
    if counts[ResultStatus.MISSING] > 0:
        parts.append(f"{counts[ResultStatus.MISSING]} missing")

    return ", ".join(parts) if parts else "no results"
