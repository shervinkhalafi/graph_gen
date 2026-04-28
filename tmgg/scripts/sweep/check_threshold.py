"""Threshold-checking library for the smallest-config sweep.

Rationale (per spec §7): missing keys raise loudly; the strict-AND
rule with per-metric tolerance and direction is the single source of
truth for "did this run pass."
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

METRIC_PREFIX = "gen-val/"


class AnchorMissingError(KeyError):
    """Raised when a dataset has no entry in anchors.yaml."""


class MetricMissingError(KeyError):
    """Raised when a required metric is absent from the run summary."""


def _limit(target: float, tolerance_x: float, direction: str) -> float:
    if direction == "smaller_is_better":
        return target * tolerance_x
    if direction == "larger_is_better":
        return target  # tolerance_x must be 1.0 by spec; ignored here
    raise ValueError(f"unknown direction: {direction!r}")


def _passes(value: float, limit: float, direction: str) -> bool:
    if direction == "smaller_is_better":
        return value <= limit
    if direction == "larger_is_better":
        return value >= limit
    raise ValueError(f"unknown direction: {direction!r}")


def check_run(
    *,
    metrics: dict[str, float],
    dataset: str,
    anchors_path: Path,
) -> dict[str, Any]:
    """Apply the strict-AND threshold rule.

    Parameters
    ----------
    metrics
        Mapping of W&B summary keys (``gen-val/<name>``) to numeric values.
    dataset
        Dataset key matching a top-level entry in ``anchors_path``.
    anchors_path
        Path to ``anchors.yaml``.

    Returns
    -------
    dict with keys ``pass`` (bool) and ``per_metric`` (dict of breakdowns).

    Raises
    ------
    AnchorMissingError
        If ``dataset`` has no entry in ``anchors_path``.
    MetricMissingError
        If any anchored metric is absent from ``metrics``.
    """
    with open(anchors_path) as fh:
        anchors = yaml.safe_load(fh)

    if dataset not in anchors:
        raise AnchorMissingError(
            f"dataset {dataset!r} not in {anchors_path}; "
            f"available: {sorted(anchors)}"
        )

    per_metric: dict[str, dict[str, Any]] = {}
    overall = True
    for metric_name, anchor in anchors[dataset].items():
        key = METRIC_PREFIX + metric_name
        if key not in metrics:
            raise MetricMissingError(
                f"required metric {key!r} missing from run summary for dataset {dataset!r}"
            )
        value = float(metrics[key])
        target = float(anchor["target"])
        tolerance_x = float(anchor["tolerance_x"])
        direction = str(anchor["direction"])
        limit = _limit(target, tolerance_x, direction)
        ok = _passes(value, limit, direction)
        overall = overall and ok
        per_metric[metric_name] = {
            "target": target,
            "tolerance_x": tolerance_x,
            "direction": direction,
            "limit": limit,
            "value": value,
            "pass": ok,
        }

    return {"pass": overall, "per_metric": per_metric}
