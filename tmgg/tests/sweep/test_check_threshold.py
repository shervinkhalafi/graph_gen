"""Tests for scripts.sweep.check_threshold.

Rationale (per CLAUDE.md): each branch of the threshold rule is
exercised by an independent test. Failure modes (missing key,
missing anchor entry) raise loudly per the §7 invariants.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from scripts.sweep.check_threshold import (
    AnchorMissingError,
    MetricMissingError,
    check_run,
)


def test_smaller_is_better_pass_within_tolerance(anchors_path: Path) -> None:
    """value=0.0014 vs target=0.0013, tol=1.5 → 0.0013*1.5=0.00195, pass."""
    metrics = {
        "gen-val/degree_mmd": 0.0014,
        "gen-val/clustering_mmd": 0.05,
        "gen-val/orbit_mmd": 0.04,
        "gen-val/sbm_accuracy": 0.85,
    }
    breakdown = check_run(
        metrics=metrics, dataset="spectre_sbm", anchors_path=anchors_path
    )
    assert breakdown["pass"] is True
    assert breakdown["per_metric"]["degree_mmd"]["pass"] is True


def test_smaller_is_better_fail_above_tolerance(anchors_path: Path) -> None:
    """value=0.002 vs target=0.0013, tol=1.5 → 0.00195, value > limit, fail."""
    metrics = {
        "gen-val/degree_mmd": 0.002,
        "gen-val/clustering_mmd": 0.05,
        "gen-val/orbit_mmd": 0.04,
        "gen-val/sbm_accuracy": 0.85,
    }
    breakdown = check_run(
        metrics=metrics, dataset="spectre_sbm", anchors_path=anchors_path
    )
    assert breakdown["pass"] is False
    assert breakdown["per_metric"]["degree_mmd"]["pass"] is False


def test_larger_is_better_strict_no_tolerance(anchors_path: Path) -> None:
    """sbm_accuracy=0.79 < 0.8 → fail (no tolerance on accuracies)."""
    metrics = {
        "gen-val/degree_mmd": 0.0014,
        "gen-val/clustering_mmd": 0.05,
        "gen-val/orbit_mmd": 0.04,
        "gen-val/sbm_accuracy": 0.79,
    }
    breakdown = check_run(
        metrics=metrics, dataset="spectre_sbm", anchors_path=anchors_path
    )
    assert breakdown["pass"] is False
    assert breakdown["per_metric"]["sbm_accuracy"]["pass"] is False


def test_missing_metric_raises(anchors_path: Path) -> None:
    """Missing required metric raises MetricMissingError loudly."""
    metrics = {
        "gen-val/degree_mmd": 0.0014,
        # clustering_mmd and orbit_mmd intentionally missing
        "gen-val/sbm_accuracy": 0.85,
    }
    with pytest.raises(MetricMissingError) as excinfo:
        check_run(metrics=metrics, dataset="spectre_sbm", anchors_path=anchors_path)
    assert "clustering_mmd" in str(excinfo.value)


def test_missing_anchor_dataset_raises(anchors_path: Path) -> None:
    """Dataset not in anchors.yaml raises AnchorMissingError."""
    metrics = {"gen-val/degree_mmd": 0.0014}
    with pytest.raises(AnchorMissingError) as excinfo:
        check_run(metrics=metrics, dataset="not_a_dataset", anchors_path=anchors_path)
    assert "not_a_dataset" in str(excinfo.value)
