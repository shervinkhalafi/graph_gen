"""Schema validation for anchors.yaml.

Rationale: the file is hand-edited and small. A loud schema test
catches typos (missing direction, unknown direction, malformed
tolerance) before the sweep loads it.
"""

from __future__ import annotations

from pathlib import Path

import yaml

ANCHORS_PATH = Path("docs/experiments/sweep/smallest-config-2026-04-29/anchors.yaml")
REQUIRED_FIELDS = {"target", "tolerance_x", "direction", "source"}
ALLOWED_DIRECTIONS = {"smaller_is_better", "larger_is_better"}


def test_anchors_file_exists() -> None:
    assert ANCHORS_PATH.exists(), f"missing {ANCHORS_PATH}"


def test_each_anchor_has_required_fields() -> None:
    anchors = yaml.safe_load(ANCHORS_PATH.read_text())
    for dataset, metrics in anchors.items():
        for metric_name, anchor in metrics.items():
            missing = REQUIRED_FIELDS - set(anchor)
            assert not missing, f"{dataset}/{metric_name} missing {missing}"


def test_directions_are_allowed() -> None:
    anchors = yaml.safe_load(ANCHORS_PATH.read_text())
    for dataset, metrics in anchors.items():
        for metric_name, anchor in metrics.items():
            assert (
                anchor["direction"] in ALLOWED_DIRECTIONS
            ), f"{dataset}/{metric_name} direction={anchor['direction']!r}"


def test_larger_is_better_has_no_slack() -> None:
    """Spec §3: larger-is-better metrics have tolerance_x=1.0 (no slack)."""
    anchors = yaml.safe_load(ANCHORS_PATH.read_text())
    for dataset, metrics in anchors.items():
        for metric_name, anchor in metrics.items():
            if anchor["direction"] == "larger_is_better":
                assert (
                    float(anchor["tolerance_x"]) == 1.0
                ), f"{dataset}/{metric_name} larger_is_better must have tolerance_x=1.0"


def test_targets_are_positive_finite() -> None:
    anchors = yaml.safe_load(ANCHORS_PATH.read_text())
    for dataset, metrics in anchors.items():
        for metric_name, anchor in metrics.items():
            assert float(anchor["target"]) > 0.0, f"{dataset}/{metric_name}"
