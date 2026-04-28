"""Fixtures for sweep-tooling tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture()
def anchors_path(tmp_path: Path) -> Path:
    """Synthetic anchors.yaml with one dataset and four metrics.

    The numbers mirror the DiGress paper SBM row (degree, clustering,
    orbit) plus a binary metric (sbm_accuracy) so both branches of the
    threshold rule are exercised.
    """
    anchors = {
        "spectre_sbm": {
            "degree_mmd": {
                "target": 0.0013,
                "tolerance_x": 1.5,
                "direction": "smaller_is_better",
                "source": "Vignac 2022 Table 5",
            },
            "clustering_mmd": {
                "target": 0.0498,
                "tolerance_x": 1.5,
                "direction": "smaller_is_better",
                "source": "Vignac 2022 Table 5",
            },
            "orbit_mmd": {
                "target": 0.0433,
                "tolerance_x": 1.5,
                "direction": "smaller_is_better",
                "source": "Vignac 2022 Table 5",
            },
            "sbm_accuracy": {
                "target": 0.8,
                "tolerance_x": 1.0,
                "direction": "larger_is_better",
                "source": "spec §3 default",
            },
        },
    }
    out = tmp_path / "anchors.yaml"
    out.write_text(yaml.safe_dump(anchors))
    return out
