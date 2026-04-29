"""Unit tests for fetch_outcomes pure logic.

The W&B query layer is integration territory; here we test
``find_pending_launches``, ``fetch_block_keyed_diagnostic``,
``build_outcome_row`` against a hand-built fake Run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from scripts.sweep.fetch_outcomes import (
    build_outcome_row,
    fetch_block_keyed_diagnostic,
    find_pending_launches,
    read_rounds,
)
from wandb.apis.public.runs import Run as WandbRun


@dataclass
class FakeRun:
    id: str
    state: str
    summary: dict[str, Any]


def test_find_pending_launches_returns_unmatched(tmp_path: Path) -> None:
    rows = [
        {"kind": "schema", "version": 1},
        {"kind": "launched", "run_uid": "a"},
        {"kind": "launched", "run_uid": "b"},
        {"kind": "outcome", "run_uid": "a"},
    ]
    pending = find_pending_launches(rows)
    assert [r["run_uid"] for r in pending] == ["b"]


def test_read_rounds_skips_schema_and_blanks(tmp_path: Path) -> None:
    p = tmp_path / "rounds.jsonl"
    p.write_text(
        '{"kind":"schema","version":1}\n' "\n" '{"kind":"launched","run_uid":"a"}\n'
    )
    rows = read_rounds(p)
    assert [r["run_uid"] for r in rows] == ["a"]


def test_fetch_block_keyed_diagnostic_filters_prefix() -> None:
    run = FakeRun(
        id="abc",
        state="finished",
        summary={
            "diagnostics-train/opt-health/grad_snr/transformer_block_0": 0.34,
            "diagnostics-train/opt-health/grad_snr/transformer_block_4": 0.12,
            "diagnostics-train/opt-health/grad_cosine/transformer_block_0": 0.18,
            "gen-val/sbm_accuracy": 0.85,
        },
    )
    out = fetch_block_keyed_diagnostic(
        cast(WandbRun, cast(object, run)),
        "diagnostics-train/opt-health/grad_snr",
    )
    assert out == {"transformer_block_0": 0.34, "transformer_block_4": 0.12}


def test_build_outcome_row_passes_when_metrics_within_tolerance(
    tmp_path: Path, anchors_path: Path
) -> None:
    launched = {
        "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        "dataset": "spectre_sbm",
        "step_cap": 100_000,
    }
    run = FakeRun(
        id="abc123xyz",
        state="finished",
        summary={
            "gen-val/degree_mmd": 0.0014,
            "gen-val/clustering_mmd": 0.05,
            "gen-val/orbit_mmd": 0.04,
            "gen-val/sbm_accuracy": 0.85,
            "_step": 100_000,
            "diagnostics-train/opt-health/grad_snr/transformer_block_0": 0.34,
            "diagnostics-train/opt-health/grad_cosine/transformer_block_0": 0.18,
            "diagnostics-train/opt-health/update_to_weight/transformer_block_0": 4.1e-4,
        },
    )
    row = build_outcome_row(
        launched=launched,
        run=cast(WandbRun, cast(object, run)),
        anchors_path=anchors_path,
    )
    assert row["kind"] == "outcome"
    assert row["status"] == "finished"
    assert row["threshold_pass"] is True
    assert row["wandb_run_id"] == "abc123xyz"


def test_build_outcome_row_failed_run_returns_failed_status(
    tmp_path: Path, anchors_path: Path
) -> None:
    launched = {
        "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        "dataset": "spectre_sbm",
        "step_cap": 100_000,
    }
    run = FakeRun(id="x", state="crashed", summary={"_step": 50})
    row = build_outcome_row(
        launched=launched,
        run=cast(WandbRun, cast(object, run)),
        anchors_path=anchors_path,
    )
    assert row["status"] == "failed"
    assert "failure_kind" in row
    assert row["last_logged_step"] == 50
