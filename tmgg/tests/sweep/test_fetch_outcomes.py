"""Unit tests for fetch_outcomes pure logic.

The W&B query layer is integration territory; here we test
``find_pending_launches``, ``fetch_block_keyed_diagnostic``,
``build_outcome_row`` against a hand-built fake Run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
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
    history_rows: list[dict[str, Any]] = field(default_factory=list)
    scan_history_calls: list[dict[str, Any]] = field(default_factory=list)

    def scan_history(
        self,
        keys: list[str] | None = None,
        page_size: int | None = None,
    ) -> list[dict[str, Any]]:
        # Record the call so tests can introspect what fetch_outcomes asked for.
        self.scan_history_calls.append({"keys": keys, "page_size": page_size})
        return list(self.history_rows)


def _write_manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a JSONL manifest with the conventional schema header row."""
    with path.open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"kind": "schema", "version": 1}) + "\n")
        for row in rows:
            fh.write(json.dumps(row) + "\n")


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
    p.write_text('{"kind":"schema","version":1}\n\n{"kind":"launched","run_uid":"a"}\n')
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


# ---------------------------------------------------------------------------
# Async-eval extensions: manifest cross-check + per-step metrics from history.
# ---------------------------------------------------------------------------


def _good_metrics(step: int) -> dict[str, Any]:
    """Pass-grade SBM metrics at ``step``."""
    return {
        "trainer/global_step": step,
        "gen-val/degree_mmd": 0.0014,
        "gen-val/clustering_mmd": 0.05,
        "gen-val/orbit_mmd": 0.04,
        "gen-val/sbm_accuracy": 0.85,
    }


def _bad_metrics(step: int) -> dict[str, Any]:
    """Fail-grade metrics (degree_mmd above tolerance)."""
    return {
        "trainer/global_step": step,
        "gen-val/degree_mmd": 0.10,
        "gen-val/clustering_mmd": 0.50,
        "gen-val/orbit_mmd": 0.50,
        "gen-val/sbm_accuracy": 0.10,
    }


def test_build_outcome_row_with_manifest_emits_metrics_per_step(
    tmp_path: Path, anchors_path: Path
) -> None:
    """When a manifest path is supplied, emit per-step metrics + completeness."""
    schedule = [1000, 3000, 5000]
    manifest_path = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest_path,
        [
            {
                "kind": "eval_event",
                "run_uid": "u",
                "scheduled_step": s,
                "global_step": s,
                "ts_utc": f"2026-04-29T00:00:0{i}Z",
                "status": "completed",
                "metrics": _good_metrics(s),
            }
            for i, s in enumerate(schedule)
        ],
    )

    launched = {
        "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        "dataset": "spectre_sbm",
        "step_cap": 5_000,
    }
    run = FakeRun(
        id="abc",
        state="finished",
        summary={"_step": 5000},
        history_rows=[_good_metrics(s) for s in schedule],
    )

    row = build_outcome_row(
        launched=launched,
        run=cast(WandbRun, cast(object, run)),
        anchors_path=anchors_path,
        manifest_path=manifest_path,
        expected_schedule=schedule,
    )

    assert row["metrics_per_step"] is not None
    assert sorted(row["metrics_per_step"].keys()) == schedule
    completeness = row["evals_completeness"]
    assert completeness is not None
    assert completeness["completed"] == 3
    assert completeness["missing"] == []


def test_build_outcome_row_uses_terminal_step_for_threshold(
    tmp_path: Path, anchors_path: Path
) -> None:
    """Threshold check must read the highest fully-anchored step's metrics."""
    schedule = [1000, 5000]
    manifest_path = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest_path,
        [
            {
                "kind": "eval_event",
                "run_uid": "u",
                "scheduled_step": s,
                "global_step": s,
                "ts_utc": f"2026-04-29T00:00:0{i}Z",
                "status": "completed",
                "metrics": (_bad_metrics(s) if s == 1000 else _good_metrics(s)),
            }
            for i, s in enumerate(schedule)
        ],
    )

    launched = {
        "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        "dataset": "spectre_sbm",
        "step_cap": 5_000,
    }
    # Step 1000 fails, step 5000 passes — the terminal step must win.
    run = FakeRun(
        id="abc",
        state="finished",
        summary={
            "_step": 5000,
            # Summary contains stale or otherwise-wrong values; the
            # terminal-history values must override them.
            "gen-val/degree_mmd": 99.0,
        },
        history_rows=[_bad_metrics(1000), _good_metrics(5000)],
    )

    row = build_outcome_row(
        launched=launched,
        run=cast(WandbRun, cast(object, run)),
        anchors_path=anchors_path,
        manifest_path=manifest_path,
        expected_schedule=schedule,
    )

    assert row["threshold_pass"] is True, row.get("threshold_breakdown")
    # The metrics field must reflect step-5000's history values, not summary.
    assert row["metrics"]["gen-val/degree_mmd"] == 0.0014
    assert row["metrics"]["gen-val/sbm_accuracy"] == 0.85


def test_build_outcome_row_refuses_pass_when_expected_eval_missing(
    tmp_path: Path, anchors_path: Path
) -> None:
    """Schedule [100, 200, 300]; manifest covers 100+200; history has no 300."""
    schedule = [100, 200, 300]
    manifest_path = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest_path,
        [
            {
                "kind": "eval_event",
                "run_uid": "u",
                "scheduled_step": s,
                "global_step": s,
                "ts_utc": f"2026-04-29T00:00:0{i}Z",
                "status": "completed",
                "metrics": _good_metrics(s),
            }
            for i, s in enumerate(schedule[:2])
        ],
    )

    launched = {
        "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        "dataset": "spectre_sbm",
        "step_cap": 300,
    }
    # History also lacks step 300 — neither manifest nor W&B know about it.
    run = FakeRun(
        id="abc",
        state="finished",
        summary={"_step": 300},
        history_rows=[_good_metrics(100), _good_metrics(200)],
    )

    row = build_outcome_row(
        launched=launched,
        run=cast(WandbRun, cast(object, run)),
        anchors_path=anchors_path,
        manifest_path=manifest_path,
        expected_schedule=schedule,
    )

    assert row["threshold_pass"] is False
    assert row["gate_reason"] == "expected_evals_missing"
    completeness = row["evals_completeness"]
    assert completeness is not None
    assert completeness["missing"] == [300]


def test_build_outcome_row_legacy_path_when_no_manifest(
    tmp_path: Path, anchors_path: Path
) -> None:
    """No manifest_path -> behave exactly like the pre-async-eval implementation."""
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
        },
    )
    row = build_outcome_row(
        launched=launched,
        run=cast(WandbRun, cast(object, run)),
        anchors_path=anchors_path,
    )

    assert row["threshold_pass"] is True
    assert row["metrics_per_step"] is None
    assert row["evals_completeness"] is None
    assert row["gate_reason"] is None
    # scan_history must NOT be called in the legacy path.
    assert run.scan_history_calls == []


def test_scan_history_keyed_by_trainer_global_step(
    tmp_path: Path, anchors_path: Path
) -> None:
    """When a manifest is supplied, scan_history must be called with the
    trainer/global_step axis plus the gen-val anchored metrics."""
    schedule = [1000]
    manifest_path = tmp_path / "eval_manifest.jsonl"
    _write_manifest(
        manifest_path,
        [
            {
                "kind": "eval_event",
                "run_uid": "u",
                "scheduled_step": 1000,
                "global_step": 1000,
                "ts_utc": "2026-04-29T00:00:00Z",
                "status": "completed",
                "metrics": _good_metrics(1000),
            }
        ],
    )

    launched = {
        "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        "dataset": "spectre_sbm",
        "step_cap": 1000,
    }
    run = FakeRun(
        id="abc",
        state="finished",
        summary={"_step": 1000},
        history_rows=[_good_metrics(1000)],
    )

    _ = build_outcome_row(
        launched=launched,
        run=cast(WandbRun, cast(object, run)),
        anchors_path=anchors_path,
        manifest_path=manifest_path,
        expected_schedule=schedule,
    )

    assert len(run.scan_history_calls) == 1
    keys = run.scan_history_calls[0]["keys"]
    assert keys is not None
    assert "trainer/global_step" in keys
    # The four anchored gen-val metrics must be present so callers can
    # build a per-step dict for the saturation fit.
    for required in (
        "gen-val/sbm_accuracy",
        "gen-val/degree_mmd",
        "gen-val/clustering_mmd",
        "gen-val/orbit_mmd",
    ):
        assert required in keys
