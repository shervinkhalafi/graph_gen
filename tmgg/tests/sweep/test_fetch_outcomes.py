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
        {"kind": "launched", "run_uid": "a", "ts_utc": "2026-04-29T10:00:00+00:00"},
        {"kind": "launched", "run_uid": "b", "ts_utc": "2026-04-29T10:01:00+00:00"},
        {"kind": "outcome", "run_uid": "a", "ts_utc": "2026-04-29T11:00:00+00:00"},
    ]
    pending = find_pending_launches(rows)
    assert [r["run_uid"] for r in pending] == ["b"]


def test_find_pending_launches_treats_relaunch_after_kill_as_pending(
    tmp_path: Path,
) -> None:
    """A relaunch row with the same run_uid but a later ts_utc must be pending.

    The smallest-config sweep's relaunch-after-kill flow (e.g. round 1's
    SBM wrapper-misconfig kill + respawn) appends:
      1. an original ``launched`` row at T1,
      2. a ``failed`` outcome row at T2 (T2 > T1) marking the cancellation,
      3. a fresh ``launched`` row at T3 (T3 > T2) for the respawn.
    All three share a ``run_uid`` because the config hash is identical.
    Pairing by uid alone would mark the relaunch as already-resolved by the
    cancel-outcome and silently skip it; pairing by (uid, latest-outcome-ts)
    correctly identifies the relaunch as pending while still marking the
    original as paired.
    """
    rows = [
        {"kind": "schema", "version": 1},
        {
            "kind": "launched",
            "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/abcd1234",
            "ts_utc": "2026-04-30T04:33:28+00:00",
            "modal_function_call_id": "fc-original",
        },
        {
            "kind": "outcome",
            "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/abcd1234",
            "ts_utc": "2026-04-30T04:40:00+00:00",
            "status": "failed",
            "failure_kind": "cancelled_wrapper_misconfig",
        },
        {
            "kind": "launched",
            "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/abcd1234",
            "ts_utc": "2026-04-30T04:45:00+00:00",
            "modal_function_call_id": "fc-relaunch",
        },
    ]
    pending = find_pending_launches(rows)
    assert (
        len(pending) == 1
    ), "the relaunch row must be pending; cancel-outcome should not shadow it"
    assert pending[0]["modal_function_call_id"] == "fc-relaunch"


def test_find_pending_launches_pairs_by_call_id_when_outcome_lands_after_relaunch(
    tmp_path: Path,
) -> None:
    """Cancel-outcome for the OLD pod must not shadow the relaunch.

    Real failure mode (round-2 relaunch, 2026-04-30): operator killed
    three pods, immediately relaunched, then appended cancel-outcome
    rows for the killed pods — but the cancel-outcomes carry timestamps
    AFTER the new launched rows because the relaunch invocation ran
    first. Pure ts-based pairing then incorrectly marks the relaunches
    as already-paired by their predecessors' cancel-outcomes.

    The fix is to match on ``modal_function_call_id`` first: each
    cancel-outcome carries the killed pod's fc-id, so it pairs to that
    specific launched row regardless of ts ordering. The relaunched
    rows have NEW fc-ids and stay pending.
    """
    rows = [
        {
            "kind": "launched",
            "run_uid": "smallest-cfg/spectre_sbm/r2/anchor/abcd1234",
            "ts_utc": "2026-04-30T09:03:38+00:00",
            "modal_function_call_id": "fc-OLD",
        },
        # Operator kills, then relaunches (fc-NEW spawns):
        {
            "kind": "launched",
            "run_uid": "smallest-cfg/spectre_sbm/r2/anchor/abcd1234",
            "ts_utc": "2026-04-30T10:25:28+00:00",
            "modal_function_call_id": "fc-NEW",
        },
        # ... and only THEN appends the cancel-outcome for the killed pod.
        # ts is after the relaunch, but modal_function_call_id pins to fc-OLD.
        {
            "kind": "outcome",
            "run_uid": "smallest-cfg/spectre_sbm/r2/anchor/abcd1234",
            "ts_utc": "2026-04-30T10:26:23+00:00",
            "status": "failed",
            "failure_kind": "killed_misread_perf",
            "modal_function_call_id": "fc-OLD",
        },
    ]
    pending = find_pending_launches(rows)
    assert (
        len(pending) == 1
    ), "the relaunch (fc-NEW) must stay pending; cancel-outcome targets fc-OLD only"
    assert pending[0]["modal_function_call_id"] == "fc-NEW"


def test_find_pending_launches_skips_relaunch_once_its_outcome_lands(
    tmp_path: Path,
) -> None:
    """After the relaunch finishes, both launched rows are paired.

    Sequencing: launched(T1) → outcome(T2) → launched(T3) → outcome(T4).
    The latest outcome's ts (T4) is >= the relaunch's ts (T3), so neither
    launched row is pending.
    """
    rows = [
        {"kind": "launched", "run_uid": "x", "ts_utc": "2026-04-30T01:00:00+00:00"},
        {
            "kind": "outcome",
            "run_uid": "x",
            "ts_utc": "2026-04-30T02:00:00+00:00",
            "status": "failed",
        },
        {"kind": "launched", "run_uid": "x", "ts_utc": "2026-04-30T03:00:00+00:00"},
        {
            "kind": "outcome",
            "run_uid": "x",
            "ts_utc": "2026-04-30T04:00:00+00:00",
            "status": "finished",
        },
    ]
    assert find_pending_launches(rows) == []


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
