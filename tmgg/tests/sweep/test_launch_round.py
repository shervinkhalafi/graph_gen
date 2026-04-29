"""Unit tests for the pure helpers in scripts.sweep.launch_round."""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from scripts.sweep.launch_round import (
    build_wrapper_invocation,
    config_hash,
    launch_one,
    load_async_eval_schedule,
    make_run_uid,
    parse_modal_app_id,
    parse_modal_function_call_id,
)


def test_config_hash_stable_across_key_order() -> None:
    h1 = config_hash({"a": 1, "b": 2})
    h2 = config_hash({"b": 2, "a": 1})
    assert h1 == h2
    assert len(h1) == 8


def test_config_hash_changes_with_value() -> None:
    h1 = config_hash({"a": 1})
    h2 = config_hash({"a": 2})
    assert h1 != h2


def test_run_uid_format() -> None:
    uid = make_run_uid(
        dataset="spectre_sbm", round_no=2, axis_changed="dx", cfg_hash="deadbeef"
    )
    assert uid == "smallest-cfg/spectre_sbm/r2/dx/deadbeef"


def test_parse_modal_app_id_finds_ap_prefix() -> None:
    text = "Spawned ap-9q8fJxK2Lm successfully\nrunning..."
    assert parse_modal_app_id(text) == "ap-9q8fJxK2Lm"


def test_parse_modal_app_id_returns_none_when_absent() -> None:
    assert parse_modal_app_id("no app id here") is None


def test_build_wrapper_invocation_includes_overrides() -> None:
    cmd = build_wrapper_invocation(
        dataset="spectre_sbm",
        run_uid="smallest-cfg/spectre_sbm/r2/dx/deadbeef",
        seed=0,
        overrides={"model.model.hidden_dims.dx": 128, "trainer.max_steps": 100000},
    )
    assert cmd[0] == "./run-upstream-digress-sbm-modal-a100.zsh"
    assert "seed=0" in cmd
    assert "+wandb_name=smallest-cfg/spectre_sbm/r2/dx/deadbeef" in cmd
    assert "model.model.hidden_dims.dx=128" in cmd
    assert "trainer.max_steps=100000" in cmd


def test_build_wrapper_invocation_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError):
        build_wrapper_invocation(
            dataset="not_a_dataset",
            run_uid="smallest-cfg/not_a_dataset/r1/dx/aa",
            seed=0,
            overrides={},
        )


# -----------------------------------------------------------------------------
# Async-eval flag plumbing (Step 8 of the async-eval plan).
#
# These tests pin the contract for ``--async-eval``:
#   * the wrapper-invocation builder appends the Hydra overrides that activate
#     the ``default_with_async_eval`` callback group and inline the schedule,
#   * the launched JSONL row records the schedule-path + gpu-tier metadata,
#   * legacy launches without the flag remain unaffected.
# -----------------------------------------------------------------------------


def _write_fake_schedule(tmp_path: Path, schedule: list[int]) -> Path:
    """Write a minimal eval_schedule_*.yaml fixture matching the real schema."""
    payload = {
        "dataset": "spectre_sbm",
        "n_evals": len(schedule),
        "total_steps": 100000,
        "params": {"rho_min": 5e-5, "rho_max": 2.5e-4, "s_p": 35000},
        "schedule": schedule,
        "doc": "test fixture",
    }
    out = tmp_path / "eval_schedule_test.yaml"
    out.write_text(yaml.safe_dump(payload, sort_keys=False))
    return out


def test_load_schedule_reads_eval_schedule_yaml(tmp_path: Path) -> None:
    """``load_async_eval_schedule`` returns the integer ``schedule`` list."""
    path = _write_fake_schedule(tmp_path, [100, 200, 300])
    assert load_async_eval_schedule(path) == [100, 200, 300]


def test_build_wrapper_invocation_with_async_eval_appends_callback_override(
    tmp_path: Path,
) -> None:
    """The async-eval flag injects the callback-group override + enabled flag.

    Hydra's existing ``base/callbacks: default`` binding (in
    ``_base_infra.yaml``) means the right syntax is
    ``base/callbacks=default_with_async_eval`` — *not* ``+callbacks=...``,
    which would attempt to append a fresh key and fail.
    """
    schedule_path = _write_fake_schedule(tmp_path, [100, 200, 300])
    cmd = build_wrapper_invocation(
        dataset="spectre_sbm",
        run_uid="smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        seed=0,
        overrides={"trainer.max_steps": 100000},
        async_eval_schedule_path=schedule_path,
        async_eval_gpu_tier="standard",
    )
    assert "base/callbacks=default_with_async_eval" in cmd
    assert "callbacks.async_eval_spawn.enabled=true" in cmd
    assert (
        "callbacks.async_eval_spawn.run_uid=smallest-cfg/spectre_sbm/r1/anchor/aabbccdd"
        in cmd
    )
    assert "callbacks.async_eval_spawn.gpu_tier=standard" in cmd


def test_build_wrapper_invocation_with_async_eval_inlines_schedule(
    tmp_path: Path,
) -> None:
    """The schedule list is inlined as ``key=[a,b,c]`` (Hydra list literal).

    No spaces in the bracketed list — Hydra accepts this form via the CLI.
    """
    schedule_path = _write_fake_schedule(tmp_path, [100, 200, 300])
    cmd = build_wrapper_invocation(
        dataset="spectre_sbm",
        run_uid="smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        seed=0,
        overrides={},
        async_eval_schedule_path=schedule_path,
        async_eval_gpu_tier="standard",
    )
    assert "callbacks.async_eval_spawn.schedule=[100,200,300]" in cmd


def test_launched_row_records_async_eval_metadata(tmp_path: Path) -> None:
    """Dry-run launch with async-eval enabled records schedule path + tier."""
    schedule_path = _write_fake_schedule(tmp_path, [100, 200, 300])
    rounds_jsonl = tmp_path / "rounds.jsonl"
    row = launch_one(
        dataset="spectre_sbm",
        round_no=1,
        axis_changed="anchor",
        axis_value="full",
        seed=0,
        step_cap=100000,
        overrides={"trainer.max_steps": 100000},
        rounds_jsonl=rounds_jsonl,
        session_tag="test",
        dry_run=True,
        async_eval_schedule_path=schedule_path,
        async_eval_gpu_tier="standard",
    )
    assert row.get("async_eval_enabled") is True
    assert row.get("async_eval_schedule_path") == str(schedule_path)
    assert row.get("async_eval_gpu_tier") == "standard"


# -----------------------------------------------------------------------------
# Manual-kill workflow: capture the trainer's FunctionCall ID at spawn time.
#
# Rationale
# ---------
# ``scripts.sweep.kill_call`` accepts an ``fc-...`` ID and cancels the
# corresponding Modal call. The trainer's call ID is otherwise invisible to
# the launcher (which sees only Modal *app* IDs in the wrapper's stdout); to
# make the manual-kill flow self-serviceable, the wrapper now prints a stable
# ``MODAL_FUNCTION_CALL_ID=fc-...`` marker, ``parse_modal_function_call_id``
# extracts it, and ``launch_one`` records it on the launched JSONL row under
# ``modal_function_call_id``. The watcher's flowchart then references that
# field directly.
# -----------------------------------------------------------------------------


def test_parse_modal_function_call_id_finds_marker() -> None:
    """The wrapper prints ``MODAL_FUNCTION_CALL_ID=fc-...`` for kill_call."""
    text = (
        "Spawning experiment (detached)...\n"
        "Spawned: run_id=abc123, gpu=fast\n"
        "MODAL_FUNCTION_CALL_ID=fc-01KQDXYZABC\n"
        "Check Modal dashboard for progress.\n"
    )
    assert parse_modal_function_call_id(text) == "fc-01KQDXYZABC"


def test_parse_modal_function_call_id_returns_none_when_absent() -> None:
    """Older wrapper output (or non-detached path) has no marker — return None."""
    text = "Spawned ap-9q8fJxK2Lm successfully\n"
    assert parse_modal_function_call_id(text) is None


def test_launched_row_records_modal_function_call_id(tmp_path: Path) -> None:
    """Production path threads the trainer's ``fc-...`` ID onto the launched row.

    Mocks the wrapper subprocess so we can verify ``launch_one`` parses
    the ``MODAL_FUNCTION_CALL_ID=...`` marker and records it on the
    JSONL row. This is the field ``kill_call.py`` reads when an operator
    needs to terminate the trainer manually.
    """
    rounds_jsonl = tmp_path / "rounds.jsonl"
    fake_stdout = (
        "Deploying Modal app...\n"
        "Spawned ap-9q8fJxK2Lm successfully\n"
        "MODAL_FUNCTION_CALL_ID=fc-01KQDABCDEF\n"
    )
    fake_proc = subprocess.CompletedProcess(
        args=["./run-upstream-digress-sbm-modal-a100.zsh"],
        returncode=0,
        stdout=fake_stdout,
        stderr="",
    )
    with patch("scripts.sweep.launch_round.subprocess.run", return_value=fake_proc):
        row = launch_one(
            dataset="spectre_sbm",
            round_no=1,
            axis_changed="anchor",
            axis_value="full",
            seed=0,
            step_cap=100000,
            overrides={"trainer.max_steps": 100000},
            rounds_jsonl=rounds_jsonl,
            session_tag="test",
            dry_run=False,
        )

    assert row["modal_function_call_id"] == "fc-01KQDABCDEF"
    assert row["modal_app_id"] == "ap-9q8fJxK2Lm"
    # Persisted to JSONL too.
    persisted = rounds_jsonl.read_text(encoding="utf-8").strip()
    assert '"modal_function_call_id": "fc-01KQDABCDEF"' in persisted


def test_launched_row_modal_function_call_id_null_when_marker_missing(
    tmp_path: Path,
) -> None:
    """If the wrapper didn't emit the marker, the row records ``null`` — never crashes.

    Robustness against legacy wrapper output and the edge case where Modal's
    spawn output races with the parser. Downstream tooling treats null as "no
    direct cancel possible; fall back to ``modal app stop <app_id>``."
    """
    rounds_jsonl = tmp_path / "rounds.jsonl"
    fake_stdout = "Spawned ap-9q8fJxK2Lm successfully\n"  # no MODAL_FUNCTION_CALL_ID
    fake_proc = subprocess.CompletedProcess(
        args=["./run-upstream-digress-sbm-modal-a100.zsh"],
        returncode=0,
        stdout=fake_stdout,
        stderr="",
    )
    with patch("scripts.sweep.launch_round.subprocess.run", return_value=fake_proc):
        row = launch_one(
            dataset="spectre_sbm",
            round_no=1,
            axis_changed="anchor",
            axis_value="full",
            seed=0,
            step_cap=100000,
            overrides={},
            rounds_jsonl=rounds_jsonl,
            session_tag="test",
            dry_run=False,
        )

    assert row["modal_function_call_id"] is None
    assert row["modal_app_id"] == "ap-9q8fJxK2Lm"


def test_legacy_launch_omits_async_eval_fields(tmp_path: Path) -> None:
    """Without the flag, async-eval metadata is absent (or explicitly false).

    The dry-run path returns a sentinel row that does not carry the
    async-eval keys; the production path (covered by the schedule-path
    test above) only sets them when the flag is present.
    """
    rounds_jsonl = tmp_path / "rounds.jsonl"
    row = launch_one(
        dataset="spectre_sbm",
        round_no=1,
        axis_changed="anchor",
        axis_value="full",
        seed=0,
        step_cap=100000,
        overrides={},
        rounds_jsonl=rounds_jsonl,
        session_tag="test",
        dry_run=True,
    )
    # Either absent entirely or explicitly false — both are acceptable.
    assert not row.get("async_eval_enabled", False)
    assert (
        "async_eval_schedule_path" not in row or row["async_eval_schedule_path"] is None
    )
