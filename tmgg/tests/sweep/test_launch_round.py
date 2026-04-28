"""Unit tests for the pure helpers in scripts.sweep.launch_round."""

from __future__ import annotations

import pytest
from scripts.sweep.launch_round import (
    build_wrapper_invocation,
    config_hash,
    make_run_uid,
    parse_modal_app_id,
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
    assert "wandb_name=smallest-cfg/spectre_sbm/r2/dx/deadbeef" in cmd
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
