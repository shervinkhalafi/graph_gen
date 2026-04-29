"""Hydra compose tests for the async-eval callback group.

Test rationale
--------------
Step 4 of the async-eval plan
(``/home/igork/.claude/plans/compressed-tumbling-whale.md``) introduces
two new Hydra configs:

* ``base/callbacks/async_eval_spawn.yaml`` — the
  ``AsyncEvalSpawnCallback`` payload, instantiable via
  ``hydra.utils.instantiate(cfg.callbacks.async_eval_spawn)``.
* ``base/callbacks/default_with_async_eval.yaml`` — composes the
  existing ``default`` callback group with the new
  ``async_eval_spawn`` config so a launcher can enable async eval with
  a single override:
  ``base/callbacks=default_with_async_eval
  callbacks.async_eval_spawn.enabled=true ...``. The override path
  matches the defaults entry in ``_base_infra.yaml``
  (``base/callbacks: default``); a plain ``callbacks=...`` would not
  resolve to this nested group.

These tests verify the compose path resolves cleanly and that the
resulting config carries both the new ``_target_`` for the async-eval
callback and the preserved keys (``early_stopping``, ``checkpoint``)
that ``create_callbacks()`` reads.

Assumed starting state
----------------------
* ``base_config_spectral_arch`` is a known-good entry-point config
  (mirrors ``tests/test_config_composition.py``).
* ``run_id`` is set explicitly via override since
  ``async_eval_spawn.run_uid: ${run_id}`` would otherwise rely on
  runtime resolution.

Invariants
----------
* Composing with ``base/callbacks=default_with_async_eval`` succeeds.
* ``cfg.callbacks.async_eval_spawn._target_`` points at the callback
  class.
* ``cfg.callbacks.early_stopping`` and ``cfg.callbacks.checkpoint``
  remain present (default callbacks not lost).
* Override of ``callbacks.async_eval_spawn.schedule=[1000,2000]``
  populates the list; ``enabled=true`` flips the gate.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

EXP_CONFIG_PATH = (
    Path(__file__).resolve().parents[3] / "src" / "tmgg" / "experiments" / "exp_configs"
)


@pytest.fixture(autouse=True)
def _clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def _minimal_overrides(tmp_path: Path) -> list[str]:
    """Match ``tests/test_config_composition.get_minimal_overrides``."""
    return [
        f"paths.output_dir={tmp_path}",
        f"paths.results_dir={tmp_path}/results",
        "trainer.max_steps=2",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        "~logger",
        "data.batch_size=2",
        "data.num_workers=0",
        f"hydra.run.dir={tmp_path}",
        # ``run_id`` is auto-generated at runtime, but the
        # async_eval_spawn YAML interpolates ``${run_id}`` for ``run_uid``.
        # Pin it for the compose-time resolution.
        "+run_id=test-run-id",
    ]


def test_default_with_async_eval_composes(tmp_path: Path) -> None:
    """The ``default_with_async_eval`` group resolves and carries the new target."""
    overrides = _minimal_overrides(tmp_path) + [
        "base/callbacks=default_with_async_eval",
        "callbacks.async_eval_spawn.enabled=true",
        "callbacks.async_eval_spawn.schedule=[1000, 2000]",
        "callbacks.async_eval_spawn.run_uid=test-uid",
        "callbacks.async_eval_spawn.manifest_path=/tmp/test_manifest.jsonl",
    ]

    with initialize_config_dir(version_base=None, config_dir=str(EXP_CONFIG_PATH)):
        cfg = compose(
            config_name="base_config_spectral_arch",
            overrides=overrides,
        )

    # New callback registered with the expected target.
    assert "async_eval_spawn" in cfg.callbacks, (
        "default_with_async_eval should add an async_eval_spawn entry; "
        f"got keys: {list(cfg.callbacks.keys())}"
    )
    assert (
        cfg.callbacks.async_eval_spawn._target_
        == "tmgg.training.callbacks.async_eval_spawn.AsyncEvalSpawnCallback"
    )
    assert cfg.callbacks.async_eval_spawn.enabled is True
    assert list(cfg.callbacks.async_eval_spawn.schedule) == [1000, 2000]
    assert cfg.callbacks.async_eval_spawn.run_uid == "test-uid"
    assert cfg.callbacks.async_eval_spawn.manifest_path == "/tmp/test_manifest.jsonl"

    # Existing default callback keys must survive composition.
    assert "early_stopping" in cfg.callbacks, (
        "default callbacks lost after composing default_with_async_eval; "
        f"got keys: {list(cfg.callbacks.keys())}"
    )
    assert "checkpoint" in cfg.callbacks
    # Sanity-check default values still flow through.
    assert cfg.callbacks.early_stopping.monitor == "val/loss"
    assert cfg.callbacks.checkpoint.monitor == "val/loss"


def test_async_eval_spawn_defaults(tmp_path: Path) -> None:
    """Defaults in async_eval_spawn.yaml match the callback's __init__ defaults."""
    overrides = _minimal_overrides(tmp_path) + [
        "base/callbacks=default_with_async_eval",
    ]

    with initialize_config_dir(version_base=None, config_dir=str(EXP_CONFIG_PATH)):
        cfg = compose(
            config_name="base_config_spectral_arch",
            overrides=overrides,
        )

    aes = cfg.callbacks.async_eval_spawn
    assert aes.enabled is False, "async_eval_spawn must default to disabled"
    assert list(aes.schedule) == [], "schedule defaults to empty list"
    assert aes.gpu_tier == "standard"
    assert aes.eval_drain_idle_timeout_s == 900
    assert aes.eval_drain_poll_s == 30
    assert aes.keep_step_checkpoints is True
    assert aes.num_samples == 40
    assert aes.num_steps == 1000
    assert aes.modal_app_name == "tmgg-spectral"
