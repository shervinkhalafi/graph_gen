"""Tests for ``modal/_lib/evaluate_async.py`` (Step 1 of async-eval plan).

Test rationale
--------------
``evaluate_mmd_async`` is the eval-side worker for the async-eval
architecture. It runs in a separate Modal container (cheaper A10G
``standard`` tier), attaches to the trainer's existing W&B run via
``resume="must"``, runs the MMD evaluation by reusing
``run_mmd_evaluation`` from ``evaluate.py``, logs results back to W&B
under the trainer's custom step axis (``trainer/global_step``), and
appends a manifest row recording the outcome.

Invariants
----------
1. The W&B log call MUST include ``trainer/global_step`` as a value in
   the dict (NOT a ``step=`` kwarg) so W&B's custom-step routing places
   gen-val/* metrics at the right step. See the plan §2 "as if in-band".
2. ``wandb.init`` MUST be called with ``resume="must"`` so a stale
   run_id fails loudly (no silent run-creation).
3. The manifest gains exactly one row per call: ``status="completed"``
   on success, ``status="failed"`` on exception (with ``error_tail``
   set and the exception re-raised).
4. ``wandb.define_metric`` MUST NOT be called by the eval worker — the
   trainer owns that contract; double-defining is a recipe for
   confusion.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch


def _make_task(tmp_path: Path, **overrides: Any) -> dict[str, Any]:
    """Build a baseline task dict for the async-eval worker."""
    base: dict[str, Any] = {
        "run_id": "smallest-cfg-r1-anchor-aabbccdd",
        "run_uid": "smallest-cfg/spectre_sbm/r1/anchor/aabbccdd",
        "checkpoint_path": "/data/outputs/smallest-cfg-r1-anchor-aabbccdd/checkpoints/step_5240.ckpt",
        "wandb_run_id": "abc1234",
        "wandb_project": "tmgg-spectral",
        "wandb_entity": "graph_denoise_team",
        "global_step": 5240,
        "num_samples": 40,
        "num_steps": 100,
        "scheduled_step": 5237,
        "manifest_path": str(tmp_path / "eval_manifest.jsonl"),
    }
    base.update(overrides)
    return base


def _success_eval_output(task: dict[str, Any]) -> dict[str, Any]:
    """Build a fake EvaluationOutput dict that the worker would receive
    from ``run_mmd_evaluation`` on success."""
    return {
        "run_id": task["run_id"],
        "checkpoint_name": "step_5240",
        "status": "completed",
        "results": {
            "eval": {
                "gen-val/sbm_accuracy": 0.62,
                "gen-val/degree_mmd": 0.012,
                "gen-val/clustering_mmd": 0.034,
            }
        },
        "error_message": None,
        "evaluation_params": {
            "num_samples": task["num_samples"],
            "num_steps": task["num_steps"],
            "mmd_kernel": "gaussian_tv",
            "mmd_sigma": 1.0,
            "seed": 42,
        },
        "timestamp": "2026-04-29T12:00:00",
    }


class TestAsyncEvalLogsAtTrainerStep:
    """The eval worker MUST log gen-val metrics with trainer/global_step
    in the payload so W&B's custom-step axis routes correctly."""

    def test_evaluate_mmd_async_logs_with_trainer_global_step(
        self, tmp_path: Path
    ) -> None:
        """``wandb.log`` must be called with a dict containing
        ``trainer/global_step`` AND the gen-val/* metrics, NOT a
        separate ``step=`` kwarg."""
        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        eval_output = _success_eval_output(task)

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=eval_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init") as mock_init,
            patch("tmgg.modal._lib.evaluate_async.wandb.log") as mock_log,
            patch("tmgg.modal._lib.evaluate_async.wandb.finish") as mock_finish,
        ):
            evaluate_async.evaluate_mmd_async(task)

        assert (
            mock_log.call_count == 1
        ), f"wandb.log must be called exactly once; got {mock_log.call_count}"
        call_args = mock_log.call_args
        # The single positional arg is the metrics dict
        logged_dict = (
            call_args.args[0] if call_args.args else call_args.kwargs.get("data")
        )
        assert logged_dict is not None
        assert (
            logged_dict["trainer/global_step"] == 5240
        ), "trainer/global_step must equal the task's global_step"
        assert "gen-val/sbm_accuracy" in logged_dict
        assert logged_dict["gen-val/sbm_accuracy"] == 0.62
        assert "gen-val/degree_mmd" in logged_dict

        # Crucially: no `step=` kwarg — let W&B's custom-step axis route it.
        assert "step" not in call_args.kwargs, (
            "Per the plan, log without step= kwarg; trainer/global_step in the "
            "payload routes via define_metric. step= bypasses that routing."
        )

        # init/finish must be invoked
        assert mock_init.called
        assert mock_finish.called


class TestAsyncEvalUsesResumeMust:
    """``wandb.init`` must use ``resume="must"`` to refuse stale run IDs."""

    def test_evaluate_mmd_async_uses_resume_must(self, tmp_path: Path) -> None:
        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        eval_output = _success_eval_output(task)

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=eval_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init") as mock_init,
            patch("tmgg.modal._lib.evaluate_async.wandb.log"),
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
        ):
            evaluate_async.evaluate_mmd_async(task)

        assert mock_init.called
        kwargs = mock_init.call_args.kwargs
        assert kwargs.get("id") == "abc1234"
        assert kwargs.get("project") == "tmgg-spectral"
        assert kwargs.get("entity") == "graph_denoise_team"
        assert kwargs.get("resume") == "must"


class TestAsyncEvalManifestWrite:
    """One manifest row per call, with the right schema."""

    def test_evaluate_mmd_async_appends_completed_manifest_row(
        self, tmp_path: Path
    ) -> None:
        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        eval_output = _success_eval_output(task)

        manifest_path = Path(task["manifest_path"])
        assert not manifest_path.exists()

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=eval_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init"),
            patch("tmgg.modal._lib.evaluate_async.wandb.log"),
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
        ):
            evaluate_async.evaluate_mmd_async(task)

        assert manifest_path.exists()
        rows = [
            json.loads(line) for line in manifest_path.read_text().splitlines() if line
        ]
        assert len(rows) == 1, f"expected 1 row, got {len(rows)}: {rows}"
        row = rows[0]
        assert row["kind"] == "eval_event"
        assert row["status"] == "completed"
        assert row["run_uid"] == task["run_uid"]
        assert row["wandb_run_id"] == "abc1234"
        assert row["scheduled_step"] == 5237
        assert row["global_step"] == 5240
        assert row["checkpoint_path"] == task["checkpoint_path"]
        assert row["modal_call_id"] is None  # filled by trainer-side spawn record
        assert row["error_tail"] is None
        assert "gen-val/sbm_accuracy" in row["metrics"]
        assert "ts_utc" in row and isinstance(row["ts_utc"], str)


class TestAsyncEvalFailureSemantics:
    """On internal failure, write a failed-row AND re-raise (Modal needs
    to record the failure on its side too)."""

    def test_evaluate_mmd_async_on_failure_writes_failed_row_and_reraises(
        self, tmp_path: Path
    ) -> None:
        import pytest

        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        manifest_path = Path(task["manifest_path"])

        with (
            patch.object(
                evaluate_async,
                "run_mmd_evaluation",
                side_effect=RuntimeError("boom"),
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init"),
            patch("tmgg.modal._lib.evaluate_async.wandb.log") as mock_log,
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
            pytest.raises(RuntimeError, match="boom"),
        ):
            evaluate_async.evaluate_mmd_async(task)

        # Manifest must record the failure.
        assert manifest_path.exists()
        rows = [
            json.loads(line) for line in manifest_path.read_text().splitlines() if line
        ]
        assert len(rows) == 1
        row = rows[0]
        assert row["status"] == "failed"
        assert row["error_tail"] is not None
        assert "boom" in row["error_tail"]
        assert row["global_step"] == task["global_step"]
        assert row["scheduled_step"] == task["scheduled_step"]
        # No metrics logged on failure path
        assert mock_log.call_count == 0


class TestAsyncEvalDoesNotDefineMetric:
    """The trainer owns ``define_metric``. The eval worker MUST NOT
    call it — that would race the trainer's setup."""

    def test_evaluate_mmd_async_does_not_call_define_metric(
        self, tmp_path: Path
    ) -> None:
        from tmgg.modal._lib import evaluate_async

        task = _make_task(tmp_path)
        eval_output = _success_eval_output(task)

        with (
            patch.object(
                evaluate_async, "run_mmd_evaluation", return_value=eval_output
            ),
            patch("tmgg.modal._lib.evaluate_async.wandb.init"),
            patch("tmgg.modal._lib.evaluate_async.wandb.log"),
            patch("tmgg.modal._lib.evaluate_async.wandb.finish"),
            patch("tmgg.modal._lib.evaluate_async.wandb.define_metric") as mock_dm,
        ):
            evaluate_async.evaluate_mmd_async(task)

        assert (
            mock_dm.call_count == 0
        ), "Eval worker must not call define_metric; trainer owns that contract."
