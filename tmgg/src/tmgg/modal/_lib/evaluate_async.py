"""Async-evaluation worker that logs back into the trainer's W&B run.

Step 1 of the async-eval plan
(``/home/igork/.claude/plans/compressed-tumbling-whale.md``). The trainer
fires-and-forgets a Modal call against this module on every scheduled
eval step; the eval container then resumes the trainer's W&B run, runs
MMD evaluation, logs gen-val/* metrics under the trainer's custom step
axis (``trainer/global_step``), and appends a row to the per-run eval
manifest JSONL.

Two design decisions are load-bearing:

1. **No ``step=`` kwarg on ``wandb.log``.** Per the user's "log as if
   running in band" clarification, gen-val/* metrics travel with
   ``trainer/global_step`` as a *value* in the dict. The trainer calls
   ``wandb.define_metric("gen-val/*", step_metric="trainer/global_step")``
   once at start; W&B's custom-step routing then places these metrics
   on the correct training step. Passing ``step=`` would bypass that
   routing.
2. **Re-raise on failure.** Manifest is written with
   ``status="failed"`` first, then the exception propagates so Modal
   records the function-call failure too.

This module reuses ``run_mmd_evaluation`` from ``evaluate.py`` without
modifying it. The Modal ``@app.function`` decorator wrapping happens in
``_functions.py`` (Step 2 of the plan), not here.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import wandb

from tmgg.modal._lib.evaluate import EvaluationInput, run_mmd_evaluation


def _append_manifest_row(manifest_path: str, row: dict[str, Any]) -> None:
    """Append a single JSON row to the manifest file.

    A single ``open(...).write(...)`` call is used so the row hits disk
    atomically on POSIX for the size we care about (well under
    ``PIPE_BUF``). The Modal volume commit happens elsewhere in the
    spawn pipeline.
    """
    with open(manifest_path, "a") as f:
        f.write(json.dumps(row) + "\n")


def evaluate_mmd_async(task: dict[str, Any]) -> dict[str, Any]:
    """Run async MMD eval and log to an existing W&B run.

    Parameters
    ----------
    task
        Dict with keys ``run_id`` (str), ``run_uid`` (str),
        ``checkpoint_path`` (str), ``wandb_run_id`` (str),
        ``wandb_project`` (str), ``wandb_entity`` (str),
        ``global_step`` (int), ``num_samples`` (int), ``num_steps``
        (int), ``scheduled_step`` (int), and ``manifest_path`` (str —
        ``/data/outputs/{run_id}/eval_manifest.jsonl``).

    Returns
    -------
    dict
        The serialized ``EvaluationOutput`` returned by
        ``run_mmd_evaluation``.

    Raises
    ------
    Exception
        Any exception raised by ``run_mmd_evaluation`` or by the W&B
        attach is re-raised after the failed-row is appended to the
        manifest, so Modal records the failure too.
    """
    # TODO(smoke-run): verify whether Modal's container-start auto-reload
    # is sufficient for reading the just-written step-stamped checkpoint.
    # If a stale checkpoint is observed in the smoke run, add an explicit
    # ``volume.reload()`` here.

    manifest_path = task["manifest_path"]
    run_uid = task["run_uid"]
    wandb_run_id = task["wandb_run_id"]
    scheduled_step = task["scheduled_step"]
    global_step = task["global_step"]
    checkpoint_path = task["checkpoint_path"]

    try:
        eval_input = EvaluationInput(
            run_id=task["run_id"],
            checkpoint_path=checkpoint_path,
            num_samples=task["num_samples"],
            num_steps=task["num_steps"],
        )
        eval_output = run_mmd_evaluation(
            {
                "run_id": eval_input.run_id,
                "checkpoint_path": eval_input.checkpoint_path,
                "num_samples": eval_input.num_samples,
                "num_steps": eval_input.num_steps,
                "mmd_kernel": eval_input.mmd_kernel,
                "mmd_sigma": eval_input.mmd_sigma,
                "seed": eval_input.seed,
            }
        )
    except Exception as exc:
        # Record the failure before re-raising so the manifest stays the
        # source of truth even if Modal's traceback is the only artifact.
        failed_row = {
            "kind": "eval_event",
            "run_uid": run_uid,
            "wandb_run_id": wandb_run_id,
            "scheduled_step": scheduled_step,
            "global_step": global_step,
            "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "status": "failed",
            "modal_call_id": None,
            "checkpoint_path": checkpoint_path,
            "metrics": None,
            "error_tail": str(exc)[:1000],
        }
        _append_manifest_row(manifest_path, failed_row)
        raise

    # ------------------------------------------------------------------
    # Attach to the trainer's W&B run and log under the custom step axis.
    # ------------------------------------------------------------------
    # Flatten the nested ``results`` dict from EvaluationOutput. The CLI
    # subprocess returns ``{"eval": {metric_name: value, ...}}``; we keep
    # metric names as-is (callers already prefix with ``gen-val/``).
    metrics_dict: dict[str, Any] = {}
    for _label, label_metrics in eval_output.get("results", {}).items():
        for metric_name, value in label_metrics.items():
            metrics_dict[metric_name] = value

    wandb.init(
        id=wandb_run_id,
        project=task["wandb_project"],
        entity=task["wandb_entity"],
        resume="must",
    )
    try:
        # No step= kwarg: trainer/global_step in the payload routes via
        # define_metric (which the trainer calls at on_train_start).
        wandb.log({**metrics_dict, "trainer/global_step": global_step})
    finally:
        wandb.finish()

    completed_row = {
        "kind": "eval_event",
        "run_uid": run_uid,
        "wandb_run_id": wandb_run_id,
        "scheduled_step": scheduled_step,
        "global_step": global_step,
        "ts_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "status": "completed",
        "modal_call_id": None,
        "checkpoint_path": checkpoint_path,
        "metrics": metrics_dict,
        "error_tail": None,
    }
    _append_manifest_row(manifest_path, completed_row)

    return eval_output
