"""``tmgg-modal eval-all`` CLI subcommand.

Usage::

    tmgg-modal eval-all <experiment_dir> [--gpu-tier=standard]
                                          [--num-samples=500]
                                          [--num-steps=1000]
                                          [--no-skip-last]
                                          [--detach/--no-detach]

``<experiment_dir>`` is the run-id directory **on the shared
``tmgg-outputs`` volume**, e.g.
``/data/outputs/discrete_qm9_digress_repro/discrete_qm9_..._fresh_...``.

Looks up the matching ``eval_all_checkpoints[_<tier>]`` function
deployed in the ``tmgg-eval-all`` Modal app, calls it remotely, and
prints the structured aggregate report. ``--detach`` switches to
fire-and-forget so the worker keeps running after the local command
exits — useful for long panels (one-shot evals on 50+ ckpts can take
2-3h).
"""

from __future__ import annotations

import json
from typing import Any

import click

from tmgg.modal._eval_all_functions import EVAL_ALL_APP_NAME

GPU_TIER_CHOICES = click.Choice(["debug", "standard", "fast"], case_sensitive=False)


def _function_name(gpu_tier: str) -> str:
    if gpu_tier == "standard":
        return "eval_all_checkpoints"
    return f"eval_all_checkpoints_{gpu_tier}"


def _lookup_function(name: str) -> Any:
    """Resolve the deployed Modal function handle by name."""
    import modal

    return modal.Function.from_name(EVAL_ALL_APP_NAME, name)


def _print_report(report: dict[str, Any]) -> None:
    click.echo(json.dumps(report, indent=2, sort_keys=True, default=str))


@click.command("eval-all")
@click.argument("experiment_dir", type=str)
@click.option(
    "--gpu-tier",
    type=GPU_TIER_CHOICES,
    default="standard",
    show_default=True,
    help="GPU tier — debug=T4, standard=A10G, fast=A100-40GB.",
)
@click.option(
    "--num-samples",
    type=int,
    default=500,
    show_default=True,
    help="Generated graphs per checkpoint for MMD evaluation.",
)
@click.option(
    "--num-steps",
    type=int,
    default=1000,
    show_default=True,
    help="Denoising steps per generated graph.",
)
@click.option(
    "--skip-last/--no-skip-last",
    default=True,
    show_default=True,
    help="Skip last.ckpt (usually duplicates the latest stepped ckpt).",
)
@click.option(
    "--detach/--no-detach",
    default=True,
    show_default=True,
    help=(
        "Fire-and-forget. With --no-detach the local CLI blocks until the "
        "worker finishes — fine for short panels, painful for 50+ ckpts."
    ),
)
def eval_all(
    experiment_dir: str,
    gpu_tier: str,
    num_samples: int,
    num_steps: int,
    skip_last: bool,
    detach: bool,
) -> None:
    """Run MMD evaluation on every checkpoint in a finished run dir."""
    fn_name = _function_name(gpu_tier)
    fn = _lookup_function(fn_name)
    kwargs = {
        "num_samples": num_samples,
        "num_steps": num_steps,
        "skip_last": skip_last,
    }
    if detach:
        call = fn.spawn(experiment_dir, **kwargs)
        click.echo(
            json.dumps(
                {
                    "spawned": True,
                    "function_call_id": call.object_id,
                    "function": fn_name,
                    "experiment_dir": experiment_dir,
                    "kwargs": kwargs,
                    "app": EVAL_ALL_APP_NAME,
                    "tail_logs": (f"modal app logs {EVAL_ALL_APP_NAME}"),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    report = fn.remote(experiment_dir, **kwargs)
    _print_report(report)
