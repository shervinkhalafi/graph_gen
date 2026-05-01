"""Two profile runners reused by host CLIs and Modal wrappers.

* ``run_train_profile`` — exercises the *training* inner loop. Builds
  the same Hydra config the trainer would build, swaps in a tiny
  ``trainer.max_steps`` so the profile finishes quickly, attaches
  Lightning's ``PyTorchProfiler`` (which wraps ``torch.profiler``) as
  the trainer's ``profiler``, runs ``trainer.fit`` once, returns paths
  to the trace + summary artefacts.

* ``run_eval_profile`` — exercises the *eval/sampling* path on an
  existing checkpoint. Wraps a single ``evaluate_checkpoint`` call in
  ``torch.profiler.profile``, dumps the chrome trace + table.

Both runners write to ``<output_dir>/{trace.json, summary.txt}``. The
caller is responsible for wiring ``output_dir`` to a path that
survives the run (e.g. a Modal volume mount).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.profiler import ProfilerActivity, profile, schedule


def _activities() -> list[ProfilerActivity]:
    activities = [ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(ProfilerActivity.CUDA)
    return activities


def run_train_profile(
    *,
    output_dir: Path,
    overrides: list[str],
    config_dir: Path,
    config_name: str = "base_config_discrete_diffusion_generative",
    num_steps: int = 100,
    warmup_steps: int = 5,
    active_steps: int = 20,
) -> dict[str, Any]:
    """Run a short training profile and dump trace + summary.

    Parameters
    ----------
    output_dir
        Directory to write ``trace.json`` and ``summary.txt`` into.
        Must be writable; created if missing.
    overrides
        Hydra-style overrides (e.g.
        ``["+data=spectre_sbm", "model.model.n_layers=4"]``). These
        are layered on top of ``config_name``.
    config_dir
        Absolute path to the Hydra config directory (the directory
        that *contains* ``config_name.yaml``).
    config_name
        Base config name; default matches the discrete-diffusion CLI.
    num_steps
        Total ``trainer.max_steps`` for the profiled fit. Profile
        captures only the active window (warmup + active).
    warmup_steps, active_steps
        Profiler schedule. Skip → warmup → active → repeat. We use
        ``wait=0, warmup=warmup_steps, active=active_steps, repeat=1``
        so the trace covers a clean steady-state slice.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    import hydra
    import pytorch_lightning as pl
    from pytorch_lightning.profilers import PyTorchProfiler

    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        config = compose(config_name=config_name, overrides=overrides)

    OmegaConf.set_struct(config, False)
    config.trainer.max_steps = num_steps
    OmegaConf.set_struct(config, True)

    # Print resolved overrides so the user can see what was profiled.
    print(f"# train profile config: max_steps={num_steps}")
    print(f"# overrides: {overrides}")

    data_module = hydra.utils.instantiate(config.data)
    model = hydra.utils.instantiate(config.model)

    # Lightning's PyTorchProfiler wraps torch.profiler with the trainer
    # hooks that mark per-step boundaries (so the trace shows distinct
    # train_step / val_step rows), and dumps both .pt.trace.json and
    # the `key_averages().table()` summary on teardown.
    pl_profiler = PyTorchProfiler(
        dirpath=str(output_dir),
        filename="train",
        export_to_chrome=True,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        sort_by_key="cuda_time_total"
        if torch.cuda.is_available()
        else "cpu_time_total",
        row_limit=30,
        activities=_activities(),
        schedule=schedule(
            wait=0,
            warmup=warmup_steps,
            active=active_steps,
            repeat=1,
        ),
    )

    trainer = pl.Trainer(
        max_steps=num_steps,
        accelerator="auto",
        devices=1,
        precision=config.trainer.get("precision", "bf16-mixed"),
        profiler=pl_profiler,
        logger=False,
        enable_checkpointing=False,
        num_sanity_val_steps=0,
        limit_val_batches=0,  # PURE train; no eval cycles
    )

    t0 = time.time()
    trainer.fit(model, data_module)
    elapsed = time.time() - t0

    # Lightning writes the summary table to a sibling file when fit
    # ends. We additionally export a clean key_averages table for easy
    # `tail` reading.
    summary_path = output_dir / "summary.txt"
    if hasattr(pl_profiler, "summary"):
        try:
            summary_path.write_text(pl_profiler.summary())
        except Exception as exc:  # noqa: BLE001 — diagnostic, broad-catch is OK
            summary_path.write_text(f"# summary() failed: {exc}\n")

    return {
        "kind": "train_profile",
        "output_dir": str(output_dir),
        "elapsed_s": elapsed,
        "num_steps": num_steps,
        "warmup_steps": warmup_steps,
        "active_steps": active_steps,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }


def run_eval_profile(
    *,
    output_dir: Path,
    checkpoint_path: Path,
    num_samples: int = 32,
    val_batch_limit: int | None = None,
    viz_count: int = 4,
    device: str = "cuda",
    use_ema: str = "auto",
) -> dict[str, Any]:
    """Run one ``evaluate_checkpoint`` cycle wrapped in ``torch.profiler``.

    Parameters
    ----------
    output_dir
        Directory to write ``trace.json`` and ``summary.txt`` into.
    checkpoint_path
        Path to a ``*.ckpt`` from a discrete-diffusion training run.
    num_samples
        Number of graphs to sample during the profiled cycle. Match
        the per-eval cost of the real sweep (32) for a comparable
        wall-time read; reduce to 8-16 if the profile run-time is
        prohibitive on cheap-tier.
    val_batch_limit, viz_count, device, use_ema
        Forwarded to ``evaluate_checkpoint`` unchanged.
    """
    from tmgg.experiments.discrete_diffusion_generative.evaluate_cli import (
        evaluate_checkpoint,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"# eval profile checkpoint: {checkpoint_path}")
    print(f"# num_samples={num_samples}, device={device}")

    t0 = time.time()
    with profile(
        activities=_activities(),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as prof:
        result = evaluate_checkpoint(
            checkpoint_path=Path(checkpoint_path),
            num_samples=num_samples,
            reference_set="val",
            use_ema=use_ema,  # type: ignore[arg-type]
            device=device,
            output_dir=output_dir / "eval_dump",
            val_batch_limit=val_batch_limit,
            viz_count=viz_count,
        )
    elapsed = time.time() - t0

    trace_path = output_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))

    sort_key = "cuda_time_total" if torch.cuda.is_available() else "cpu_time_total"
    summary = prof.key_averages().table(sort_by=sort_key, row_limit=30)
    (output_dir / "summary.txt").write_text(summary)

    return {
        "kind": "eval_profile",
        "output_dir": str(output_dir),
        "trace_path": str(trace_path),
        "elapsed_s": elapsed,
        "num_samples": num_samples,
        "device": device,
        "checkpoint_path": str(checkpoint_path),
        "metrics_keys": sorted(result.get("metrics", {}).keys())[:20]
        if isinstance(result, dict)
        else [],
    }
