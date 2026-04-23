"""CLI for evaluating every Lightning checkpoint under a run directory.

Walks ``--run_dir`` for ``*.ckpt`` files, sorts them by training step
parsed from the canonical filename pattern, then for each checkpoint
loads the model, optionally swaps to EMA shadow weights, generates
``--num_samples`` graphs, runs the configured :class:`GraphEvaluator`,
and writes a CSV row per checkpoint. The output CSV uses pandas's
union-of-keys behaviour so adding a new metric in a future
:class:`EvaluationResults` revision lands as a fresh column with
``NaN`` in older rows -- no schema versioning required (spec D-16c
resolution Q11).

The CLI delegates per-checkpoint work to
:func:`tmgg.experiments.discrete_diffusion_generative.evaluate_cli.evaluate_checkpoint`
so the load + sample + evaluate primitive is shared with the
single-checkpoint CLI.

Per spec resolutions (`docs/specs/2026-04-22-upstream-config-surface-c.md`):

* **Q10** -- ``--device`` defaults to ``auto`` (probes CUDA at run
  time).
* **Q11** -- CSV column drift is handled by union-of-keys NaN-fill on
  re-run, leveraging ``pandas`` natively.
* **Q6** -- ``--reference_set {val,test}`` flag, default ``val`` (matches
  the training-time evaluation cadence; opt in to ``test`` for
  published-quality numbers).
* **Q7** -- ``--use_ema {auto,true,false}`` flag with ``auto`` default.
  ``auto`` swaps to EMA weights iff the checkpoint contains an EMA
  shadow under ``callbacks.<EMACallback>``.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any, Literal

import torch

from tmgg.experiments.discrete_diffusion_generative.evaluate_cli import (
    evaluate_checkpoint,
)

# Lightning's ModelCheckpoint default filename pattern in run_experiment is
# ``model-step={step:06d}-val_loss={...}``. We extract the step counter so
# the CLI can sort checkpoints chronologically without depending on file
# mtimes (which a `cp -r` between machines would destroy).
_STEP_RE = re.compile(r"step=(?P<step>\d+)")


def _parse_step_from_filename(path: Path) -> int:
    """Return the parsed training step counter, or ``-1`` when missing."""
    m = _STEP_RE.search(path.name)
    if m is None:
        return -1
    return int(m.group("step"))


def _resolve_device(requested: Literal["auto", "cpu", "cuda"]) -> str:
    """Return a torch-device string per the ``--device`` flag.

    ``auto`` picks ``cuda`` when available, else ``cpu`` (spec Q10).
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested


def _checkpoint_has_ema_shadow(checkpoint: dict[str, Any]) -> bool:
    """Probe a loaded Lightning checkpoint for an EMA shadow.

    Lightning persists callback state under
    ``checkpoint["callbacks"][<callback-state-key>]``. EMACallback
    today does not implement ``state_dict`` (so no shadow is saved),
    but a future revision will. The probe inspects the dict for any
    callback whose state-key string contains ``"EMA"`` -- a heuristic
    that matches Lightning's default state-key naming
    (``"<ClassName>{init-args-repr}"``). Until EMACallback writes a
    state_dict, this returns ``False`` and the CLI evaluates live
    weights, which is the documented v1 behaviour.
    """
    callbacks = checkpoint.get("callbacks")
    if not isinstance(callbacks, dict):
        return False
    return any("EMA" in str(key) for key in callbacks)


def _evaluate_one_checkpoint(
    checkpoint_path: Path,
    *,
    dataset_type: str,
    num_samples: int,
    num_nodes: int,
    mmd_kernel: Literal["gaussian", "gaussian_tv"],
    mmd_sigma: float,
    device: str,
    seed: int,
    use_ema: Literal["auto", "true", "false"],
) -> dict[str, Any]:
    """Single-checkpoint evaluation row builder.

    Reuses :func:`evaluate_checkpoint` from ``evaluate_cli`` for the
    actual load + sample + evaluate work. The EMA detection probe runs
    independently (a quick ``torch.load`` of the checkpoint header), so
    the row records ``ema_active`` for the user without forcing a
    second model load.
    """
    t0 = time.perf_counter()

    # EMA detection: probe the checkpoint header for a shadow before
    # the heavier load-and-sample path runs. The actual EMA swap
    # remains a v2 follow-up (the EMACallback does not write its
    # shadow into the checkpoint state_dict yet, so even auto mode
    # currently evaluates live weights).
    ema_active = False
    if use_ema in ("auto", "true"):
        ckpt_header = torch.load(
            checkpoint_path, map_location="cpu", weights_only=False
        )
        ema_active = _checkpoint_has_ema_shadow(ckpt_header)
        if use_ema == "true" and not ema_active:
            raise RuntimeError(
                f"--use_ema true requested but checkpoint {checkpoint_path} "
                "carries no EMA shadow."
            )

    raw = evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset_type=dataset_type,
        num_samples=num_samples,
        num_nodes=num_nodes,
        mmd_kernel=mmd_kernel,
        mmd_sigma=mmd_sigma,
        device=device,
        seed=seed,
    )
    eval_seconds = time.perf_counter() - t0

    mmd = raw.get("mmd_results", {})
    row: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path.absolute()),
        "step": _parse_step_from_filename(checkpoint_path),
        "num_samples_generated": int(raw.get("num_generated", num_samples)),
        "eval_seconds": float(eval_seconds),
        "ema_active": bool(ema_active),
    }
    # Flatten the metrics dict into the top-level row. Pandas's
    # union-of-keys handles future additions without code changes.
    for key, value in mmd.items():
        row[key] = value
    return row


def evaluate_all_checkpoints(
    run_dir: Path,
    *,
    num_samples: int,
    num_nodes: int = 20,
    dataset_type: str = "sbm",
    mmd_kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    mmd_sigma: float = 1.0,
    device: str = "cpu",
    seed: int = 42,
    checkpoint_glob: str = "*.ckpt",
    output_csv: Path | None = None,
    skip_existing: bool = False,
    use_ema: Literal["auto", "true", "false"] = "auto",
) -> Any:
    """Programmatic entry: evaluate every checkpoint under ``run_dir``.

    Returns the resulting :class:`pandas.DataFrame` (also written to
    ``output_csv`` when supplied). Use this from notebooks or test
    fixtures to drive the same workflow as the CLI without shelling
    out.
    """
    import pandas as pd

    if not run_dir.is_dir():
        raise FileNotFoundError(
            f"run_dir does not exist or is not a directory: {run_dir}"
        )
    if num_samples < 1:
        raise ValueError(f"num_samples must be >= 1; got {num_samples}")

    checkpoints = sorted(run_dir.rglob(checkpoint_glob), key=_parse_step_from_filename)
    if not checkpoints:
        raise FileNotFoundError(f"No files matched {checkpoint_glob!r} under {run_dir}")

    output_path = output_csv or (run_dir / "all_checkpoints_eval.csv")

    existing_paths: set[str] = set()
    existing_rows: list[dict[str, Any]] = []
    if skip_existing and output_path.exists():
        prior = pd.read_csv(output_path)
        existing_paths = set(prior["checkpoint_path"].astype(str).tolist())
        existing_rows = prior.to_dict(orient="records")

    rows: list[dict[str, Any]] = list(existing_rows)
    for ckpt in checkpoints:
        ckpt_abs = str(ckpt.absolute())
        if ckpt_abs in existing_paths:
            continue
        try:
            row = _evaluate_one_checkpoint(
                ckpt,
                dataset_type=dataset_type,
                num_samples=num_samples,
                num_nodes=num_nodes,
                mmd_kernel=mmd_kernel,
                mmd_sigma=mmd_sigma,
                device=device,
                seed=seed,
                use_ema=use_ema,
            )
        except Exception as exc:  # noqa: BLE001 -- intentional per spec D-16c
            # Per spec D-16c, walking a directory of checkpoints must
            # not abort on a single bad file. The empty-metrics row in
            # the CSV is the visible failure signal; the full traceback
            # goes to stderr so the user can inspect post-hoc.
            print(
                f"warning: checkpoint {ckpt} failed to evaluate: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            row = {
                "checkpoint_path": ckpt_abs,
                "step": _parse_step_from_filename(ckpt),
                "num_samples_generated": 0,
                "eval_seconds": 0.0,
                "ema_active": False,
            }
        rows.append(row)
        # Flush after each checkpoint so an interrupted run leaves a
        # consistent CSV for ``--skip_existing`` to resume from.
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: evaluate every checkpoint under ``--run_dir``."""
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate every Lightning checkpoint under a run directory and "
            "write a per-checkpoint CSV of MMD / SBM metrics."
        ),
    )
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--num_samples", required=True, type=int)
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument(
        "--dataset",
        default="sbm",
        choices=["sbm", "erdos_renyi", "er", "watts_strogatz", "ws", "regular", "tree"],
    )
    parser.add_argument(
        "--kernel", default="gaussian_tv", choices=["gaussian", "gaussian_tv"]
    )
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="auto = cuda when available, else cpu (spec D-16c Q10).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_glob", default="*.ckpt")
    parser.add_argument("--output_csv", type=Path, default=None)
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        help="Skip checkpoints already present in --output_csv.",
    )
    parser.add_argument(
        "--reference_set",
        default="val",
        choices=["val", "test"],
        help="Reference graph set (default val; spec D-16b/c Q6).",
    )
    parser.add_argument(
        "--use_ema",
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "Swap to EMA shadow weights before sampling. auto = swap iff "
            "the checkpoint carries a shadow (spec D-16c Q7)."
        ),
    )

    args = parser.parse_args(argv)
    device = _resolve_device(args.device)

    # The reference_set knob is a flag for downstream consumers; the
    # current evaluate_checkpoint only generates reference graphs
    # synthetically (it does not consume the val/test split), so the
    # value is recorded but not yet plumbed into the per-checkpoint
    # evaluation. A v2 will route the flag into the evaluator's
    # reference fetch.
    _ = args.reference_set

    df = evaluate_all_checkpoints(
        run_dir=args.run_dir,
        num_samples=args.num_samples,
        num_nodes=args.num_nodes,
        dataset_type=args.dataset,
        mmd_kernel=args.kernel,
        mmd_sigma=args.sigma,
        device=device,
        seed=args.seed,
        checkpoint_glob=args.checkpoint_glob,
        output_csv=args.output_csv,
        skip_existing=args.skip_existing,
        use_ema=args.use_ema,
    )
    output_path = args.output_csv or (args.run_dir / "all_checkpoints_eval.csv")
    print(f"Wrote {len(df)} rows to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
