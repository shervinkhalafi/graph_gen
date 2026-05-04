"""Launch the train + eval profile pair in parallel on Modal.

Two ``@app.function``s in ``tmgg.modal._profile_functions`` are
``.spawn()``-ed (fire-and-forget); their FunctionCall ids are printed
so progress can be polled via Modal's dashboard. When both complete,
artefacts live under::

    <tmgg-outputs volume>/profiles/<run_tag>/{train,eval}/{trace.json, summary.txt}

Pull them back to the host with ``modal volume get tmgg-outputs
profiles/<run_tag>/ ./local_profiles`` (the leading ``/data/outputs``
maps to the volume root, so volume-side paths drop that prefix).

Defaults profile the round-5 winning config (Greedy: n_layers=4,
dx=16, dim_ffX=16, dim_ffy=32) for training, and the latest Greedy
checkpoint for eval. Override with ``--overrides`` and
``--checkpoint-path``.

Invocation::

    doppler run -- uv run modal deploy -m tmgg.modal._profile_functions
    doppler run -- uv run python -m scripts.profile.launch_profile

Use ``--no-eval`` or ``--no-train`` to spawn just one half.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

# Greedy config — matches docs/.../round-5.yaml's combined_L4_dx16 pod.
GREEDY_OVERRIDES_BASE = [
    "models/discrete@model=discrete_sbm_official",
    "+data=spectre_sbm",
    "trainer.precision=bf16-mixed",
    "trainer.max_steps=100",
    "model.model.n_layers=4",
    "model.model.hidden_dims.dx=16",
    "model.model.hidden_dims.de=8",
    "model.model.hidden_dims.dy=8",
    "model.model.hidden_dims.n_head=8",
    "model.model.hidden_dims.dim_ffX=16",
    "model.model.hidden_dims.dim_ffE=8",
    "model.model.hidden_dims.dim_ffy=32",
    "+model.model.extra_features._target_=tmgg.models.digress.extra_features.ExtraFeatures",
    "+model.model.extra_features.extra_features_type=all",
    "+model.model.extra_features.max_n_nodes=200",
]

DEFAULT_GREEDY_CKPT = (
    "/data/outputs/discrete_diffusion/discrete_diffusion_DiffusionModule_dSpectreSBMDataModule_lr2e-4_wd1e-12_L4_s0_fresh_20260501T111030/"
    "checkpoints/last.ckpt"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _ = p.add_argument(
        "--run-tag",
        default=dt.datetime.now(dt.UTC).strftime("greedy_%Y%m%dT%H%M%S"),
        help="Output subdirectory under /outputs/profiles/.",
    )
    _ = p.add_argument(
        "--overrides",
        default=None,
        help=(
            "Hydra override list (comma-separated). Defaults to round-5 "
            "Greedy config."
        ),
    )
    _ = p.add_argument(
        "--checkpoint-path",
        default=DEFAULT_GREEDY_CKPT,
        help="Modal-volume path to the *.ckpt for eval profile.",
    )
    _ = p.add_argument("--num-train-steps", type=int, default=100)
    _ = p.add_argument("--warmup-steps", type=int, default=5)
    _ = p.add_argument("--active-steps", type=int, default=20)
    _ = p.add_argument("--num-eval-samples", type=int, default=32)
    _ = p.add_argument("--no-train", action="store_true")
    _ = p.add_argument("--no-eval", action="store_true")
    _ = p.add_argument(
        "--eval-compile",
        action="store_true",
        help="Wrap module.model in torch.compile post-load for the eval profile.",
    )
    _ = p.add_argument(
        "--eval-compile-mode",
        default="default",
        help="torch.compile mode for the eval profile (default, reduce-overhead, max-autotune).",
    )
    _ = p.add_argument(
        "--eval-sample-chunk-size",
        type=int,
        default=None,
        help=(
            "Chunk the sample loop into batches of this size (default: "
            "no chunking). Set to the train batch_size when running with "
            "--eval-compile so the compiled trace is reusable across the "
            "sample loop instead of recompiling for num_samples."
        ),
    )
    _ = p.add_argument(
        "--rounds-jsonl",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/rounds.jsonl"),
        help="Where to append the launched-row metadata (audit trail).",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # Lazy-import modal so the script is harmless to import for --help
    # without modal installed.
    import modal

    overrides_list = (
        [s.strip() for s in args.overrides.split(",") if s.strip()]
        if args.overrides
        else GREEDY_OVERRIDES_BASE
    )
    overrides_list = [
        o for o in overrides_list if not o.startswith("trainer.max_steps=")
    ]
    overrides_list.append(f"trainer.max_steps={args.num_train_steps}")

    output_root = f"/data/outputs/profiles/{args.run_tag}"
    print(f"# run_tag={args.run_tag}")
    print(f"# output_root={output_root}")

    spawned: list[dict[str, object]] = []

    if not args.no_train:
        profile_train = modal.Function.from_name("tmgg-profile", "profile_train")
        train_call = profile_train.spawn(
            overrides=overrides_list,
            output_dir_on_volume=f"{output_root}/train",
            num_steps=args.num_train_steps,
            warmup_steps=args.warmup_steps,
            active_steps=args.active_steps,
        )
        print(f"  TRAIN spawned: call_id={train_call.object_id}")
        spawned.append(
            {
                "kind": "profile_train_launched",
                "ts_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "run_tag": args.run_tag,
                "modal_function_call_id": train_call.object_id,
                "output_path": f"{output_root}/train",
                "overrides": overrides_list,
                "num_steps": args.num_train_steps,
            }
        )

    if not args.no_eval:
        profile_eval = modal.Function.from_name("tmgg-profile", "profile_eval")
        eval_call = profile_eval.spawn(
            checkpoint_path=args.checkpoint_path,
            output_dir_on_volume=f"{output_root}/eval",
            num_samples=args.num_eval_samples,
            compile_model=args.eval_compile,
            compile_mode=args.eval_compile_mode,
            sample_chunk_size=args.eval_sample_chunk_size,
        )
        print(f"  EVAL  spawned: call_id={eval_call.object_id}")
        spawned.append(
            {
                "kind": "profile_eval_launched",
                "ts_utc": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
                "run_tag": args.run_tag,
                "modal_function_call_id": eval_call.object_id,
                "output_path": f"{output_root}/eval",
                "checkpoint_path": args.checkpoint_path,
                "num_samples": args.num_eval_samples,
            }
        )

    if not spawned:
        print("# nothing spawned; both --no-train and --no-eval were set")
        return 1

    if args.rounds_jsonl.exists():
        with args.rounds_jsonl.open("a") as fh:
            for row in spawned:
                fh.write(json.dumps(row) + "\n")
        print(f"# audit rows appended to {args.rounds_jsonl}")

    print()
    print("# Pull artefacts when complete:")
    print(f"  modal volume get tmgg-outputs profiles/{args.run_tag}/ ./local_profiles")
    print()
    print("# Inspect the chrome traces locally:")
    print("  open chrome://tracing/ → load .json")
    print(
        f"  cat local_profiles/{args.run_tag}/train/summary.txt   "
        "# train summary table"
    )
    print(
        f"  cat local_profiles/{args.run_tag}/eval/summary.txt    "
        "# eval summary table"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
