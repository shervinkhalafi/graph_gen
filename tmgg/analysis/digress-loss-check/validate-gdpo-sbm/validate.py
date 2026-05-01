"""Validate TMGG sampling with GDPO's pretrained DiGress-SBM checkpoint.

Loads the GDPO SBM checkpoint (matches Vignac's paper SBM config bit-for-bit),
runs our sampler + noise process end-to-end, and compares the resulting
metrics against Vignac's pinned numbers from ``cvignac/DiGress/README.md``.

Purpose: isolate whether sampler + noise process + extras pipeline are
correct, independent of any training-run-specific differences.

Safe-loads the checkpoint with ``weights_only=True`` (no arbitrary Python
executed at load time; satisfies the pickle-security rule in
``~/.claude/CLAUDE.md``).

Usage
-----
    uv run python3 analysis/digress-loss-check/validate-gdpo-sbm/validate.py
    uv run python3 analysis/digress-loss-check/validate-gdpo-sbm/validate.py --device cuda
    uv run python3 analysis/digress-loss-check/validate-gdpo-sbm/validate.py --num-samples 4  # smoke
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from tqdm import tqdm

from tmgg.data.data_modules.spectre_sbm import SpectreSBMDataModule
from tmgg.diffusion.noise_process import CategoricalNoiseProcess
from tmgg.diffusion.sampler import Sampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.evaluation.graph_evaluator import GraphEvaluator
from tmgg.models.digress.extra_features import ExtraFeatures
from tmgg.models.digress.transformer_model import GraphTransformer

# Vignac's SBM numbers pinned in cvignac/DiGress README (Test NLL + MMD/accuracy).
# Used as comparison targets; exact reproduction not expected because GDPO's ckpt
# was bitwise-confirmed to be an independent training run.
VIGNAC_SBM_REFERENCE: dict[str, float] = {
    "degree_mmd": 0.0060,  # README labels this "spectre" — it's the degree MMD
    "clustering_mmd": 0.0502,
    "orbit_mmd": 0.0462,
    "sbm_accuracy": 0.675,
    "uniqueness": 1.0,
    "frac_unique_non_iso": 1.0,
    "frac_unic_non_iso_valid": 0.625,
    "test_nll": 4757.903,
}


def build_graph_transformer(device: torch.device) -> GraphTransformer:
    """Instantiate a GraphTransformer matching GDPO / Vignac's SBM config.

    ``input_dims`` is the raw per-class dims before extras and timestep
    augmentation; :class:`GraphTransformer` internally calls
    ``ExtraFeatures.adjust_dims`` (+6 on X, +11 on y for ``"all"`` mode on
    abstract graphs) and appends one dim to y when ``use_timestep=True``.
    Net result matches GDPO's ``mlp_in_X.0.weight=(128,7)``,
    ``mlp_in_y.0.weight=(128,12)`` exactly.

    Output dims match GDPO's heads: 1 node class, 2 edge classes
    (absent/present), unconditional y.
    """
    # DiGress SBM paper config (configs/experiment/sbm.yaml) with the
    # exception of ``dim_ffy`` — GDPO's checkpoint was trained with the
    # upstream-DiGress default (2048) rather than Vignac's SBM override
    # (256). We match the checkpoint's actual shape to load cleanly.
    model = GraphTransformer(
        n_layers=8,
        input_dims={"X": 1, "E": 2, "y": 0},
        hidden_mlp_dims={"X": 128, "E": 64, "y": 128},
        hidden_dims={
            "dx": 256,
            "de": 64,
            "dy": 64,
            "n_head": 8,
            "dim_ffX": 256,
            "dim_ffE": 64,
            "dim_ffy": 2048,
        },
        output_dims={"X": 1, "E": 2, "y": 0},
        extra_features=ExtraFeatures("all", max_n_nodes=200),
        use_timestep=True,
    )
    model.to(device)
    return model


def load_gdpo_weights(
    model: GraphTransformer,
    ckpt_path: Path,
    device: torch.device,
) -> dict[str, list[str]]:
    """Load GDPO's bare state_dict into our GraphTransformer.

    GDPO's state_dict keys are prefixed ``model.*`` (because in GDPO's
    own Lightning wrapper the transformer lived at ``self.model``).
    Upstream DiGress prefixes the same way. Our :class:`GraphTransformer`
    wraps the inner ``_GraphTransformer`` under ``self.transformer`` —
    so the mapping is::

        GDPO: model.mlp_in_X.0.weight
        Ours: transformer.mlp_in_X.0.weight

    This function strips the ``model.`` prefix and prepends
    ``transformer.``, then loads with ``strict=True``.

    Uses ``weights_only=True`` because GDPO's file is a bare state_dict
    with no arbitrary Python metadata; this avoids running any embedded
    code at load time.
    """
    raw = torch.load(ckpt_path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise TypeError(
            f"Expected dict state_dict from {ckpt_path}, got {type(raw).__name__}"
        )

    # Partition keys. GDPO's bare state_dict contains the wrapped GraphTransformer
    # weights at ``model.*`` plus a few auxiliary buffers at ``noise_schedule.*``
    # and optionally ``sampling_metrics.*`` (molecular ckpts only; absent for SBM).
    # We consume the ``model.*`` keys into our transformer and drop the rest
    # — our noise schedule is rebuilt from scratch with the same cosine spec,
    # so its buffers are deterministic.
    remapped: dict[str, torch.Tensor] = {}
    dropped: list[str] = []
    for k, v in raw.items():
        if k.startswith("model."):
            remapped[f"transformer.{k.removeprefix('model.')}"] = v
        else:
            dropped.append(k)

    incompatible = model.load_state_dict(remapped, strict=True)
    return {
        "missing": list(incompatible.missing_keys),
        "unexpected": list(incompatible.unexpected_keys),
        "dropped_auxiliary": dropped,
    }


def init_noise_process(
    datamodule: SpectreSBMDataModule,
    device: torch.device,
) -> tuple[NoiseSchedule, CategoricalNoiseProcess]:
    """Build paper-matching noise schedule + categorical process.

    Uses cosine schedule at T=1000 and empirical-marginal stationary
    distribution (upstream DiGress SBM default per ``configs/experiment/sbm.yaml``
    with ``transition='marginal'`` on ``configs/model/discrete.yaml``).

    The empirical marginal must be initialised from the training split
    before sampling; we pull the raw PyG loader and run the estimator.
    """
    schedule = NoiseSchedule(
        schedule_type="cosine_iddpm",
        timesteps=1000,
        num_edge_classes=2,  # SBM abstract edges: absent / present
    )
    # 1 node class, 2 edge classes, 0 y classes, empirical marginal transitions.
    # With the single-class smoothing in ``count_node_classes_sparse`` the
    # default initializer handles SBM's trivial node marginal (all mass on
    # the only class) without needing a manual bypass.
    noise_process = CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=1,
        e_classes=2,
        y_classes=0,
        limit_distribution="empirical_marginal",
    )
    noise_process.initialize_from_data(datamodule.train_dataloader_raw_pyg())

    schedule.to(device)
    noise_process.to(device)
    return schedule, noise_process


def sample_from_distribution(
    model: GraphTransformer,
    noise_process: CategoricalNoiseProcess,
    datamodule: SpectreSBMDataModule,
    num_samples: int,
    device: torch.device,
    seed: int,
    batch_size: int,
) -> list[nx.Graph]:
    """Sample *num_samples* graphs with node counts drawn from the train+val size distribution.

    Upstream DiGress samples node counts from an empirical size distribution
    at inference (``configs/experiment/sbm.yaml`` + SPECTRE's variable-size
    fixture). We draw from the train split — mirrors the upstream behaviour
    exactly, using :meth:`BaseGraphDataModule.get_size_distribution`.

    Samples are generated in micro-batches of size *batch_size* to keep
    the attention intermediate ``(bs, n, n, n_head, df)`` tensor within
    memory bounds (at n=180, n_head=8, df=32 this is ~1.3 GB per batch;
    8 GB laptop GPUs can't fit more than ~8 graphs in one pass).
    """
    size_dist = datamodule.get_size_distribution("train")
    gen = torch.Generator().manual_seed(seed)
    num_nodes_all = size_dist.sample(num_samples, generator=gen)

    sampler = Sampler(assert_symmetric_e=True)
    model.eval()

    # Wrap model.forward to advance a tqdm bar per reverse step. The sampler
    # calls ``model(z_t, t=condition)`` exactly once per timestep, so the
    # bar tracks every step without modifying the sampler. Zero compute
    # overhead beyond the ``pbar.update(1)`` call.
    timesteps = noise_process.timesteps
    total_steps = sum(timesteps for _ in range(0, num_samples, batch_size))
    pbar = tqdm(total=total_steps, desc="reverse diffusion steps", dynamic_ncols=True)
    original_forward = model.forward

    def _progress_forward(*args, **kwargs):  # type: ignore[no-untyped-def]
        pbar.update(1)
        return original_forward(*args, **kwargs)

    model.forward = _progress_forward  # type: ignore[method-assign]

    try:
        all_graph_data: list = []
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            chunk_n_nodes = num_nodes_all[start:end].to(device)
            chunk_size = int(chunk_n_nodes.numel())
            pbar.set_postfix_str(
                f"chunk {1 + start // batch_size} " f"(nodes {chunk_n_nodes.tolist()})"
            )
            graph_data_list = sampler.sample(
                model=model,
                noise_process=noise_process,
                num_graphs=chunk_size,
                num_nodes=chunk_n_nodes,
                device=device,
            )
            all_graph_data.extend(graph_data_list)
            if device.type == "cuda":
                torch.cuda.empty_cache()
    finally:
        pbar.close()
        model.forward = original_forward  # type: ignore[method-assign]

    # Convert to networkx via the evaluator's canonical helper so we pick
    # up the E_class -> binary adjacency rule used by metric functions.
    tmp_evaluator = GraphEvaluator(eval_num_samples=num_samples)
    return tmp_evaluator.to_networkx_graphs(all_graph_data)


def render_report(
    metrics: dict[str, float | None],
    gen_graphs: list[nx.Graph],
    ref_graphs: list[nx.Graph],
    ckpt_path: Path,
    out_dir: Path,
    reference: dict[str, float],
    load_report: dict[str, list[str]],
    elapsed: float,
    device: str,
    seed: int,
) -> None:
    """Write metrics.json, metrics_vs_reference.png, and report.md to *out_dir*.

    The PNG is a simple side-by-side bar plot: our metric next to
    Vignac's pinned value for the three MMDs plus SBM accuracy. The
    report.md tabulates all metrics with absolute delta and a
    pass/fail column thresholded at |delta/ref| < 0.5 for MMDs and
    |delta| < 0.2 for sbm_accuracy.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # JSON dump of raw metrics
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "metrics": metrics,
                "reference": reference,
                "ckpt_path": str(ckpt_path),
                "num_generated": len(gen_graphs),
                "num_reference": len(ref_graphs),
                "elapsed_seconds": elapsed,
                "device": device,
                "seed": seed,
                "state_dict_load": load_report,
            },
            indent=2,
            sort_keys=True,
        )
    )

    # Side-by-side bar plot
    compare_keys = [
        "degree_mmd",
        "clustering_mmd",
        "spectral_mmd",
        "orbit_mmd",
        "sbm_accuracy",
        "uniqueness",
    ]
    labels = [k for k in compare_keys if metrics.get(k) is not None]
    ours = [metrics[k] for k in labels]
    ref = [reference.get(k, float("nan")) for k in labels]

    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, ours, width, label="TMGG + GDPO ckpt")
    ax.bar(x + width / 2, ref, width, label="Vignac README (original run)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("metric value (lower is better for MMD; higher for accuracy)")
    ax.set_title(
        f"SPECTRE SBM — our sampler with GDPO pretrained weights vs Vignac reference\n"
        f"n_gen={len(gen_graphs)}, n_ref={len(ref_graphs)}, device={device}"
    )
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_vs_reference.png", dpi=140)
    plt.close(fig)

    # Markdown report
    lines: list[str] = []
    lines.append("# Validate TMGG sampling with GDPO's DiGress-SBM pretrained weights")
    lines.append("")
    lines.append(f"- Checkpoint: `{ckpt_path}`")
    lines.append(f"- Device: `{device}`, seed: `{seed}`, wall-clock: `{elapsed:.1f}s`")
    lines.append(
        f"- Generated: **{len(gen_graphs)}** graphs; reference test set: "
        f"**{len(ref_graphs)}** graphs."
    )
    lines.append(
        "- State-dict load: "
        f"{len(load_report['missing'])} missing, "
        f"{len(load_report['unexpected'])} unexpected keys "
        "(both should be 0 for upstream parity)."
    )
    lines.append("")
    lines.append("## Metrics")
    lines.append("")
    lines.append("| metric | ours | Vignac (pinned) | delta | comment |")
    lines.append("|---|---|---|---|---|")
    for key in [
        "degree_mmd",
        "clustering_mmd",
        "spectral_mmd",
        "orbit_mmd",
        "sbm_accuracy",
        "planarity_accuracy",
        "uniqueness",
        "novelty",
    ]:
        ours_val = metrics.get(key)
        ref_val = reference.get(key)
        delta: float | None = None
        comment = ""
        ours_str = "—" if ours_val is None else f"{ours_val:.4f}"
        ref_str = "—" if ref_val is None else f"{ref_val:.4f}"
        if ours_val is not None and ref_val is not None:
            delta = ours_val - ref_val
            delta_str = f"{delta:+.4f}"
            if key == "sbm_accuracy":
                comment = "PASS" if abs(delta) < 0.2 else "REVIEW"
            elif key.endswith("_mmd") and ref_val > 0:
                comment = "PASS" if abs(delta / ref_val) < 0.5 else "REVIEW"
            else:
                comment = "PASS" if abs(delta) < 0.1 else "REVIEW"
        else:
            delta_str = "—"
        lines.append(f"| {key} | {ours_str} | {ref_str} | {delta_str} | {comment} |")
    lines.append("")

    # Node-count distribution sanity check
    gen_sizes = sorted(g.number_of_nodes() for g in gen_graphs)
    ref_sizes = sorted(g.number_of_nodes() for g in ref_graphs)
    lines.append("## Sanity: node-count distributions")
    lines.append("")
    lines.append(
        f"- Generated: min={min(gen_sizes)}, median={gen_sizes[len(gen_sizes)//2]}, "
        f"max={max(gen_sizes)}"
    )
    lines.append(
        f"- Reference (SPECTRE test): min={min(ref_sizes)}, "
        f"median={ref_sizes[len(ref_sizes)//2]}, max={max(ref_sizes)}"
    )
    lines.append("")
    lines.append(
        "Generated node counts are drawn from the train-split "
        "``SizeDistribution`` (matches upstream DiGress's variable-size "
        "sampling behaviour). Ranges should overlap heavily with the "
        "SPECTRE test set; otherwise the sampler is misreading the size "
        "distribution."
    )
    lines.append("")
    lines.append("## Context")
    lines.append("")
    lines.append(
        "GDPO's SBM checkpoint was bitwise-confirmed to be an "
        "independent training run from Vignac's original weights "
        "(see `.local-storage/digress-checkpoints/README.md` for the "
        "comparison methodology). Identical architecture and training "
        "config, but a fresh random init, so exact reproduction of "
        "Vignac's numbers is not expected. A `REVIEW` flag above "
        "suggests either (a) a real bug in our sampler / noise process "
        "/ extras pipeline, or (b) larger-than-typical run-to-run "
        "variance. Look at the `load_report.missing`/`unexpected` "
        "counts first — if non-zero, that's the primary parity signal."
    )
    lines.append("")

    (out_dir / "report.md").write_text("\n".join(lines))


def run(
    *,
    ckpt_path: Path,
    out_dir: Path,
    num_samples: int = 40,
    device_str: str = "cpu",
    seed: int = 42,
    batch_size: int = 4,
) -> dict:
    """Run the full validate-sample-evaluate-report pipeline end-to-end.

    Factored out of :func:`main` so the same logic can be invoked from a
    Modal function (``modal_app.py``) without re-parsing argparse.

    Returns the metrics dict as produced by :class:`GraphEvaluator`.
    """
    ckpt_path = Path(ckpt_path).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found at {ckpt_path}")

    device = torch.device(device_str)
    torch.manual_seed(seed)

    t_start = time.perf_counter()

    print("[1/5] Building SpectreSBMDataModule (auto-downloads fixture if missing)...")
    datamodule = SpectreSBMDataModule(batch_size=12, num_workers=0)
    datamodule.setup()
    # get_reference_graphs returns list[GraphData] post 2026-05-01;
    # this script is nx-native (uses g.number_of_nodes throughout).
    ref_graphs = [
        gd.to_networkx() for gd in datamodule.get_reference_graphs("test", 40)
    ]
    print(
        f"      Reference test graphs: {len(ref_graphs)} "
        f"(node count range {min(g.number_of_nodes() for g in ref_graphs)} — "
        f"{max(g.number_of_nodes() for g in ref_graphs)})"
    )

    print("[2/5] Building GraphTransformer with SBM paper config...")
    model = build_graph_transformer(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"      GraphTransformer parameters: {num_params / 1e6:.2f} M")

    print(f"[3/5] Loading GDPO state_dict from {ckpt_path} (weights_only=True)...")
    load_report = load_gdpo_weights(model, ckpt_path, device)
    print(
        f"      Missing keys: {len(load_report['missing'])}, "
        f"Unexpected keys: {len(load_report['unexpected'])}, "
        f"Dropped auxiliary: {len(load_report['dropped_auxiliary'])}"
    )
    if load_report["dropped_auxiliary"]:
        print(
            f"      Dropped (auxiliary non-transformer buffers, safe): "
            f"{load_report['dropped_auxiliary'][:5]}"
        )
    if load_report["missing"] or load_report["unexpected"]:
        print(
            "      WARN: state_dict did not load cleanly.\n"
            f"      Missing sample: {load_report['missing'][:3]}\n"
            f"      Unexpected sample: {load_report['unexpected'][:3]}"
        )

    print(
        "[4/5] Building noise schedule + categorical process, fitting empirical marginals..."
    )
    _schedule, noise_process = init_noise_process(datamodule, device)

    print(
        f"[4b/5] Sampling {num_samples} graphs via reverse diffusion "
        "(T=1000; this is the slow step)..."
    )
    gen_graphs = sample_from_distribution(
        model=model,
        noise_process=noise_process,
        datamodule=datamodule,
        num_samples=num_samples,
        device=device,
        seed=seed,
        batch_size=batch_size,
    )
    print(
        f"      Generated {len(gen_graphs)} graphs. "
        f"Node count range: "
        f"{min(g.number_of_nodes() for g in gen_graphs)} — "
        f"{max(g.number_of_nodes() for g in gen_graphs)}"
    )

    # Persist samples alongside metrics so downstream analysis can re-score
    # without re-sampling. JSONL format (one graph per line as num_nodes +
    # sorted edge list) keeps us off pickle and is trivial to load with
    # networkx.from_edgelist.
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w") as f:
        for graph in gen_graphs:
            edges = sorted((int(min(u, v)), int(max(u, v))) for u, v in graph.edges())
            f.write(
                json.dumps({"num_nodes": graph.number_of_nodes(), "edges": edges})
                + "\n"
            )
    print(f"      Wrote samples to {samples_path} ({len(gen_graphs)} graphs)")

    print(
        "[5/5] Running GraphEvaluator (MMDs + SBM accuracy + uniqueness + planarity)..."
    )
    evaluator = GraphEvaluator(
        eval_num_samples=max(num_samples, len(ref_graphs)),
        # Paper-matching bandwidths (upstream ships clustering_sigma=0.1,
        # others at 1.0 for gaussian_tv; see GraphEvaluator docstring).
        kernel="gaussian_tv",
        sigma=1.0,
        clustering_sigma=0.1,
    )
    results = evaluator.evaluate(ref_graphs, gen_graphs)
    if results is None:
        raise RuntimeError("evaluator returned None (fewer than 2 graphs).")
    metrics = asdict(results)

    elapsed = time.perf_counter() - t_start

    print(f"      Metrics: {json.dumps(metrics, indent=2, default=str)}")
    print(f"      Elapsed: {elapsed:.1f}s")

    render_report(
        metrics=metrics,
        gen_graphs=gen_graphs,
        ref_graphs=ref_graphs,
        ckpt_path=ckpt_path,
        out_dir=out_dir,
        reference=VIGNAC_SBM_REFERENCE,
        load_report=load_report,
        elapsed=elapsed,
        device=str(device),
        seed=seed,
    )
    print(f"Wrote report to {out_dir / 'report.md'}")
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=Path(".local-storage/digress-checkpoints/gdpo_sbm/gdpo_sbm.ckpt"),
        help="Path to GDPO SBM checkpoint (bare state_dict).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
        help="Directory for metrics.json, report.md, PNG.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=40,
        help="Number of graphs to generate (defaults to SPECTRE test-set size).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="'cpu' or 'cuda'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for node-count sampling and reverse-diffusion.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help=(
            "Micro-batch size for sampling. Attention memory scales as "
            "bs*n^2*n_head; 4 graphs fits 8 GB VRAM at n=180. Raise on "
            "larger GPUs, lower if still OOM."
        ),
    )
    args = parser.parse_args()
    try:
        run(
            ckpt_path=args.ckpt,
            out_dir=args.out_dir,
            num_samples=args.num_samples,
            device_str=args.device,
            seed=args.seed,
            batch_size=args.batch_size,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
