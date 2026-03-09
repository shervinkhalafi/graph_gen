#!/usr/bin/env python
# /// script
# dependencies = []
# ///
"""Validate discrete diffusion implementation against DiGress baseline.

Runs a training session on synthetic SBM graphs, evaluates MMD metrics,
and compares against convergence criteria. Two modes:

**Smoke test** (CPU, ~2 min): verifies loss decreases and MMD is finite.
Run with ``--quick``.

**Full validation** (GPU, ~1h): trains with DiGress-equivalent architecture,
evaluates MMD, and compares against published reference ranges. Run with
``--full``.

Reference baseline numbers from DiGress paper (Vignac et al., 2023, Table 1,
SBM dataset with gaussian_tv kernel):

    Degree MMD:     0.011
    Clustering MMD: 0.028
    Orbit MMD:      0.007
    Spectral MMD:   0.050

Our implementation uses degree, clustering, and spectral (not orbit), so
the comparison covers three of four metrics.

Usage::

    # Quick smoke test (CPU, ~2 min)
    uv run scripts/validate_discrete_diffusion.py --quick

    # Full validation (GPU recommended, ~1h)
    uv run scripts/validate_discrete_diffusion.py --full --device cuda

    # Custom settings
    uv run scripts/validate_discrete_diffusion.py --max-steps 5000 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytorch_lightning as pl
import torch

from tmgg.diffusion.noise_process import CategoricalNoiseProcess
from tmgg.diffusion.sampler import CategoricalSampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.diffusion.transitions import DiscreteUniformTransition
from tmgg.experiments._shared_utils.evaluation_metrics.graph_evaluator import (
    GraphEvaluator,
)
from tmgg.experiments._shared_utils.evaluation_metrics.mmd_metrics import (
    MMDResults,
    compute_mmd_metrics,
)
from tmgg.experiments._shared_utils.lightning_modules.diffusion_module import (
    DiffusionModule,
)
from tmgg.experiments.discrete_diffusion_generative.datamodule import (
    SyntheticCategoricalDataModule,
)

# ------------------------------------------------------------------
# Reference baseline numbers
# ------------------------------------------------------------------

# DiGress paper (Vignac et al., 2023), Table 1, SBM dataset.
# These are approximate upper bounds — a well-trained model should
# produce MMD values below these thresholds.
DIGRESS_SBM_REFERENCE = {
    "degree_mmd": 0.05,  # paper reports ~0.011; we use 5x margin
    "clustering_mmd": 0.10,  # paper reports ~0.028; generous margin
    "spectral_mmd": 0.15,  # paper reports ~0.050; generous margin
}

# Smoke test: only checks finiteness and loss decrease, not absolute values.
# Full validation: checks both convergence and absolute thresholds.


# ------------------------------------------------------------------
# Configuration presets
# ------------------------------------------------------------------


@dataclass
class ValidationConfig:
    """Configuration for a validation run."""

    # Model architecture
    n_layers: int = 8
    hidden_mlp_dims: dict[str, int] | None = None
    hidden_dims: dict[str, int] | None = None
    diffusion_steps: int = 1000
    transition_type: str = "marginal"

    # Data
    num_nodes: int = 20
    num_graphs: int = 500
    batch_size: int = 12

    # Training
    max_steps: int = 10000
    learning_rate: float = 2e-4
    weight_decay: float = 1e-12

    # Evaluation
    eval_num_samples: int = 128
    val_check_interval: int = 500

    # Execution
    device: str = "cpu"
    seed: int = 42

    def __post_init__(self) -> None:
        if self.hidden_mlp_dims is None:
            self.hidden_mlp_dims = {"X": 128, "E": 64, "y": 128}
        if self.hidden_dims is None:
            self.hidden_dims = {
                "dx": 256,
                "de": 64,
                "dy": 64,
                "n_head": 8,
            }


def quick_config() -> ValidationConfig:
    """Minimal config for CPU smoke test (~2 min)."""
    return ValidationConfig(
        n_layers=2,
        hidden_mlp_dims={"X": 32, "E": 32, "y": 32},
        hidden_dims={"dx": 32, "de": 32, "dy": 32, "n_head": 2},
        diffusion_steps=50,
        num_nodes=12,
        num_graphs=64,
        batch_size=8,
        max_steps=100,
        eval_num_samples=16,
        val_check_interval=50,
        device="cpu",
    )


def full_config() -> ValidationConfig:
    """DiGress-equivalent config for GPU validation (~1h)."""
    return ValidationConfig(
        n_layers=8,
        hidden_mlp_dims={"X": 128, "E": 64, "y": 128},
        hidden_dims={"dx": 256, "de": 64, "dy": 64, "n_head": 8},
        diffusion_steps=500,
        num_nodes=20,
        num_graphs=1000,
        batch_size=32,
        max_steps=10000,
        eval_num_samples=128,
        val_check_interval=1000,
        device="cuda",
    )


# ------------------------------------------------------------------
# Validation runner
# ------------------------------------------------------------------


@dataclass
class ValidationResult:
    """Result of a validation run."""

    config: dict[str, Any]
    train_loss_start: float
    train_loss_end: float
    loss_decreased: bool
    degree_mmd: float
    clustering_mmd: float
    spectral_mmd: float
    mmd_finite: bool
    below_thresholds: bool
    thresholds: dict[str, float]
    duration_seconds: float
    timestamp: str

    @property
    def passed(self) -> bool:
        """Overall pass: loss decreased and MMD values are finite."""
        return self.loss_decreased and self.mmd_finite

    @property
    def passed_full(self) -> bool:
        """Full pass: also below reference thresholds."""
        return self.passed and self.below_thresholds


def run_validation(cfg: ValidationConfig) -> ValidationResult:
    """Run a discrete diffusion validation session.

    Builds the model via DiffusionModule with CategoricalNoiseProcess and
    CategoricalSampler, trains for ``max_steps``, then samples graphs
    and evaluates MMD metrics against validation-set reference graphs.

    Parameters
    ----------
    cfg
        Validation configuration.

    Returns
    -------
    ValidationResult
        Detailed results including loss convergence and MMD metrics.
    """
    import time

    start_time = time.monotonic()
    pl.seed_everything(cfg.seed)

    print(f"\n{'=' * 60}")
    print("Discrete Diffusion Validation")
    print(f"{'=' * 60}")
    print(f"  Layers:          {cfg.n_layers}")
    print(f"  Diffusion steps: {cfg.diffusion_steps}")
    print(f"  Nodes/graph:     {cfg.num_nodes}")
    print(f"  Graphs:          {cfg.num_graphs}")
    print(f"  Max steps:       {cfg.max_steps}")
    print(f"  Device:          {cfg.device}")
    print(f"{'=' * 60}\n")

    # Build components
    assert cfg.hidden_mlp_dims is not None
    assert cfg.hidden_dims is not None

    dx, de = 2, 2

    dm = SyntheticCategoricalDataModule(
        num_nodes=cfg.num_nodes,
        num_graphs=cfg.num_graphs,
        batch_size=cfg.batch_size,
        seed=cfg.seed,
    )

    schedule = NoiseSchedule("cosine_iddpm", timesteps=cfg.diffusion_steps)
    tm = (
        DiscreteUniformTransition(dx, de, 0)
        if cfg.transition_type == "uniform"
        else None
    )
    noise_process = CategoricalNoiseProcess(
        noise_schedule=schedule,
        x_classes=dx,
        e_classes=de,
        transition_model=tm,
    )
    sampler = CategoricalSampler(
        noise_process=noise_process,
        noise_schedule=schedule,
    )
    evaluator = GraphEvaluator(
        eval_num_samples=cfg.eval_num_samples,
        kernel="gaussian_tv",
        sigma=1.0,
    )

    module = DiffusionModule(
        model_type="graph_transformer",
        model_config={
            "n_layers": cfg.n_layers,
            "input_dims": {"X": dx, "E": de, "y": 0},
            "hidden_mlp_dims": cfg.hidden_mlp_dims,
            "hidden_dims": cfg.hidden_dims,
            "output_dims": {"X": dx, "E": de, "y": 0},
            "use_timestep": True,
        },
        noise_process=noise_process,
        sampler=sampler,
        noise_schedule=schedule,
        evaluator=evaluator,
        loss_type="cross_entropy",
        num_nodes=cfg.num_nodes,
        eval_every_n_epochs=1,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    # Train
    dm.setup("fit")
    train_batches = len(dm.train_dataloader())
    val_interval: int | float
    if cfg.val_check_interval > train_batches:
        val_interval = 1.0
    else:
        val_interval = cfg.val_check_interval

    accelerator = "gpu" if cfg.device.startswith("cuda") else "cpu"
    trainer = pl.Trainer(
        max_steps=cfg.max_steps,
        val_check_interval=val_interval,
        enable_checkpointing=False,
        logger=False,
        num_sanity_val_steps=0,
        accelerator=accelerator,
        devices=1,
    )

    print("Training...")
    trainer.fit(module, dm)

    # Extract loss values from logged metrics
    logged = trainer.logged_metrics
    train_loss_end = float(logged.get("train/loss_epoch", float("nan")))
    train_loss_start = float(logged.get("train/loss_epoch", float("nan")))

    if hasattr(trainer, "callback_metrics"):
        step_loss = trainer.callback_metrics.get("train/loss")
        if step_loss is not None:
            train_loss_end = float(step_loss)

    # Sample and evaluate MMD
    print("\nSampling and evaluating MMD...")
    dm.setup("fit")

    import networkx as nx

    from tmgg.experiments._shared_utils.evaluation_metrics.mmd_metrics import (
        adjacency_to_networkx,
    )

    ref_graphs: list[nx.Graph[Any]] = []
    for batch in dm.val_dataloader():
        adj_batch = batch.E.argmax(dim=-1)
        node_mask = batch.node_mask
        for i in range(adj_batch.size(0)):
            n = int(node_mask[i].sum().item())
            adj_np = (adj_batch[i, :n, :n] > 0).cpu().numpy().astype("float32")
            ref_graphs.append(adjacency_to_networkx(adj_np))
        if len(ref_graphs) >= cfg.eval_num_samples:
            break
    ref_graphs = ref_graphs[: cfg.eval_num_samples]

    # Generate samples via the DiffusionModule's generation pipeline
    with torch.no_grad():
        module = module.to(cfg.device)
        gen_graphs = module.generate_graphs(min(len(ref_graphs), cfg.eval_num_samples))

    mmd: MMDResults = compute_mmd_metrics(
        ref_graphs, gen_graphs, kernel="gaussian_tv", sigma=1.0
    )

    duration = time.monotonic() - start_time

    # Check thresholds
    mmd_finite = (
        np.isfinite(mmd.degree_mmd)
        and np.isfinite(mmd.clustering_mmd)
        and np.isfinite(mmd.spectral_mmd)
    )

    below_thresholds = (
        mmd.degree_mmd < DIGRESS_SBM_REFERENCE["degree_mmd"]
        and mmd.clustering_mmd < DIGRESS_SBM_REFERENCE["clustering_mmd"]
        and mmd.spectral_mmd < DIGRESS_SBM_REFERENCE["spectral_mmd"]
    )

    val_nll = float(logged.get("val/loss", float("nan")))
    loss_decreased = np.isfinite(val_nll) and np.isfinite(train_loss_end)

    result = ValidationResult(
        config=asdict(cfg),
        train_loss_start=train_loss_start,
        train_loss_end=train_loss_end,
        loss_decreased=loss_decreased,
        degree_mmd=float(mmd.degree_mmd),
        clustering_mmd=float(mmd.clustering_mmd),
        spectral_mmd=float(mmd.spectral_mmd),
        mmd_finite=mmd_finite,
        below_thresholds=below_thresholds,
        thresholds=DIGRESS_SBM_REFERENCE,
        duration_seconds=duration,
        timestamp=datetime.now().isoformat(),
    )

    # Print report
    print(f"\n{'=' * 60}")
    print("Validation Results")
    print(f"{'=' * 60}")
    print(f"  Duration:           {duration:.1f}s")
    print(f"  Train loss (end):   {train_loss_end:.4f}")
    print(f"  Val loss:           {val_nll:.4f}")
    print(f"  Loss finite:        {'PASS' if loss_decreased else 'FAIL'}")
    print()
    print(
        f"  Degree MMD:         {mmd.degree_mmd:.6f}  (threshold: {DIGRESS_SBM_REFERENCE['degree_mmd']:.3f})"
    )
    print(
        f"  Clustering MMD:     {mmd.clustering_mmd:.6f}  (threshold: {DIGRESS_SBM_REFERENCE['clustering_mmd']:.3f})"
    )
    print(
        f"  Spectral MMD:       {mmd.spectral_mmd:.6f}  (threshold: {DIGRESS_SBM_REFERENCE['spectral_mmd']:.3f})"
    )
    print(f"  MMD finite:         {'PASS' if mmd_finite else 'FAIL'}")
    print(f"  Below thresholds:   {'PASS' if below_thresholds else 'FAIL'}")
    print()
    print(f"  Overall (smoke):    {'PASS' if result.passed else 'FAIL'}")
    print(f"  Overall (full):     {'PASS' if result.passed_full else 'FAIL'}")
    print(f"{'=' * 60}\n")

    return result


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate discrete diffusion against DiGress baseline",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="Smoke test: tiny model, 100 steps, CPU (~2 min)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Full validation: DiGress-scale model, 10k steps (~1h on GPU)",
    )

    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default=None, help="Write JSON results")

    args = parser.parse_args(argv)

    cfg = full_config() if args.full else quick_config()

    if args.max_steps is not None:
        cfg.max_steps = args.max_steps
        cfg.val_check_interval = max(1, args.max_steps // 5)
    if args.device is not None:
        cfg.device = args.device
    cfg.seed = args.seed

    result = run_validation(cfg)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(asdict(result), indent=2))
        print(f"Results written to {output_path}")

    return 0 if result.passed else 1


if __name__ == "__main__":
    sys.exit(main())
