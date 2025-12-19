#!/usr/bin/env python
"""Debug script for running experiments with constant noise level.

This script helps diagnose why models might be stuck at 0.5 output by:
1. Using a single constant noise level (eliminating variance)
2. Running with enhanced debug logging
3. Testing multiple model types for comparison

Usage:
    # Test LinearPE with eps=0.1
    uv run python scripts/debug_constant_noise.py --model linear_pe --eps 0.1

    # Test identity task (eps=0)
    uv run python scripts/debug_constant_noise.py --model linear_pe --eps 0.0

    # Test all models
    uv run python scripts/debug_constant_noise.py --model all --eps 0.1

    # Test baselines only
    uv run python scripts/debug_constant_noise.py --model baselines --eps 0.1
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import TensorBoardLogger

# Add src to path for local imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tmgg.experiment_utils import DebugCallback
from tmgg.experiment_utils.data import add_digress_noise
from tmgg.models.baselines import LinearBaseline, MLPBaseline
from tmgg.models.spectral_denoisers import (
    GraphFilterBank,
    LinearPE,
    SelfAttentionDenoiser,
)


def create_model(model_type: str, max_nodes: int = 32, k: int = 8) -> torch.nn.Module:
    """Create model by type name."""
    models: dict[str, Callable[[], torch.nn.Module]] = {
        "linear_pe": lambda: LinearPE(k=k, max_nodes=max_nodes),
        "filter_bank": lambda: GraphFilterBank(k=k, polynomial_degree=5),
        "self_attention": lambda: SelfAttentionDenoiser(k=k, d_k=64),
        "linear_baseline": lambda: LinearBaseline(max_nodes=max_nodes),
        "mlp_baseline": lambda: MLPBaseline(max_nodes=max_nodes, hidden_dim=256),
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. Valid: {list(models.keys())}"
        )
    return models[model_type]()


class DebugLightningModule(pl.LightningModule):
    """Minimal Lightning module for debugging with constant noise."""

    model: torch.nn.Module
    eps: float
    learning_rate: float

    def __init__(
        self,
        model: torch.nn.Module,
        eps: float = 0.1,
        learning_rate: float = 1e-3,
    ) -> None:
        super().__init__()
        self.model = model
        self.eps = eps
        self.learning_rate = learning_rate
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def training_step(
        self, batch: torch.Tensor, batch_idx: int
    ) -> dict[str, torch.Tensor]:
        # Add constant noise
        batch_noisy = add_digress_noise(batch, p=self.eps) if self.eps > 0 else batch

        # Forward pass
        logits = self(batch_noisy)
        loss = F.binary_cross_entropy_with_logits(logits, batch)

        # Log detailed stats
        with torch.no_grad():
            probs = torch.sigmoid(logits)
            self.log_dict(
                {
                    "train_loss": loss,
                    "logit_mean": logits.mean(),
                    "logit_std": logits.std(),
                    "logit_abs_mean": logits.abs().mean(),
                    "pred_mean": probs.mean(),
                    "pred_std": probs.std(),
                    "mae": (probs - batch).abs().mean(),
                },
                prog_bar=True,
            )

        return {"loss": loss, "logits": logits}

    def validation_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        batch_noisy = add_digress_noise(batch, p=self.eps) if self.eps > 0 else batch

        logits = self(batch_noisy)
        loss = F.binary_cross_entropy_with_logits(logits, batch)

        with torch.no_grad():
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            accuracy = (preds == batch).float().mean()

        self.log_dict(
            {
                "val_loss": loss,
                "val_accuracy": accuracy,
                "val_pred_mean": probs.mean(),
            },
            prog_bar=True,
        )

        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-2,
        )


def run_debug_experiment(
    model_type: str,
    eps: float,
    max_steps: int = 500,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    num_nodes: int = 32,
) -> DebugLightningModule:
    """Run a single debug experiment."""
    print(f"\n{'='*60}")
    print(f"Model: {model_type}, Noise: eps={eps}")
    print(f"{'='*60}")

    # Create model
    model = create_model(model_type, max_nodes=num_nodes)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create Lightning module
    pl_module = DebugLightningModule(
        model=model,
        eps=eps,
        learning_rate=learning_rate,
    )

    # Create simple data (random graphs)
    _ = torch.manual_seed(42)
    train_graphs: list[torch.Tensor] = []
    for _ in range(100):
        A = (torch.rand(num_nodes, num_nodes) > 0.5).float()
        A = (A + A.T) / 2
        _ = A.fill_diagonal_(0)
        train_graphs.append(A)
    train_data = torch.stack(train_graphs)

    val_graphs: list[torch.Tensor] = []
    for _ in range(20):
        A = (torch.rand(num_nodes, num_nodes) > 0.5).float()
        A = (A + A.T) / 2
        _ = A.fill_diagonal_(0)
        val_graphs.append(A)
    val_data = torch.stack(val_graphs)

    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size)

    # Create trainer with debug callback
    logger = TensorBoardLogger("debug_logs", name=f"{model_type}_eps{eps}")
    debug_callback = DebugCallback(log_interval=10)

    trainer = pl.Trainer(
        max_steps=max_steps,
        accelerator="auto",
        logger=logger,
        callbacks=[debug_callback],
        enable_progress_bar=True,
        log_every_n_steps=10,
        val_check_interval=100,
    )

    # Train
    _ = trainer.fit(pl_module, train_loader, val_loader)

    # Final evaluation
    print(f"\nFinal Results for {model_type}:")
    _ = pl_module.eval()
    with torch.no_grad():
        sample = val_data[:4]
        sample_noisy = add_digress_noise(sample, p=eps) if eps > 0 else sample

        logits = pl_module(sample_noisy)
        probs = torch.sigmoid(logits)

        print(f"  Logit range: [{logits.min():.3f}, {logits.max():.3f}]")
        print(f"  Logit mean: {logits.mean():.4f}, std: {logits.std():.4f}")
        print(f"  Pred range: [{probs.min():.3f}, {probs.max():.3f}]")
        print(f"  Pred mean: {probs.mean():.4f}")

        accuracy = ((probs > 0.5).float() == sample).float().mean()
        print(f"  Accuracy: {accuracy:.2%}")

    return pl_module


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug training with constant noise")
    parser.add_argument(
        "--model",
        type=str,
        default="linear_pe",
        choices=[
            "linear_pe",
            "filter_bank",
            "self_attention",
            "linear_baseline",
            "mlp_baseline",
            "all",
            "baselines",
            "spectral",
        ],
        help="Model type to test",
    )
    parser.add_argument("--eps", type=float, default=0.1, help="Noise level")
    parser.add_argument("--steps", type=int, default=500, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--nodes", type=int, default=32, help="Number of nodes")

    args = parser.parse_args()

    # Determine which models to run
    models: list[str]
    if args.model == "all":
        models = [
            "linear_pe",
            "filter_bank",
            "self_attention",
            "linear_baseline",
            "mlp_baseline",
        ]
    elif args.model == "baselines":
        models = ["linear_baseline", "mlp_baseline"]
    elif args.model == "spectral":
        models = ["linear_pe", "filter_bank", "self_attention"]
    else:
        models = [args.model]

    # Run experiments
    results: dict[str, DebugLightningModule] = {}
    for model_type in models:
        module = run_debug_experiment(
            model_type=model_type,
            eps=args.eps,
            max_steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            num_nodes=args.nodes,
        )
        results[model_type] = module

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for model_type, module in results.items():
        print(f"\n{model_type}:")
        with torch.no_grad():
            # Quick check on a sample
            sample = torch.rand(1, args.nodes, args.nodes)
            sample = (sample + sample.transpose(-2, -1)) / 2
            logits = module(sample)
            print(f"  Output logit mean: {logits.mean():.4f}")
            print(f"  Output logit std: {logits.std():.4f}")
            print(f"  Stuck at 0.5?: {logits.abs().mean() < 0.1}")


if __name__ == "__main__":
    main()
