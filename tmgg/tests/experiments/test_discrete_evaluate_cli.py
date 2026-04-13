"""Tests for discrete checkpoint evaluation CLI helpers.

Test rationale
--------------
The evaluation helper is a small orchestration layer: it loads a checkpoint,
generates reference graphs, samples generated graphs, and delegates the metric
computation. The regression here is about module structure rather than metric
math: ``compute_mmd_metrics`` should live at module scope so callers and tests
can patch one stable symbol instead of relying on an inline import hidden in
the function body.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from tmgg.experiments.discrete_diffusion_generative import evaluate_cli


@dataclass
class _DummyMMDResults:
    degree_mmd: float = 0.1
    clustering_mmd: float = 0.2
    spectral_mmd: float = 0.3

    def to_dict(self) -> dict[str, float]:
        return {
            "degree_mmd": self.degree_mmd,
            "clustering_mmd": self.clustering_mmd,
            "spectral_mmd": self.spectral_mmd,
        }


class _DummyModule:
    def to(self, device: str) -> _DummyModule:
        assert device == "cpu"
        return self

    def eval(self) -> _DummyModule:
        return self

    def generate_graphs(self, num_samples: int) -> list[nx.Graph]:
        return [nx.path_graph(4) for _ in range(num_samples)]


def _train_tiny_discrete_checkpoint(tmp_path: Path) -> Path:
    """Train one tiny diffusion step and persist a real Lightning checkpoint.

    Test rationale
    --------------
    The evaluation CLI must load the same checkpoint format produced by the
    discrete training stack. A synthetic or hand-built checkpoint would miss
    the real hyperparameter layout that previously triggered the load failure.
    """
    from tmgg.data.data_modules.synthetic_categorical import (
        SyntheticCategoricalDataModule,
    )
    from tmgg.diffusion.noise_process import CategoricalNoiseProcess
    from tmgg.diffusion.sampler import CategoricalSampler
    from tmgg.diffusion.schedule import NoiseSchedule
    from tmgg.evaluation.graph_evaluator import GraphEvaluator
    from tmgg.models.digress.transformer_model import GraphTransformer
    from tmgg.training.lightning_modules.diffusion_module import DiffusionModule

    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename="tiny-step={step}",
        save_top_k=0,
        save_last=True,
        every_n_train_steps=1,
    )

    schedule = NoiseSchedule("cosine_iddpm", timesteps=5)
    noise_process = CategoricalNoiseProcess(
        schedule=schedule,
        x_classes=2,
        e_classes=2,
        limit_distribution="uniform",
    )

    module = DiffusionModule(
        model=GraphTransformer(
            n_layers=2,
            input_dims={"X": 2, "E": 2, "y": 0},
            hidden_mlp_dims={"X": 16, "E": 16, "y": 16},
            hidden_dims={"dx": 16, "de": 16, "dy": 16, "n_head": 2},
            output_dims={"X": 2, "E": 2, "y": 0},
            use_timestep=True,
        ),
        noise_process=noise_process,
        sampler=CategoricalSampler(),
        noise_schedule=schedule,
        evaluator=GraphEvaluator(eval_num_samples=2, kernel="gaussian", sigma=1.0),
        loss_type="cross_entropy",
        num_nodes=8,
        eval_every_n_steps=100,
    )
    datamodule = SyntheticCategoricalDataModule(
        num_nodes=8,
        num_graphs=8,
        batch_size=4,
        seed=42,
    )
    trainer = pl.Trainer(
        max_epochs=1,
        max_steps=1,
        limit_train_batches=1,
        limit_val_batches=1,
        accelerator="cpu",
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
        enable_checkpointing=True,
        callbacks=[checkpoint_callback],
        default_root_dir=tmp_path,
        num_sanity_val_steps=0,
    )
    trainer.fit(module, datamodule)

    checkpoint_path = checkpoint_dir / "last.ckpt"
    assert checkpoint_path.exists()
    return checkpoint_path


def test_evaluate_checkpoint_uses_module_level_mmd_symbol(monkeypatch) -> None:
    """evaluate_checkpoint should delegate via the module-level MMD helper.

    The test patches the symbol directly on ``evaluate_cli``. That must be
    enough to steer the evaluation path; otherwise the implementation has
    regressed to an inline import that bypasses the module surface.
    """
    captured_ref_graphs: list[object] = []
    captured_gen_graphs: list[nx.Graph] = []
    captured_kernel = ""
    captured_sigma = 0.0

    monkeypatch.setattr(
        evaluate_cli.DiffusionModule,
        "load_from_checkpoint",
        lambda path, map_location: _DummyModule(),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "generate_reference_graphs",
        lambda **kwargs: [torch.zeros(4, 4) for _ in range(kwargs["num_graphs"])],
    )
    monkeypatch.setattr(
        evaluate_cli,
        "adjacency_to_networkx",
        lambda adj: f"ref:{tuple(adj.shape)}",
    )

    def fake_compute(
        ref_graphs: list[object],
        gen_graphs: list[nx.Graph],
        *,
        kernel: str,
        sigma: float,
    ) -> _DummyMMDResults:
        nonlocal captured_ref_graphs, captured_gen_graphs
        nonlocal captured_kernel, captured_sigma

        captured_ref_graphs = ref_graphs
        captured_gen_graphs = gen_graphs
        captured_kernel = kernel
        captured_sigma = sigma
        return _DummyMMDResults()

    monkeypatch.setattr(evaluate_cli, "compute_mmd_metrics", fake_compute)

    results = evaluate_cli.evaluate_checkpoint(
        checkpoint_path="dummy.ckpt",
        dataset_type="sbm",
        num_samples=3,
        num_nodes=4,
        mmd_kernel="gaussian_tv",
        mmd_sigma=0.5,
        device="cpu",
        seed=7,
    )

    assert captured_ref_graphs == ["ref:(4, 4)"] * 3
    assert len(captured_gen_graphs) == 3
    assert captured_kernel == "gaussian_tv"
    assert captured_sigma == 0.5
    assert results["mmd_results"] == {
        "degree_mmd": 0.1,
        "clustering_mmd": 0.2,
        "spectral_mmd": 0.3,
    }


def test_evaluate_checkpoint_loads_real_diffusion_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Discrete evaluation should load a real tiny training checkpoint.

    Starting state
    --------------
    ``DiffusionModule`` checkpoints do not store the nested graph model as a
    constructor hyperparameter. The evaluation helper therefore has to rebuild
    that model from checkpoint metadata before calling
    ``load_from_checkpoint``.
    """
    checkpoint_path = _train_tiny_discrete_checkpoint(tmp_path)

    monkeypatch.setattr(
        evaluate_cli,
        "generate_reference_graphs",
        lambda **kwargs: [torch.zeros(8, 8) for _ in range(kwargs["num_graphs"])],
    )
    monkeypatch.setattr(
        evaluate_cli,
        "adjacency_to_networkx",
        lambda adj: nx.from_numpy_array(adj.numpy()),
    )
    monkeypatch.setattr(
        evaluate_cli,
        "compute_mmd_metrics",
        lambda *args, **kwargs: _DummyMMDResults(),
    )

    results = evaluate_cli.evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        dataset_type="sbm",
        num_samples=2,
        num_nodes=8,
        mmd_kernel="gaussian_tv",
        mmd_sigma=0.5,
        device="cpu",
        seed=7,
    )

    assert results["checkpoint_name"] == "last.ckpt"
    assert results["num_generated"] == 2
    assert results["mmd_results"] == {
        "degree_mmd": 0.1,
        "clustering_mmd": 0.2,
        "spectral_mmd": 0.3,
    }
