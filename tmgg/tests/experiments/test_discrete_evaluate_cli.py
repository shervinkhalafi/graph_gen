"""Tests for discrete checkpoint evaluation CLI helpers.

Test rationale
--------------
The evaluation helper is a small orchestration layer: it loads a checkpoint,
optionally swaps in EMA shadow weights, fetches reference graphs from the
training-time datamodule (``--reference_set val|test``), samples generated
graphs, and delegates the metric computation. Tests cover:

* The MMD compute symbol is module-level so callers can patch it.
* The evaluator drives the datamodule through ``setup`` + ``get_reference_graphs``
  rather than the deprecated synthetic ``generate_reference_graphs`` path
  (regression for parity D-16c cleanup; the synthetic shortcut bypassed
  the val/test distinction).
* ``--reference_set`` flag plumbs through to the datamodule call.
* ``--use_ema {auto,true,false}`` semantics: auto swaps when shadow
  present, true raises when shadow absent, false skips the swap.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import networkx as nx
import pytest
import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
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
    """Stand-in for DiffusionModule with enough surface for evaluate_checkpoint."""

    def __init__(self) -> None:
        self.model = torch.nn.Linear(2, 2)

    def to(self, device: str) -> _DummyModule:
        assert device == "cpu"
        return self

    def eval(self) -> _DummyModule:
        return self

    def generate_graphs(self, num_samples: int) -> list[nx.Graph[Any]]:
        return [nx.path_graph(4) for _ in range(num_samples)]


class _DummyDatamodule:
    """Records setup calls and serves canned NetworkX references."""

    def __init__(self, ref_graphs: list[nx.Graph[Any]]) -> None:
        self._refs = ref_graphs
        self.setup_calls: list[str] = []
        self.get_reference_calls: list[tuple[str, int]] = []

    def setup(self, stage: str) -> None:
        self.setup_calls.append(stage)

    def get_reference_graphs(self, split: str, max_graphs: int) -> list[nx.Graph[Any]]:
        self.get_reference_calls.append((split, max_graphs))
        return self._refs[:max_graphs]


def _write_dummy_config_yaml(checkpoint_path: Path, *, _target_: str) -> Path:
    """Write a sibling config.yaml with a `data:` block containing _target_.

    The CLI walks one directory up from the checkpoint to find this file.
    Returns the config path written.
    """
    run_dir = checkpoint_path.parent.parent
    run_dir.mkdir(parents=True, exist_ok=True)
    config = OmegaConf.create({"data": {"_target_": _target_, "batch_size": 4}})
    config_path = run_dir / "config.yaml"
    OmegaConf.save(config, config_path)
    return config_path


# ---------------------------------------------------------------------------
# Refactor regression tests: datamodule path, no synthetic fallback
# ---------------------------------------------------------------------------


class TestDatamoduleReferencePath:
    """evaluate_checkpoint must drive the training-time datamodule."""

    def test_uses_val_dataloader_when_reference_set_val(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reference_set='val' -> datamodule.setup('fit') + get_reference_graphs('val', N)."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10-val_loss=0.5.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        _write_dummy_config_yaml(ckpt_path, _target_="dummy.module.DataModule")

        dm = _DummyDatamodule(ref_graphs=[nx.cycle_graph(5) for _ in range(3)])

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        monkeypatch.setattr(evaluate_cli, "torch", _build_torch_stub(callbacks={}))
        monkeypatch.setattr(evaluate_cli.hydra.utils, "instantiate", lambda cfg: dm)
        monkeypatch.setattr(
            evaluate_cli, "compute_mmd_metrics", lambda *a, **kw: _DummyMMDResults()
        )

        result = evaluate_cli.evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            num_samples=3,
            reference_set="val",
        )

        assert dm.setup_calls == ["fit"]
        assert dm.get_reference_calls == [("val", 3)]
        assert result["reference_set"] == "val"
        assert result["num_reference"] == 3

    def test_uses_test_dataloader_when_reference_set_test(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """reference_set='test' -> datamodule.setup('test') + get_reference_graphs('test', N)."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        _write_dummy_config_yaml(ckpt_path, _target_="dummy.module.DataModule")

        dm = _DummyDatamodule(ref_graphs=[nx.path_graph(4) for _ in range(2)])

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        monkeypatch.setattr(evaluate_cli, "torch", _build_torch_stub(callbacks={}))
        monkeypatch.setattr(evaluate_cli.hydra.utils, "instantiate", lambda cfg: dm)
        monkeypatch.setattr(
            evaluate_cli, "compute_mmd_metrics", lambda *a, **kw: _DummyMMDResults()
        )

        result = evaluate_cli.evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            num_samples=2,
            reference_set="test",
        )

        assert dm.setup_calls == ["test"]
        assert dm.get_reference_calls == [("test", 2)]
        assert result["reference_set"] == "test"

    def test_missing_config_yaml_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing sibling config.yaml fails loud; no synthetic-fallback path exists."""
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        # Note: no config.yaml written.

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        monkeypatch.setattr(evaluate_cli, "torch", _build_torch_stub(callbacks={}))

        with pytest.raises(FileNotFoundError, match="No config.yaml found"):
            evaluate_cli.evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                num_samples=2,
                reference_set="val",
            )


# ---------------------------------------------------------------------------
# EMA wiring (parity D-16c Q7)
# ---------------------------------------------------------------------------


class TestEmaWiring:
    """evaluate_checkpoint honours the use_ema flag against checkpoint state."""

    def test_use_ema_true_with_no_shadow_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        _write_dummy_config_yaml(ckpt_path, _target_="dummy.module.DataModule")

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        # Checkpoint has callbacks but no EMA entry.
        monkeypatch.setattr(
            evaluate_cli,
            "torch",
            _build_torch_stub(callbacks={"ModelCheckpoint{...}": {}}),
        )

        with pytest.raises(RuntimeError, match="use_ema=true requested"):
            evaluate_cli.evaluate_checkpoint(
                checkpoint_path=ckpt_path,
                num_samples=2,
                reference_set="val",
                use_ema="true",
            )

    def test_use_ema_auto_with_no_shadow_records_inactive(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        _write_dummy_config_yaml(ckpt_path, _target_="dummy.module.DataModule")

        dm = _DummyDatamodule(ref_graphs=[nx.cycle_graph(5)])

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        monkeypatch.setattr(evaluate_cli, "torch", _build_torch_stub(callbacks={}))
        monkeypatch.setattr(evaluate_cli.hydra.utils, "instantiate", lambda cfg: dm)
        monkeypatch.setattr(
            evaluate_cli, "compute_mmd_metrics", lambda *a, **kw: _DummyMMDResults()
        )

        result = evaluate_cli.evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            num_samples=1,
            reference_set="val",
            use_ema="auto",
        )

        assert result["ema_active"] is False

    def test_use_ema_false_skips_swap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        ckpt_path = ckpt_dir / "model-step=10.ckpt"
        ckpt_path.write_bytes(b"\x80\x04N.")
        _write_dummy_config_yaml(ckpt_path, _target_="dummy.module.DataModule")

        dm = _DummyDatamodule(ref_graphs=[nx.cycle_graph(5)])

        monkeypatch.setattr(
            evaluate_cli, "_load_diffusion_module", lambda *a, **k: _DummyModule()
        )
        # Even with an EMA entry, use_ema='false' must not swap.
        monkeypatch.setattr(
            evaluate_cli,
            "torch",
            _build_torch_stub(
                callbacks={
                    "EMACallback{decay=0.999}": {
                        "shadow_params": [],
                        "decay": 0.999,
                    }
                }
            ),
        )
        monkeypatch.setattr(evaluate_cli.hydra.utils, "instantiate", lambda cfg: dm)
        monkeypatch.setattr(
            evaluate_cli, "compute_mmd_metrics", lambda *a, **kw: _DummyMMDResults()
        )

        result = evaluate_cli.evaluate_checkpoint(
            checkpoint_path=ckpt_path,
            num_samples=1,
            reference_set="val",
            use_ema="false",
        )

        assert result["ema_active"] is False


def _build_torch_stub(*, callbacks: dict[str, Any]) -> Any:
    """Return a `torch`-shaped namespace whose `torch.load` returns a stub ckpt.

    The CLI calls ``torch.load(checkpoint_path, ...)`` after Lightning's
    ``load_from_checkpoint`` to inspect the callback section. Stub it
    out with a dict containing the desired callbacks block; everything
    else (no_grad, the `torch` namespace) is forwarded to the real
    torch module so generate_graphs / model code keeps working.
    """
    real_torch = torch

    class _TorchStub:
        @staticmethod
        def load(*_a: Any, **_k: Any) -> dict[str, Any]:
            return {"callbacks": callbacks}

        # Forward attributes used downstream.
        def __getattr__(self, name: str) -> Any:
            return getattr(real_torch, name)

    return _TorchStub()


# ---------------------------------------------------------------------------
# Real-checkpoint integration
# ---------------------------------------------------------------------------


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
        num_graphs=40,
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

    # Mirror run_experiment: persist the data block as sibling config.yaml
    # so evaluate_checkpoint can rebuild the datamodule.
    config = OmegaConf.create(
        {
            "data": {
                "_target_": "tmgg.data.data_modules.synthetic_categorical.SyntheticCategoricalDataModule",
                "num_nodes": 8,
                "num_graphs": 40,
                "batch_size": 4,
                "seed": 42,
            }
        }
    )
    OmegaConf.save(config, tmp_path / "config.yaml")
    return checkpoint_path


def test_evaluate_checkpoint_loads_real_diffusion_checkpoint(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Discrete evaluation should load a real tiny training checkpoint.

    Starting state
    --------------
    ``DiffusionModule`` checkpoints do not store the nested graph model as a
    constructor hyperparameter. The evaluation helper rebuilds that model
    from checkpoint metadata before calling ``load_from_checkpoint``, then
    rebuilds the datamodule from the sibling config.yaml.
    """
    checkpoint_path = _train_tiny_discrete_checkpoint(tmp_path)

    monkeypatch.setattr(
        evaluate_cli,
        "compute_mmd_metrics",
        lambda *args, **kwargs: _DummyMMDResults(),
    )

    results = evaluate_cli.evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        num_samples=2,
        reference_set="val",
        mmd_kernel="gaussian_tv",
        mmd_sigma=0.5,
        device="cpu",
    )

    assert results["checkpoint_name"] == "last.ckpt"
    assert results["num_generated"] >= 1
    assert results["reference_set"] == "val"
    assert results["mmd_results"] == {
        "degree_mmd": 0.1,
        "clustering_mmd": 0.2,
        "spectral_mmd": 0.3,
    }
