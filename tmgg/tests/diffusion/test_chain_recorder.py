"""Tests for ChainRecorder, the per-step PMF snapshot capture for sampling.

Test rationale
--------------
:class:`tmgg.diffusion.chain_recorder.ChainRecorder` hooks the sampler's
reverse loop to record post-symmetrisation ``z_t`` snapshots at a
configurable cadence. The class is small and pure (no I/O); the
expectations are:

* Construction rejects nonsensical configs (``num_chains_to_save < 1``,
  ``snapshot_step_interval < 1``).
* :meth:`maybe_record` is gated on ``step_index %
  snapshot_step_interval``: only multiples of the interval fire.
* :meth:`maybe_record` raises when the batch is too small to slice the
  configured number of chains.
* :meth:`finalize` produces the documented schema with correct shapes,
  honouring the optional ``X_class`` field and the ``field_prefix``
  namespace (used for CompositeNoiseProcess fan-out per spec resolution
  Q5).
* GPU tensors stay on GPU during accumulation (per spec resolution Q3)
  and are moved to CPU exactly once at finalise.
* Roundtrip via ``torch.save`` / ``torch.load`` preserves the dict.

Sampler integration is tested with a tiny CategoricalSampler reverse
loop that ensures the recorder receives at least the expected number of
calls and produces a saved file that loads back to the documented
shape.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch
from torch import Tensor

from tests._helpers.graph_builders import binary_graphdata
from tmgg.data.datasets.graph_types import GraphData
from tmgg.diffusion.chain_recorder import ChainRecorder, merge_chain_snapshots
from tmgg.diffusion.noise_process import CategoricalNoiseProcess
from tmgg.diffusion.sampler import CategoricalSampler
from tmgg.diffusion.schedule import NoiseSchedule
from tmgg.models.base import GraphModel

# ---------------------------------------------------------------------------
# Helper builders
# ---------------------------------------------------------------------------


def _make_pmf_graph(
    bs: int, n: int, dx: int | None, de: int, device: str = "cpu"
) -> GraphData:
    """Build a batched GraphData with a flat-PMF E_class (and X_class if dx)."""
    node_mask = torch.ones(bs, n, dtype=torch.bool, device=device)
    e_class = torch.full((bs, n, n, de), 1.0 / de, device=device)
    x_class: Tensor | None = None
    if dx is not None:
        x_class = torch.full((bs, n, dx), 1.0 / dx, device=device)
    return GraphData(
        y=torch.zeros(bs, 0, device=device),
        node_mask=node_mask,
        X_class=x_class,
        E_class=e_class,
    )


# ---------------------------------------------------------------------------
# Construction validity
# ---------------------------------------------------------------------------


class TestConstructionValidity:
    def test_rejects_zero_chains(self) -> None:
        with pytest.raises(ValueError, match="num_chains_to_save must be >= 1"):
            ChainRecorder(num_chains_to_save=0, snapshot_step_interval=1)

    def test_rejects_negative_chains(self) -> None:
        with pytest.raises(ValueError, match="num_chains_to_save must be >= 1"):
            ChainRecorder(num_chains_to_save=-1, snapshot_step_interval=1)

    def test_rejects_zero_interval(self) -> None:
        with pytest.raises(ValueError, match="snapshot_step_interval must be >= 1"):
            ChainRecorder(num_chains_to_save=1, snapshot_step_interval=0)


# ---------------------------------------------------------------------------
# maybe_record gating + slicing
# ---------------------------------------------------------------------------


class TestMaybeRecord:
    def test_only_fires_at_snapshot_steps(self) -> None:
        rec = ChainRecorder(num_chains_to_save=2, snapshot_step_interval=3)
        graph = _make_pmf_graph(bs=4, n=5, dx=3, de=2)
        # step 0 fires (0 % 3 == 0), step 1 and 2 are skipped, step 3 fires.
        rec.maybe_record(0, graph)
        rec.maybe_record(1, graph)
        rec.maybe_record(2, graph)
        rec.maybe_record(3, graph)
        out = rec.finalize()
        assert out["E_chain"].shape[0] == 2
        assert out["step_indices"].tolist() == [0, 3]

    def test_slices_first_c_graphs(self) -> None:
        rec = ChainRecorder(num_chains_to_save=2, snapshot_step_interval=1)
        graph = _make_pmf_graph(bs=4, n=5, dx=3, de=2)
        rec.maybe_record(0, graph)
        out = rec.finalize()
        # E_chain shape: (S=1, C=2, n=5, n=5, de=2)
        assert out["E_chain"].shape == (1, 2, 5, 5, 2)
        assert out["X_chain"].shape == (1, 2, 5, 3)
        assert out["node_mask"].shape == (2, 5)

    def test_rejects_undersized_batch(self) -> None:
        rec = ChainRecorder(num_chains_to_save=4, snapshot_step_interval=1)
        graph = _make_pmf_graph(bs=2, n=5, dx=3, de=2)
        with pytest.raises(ValueError, match="batch must be at least"):
            rec.maybe_record(0, graph)

    def test_rejects_missing_e_class(self) -> None:
        rec = ChainRecorder(num_chains_to_save=1, snapshot_step_interval=1)
        graph = GraphData(
            y=torch.zeros(2, 0),
            node_mask=torch.ones(2, 4, dtype=torch.bool),
            E_feat=torch.zeros(2, 4, 4, 1),  # populates one edge field
        )
        with pytest.raises(RuntimeError, match="E_class to be populated"):
            rec.maybe_record(0, graph)

    def test_rejects_inconsistent_x_presence(self) -> None:
        rec = ChainRecorder(num_chains_to_save=1, snapshot_step_interval=1)
        with_x = _make_pmf_graph(bs=2, n=4, dx=3, de=2)
        without_x = _make_pmf_graph(bs=2, n=4, dx=None, de=2)
        rec.maybe_record(0, with_x)
        with pytest.raises(RuntimeError, match="inconsistent X_class presence"):
            rec.maybe_record(1, without_x)


# ---------------------------------------------------------------------------
# finalize semantics
# ---------------------------------------------------------------------------


class TestFinalize:
    def test_finalize_without_x_field(self) -> None:
        rec = ChainRecorder(num_chains_to_save=2, snapshot_step_interval=2)
        graph = _make_pmf_graph(bs=3, n=4, dx=None, de=2)
        rec.maybe_record(0, graph)
        rec.maybe_record(2, graph)
        out = rec.finalize()
        assert "X_chain" not in out
        assert out["E_chain"].shape == (2, 2, 4, 4, 2)
        assert out["step_indices"].dtype == torch.long
        assert out["node_mask"].dtype == torch.bool

    def test_finalize_without_records_raises(self) -> None:
        rec = ChainRecorder(num_chains_to_save=1, snapshot_step_interval=1)
        with pytest.raises(RuntimeError, match="no recorded snapshots"):
            rec.finalize()

    def test_field_prefix_namespaces_keys(self) -> None:
        rec = ChainRecorder(
            num_chains_to_save=1,
            snapshot_step_interval=1,
            field_prefix="categorical",
        )
        graph = _make_pmf_graph(bs=2, n=4, dx=3, de=2)
        rec.maybe_record(0, graph)
        out = rec.finalize()
        assert set(out.keys()) == {
            "categorical/E_chain",
            "categorical/X_chain",
            "categorical/node_mask",
            "categorical/step_indices",
        }

    def test_torch_save_load_roundtrip(self, tmp_path: Path) -> None:
        rec = ChainRecorder(num_chains_to_save=2, snapshot_step_interval=1)
        graph = _make_pmf_graph(bs=3, n=4, dx=2, de=2)
        rec.maybe_record(0, graph)
        rec.maybe_record(1, graph)
        out = rec.finalize()
        target = tmp_path / "chain.pt"
        torch.save(out, target)
        loaded: dict[str, Tensor] = torch.load(target, weights_only=True)
        assert set(loaded.keys()) == set(out.keys())
        torch.testing.assert_close(loaded["E_chain"], out["E_chain"])
        torch.testing.assert_close(loaded["X_chain"], out["X_chain"])
        torch.testing.assert_close(loaded["node_mask"], out["node_mask"])
        torch.testing.assert_close(loaded["step_indices"], out["step_indices"])


# ---------------------------------------------------------------------------
# GPU accumulation (per spec resolution Q3)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGpuAccumulation:
    def test_accumulates_on_gpu_until_finalize(self) -> None:
        rec = ChainRecorder(num_chains_to_save=2, snapshot_step_interval=1)
        graph = _make_pmf_graph(bs=2, n=4, dx=2, de=2, device="cuda")
        rec.maybe_record(0, graph)
        # Internal buffer should still live on GPU.
        assert rec._e_snapshots[0].is_cuda
        out = rec.finalize()
        # After finalize, tensors are on CPU (ready for torch.save).
        assert not out["E_chain"].is_cuda
        assert not out["node_mask"].is_cuda


# ---------------------------------------------------------------------------
# Composite key merge (spec resolution Q5)
# ---------------------------------------------------------------------------


class TestMergeChainSnapshots:
    def test_merges_disjoint_keys(self) -> None:
        cat_rec = ChainRecorder(
            num_chains_to_save=1, snapshot_step_interval=1, field_prefix="categorical"
        )
        gauss_rec = ChainRecorder(
            num_chains_to_save=1, snapshot_step_interval=1, field_prefix="gaussian"
        )
        graph = _make_pmf_graph(bs=2, n=4, dx=2, de=2)
        cat_rec.maybe_record(0, graph)
        gauss_rec.maybe_record(0, graph)
        merged = merge_chain_snapshots([cat_rec.finalize(), gauss_rec.finalize()])
        assert {
            "categorical/E_chain",
            "categorical/X_chain",
            "categorical/node_mask",
            "categorical/step_indices",
            "gaussian/E_chain",
            "gaussian/X_chain",
            "gaussian/node_mask",
            "gaussian/step_indices",
        } == set(merged.keys())

    def test_rejects_colliding_keys(self) -> None:
        rec_a = ChainRecorder(num_chains_to_save=1, snapshot_step_interval=1)
        rec_b = ChainRecorder(num_chains_to_save=1, snapshot_step_interval=1)
        graph = _make_pmf_graph(bs=2, n=4, dx=2, de=2)
        rec_a.maybe_record(0, graph)
        rec_b.maybe_record(0, graph)
        with pytest.raises(ValueError, match="duplicate key"):
            merge_chain_snapshots([rec_a.finalize(), rec_b.finalize()])


# ---------------------------------------------------------------------------
# Sampler integration smoke
# ---------------------------------------------------------------------------


class _UniformCategoricalModel(GraphModel):
    """Returns uniform PMFs, mirroring the helper in test_sampler."""

    def __init__(self, dx: int, de: int) -> None:
        super().__init__()
        self.dx = dx
        self.de = de

    def get_config(self) -> dict[str, Any]:
        return {"dx": self.dx, "de": self.de}

    def forward(self, data: GraphData, t: Tensor | None = None) -> GraphData:
        _ = t
        assert data.X_class is not None
        assert data.E_class is not None
        bs, n, _ = data.X_class.shape
        X = torch.ones(bs, n, self.dx, device=data.X_class.device) / self.dx
        E = torch.ones(bs, n, n, self.de, device=data.E_class.device) / self.de
        return GraphData(
            y=data.y,
            node_mask=data.node_mask,
            X_class=X,
            E_class=E,
        )


def test_sampler_records_at_least_one_snapshot(tmp_path: Path) -> None:
    """End-to-end: drive a tiny CategoricalSampler with a recorder.

    Verifies the recorder is invoked from inside the reverse loop, the
    saved artefact loads back, and the snapshot count matches the
    expected cadence (T=5, interval=2, snapshot at step_index in {0,2,4}).
    """
    schedule = NoiseSchedule("cosine_iddpm", timesteps=5)
    process = CategoricalNoiseProcess(
        schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
    )
    model = _UniformCategoricalModel(dx=3, de=2)
    sampler = CategoricalSampler()
    rec = ChainRecorder(num_chains_to_save=2, snapshot_step_interval=2)

    sampler.sample(
        model=model,
        noise_process=process,
        num_graphs=2,
        num_nodes=4,
        device=torch.device("cpu"),
        chain_recorder=rec,
    )

    out = rec.finalize()
    # T=5 reverse steps -> step_indices 0,1,2,3,4. interval=2 -> {0, 2, 4}.
    assert out["step_indices"].tolist() == [0, 2, 4]
    assert out["E_chain"].shape == (3, 2, 4, 4, 2)
    assert out["X_chain"].shape == (3, 2, 4, 3)

    target = tmp_path / "chain.pt"
    torch.save(out, target)
    loaded: dict[str, Tensor] = torch.load(target, weights_only=True)
    torch.testing.assert_close(loaded["E_chain"], out["E_chain"])


def test_sampler_without_recorder_unchanged() -> None:
    """Sampler with chain_recorder=None should retain prior behaviour.

    This protects acceptance criterion A2 (capture is a pure side effect).
    """
    _ = binary_graphdata  # imported for re-use elsewhere; silence linter
    schedule = NoiseSchedule("cosine_iddpm", timesteps=3)
    process = CategoricalNoiseProcess(
        schedule=schedule, x_classes=3, e_classes=2, limit_distribution="uniform"
    )
    model = _UniformCategoricalModel(dx=3, de=2)
    sampler = CategoricalSampler()

    torch.manual_seed(0)
    no_rec = sampler.sample(
        model=model,
        noise_process=process,
        num_graphs=2,
        num_nodes=4,
        device=torch.device("cpu"),
    )
    torch.manual_seed(0)
    rec = ChainRecorder(num_chains_to_save=1, snapshot_step_interval=1)
    with_rec = sampler.sample(
        model=model,
        noise_process=process,
        num_graphs=2,
        num_nodes=4,
        device=torch.device("cpu"),
        chain_recorder=rec,
    )
    assert len(no_rec) == len(with_rec)
    for a, b in zip(no_rec, with_rec, strict=True):
        if a.E_class is not None and b.E_class is not None:
            torch.testing.assert_close(a.E_class, b.E_class)
        if a.X_class is not None and b.X_class is not None:
            torch.testing.assert_close(a.X_class, b.X_class)
