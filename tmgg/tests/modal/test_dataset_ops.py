"""Tests for ``tmgg.modal._lib.dataset_ops`` impl functions.

These tests exercise the pure-Python prepare/validate path against
synthetic shard files written into a ``tmp_path`` cache. The Modal
``@app.function`` wrappers are not invoked.

Testing Strategy
----------------
- Build a fake ``preprocessed/<hash>/<split>/<shard>.pt`` tree under a
  ``tmp_path`` rooted ``cache_root``, then call the validate impl
  directly. Avoids heavyweight downloads while still exercising the
  full schema-check path on real ``.pt`` files.
- Unknown-dataset names must raise ``ValueError`` (loud failure as per
  the project's no-graceful-fallback rule).

Key Invariants
--------------
- A clean ``preprocessed/<hash>/<split>/0000.pt`` validates ``ok``.
- A truncated shard file marks the split as ``errors`` with
  ``load_error`` on the affected shard.
- A missing shard directory marks the split as ``missing``.
- The dispatcher rejects names outside ``ALL_DATASETS``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from tmgg.modal._lib.dataset_ops import (
    ALL_DATASETS,
    _validate_shard,
    prepare_dataset,
    validate_dataset,
)


@dataclass
class _FakeGraph:
    node_mask: torch.Tensor
    X_class: torch.Tensor


def _write_clean_shard(path: Path, n: int = 4) -> None:
    """Materialise ``n`` synthetic GraphData-like dataclass instances."""
    path.parent.mkdir(parents=True, exist_ok=True)
    graphs = [
        _FakeGraph(
            node_mask=torch.ones(9, dtype=torch.bool),
            X_class=torch.zeros(9, 4),
        )
        for _ in range(n)
    ]
    torch.save(graphs, path)


# ---------------------------------------------------------------------------
# Per-shard validator
# ---------------------------------------------------------------------------


class TestValidateShard:
    def test_clean_shard_returns_ok(self, tmp_path: Path) -> None:
        shard = tmp_path / "0000.pt"
        _write_clean_shard(shard, n=3)
        report = _validate_shard(shard)
        assert report["status"] == "ok"
        assert report["graphs"] == 3
        assert report["size_bytes"] > 0

    def test_missing_file_returns_missing(self, tmp_path: Path) -> None:
        shard = tmp_path / "absent.pt"
        report = _validate_shard(shard)
        assert report["status"] == "missing"

    def test_truncated_shard_returns_load_error(self, tmp_path: Path) -> None:
        shard = tmp_path / "0000.pt"
        _write_clean_shard(shard)
        # Truncate to half its length — guaranteed to corrupt the
        # internal zip directory and trigger PytorchStreamReader.
        size = shard.stat().st_size
        with open(shard, "r+b") as f:
            f.truncate(size // 2)
        report = _validate_shard(shard)
        assert report["status"] == "load_error"
        assert "detail" in report

    def test_non_list_payload_returns_schema_error(self, tmp_path: Path) -> None:
        shard = tmp_path / "0000.pt"
        torch.save({"not": "a list"}, shard)
        report = _validate_shard(shard)
        assert report["status"] == "schema_error"

    def test_empty_list_returns_schema_error(self, tmp_path: Path) -> None:
        shard = tmp_path / "0000.pt"
        torch.save([], shard)
        report = _validate_shard(shard)
        assert report["status"] == "schema_error"


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


class TestDispatcher:
    def test_unknown_name_raises_validate(self) -> None:
        with pytest.raises(ValueError, match="unknown dataset"):
            validate_dataset("not-a-dataset")

    def test_unknown_name_raises_prepare(self) -> None:
        with pytest.raises(ValueError, match="unknown dataset"):
            prepare_dataset("not-a-dataset")

    def test_all_datasets_listed(self) -> None:
        # Sanity: catches typos in the constant if a contributor adds a
        # new dataset to one helper but not the other.
        assert set(ALL_DATASETS) >= {"qm9", "moses", "guacamol", "planar", "sbm"}


# ---------------------------------------------------------------------------
# Spectre validate path (synthesise a fixture, no network)
# ---------------------------------------------------------------------------


class TestSpectreValidate:
    def test_missing_fixture_marks_missing(self, tmp_path: Path) -> None:
        # Don't write anything — validator must report 'missing' rather
        # than raise. tmp_path subdir 'spectre/' won't exist.
        report = validate_dataset("planar", cache_root=tmp_path)
        assert report["status"] == "missing"
        assert report["exists"] is False
        assert report["fixture_path"].endswith("planar_64_200.pt")

    def test_corrupt_fixture_marks_load_error(self, tmp_path: Path) -> None:
        # Write a non-torch file at the expected path.
        spectre_dir = tmp_path / "spectre"
        spectre_dir.mkdir()
        (spectre_dir / "sbm_200.pt").write_bytes(b"\x00\x01\x02 not a torch zip")
        report = validate_dataset("sbm", cache_root=tmp_path)
        assert report["status"] == "load_error"
        assert "detail" in report
