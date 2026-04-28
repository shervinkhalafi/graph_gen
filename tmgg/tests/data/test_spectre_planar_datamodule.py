"""Tests for SpectrePlanarDataModule, mirroring the SBM tests' structure."""

from __future__ import annotations

import pytest

from tmgg.data.data_modules.spectre_planar import SpectrePlanarDataModule
from tmgg.data.datasets.spectre_planar import (
    SPECTRE_PLANAR_TEST_LEN,
    SPECTRE_PLANAR_TRAIN_LEN,
    SPECTRE_PLANAR_VAL_LEN,
    split_spectre_planar,
)


def test_split_sizes_match_spec() -> None:
    splits = split_spectre_planar()
    assert len(splits["train"]) == SPECTRE_PLANAR_TRAIN_LEN
    assert len(splits["val"]) == SPECTRE_PLANAR_VAL_LEN
    assert len(splits["test"]) == SPECTRE_PLANAR_TEST_LEN
    assert SPECTRE_PLANAR_TRAIN_LEN == 128
    assert SPECTRE_PLANAR_VAL_LEN == 32
    assert SPECTRE_PLANAR_TEST_LEN == 40


def test_setup_loads_fixture() -> None:
    """Requires fixture at ~/.cache/tmgg/spectre/planar_64_200.pt."""
    pytest.importorskip("torch_geometric")
    dm = SpectrePlanarDataModule()
    dm.setup("fit")
    assert dm.num_nodes == 64
    assert dm._train_data is not None
    assert len(dm._train_data) == SPECTRE_PLANAR_TRAIN_LEN


def test_dataloader_shapes() -> None:
    """Sanity-check that the collator produces dense GraphData."""
    pytest.importorskip("torch_geometric")
    dm = SpectrePlanarDataModule(batch_size=4)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert batch.E_class is not None
    assert batch.E_class.shape[0] == 4  # batch dim
    assert batch.E_class.shape[1] == 64  # n_max
