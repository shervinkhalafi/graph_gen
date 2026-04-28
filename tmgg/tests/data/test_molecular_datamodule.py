"""Tests for the molecular DataModule batch shape + collator integration."""

from __future__ import annotations

from pathlib import Path

from tmgg.data.data_modules.molecular.base import MolecularDataModule
from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary


class _TinyDataset(MolecularGraphDataset):
    DATASET_NAME = "tiny_dm"
    DEFAULT_MAX_ATOMS = 5

    @classmethod
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.qm9(),
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    def download_smiles_split(self, split: str) -> list[str]:
        # 3 each: train/val/test
        return ["CCO", "CC(=O)O", "CCC"]


class _TinyDataModule(MolecularDataModule):
    dataset_cls = _TinyDataset


def test_dataloader_yields_graphdata(tmp_path: Path) -> None:
    dm = _TinyDataModule(batch_size=2, cache_root=str(tmp_path))
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader()))
    # batch should be a GraphData with per-batch shapes (bs, n, ...).
    assert batch.X_class is not None
    assert batch.E_class is not None
    assert batch.node_mask is not None
    assert batch.X_class.shape[0] == 2
    assert batch.E_class.shape[0] == 2
    assert batch.node_mask.shape[0] == 2


def test_size_distribution_populated(tmp_path: Path) -> None:
    dm = _TinyDataModule(batch_size=1, cache_root=str(tmp_path))
    dm.prepare_data()
    dm.setup()
    sd = dm.get_size_distribution("train")
    # Three molecules, sizes ∈ {3, 4, 3}.
    sample = sd.sample(100)
    assert sample.min() >= 3
    assert sample.max() <= 4
