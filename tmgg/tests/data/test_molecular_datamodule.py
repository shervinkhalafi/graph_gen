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


def test_train_dataloader_raw_pyg_yields_pyg_batch(tmp_path: Path) -> None:
    """``train_dataloader_raw_pyg`` must produce a PyG ``Batch``.

    Starting state: a tiny molecular DataModule with three SMILES.
    Invariant: the raw loader yields a ``torch_geometric.data.Batch``
    (the noise-process initialiser walks this once before training to
    estimate empirical (atom, bond)-class marginals). Without an
    override on ``MolecularDataModule``, this would have raised
    ``NotImplementedError`` at first iter — and previously,
    ``BaseGraphDataModule`` even allowed instantiation of subclasses
    without the override (caught only at preflight). This test pins
    both fixes (override + abstract enforcement upstream).
    """
    from torch_geometric.data import Batch

    dm = _TinyDataModule(batch_size=2, cache_root=str(tmp_path))
    dm.prepare_data()
    dm.setup()
    batch = next(iter(dm.train_dataloader_raw_pyg()))
    assert isinstance(batch, Batch)
    assert hasattr(batch, "x")
    assert hasattr(batch, "edge_index")
    # Two molecules collated; node count should be sum of per-graph
    # node counts (sizes ∈ {3, 4, 3}, so first 2 → 3 + 4 = 7).
    assert batch.num_graphs == 2
    assert batch.num_nodes >= 6  # type: ignore[attr-defined]  # PyG runtime attr


def test_basegraph_subclass_without_raw_pyg_override_cannot_instantiate() -> None:
    """Abstract-method enforcement: instantiation refused at object creation.

    Starting state: a synthetic subclass of ``BaseGraphDataModule`` that
    overrides ``setup``/``train_dataloader``/``val_dataloader``/
    ``test_dataloader`` but NOT ``train_dataloader_raw_pyg``.
    Invariant: ``Subclass()`` must raise ``TypeError`` from Python's
    abstract-class machinery, mentioning the missing method by name —
    so a future regression that drops the override surfaces at unit-
    test time, not during a Modal preflight on a remote A100.
    """
    import pytest

    from tmgg.data.data_modules.base_data_module import BaseGraphDataModule

    class _Incomplete(BaseGraphDataModule):
        def setup(self, stage: str | None = None) -> None:  # noqa: ARG002
            pass

        def train_dataloader(self):  # type: ignore[no-untyped-def]
            pass

        def val_dataloader(self):  # type: ignore[no-untyped-def]
            pass

        def test_dataloader(self):  # type: ignore[no-untyped-def]
            pass

    with pytest.raises(TypeError, match="train_dataloader_raw_pyg"):
        # ``_Incomplete`` is abstract by design here; the test asserts
        # that Python's runtime refuses the instantiation.
        _Incomplete()  # pyright: ignore[reportAbstractUsage]
