"""QM9 dataset (no-H by default) — DiGress repro Table 4.

SMILES source: DiGress's published QM9 CSV mirror (the same file the
upstream repo reads). Each row is one SMILES; ~134k molecules total.
Splits: 80 % train / 10 % val / 10 % test, mirroring upstream.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import override

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

# DiGress paper uses the standard PyG QM9 download (gdb9.sdf.csv).
# We download via the PyG mirror used by upstream DiGress.
_QM9_CSV_URL = "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz"

# Upstream split rule: random 80/10/10 with seed 0.
_QM9_SPLIT_SEED = 0


class QM9Dataset(MolecularGraphDataset):
    """QM9 (no-H) split. Atom decoder: (C, N, O, F)."""

    DATASET_NAME = "qm9"
    DEFAULT_MAX_ATOMS = 9
    # No per-split URLs — single CSV; we split locally.
    RAW_FILES: dict[str, str] = {}

    @classmethod
    @override
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.qm9(remove_h=True),
            remove_h=True,
            kekulize=True,
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    @override
    def download_smiles_split(self, split: str) -> list[str]:
        """Read all SMILES from the QM9 CSV and return the requested split."""
        raw_path = self._raw_dir() / "qm9.csv"
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path.exists():
            self._download_qm9_csv(raw_path)

        all_smiles: list[str] = []
        with raw_path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                smi = row.get("smiles") or row.get("SMILES")
                if smi:
                    all_smiles.append(smi.strip())

        rng = random.Random(_QM9_SPLIT_SEED)
        indices = list(range(len(all_smiles)))
        rng.shuffle(indices)
        n = len(indices)
        n_train = int(0.8 * n)
        n_val = int(0.1 * n)
        if split == "train":
            picked = indices[:n_train]
        elif split == "val":
            picked = indices[n_train : n_train + n_val]
        elif split == "test":
            picked = indices[n_train + n_val :]
        else:
            raise ValueError(f"Unknown split {split!r}.")
        return [all_smiles[i] for i in picked]

    def _download_qm9_csv(self, target: Path) -> None:
        """Download + extract the QM9 CSV from DeepChem's mirror."""
        import tarfile
        import urllib.request

        archive = target.with_name("qm9.tar.gz")
        if not archive.exists():
            urllib.request.urlretrieve(_QM9_CSV_URL, archive)  # noqa: S310
        with tarfile.open(archive, "r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".csv")]
            if not members:
                raise RuntimeError(f"No CSV inside {archive}")
            tar.extract(members[0], target.parent)
            extracted = target.parent / members[0].name
            extracted.rename(target)
