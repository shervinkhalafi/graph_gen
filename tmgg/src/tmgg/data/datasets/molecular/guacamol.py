"""GuacaMol dataset — DiGress repro Table 6.

SMILES source: GuacaMol's published train/val/test splits.
Atom decoder: (C, N, O, F, B, Br, Cl, I, P, S, Se, Si).
"""

from __future__ import annotations

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

# GuacaMol's published splits.
_GUACAMOL_BASE_URL = "https://figshare.com/ndownloader/files/13612760"  # train
_GUACAMOL_URLS = {
    "train": "https://figshare.com/ndownloader/files/13612760",
    "val": "https://figshare.com/ndownloader/files/13612766",
    "test": "https://figshare.com/ndownloader/files/13612757",
}


class GuacaMolDataset(MolecularGraphDataset):
    """GuacaMol split. Atom decoder: 12-element vocabulary."""

    DATASET_NAME = "guacamol"
    DEFAULT_MAX_ATOMS = 88
    RAW_FILES = _GUACAMOL_URLS

    @classmethod
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.guacamol(),
            remove_h=True,
            kekulize=True,
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    def download_smiles_split(self, split: str) -> list[str]:
        path = self._default_download(split)
        with path.open("r") as f:
            return [line.strip() for line in f if line.strip()]
