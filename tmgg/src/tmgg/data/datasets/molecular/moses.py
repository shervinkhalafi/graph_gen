"""MOSES dataset — DiGress repro Table 5.

SMILES source: the ``moses`` package (``moses.get_dataset(split)``).
Atom decoder: (C, N, S, O, F, Cl, Br, H). DiGress uses the published
MOSES train/test/scaffold split; we mirror the package's
``get_dataset`` helper.
"""

from __future__ import annotations

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.dataset import MolecularGraphDataset
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

# MOSES test split is large enough that DiGress evaluates against
# scaffold split (held-out scaffolds) for the FCD/SNN metrics. Our
# DataModule serves train + val(=test_scaffolds) + test(=test).
_MOSES_VAL_SPLIT = "test_scaffolds"


class MOSESDataset(MolecularGraphDataset):
    """MOSES split. Atom decoder: (C, N, S, O, F, Cl, Br, H)."""

    DATASET_NAME = "moses"
    DEFAULT_MAX_ATOMS = 30
    # SMILES come from the ``moses`` package, not URLs.
    RAW_FILES: dict[str, str] = {}

    @classmethod
    def make_codec(cls) -> SMILESCodec:
        return SMILESCodec(
            vocab=AtomBondVocabulary.moses(),
            remove_h=True,
            kekulize=True,
            max_atoms=cls.DEFAULT_MAX_ATOMS,
        )

    def download_smiles_split(self, split: str) -> list[str]:
        """Pull SMILES from the ``moses`` package."""
        import moses

        moses_split = {
            "train": "train",
            "val": _MOSES_VAL_SPLIT,
            "test": "test",
        }.get(split)
        if moses_split is None:
            raise ValueError(f"Unknown split {split!r}.")
        return list(moses.get_dataset(moses_split))  # pyright: ignore[reportAttributeAccessIssue]  # moses has no stubs; runtime-verified
