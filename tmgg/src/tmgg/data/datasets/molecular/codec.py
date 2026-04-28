"""SMILES ↔ GraphData codec — the only module that imports RDKit.

Parameterised by an :class:`AtomBondVocabulary`. Encodes a SMILES
string into a categorical :class:`GraphData` with ``X_class`` (atom
classes) and ``E_class`` (bond classes), or returns ``None`` on
parse failure / atom-count overflow.

The codec's :func:`__hash__` doubles as the cache-invalidation key
for preprocessed shards stored under
``<cache_root>/<dataset>/preprocessed/<codec_hash>/<split>.pt``.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from tmgg.data.datasets.graph_types import GraphData
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary

if TYPE_CHECKING:
    pass  # type: ignore[import-not-found]


# Map RDKit's BondType to our bond_decoder string for `encode_bond`.
# Populated lazily so the rdkit import stays inside the codec.
def _rdkit_bond_name(bond_type: object) -> str:
    """Map an RDKit ``Chem.BondType`` to our bond_decoder name.

    Lazy-importing here avoids hoisting rdkit into module-level
    imports and keeps the codec the sole RDKit-touching surface.
    """
    from rdkit.Chem.rdchem import BondType

    if bond_type == BondType.SINGLE:
        return "SINGLE"
    if bond_type == BondType.DOUBLE:
        return "DOUBLE"
    if bond_type == BondType.TRIPLE:
        return "TRIPLE"
    if bond_type == BondType.AROMATIC:
        return "AROMATIC"
    raise ValueError(f"Unsupported RDKit bond type: {bond_type!r}.")


def _bond_name_to_rdkit(name: str) -> object:
    from rdkit.Chem.rdchem import BondType

    return {
        "SINGLE": BondType.SINGLE,
        "DOUBLE": BondType.DOUBLE,
        "TRIPLE": BondType.TRIPLE,
        "AROMATIC": BondType.AROMATIC,
    }[name]


@dataclass(frozen=True)
class SMILESCodec:
    """SMILES ↔ GraphData round-trip parameterised by a vocabulary.

    Parameters
    ----------
    vocab
        Atom + bond vocabulary the codec encodes against.
    remove_h
        When True, hydrogens are stripped before encoding (matches
        DiGress's default for QM9/MOSES/GuacaMol).
    kekulize
        When True, aromatic bonds are kekulised so the encoder sees
        explicit SINGLE/DOUBLE pairs rather than AROMATIC. Matches
        upstream DiGress's preprocessing.
    max_atoms
        Molecules with more than ``max_atoms`` heavy atoms are dropped
        (encode returns ``None``).
    """

    vocab: AtomBondVocabulary
    remove_h: bool = True
    kekulize: bool = True
    max_atoms: int = 30

    def __hash__(self) -> int:
        return hash((self.vocab, self.remove_h, self.kekulize, self.max_atoms))

    def cache_key(self) -> str:
        """Stable SHA-256 hex digest used as the on-disk shard subdirectory."""
        material = repr((self.vocab, self.remove_h, self.kekulize, self.max_atoms))
        return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]

    # ------------------------------------------------------------------
    # encode
    # ------------------------------------------------------------------

    def encode(self, smiles: str) -> GraphData | None:
        """Encode a SMILES string into a categorical :class:`GraphData`.

        Returns ``None`` when the molecule fails to parse, has too
        many heavy atoms, or contains an atom not in the vocabulary.
        """
        from rdkit import Chem

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = Chem.RemoveHs(mol) if self.remove_h else Chem.AddHs(mol)
        if self.kekulize:
            try:
                Chem.Kekulize(mol)
            except Exception:
                return None

        atoms = list(mol.GetAtoms())
        n = len(atoms)
        if n == 0 or n > self.max_atoms:
            return None

        # Atom classes
        try:
            atom_classes = [self.vocab.encode_atom(a.GetSymbol()) for a in atoms]
        except ValueError:
            return None

        x_class = torch.zeros((1, n, self.vocab.num_atom_types), dtype=torch.float32)
        for i, cls in enumerate(atom_classes):
            x_class[0, i, cls] = 1.0

        # Bond classes (E_class[0,i,j] = one-hot over bond_decoder).
        e_class = torch.zeros((1, n, n, self.vocab.num_bond_types), dtype=torch.float32)
        # Default everywhere: NONE (class 0).
        e_class[..., 0] = 1.0
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            try:
                bond_class = self.vocab.encode_bond(
                    _rdkit_bond_name(bond.GetBondType())
                )
            except ValueError:
                return None
            # Symmetric edge.
            e_class[0, i, j] = 0.0
            e_class[0, j, i] = 0.0
            e_class[0, i, j, bond_class] = 1.0
            e_class[0, j, i, bond_class] = 1.0
        # Diagonal stays at NONE (already set).

        node_mask = torch.ones((1, n), dtype=torch.bool)

        return GraphData(
            X_class=x_class,
            X_feat=None,
            E_class=e_class,
            E_feat=None,
            y=torch.zeros((1, 0), dtype=torch.float32),
            node_mask=node_mask,
        )

    # ------------------------------------------------------------------
    # decode
    # ------------------------------------------------------------------

    def decode(self, data: GraphData) -> str | None:
        """Decode a single-graph :class:`GraphData` back to a SMILES.

        Returns ``None`` if the resulting molecule fails RDKit
        sanitisation. Used by :class:`ValidityMetric` and round-trip
        tests; not used in the training loop.
        """
        from rdkit import Chem
        from rdkit.Chem import RWMol

        if data.X_class is None or data.E_class is None:
            return None
        x_class = data.X_class
        e_class = data.E_class
        node_mask = data.node_mask
        if x_class.shape[0] != 1:
            raise ValueError(
                f"SMILESCodec.decode expects a single-graph batch; got "
                f"batch size {x_class.shape[0]}."
            )

        n_real = int(node_mask[0].sum().item())
        atom_idx = x_class[0, :n_real].argmax(dim=-1).tolist()
        bond_idx = e_class[0, :n_real, :n_real].argmax(dim=-1).tolist()

        rwmol = RWMol()
        for a in atom_idx:
            rwmol.AddAtom(Chem.Atom(self.vocab.decode_atom(int(a))))
        for i in range(n_real):
            for j in range(i + 1, n_real):
                cls = int(bond_idx[i][j])
                if cls == 0:  # NONE
                    continue
                bond_name = self.vocab.decode_bond(cls)
                rwmol.AddBond(i, j, _bond_name_to_rdkit(bond_name))  # type: ignore[arg-type]

        mol = rwmol.GetMol()
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        try:
            return Chem.MolToSmiles(mol)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # batch helper
    # ------------------------------------------------------------------

    def encode_dataset(self, smiles_iter: Iterable[str]) -> Iterator[GraphData]:
        """Iterate over SMILES, dropping parse failures silently.

        Use :meth:`encode_dataset_with_stats` if you need the
        dropped-mol counters for logging.
        """
        for smi in smiles_iter:
            data = self.encode(smi)
            if data is not None:
                yield data

    def encode_dataset_with_stats(
        self, smiles_iter: Iterable[str]
    ) -> tuple[list[GraphData], dict[str, int]]:
        """Encode a list of SMILES, returning (graphs, drop_counters).

        Counters: ``"parse_failure"``, ``"atom_count_overflow"``,
        ``"vocab_miss"``, ``"kekulize_failure"``.
        """
        graphs: list[GraphData] = []
        counters = {
            "input": 0,
            "parse_failure": 0,
            "atom_count_overflow": 0,
            "vocab_miss": 0,
            "kekulize_failure": 0,
            "kept": 0,
        }
        # Re-implement the encode() body inline so we can fill counter
        # buckets per failure mode.
        from rdkit import Chem

        for smi in smiles_iter:
            counters["input"] += 1
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                counters["parse_failure"] += 1
                continue
            mol = Chem.RemoveHs(mol) if self.remove_h else Chem.AddHs(mol)
            if self.kekulize:
                try:
                    Chem.Kekulize(mol)
                except Exception:
                    counters["kekulize_failure"] += 1
                    continue
            atoms = list(mol.GetAtoms())
            n = len(atoms)
            if n == 0 or n > self.max_atoms:
                counters["atom_count_overflow"] += 1
                continue
            try:
                # Pre-flight vocab lookup so we can bucket vocab misses
                # separately from other failures inside encode().
                _ = [self.vocab.encode_atom(a.GetSymbol()) for a in atoms]
            except ValueError:
                counters["vocab_miss"] += 1
                continue
            data = self.encode(smi)
            if data is None:
                # Edge case where encode() raised in the second pass;
                # very rare, count under parse_failure.
                counters["parse_failure"] += 1
                continue
            graphs.append(data)
            counters["kept"] += 1
        return graphs, counters
