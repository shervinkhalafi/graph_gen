"""Tests for SMILESCodec encode/decode round-trip."""

from __future__ import annotations

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary


def _qm9_codec() -> SMILESCodec:
    return SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)


def test_encode_simple_smiles() -> None:
    codec = _qm9_codec()
    data = codec.encode("CCO")
    assert data is not None
    assert data.X_class is not None
    assert data.E_class is not None
    assert data.X_class.shape == (1, 3, 4)
    assert data.E_class.shape == (1, 3, 3, 5)


def test_encode_invalid_smiles_returns_none() -> None:
    codec = _qm9_codec()
    assert codec.encode("not_a_smiles") is None


def test_encode_atom_count_overflow() -> None:
    codec = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=2)
    # ethanol = 3 heavy atoms — exceeds max_atoms=2
    assert codec.encode("CCO") is None


def test_decode_recovers_canonical() -> None:
    codec = _qm9_codec()
    data = codec.encode("CCO")
    assert data is not None
    decoded = codec.decode(data)
    # RDKit canonicalisation of "CCO" is itself "CCO".
    assert decoded == "CCO"


def test_decode_accepts_unbatched_graph_data() -> None:
    """Regression: codec must accept the unbatched shape produced by
    the sampler and by ``BaseGraphDataModule.get_reference_graphs`` post
    the 2026-05-01 universal-transport refactor (1-D node_mask, 2-D
    X_class, 3-D E_class). Pre-fix, decode raised
    ``"expects a single-graph batch"`` because it only handled the
    batched-of-one shape.
    """
    from tmgg.data.datasets.graph_types import GraphData

    codec = _qm9_codec()
    encoded = codec.encode("CCO")
    assert encoded is not None
    # Strip the leading-1 batch dim to mimic an unbatched single-graph.
    unbatched = GraphData(
        node_mask=encoded.node_mask[0],
        X_class=encoded.X_class[0] if encoded.X_class is not None else None,
        E_class=encoded.E_class[0] if encoded.E_class is not None else None,
        y=encoded.y[0] if encoded.y is not None else encoded.y,
    )
    assert codec.decode(unbatched) == "CCO"


def test_cache_key_changes_with_vocab() -> None:
    qm9 = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)
    moses = SMILESCodec(vocab=AtomBondVocabulary.moses(), max_atoms=30)
    assert qm9.cache_key() != moses.cache_key()


def test_cache_key_changes_with_max_atoms() -> None:
    a = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)
    b = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=29)
    assert a.cache_key() != b.cache_key()


def test_cache_key_stable_across_instantiations() -> None:
    a = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)
    b = SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)
    assert a.cache_key() == b.cache_key()


def test_encode_dataset_with_stats_counts() -> None:
    codec = _qm9_codec()
    smiles = ["CCO", "CC(=O)O", "not_a_smiles", "C" * 200]
    graphs, counters = codec.encode_dataset_with_stats(smiles)
    assert counters["input"] == 4
    assert counters["kept"] == 2
    assert counters["parse_failure"] == 1
    assert counters["atom_count_overflow"] == 1
    assert len(graphs) == 2
