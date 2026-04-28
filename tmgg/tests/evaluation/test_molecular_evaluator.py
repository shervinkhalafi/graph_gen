"""Tests for MolecularEvaluator + classmethod presets."""

from __future__ import annotations

from tmgg.data.datasets.molecular.codec import SMILESCodec
from tmgg.data.datasets.molecular.vocabulary import AtomBondVocabulary
from tmgg.evaluation.molecular import MolecularEvaluator
from tmgg.evaluation.molecular.rdkit_metrics import (
    UniquenessMetric,
    ValidityMetric,
)


def _qm9_codec() -> SMILESCodec:
    return SMILESCodec(vocab=AtomBondVocabulary.qm9(), max_atoms=30)


def test_for_qm9_metric_keys() -> None:
    ev = MolecularEvaluator.for_qm9()
    names = [m.name for m in ev.metrics]
    assert names == ["validity", "uniqueness", "novelty"]


def test_for_moses_metric_keys() -> None:
    ev = MolecularEvaluator.for_moses()
    names = [m.name for m in ev.metrics]
    assert names == [
        "validity",
        "uniqueness",
        "novelty",
        "fcd",
        "snn",
        "int_div",
        "filters",
        "scaffold_novelty",
    ]


def test_for_guacamol_metric_keys() -> None:
    ev = MolecularEvaluator.for_guacamol()
    names = [m.name for m in ev.metrics]
    assert names == [
        "validity",
        "uniqueness",
        "novelty",
        "kl_div_property",
        "fcd_chembl",
    ]


def test_evaluate_on_decoded_graphs() -> None:
    """End-to-end: evaluator decodes GraphData and runs metrics."""
    codec = _qm9_codec()
    g_ccol = codec.encode("CCO")
    g_acet = codec.encode("CC(=O)O")
    assert g_ccol is not None and g_acet is not None
    ev = MolecularEvaluator(
        metrics=[ValidityMetric(), UniquenessMetric()],
        codec=codec,
    )
    refs = [g_ccol]
    gen = [g_ccol, g_acet]
    results = ev.evaluate(refs, gen)
    assert "validity" in results.values
    assert "uniqueness" in results.values
    assert results.values["validity"] == 1.0
