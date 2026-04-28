"""MOSES metrics (Table 5 in DiGress): FCD/SNN/IntDiv/Filters/ScaffoldSplit.

Wraps the ``moses`` package + ``fcd_torch`` so the rest of our code
does not have to know either API. Each class is one metric.
"""

from __future__ import annotations

from collections.abc import Sequence

from tmgg.evaluation.molecular.metric import MolecularMetric


def _drop_invalid(smiles: Sequence[str]) -> list[str]:
    """Filter SMILES to those that round-trip through RDKit."""
    from rdkit import Chem

    out: list[str] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        out.append(s)
    return out


class FCDMetric(MolecularMetric):
    """Frechet ChemNet Distance via :mod:`fcd_torch`."""

    name = "fcd"

    def __init__(self, device: str = "cpu", n_jobs: int = 1) -> None:
        self.device = device
        self.n_jobs = n_jobs
        self._fcd: object | None = None

    def _ensure(self) -> object:
        if self._fcd is None:
            from fcd_torch import FCD

            self._fcd = FCD(device=self.device, n_jobs=self.n_jobs)
        return self._fcd

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("FCDMetric requires reference SMILES.")
        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return float("inf")
        fcd = self._ensure()
        # fcd_torch's FCD object is callable: __call__(gen, ref) → float.
        return float(fcd(list(gen_valid), list(reference)))  # type: ignore[operator]


class SNNMetric(MolecularMetric):
    """Average Tanimoto similarity to nearest train neighbour (MOSES)."""

    name = "snn"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("SNNMetric requires reference SMILES.")
        import moses.metrics as moses_metrics  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return 0.0
        return float(moses_metrics.SNNMetric()(list(gen_valid), list(reference)))


class IntDivMetric(MolecularMetric):
    """Internal diversity (1 − mean Tanimoto similarity)."""

    name = "int_div"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        del reference
        import moses.metrics as moses_metrics  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return 0.0
        return float(moses_metrics.internal_diversity(list(gen_valid)))


class FiltersMetric(MolecularMetric):
    """Fraction of generated mols passing the MOSES filter set."""

    name = "filters"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        del reference
        import moses.metrics as moses_metrics  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return 0.0
        return float(moses_metrics.fraction_passes_filters(list(gen_valid)))


class ScaffoldSplitMetric(MolecularMetric):
    """Fraction of generated mols whose Bemis-Murcko scaffolds are
    novel relative to the reference set's scaffolds."""

    name = "scaffold_novelty"

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("ScaffoldSplitMetric requires reference SMILES.")
        import moses.metrics as moses_metrics  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return 0.0
        ref_scaffolds = set(moses_metrics.compute_scaffolds(list(reference)).keys())
        gen_scaffolds = moses_metrics.compute_scaffolds(list(gen_valid))
        novel = [s for s in gen_scaffolds.keys() if s not in ref_scaffolds]  # noqa: SIM118
        return len(novel) / len(gen_scaffolds) if gen_scaffolds else 0.0
