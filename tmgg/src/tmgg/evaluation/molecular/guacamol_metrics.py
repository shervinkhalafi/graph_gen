"""GuacaMol Distribution-Learning metrics (Table 6 in DiGress).

Wraps :mod:`guacamol.distribution_learning_benchmark`. Two metrics:
KL divergence on physchem properties, and FCD against a ChEMBL
reference set.
"""

from __future__ import annotations

from collections.abc import Sequence

from tmgg.evaluation.molecular.metric import MolecularMetric


def _drop_invalid(smiles: Sequence[str]) -> list[str]:
    from rdkit import Chem

    out: list[str] = []
    for s in smiles:
        mol = Chem.MolFromSmiles(s)
        if mol is not None:
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            out.append(s)
    return out


class KLDivPropertyMetric(MolecularMetric):
    """KL divergence on the GuacaMol physchem-property distribution.

    Wraps ``guacamol.distribution_learning_benchmark.KLDivBenchmark``.
    Returns the benchmark's score in [0, 1] (higher = closer to ref).
    """

    name = "kl_div_property"

    def __init__(self, n_samples: int = 10_000) -> None:
        self.n_samples = n_samples

    def compute(
        self,
        generated: Sequence[str],
        reference: Sequence[str] | None = None,
    ) -> float:
        if reference is None:
            raise ValueError("KLDivPropertyMetric requires reference SMILES.")
        import guacamol.distribution_learning_benchmark as glb  # type: ignore[import-not-found]  # pyright: ignore[reportMissingImports]

        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return 0.0
        bench = glb.KLDivBenchmark(
            training_set=list(reference),
            number_samples=self.n_samples,
            name="kl_div_property",  # pyright: ignore[reportCallIssue]
        )

        # KLDivBenchmark.assess_model expects an object with
        # ``generate(n)`` returning SMILES; wrap our list in a tiny
        # adapter so we feed exactly our generated set rather than
        # re-sampling.
        class _Adapter:
            def __init__(self, smiles: list[str]) -> None:
                self.smiles = smiles

            def generate(self, _n: int) -> list[str]:
                return self.smiles

        result = bench.assess_model(_Adapter(list(gen_valid)))  # pyright: ignore[reportArgumentType]
        return float(result.score)


class FCDChEMBLMetric(MolecularMetric):
    """Frechet ChemNet Distance against the GuacaMol ChEMBL reference set."""

    name = "fcd_chembl"

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
            raise ValueError("FCDChEMBLMetric requires reference SMILES.")
        gen_valid = _drop_invalid(generated)
        if not gen_valid:
            return float("inf")
        fcd = self._ensure()
        return float(fcd(list(gen_valid), list(reference)))  # type: ignore[operator]
