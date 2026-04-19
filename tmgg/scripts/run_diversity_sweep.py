"""Phase 3 validation: improvement-gap surrogate vs. diversity knob.

Generates four SBM datasets at ``diversity ∈ {0, 0.33, 0.67, 1.0}``, runs
the noised eigenstructure collector with B-projection at four ``k`` values,
and reports the improvement-gap surrogate ``ĝ`` with both the kNN and
binning estimators. Expected outcome: monotone increase in ``ĝ`` with
diversity — if flat or non-monotone, the knob parameterisation is wrong
and Phase 2 needs revisiting before the Phase 4 headline sweep.

Run:

    uv run python scripts/run_diversity_sweep.py

Emits ``docs/reports/2026-04-19-diversity-knob-validation/`` with a
CSV of raw numbers and a markdown summary.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from loguru import logger

from tmgg.data.datasets.sbm import generate_sbm_batch
from tmgg.experiments.eigenstructure_study import (
    NoisedAnalysisComparator,
    NoisedEigenstructureCollector,
)
from tmgg.experiments.eigenstructure_study.storage import (
    save_dataset_manifest,
    save_decomposition_batch,
)
from tmgg.utils.spectral.laplacian import compute_laplacian


@dataclass(frozen=True)
class SweepRecord:
    """One (diversity, k, estimator) result row."""

    diversity: float
    k: int
    estimator: str
    g_hat: float
    trace_cov_B: float
    ratio: float
    num_graphs: int


DIVERSITIES: tuple[float, ...] = (0.0, 0.33, 0.67, 1.0)
NUM_GRAPHS = 200
NUM_NODES = 50
NOISE_LEVEL = 0.1
K_VALUES: list[int] = [4, 8, 16, 32]
BATCH_SIZE = 50
SEED = 42

# Hyperparameter ranges used at diversity > 0. Their midpoints are the
# fixed-mode reference used at diversity = 0.
NUM_BLOCKS_RANGE: tuple[int, int] = (2, 5)
P_INTRA_RANGE: tuple[float, float] = (0.3, 0.9)
P_INTER_RANGE: tuple[float, float] = (0.01, 0.2)


def _midpoint_int(lo: int, hi: int) -> int:
    return int(round((lo + hi) / 2.0))


def _midpoint_float(lo: float, hi: float) -> float:
    return (lo + hi) / 2.0


def generate_sbm_at_diversity(diversity: float) -> torch.Tensor:
    """Return a stack of ``(NUM_GRAPHS, NUM_NODES, NUM_NODES)`` adjacencies.

    At ``diversity == 0`` uses scalar midpoints (fixed mode, no tuple
    draws); at ``diversity > 0`` uses the full tuple ranges.
    """
    if diversity == 0.0:
        adjacencies = generate_sbm_batch(
            NUM_GRAPHS,
            NUM_NODES,
            num_blocks=_midpoint_int(*NUM_BLOCKS_RANGE),
            p_intra=_midpoint_float(*P_INTRA_RANGE),
            p_inter=_midpoint_float(*P_INTER_RANGE),
            seed=SEED,
        )
    else:
        adjacencies = generate_sbm_batch(
            NUM_GRAPHS,
            NUM_NODES,
            num_blocks=NUM_BLOCKS_RANGE,
            p_intra=P_INTRA_RANGE,
            p_inter=P_INTER_RANGE,
            diversity=diversity,
            seed=SEED,
        )
    return torch.from_numpy(adjacencies).float()


def build_phase1_decompositions(adjacencies: torch.Tensor, output_dir: Path) -> None:
    """Compute clean eigendecompositions and persist in safetensors batches."""
    num_graphs = adjacencies.shape[0]
    batch_idx = 0
    for start in range(0, num_graphs, BATCH_SIZE):
        end = min(start + BATCH_SIZE, num_graphs)
        A_chunk = adjacencies[start:end]

        eig_adj_vals_list: list[torch.Tensor] = []
        eig_adj_vecs_list: list[torch.Tensor] = []
        eig_lap_vals_list: list[torch.Tensor] = []
        eig_lap_vecs_list: list[torch.Tensor] = []
        for i in range(A_chunk.shape[0]):
            A_i = A_chunk[i]
            L_i = compute_laplacian(A_i)
            vals_a, vecs_a = torch.linalg.eigh(A_i)
            vals_l, vecs_l = torch.linalg.eigh(L_i)
            eig_adj_vals_list.append(vals_a)
            eig_adj_vecs_list.append(vecs_a)
            eig_lap_vals_list.append(vals_l)
            eig_lap_vecs_list.append(vecs_l)

        metadata = [{"graph_index": start + j} for j in range(A_chunk.shape[0])]
        save_decomposition_batch(
            output_dir,
            batch_idx,
            torch.stack(eig_adj_vals_list),
            torch.stack(eig_adj_vecs_list),
            torch.stack(eig_lap_vals_list),
            torch.stack(eig_lap_vecs_list),
            A_chunk,
            metadata,
        )
        batch_idx += 1

    save_dataset_manifest(
        output_dir,
        dataset_name=f"sbm_diversity_{output_dir.name}",
        dataset_config={
            "num_nodes": NUM_NODES,
            "num_graphs": num_graphs,
            "num_blocks_range": list(NUM_BLOCKS_RANGE),
            "p_intra_range": list(P_INTRA_RANGE),
            "p_inter_range": list(P_INTER_RANGE),
        },
        num_graphs=num_graphs,
        num_batches=batch_idx,
        seed=SEED,
    )


def sweep(output_root: Path) -> list[SweepRecord]:
    """Run the diversity sweep; return flat records for each (div, k, est)."""
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[SweepRecord] = []

    for diversity in DIVERSITIES:
        logger.info(f"=== diversity={diversity:.2f} ===")
        diversity_dir = output_root / f"d{diversity:.2f}"
        phase1_dir = diversity_dir / "phase1"
        noised_dir = diversity_dir / "noised"
        phase1_dir.mkdir(parents=True, exist_ok=True)
        noised_dir.mkdir(parents=True, exist_ok=True)

        adjacencies = generate_sbm_at_diversity(diversity)
        build_phase1_decompositions(adjacencies, phase1_dir)

        collector = NoisedEigenstructureCollector(
            input_dir=phase1_dir,
            output_dir=noised_dir,
            noise_type="gaussian",
            noise_levels=[NOISE_LEVEL],
            seed=SEED,
            surrogate_k_values=K_VALUES,
        )
        collector.collect()

        comparator = NoisedAnalysisComparator(
            original_dir=phase1_dir,
            noised_base_dir=noised_dir,
        )
        for k in K_VALUES:
            for estimator in ("knn", "bin"):
                result = comparator.compute_improvement_gap_surrogate(
                    NOISE_LEVEL,
                    k=k,
                    estimator=estimator,
                    knn_neighbours=10,
                    num_bins=4,
                )
                results.append(
                    SweepRecord(
                        diversity=diversity,
                        k=k,
                        estimator=estimator,
                        g_hat=result.g_hat,
                        trace_cov_B=result.trace_cov_B,
                        ratio=result.ratio,
                        num_graphs=result.num_graphs,
                    )
                )
    return results


def write_report(results: list[SweepRecord], report_dir: Path) -> None:
    """Write CSV and markdown summary side by side."""
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "diversity_sweep.csv"
    md_path = report_dir / "README.md"

    fieldnames = [
        "diversity",
        "k",
        "estimator",
        "g_hat",
        "trace_cov_B",
        "ratio",
        "num_graphs",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in results:
            writer.writerow(asdict(record))

    # Per-(k, estimator) monotonicity check over diversity.
    lines: list[str] = [
        "# 2026-04-19 — Diversity-knob validation (Phase 3)",
        "",
        (
            "Phase 3 of the improvement-gap plan "
            "(`docs/plans/2026-04-18-improvement-gap-surrogate-and-spectrum-diversity.md`). "
            "Sweeps the SBM diversity knob at four settings and reports the "
            "improvement-gap surrogate ĝ (eq. 18 in the NeurIPS draft) at "
            "`k ∈ {4, 8, 16, 32}` with both kNN (10 neighbours) and binning "
            "(4 quantile bins) estimators. The knob is validated if ĝ increases "
            "monotonically with diversity."
        ),
        "",
        "## Setup",
        "",
        f"- `num_graphs = {NUM_GRAPHS}`, `num_nodes = {NUM_NODES}`, seed = {SEED}",
        (
            f"- Hyperparameter ranges at `diversity > 0`: "
            f"`num_blocks ∈ {list(NUM_BLOCKS_RANGE)}`, "
            f"`p_intra ∈ {list(P_INTRA_RANGE)}`, "
            f"`p_inter ∈ {list(P_INTER_RANGE)}`"
        ),
        (
            f"- Fixed-mode reference (diversity=0) uses the midpoints: "
            f"`num_blocks = {_midpoint_int(*NUM_BLOCKS_RANGE)}`, "
            f"`p_intra = {_midpoint_float(*P_INTRA_RANGE):.3f}`, "
            f"`p_inter = {_midpoint_float(*P_INTER_RANGE):.3f}`"
        ),
        f"- Noise: gaussian @ ε = {NOISE_LEVEL}",
        "",
        "## Results",
        "",
        "| diversity | k | estimator | ĝ | trace(Cov B) | ratio |",
        "|-----------|---|-----------|-----|-----|-----|",
    ]
    for r in results:
        lines.append(
            f"| {r.diversity:.2f} | {r.k} | {r.estimator} | "
            f"{r.g_hat:.4f} | {r.trace_cov_B:.4f} | {r.ratio:.4f} |"
        )
    lines.append("")

    # Monotonicity verdict per (k, estimator).
    lines.append("## Monotonicity verdict")
    lines.append("")
    any_non_monotone = False
    for k in K_VALUES:
        for estimator in ("knn", "bin"):
            series: list[float] = [
                r.g_hat for r in results if r.k == k and r.estimator == estimator
            ]
            assert len(series) == len(DIVERSITIES)
            is_monotone = all(
                series[i] <= series[i + 1] for i in range(len(series) - 1)
            )
            verdict = "monotone ↑" if is_monotone else "NOT monotone"
            if not is_monotone:
                any_non_monotone = True
            formatted = ", ".join(f"{v:.4f}" for v in series)
            lines.append(f"- `k={k}`, `{estimator}`: {verdict}  (series: {formatted})")

    lines.append("")
    lines.append("## Conclusion")
    lines.append("")
    if any_non_monotone:
        lines.append(
            "At least one (k, estimator) series is non-monotone. The knob "
            "parameterisation likely needs adjustment before Phase 4."
        )
    else:
        lines.append(
            "ĝ increases monotonically with diversity across every `(k, estimator)` "
            "cell. The knob is validated and Phase 4 (full-dataset study) is unblocked."
        )
    lines.append("")
    lines.append(f"Raw data: `{csv_path.name}`")
    lines.append("")

    md_path.write_text("\n".join(lines))
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Markdown: {md_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    work_root = repo_root / ".local-storage" / "diversity_sweep_2026-04-19"
    report_dir = repo_root / "docs" / "reports" / "2026-04-19-diversity-knob-validation"
    results = sweep(work_root)
    write_report(results, report_dir)


if __name__ == "__main__":
    main()
