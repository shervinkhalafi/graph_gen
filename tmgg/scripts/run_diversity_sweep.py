"""Phase 3 validation (v2): improvement-gap surrogate vs. diversity knob.

Sweeps the SBM diversity knob at four settings and reports the
improvement-gap surrogate ĝ for multiple (frame_mode, estimator,
conditioning, target) combinations with a permutation-null control. The
knob is validated if the fraction of variance explained (FVE, not
just absolute ĝ) is monotone ↑ in diversity across frame modes, with
the permutation-null FVE staying near zero throughout.

Run:

    uv run python scripts/run_diversity_sweep.py

Overwrites ``docs/reports/2026-04-19-diversity-knob-validation/`` with a
flat CSV and a markdown summary containing the per-seed raw numbers,
per-(cell) mean/std across seeds, and explicit monotonicity verdicts.

Design choices addressing the reviewer-2 audit of the Phase 3 v1 report:

- ``num_blocks`` is frozen at 4 across all diversity levels instead of
  being drawn from ``{2,3,4,5}``. The original range had integer
  midpoint 3.5, which made the diversity=0 reference (``num_blocks=4``)
  asymmetrically biased against the diversity=1 distribution mean (3.5).
  Only ``p_intra`` and ``p_inter`` vary with the knob now, so any ĝ
  growth is unambiguously attributable to spectral-density diversity.
- Every cell is computed for both ``frame_mode=frechet`` (primary) and
  ``frame_mode=per_graph`` (legacy / diagnostic) so the frame-convention
  question is visible in the numbers.
- Every cell is computed a second time with ``permute_features=True``
  to produce the permutation-null FVE. A kNN finite-sample bias would
  show as a large null FVE independent of diversity.
- ``estimator`` sweeps ``{knn_top_k, knn_1d, bin_1d, invariants_knn}``
  so kNN(1-D) and bin(1-D) share a feature space (directly comparable),
  the invariants path (tr, ||·||_F², eigvals) serves as a frame-free
  cross-check, and the original top-k kNN remains as the headline.
- Three seeds give mean ± std per cell; monotonicity verdicts gate on
  the mean series.
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
from tmgg.experiments.eigenstructure_study.noised_collector import (
    ConditioningFeatures,
    FrameMode,
    SurrogateTarget,
)
from tmgg.experiments.eigenstructure_study.storage import (
    save_dataset_manifest,
    save_decomposition_batch,
)
from tmgg.utils.spectral.laplacian import compute_laplacian


@dataclass(frozen=True)
class EstimatorSpec:
    """One estimator configuration in the sweep.

    Attributes
    ----------
    label : str
        Short name used in the report table (e.g. ``"knn_top_k"``).
    estimator : str
        ``"knn"`` or ``"bin"``.
    conditioning : ConditioningFeatures
        Feature space: ``"top_k_eigenvalues"`` or ``"spectral_gap"``.
    target : SurrogateTarget
        ``"matrix"`` (raw ``B``) or ``"invariants"`` (frame-free summaries).
    """

    label: str
    estimator: str
    conditioning: ConditioningFeatures
    target: SurrogateTarget


@dataclass(frozen=True)
class SweepRecord:
    """One result row; matches the CSV schema."""

    seed: int
    diversity: float
    frame_mode: str
    estimator_label: str
    permuted: bool
    k: int
    g_hat: float
    trace_cov_B: float
    fve: float
    num_graphs: int


DIVERSITIES: tuple[float, ...] = (0.0, 0.33, 0.67, 1.0)
SEEDS: tuple[int, ...] = (42, 123, 2024)
NUM_GRAPHS = 200
NUM_NODES = 50
NOISE_LEVEL = 0.1
K_VALUES: list[int] = [4, 8, 16, 32]
BATCH_SIZE = 50

# Hyperparameter ranges used at diversity > 0. num_blocks is frozen
# (see module docstring) so only p_intra / p_inter vary with the knob.
FIXED_NUM_BLOCKS = 4
P_INTRA_RANGE: tuple[float, float] = (0.3, 0.9)
P_INTER_RANGE: tuple[float, float] = (0.01, 0.2)

FRAME_MODES: tuple[FrameMode, ...] = ("frechet", "per_graph")

ESTIMATORS: tuple[EstimatorSpec, ...] = (
    EstimatorSpec("knn_top_k", "knn", "top_k_eigenvalues", "matrix"),
    EstimatorSpec("knn_1d", "knn", "spectral_gap", "matrix"),
    EstimatorSpec("bin_1d", "bin", "spectral_gap", "matrix"),
    EstimatorSpec("invariants_knn", "knn", "top_k_eigenvalues", "invariants"),
)


def _midpoint_float(lo: float, hi: float) -> float:
    return (lo + hi) / 2.0


def generate_sbm_at_diversity(diversity: float, seed: int) -> torch.Tensor:
    """Stack of ``(NUM_GRAPHS, NUM_NODES, NUM_NODES)`` SBM adjacencies.

    At ``diversity == 0`` uses scalar midpoints of the tuple ranges;
    otherwise uses the full tuple ranges scaled by ``diversity``.
    ``num_blocks`` is constant at ``FIXED_NUM_BLOCKS`` regardless of
    diversity.
    """
    if diversity == 0.0:
        adjacencies = generate_sbm_batch(
            NUM_GRAPHS,
            NUM_NODES,
            num_blocks=FIXED_NUM_BLOCKS,
            p_intra=_midpoint_float(*P_INTRA_RANGE),
            p_inter=_midpoint_float(*P_INTER_RANGE),
            seed=seed,
        )
    else:
        adjacencies = generate_sbm_batch(
            NUM_GRAPHS,
            NUM_NODES,
            num_blocks=FIXED_NUM_BLOCKS,
            p_intra=P_INTRA_RANGE,
            p_inter=P_INTER_RANGE,
            diversity=diversity,
            seed=seed,
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
            "num_blocks_fixed": FIXED_NUM_BLOCKS,
            "p_intra_range": list(P_INTRA_RANGE),
            "p_inter_range": list(P_INTER_RANGE),
        },
        num_graphs=num_graphs,
        num_batches=batch_idx,
        seed=0,  # seed propagated via the runner; manifest seed is a placeholder
    )


def sweep_one_dataset(
    phase1_dir: Path,
    noised_root: Path,
    seed: int,
    diversity: float,
) -> list[SweepRecord]:
    """Run every (frame_mode × estimator × permuted) cell for one dataset."""
    records: list[SweepRecord] = []

    for frame_mode in FRAME_MODES:
        noised_dir = noised_root / f"frame_{frame_mode}"
        noised_dir.mkdir(parents=True, exist_ok=True)

        collector = NoisedEigenstructureCollector(
            input_dir=phase1_dir,
            output_dir=noised_dir,
            noise_type="gaussian",
            noise_levels=[NOISE_LEVEL],
            seed=seed,
            surrogate_k_values=K_VALUES,
            frame_mode=frame_mode,
        )
        collector.collect()

        comparator = NoisedAnalysisComparator(
            original_dir=phase1_dir,
            noised_base_dir=noised_dir,
        )

        for spec in ESTIMATORS:
            for k in K_VALUES:
                for permuted in (False, True):
                    result = comparator.compute_improvement_gap_surrogate(
                        NOISE_LEVEL,
                        k=k,
                        estimator=spec.estimator,
                        knn_neighbours=10,
                        num_bins=4,
                        target=spec.target,
                        conditioning=spec.conditioning,
                        permute_features=permuted,
                        permutation_seed=seed,
                    )
                    records.append(
                        SweepRecord(
                            seed=seed,
                            diversity=diversity,
                            frame_mode=frame_mode,
                            estimator_label=spec.label,
                            permuted=permuted,
                            k=k,
                            g_hat=result.g_hat,
                            trace_cov_B=result.trace_cov_B,
                            fve=result.fve,
                            num_graphs=result.num_graphs,
                        )
                    )
    return records


def sweep(output_root: Path) -> list[SweepRecord]:
    """Run the full diversity × seed × frame-mode × estimator sweep."""
    output_root.mkdir(parents=True, exist_ok=True)
    all_records: list[SweepRecord] = []

    for seed in SEEDS:
        for diversity in DIVERSITIES:
            logger.info(f"=== seed={seed}, diversity={diversity:.2f} ===")
            cell_dir = output_root / f"seed_{seed}" / f"d{diversity:.2f}"
            phase1_dir = cell_dir / "phase1"
            phase1_dir.mkdir(parents=True, exist_ok=True)

            adjacencies = generate_sbm_at_diversity(diversity, seed)
            build_phase1_decompositions(adjacencies, phase1_dir)

            all_records.extend(
                sweep_one_dataset(
                    phase1_dir=phase1_dir,
                    noised_root=cell_dir,
                    seed=seed,
                    diversity=diversity,
                )
            )
    return all_records


def _series_mean_std(
    records: list[SweepRecord],
    *,
    frame_mode: str,
    estimator_label: str,
    permuted: bool,
    k: int,
    field: str,
) -> list[tuple[float, float]]:
    """Return mean/std of ``field`` at each diversity level, across seeds."""
    out: list[tuple[float, float]] = []
    for div in DIVERSITIES:
        matching = [
            r
            for r in records
            if r.frame_mode == frame_mode
            and r.estimator_label == estimator_label
            and r.permuted is permuted
            and r.k == k
            and abs(r.diversity - div) < 1e-9
        ]
        vals = torch.tensor([float(getattr(r, field)) for r in matching])
        out.append(
            (
                float(vals.mean().item()),
                float(vals.std().item()) if len(vals) > 1 else 0.0,
            )
        )
    return out


def _is_monotone_increasing(series: list[tuple[float, float]]) -> bool:
    means = [m for m, _ in series]
    return all(means[i] <= means[i + 1] for i in range(len(means) - 1))


def write_report(records: list[SweepRecord], report_dir: Path) -> None:
    """Write flat CSV + narrative markdown with monotonicity verdicts."""
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "diversity_sweep.csv"
    md_path = report_dir / "README.md"

    fieldnames = [
        "seed",
        "diversity",
        "frame_mode",
        "estimator_label",
        "permuted",
        "k",
        "g_hat",
        "trace_cov_B",
        "fve",
        "num_graphs",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    lines: list[str] = [
        "# 2026-04-19 — Diversity-knob validation (Phase 3, v2)",
        "",
        (
            "Phase 3 of the improvement-gap plan "
            "(`docs/plans/2026-04-18-improvement-gap-surrogate-and-spectrum-diversity.md`). "
            "Addresses reviewer-2 audit findings: frame mode exposed, permutation null "
            "reported, FVE monotonicity (not absolute ĝ) used as the success "
            "criterion, `num_blocks` frozen, ≥3 seeds."
        ),
        "",
        "## Setup",
        "",
        f"- `num_graphs = {NUM_GRAPHS}`, `num_nodes = {NUM_NODES}`, seeds = {list(SEEDS)}",
        (
            f"- `num_blocks` frozen at {FIXED_NUM_BLOCKS} across all diversity levels; "
            f"knob varies `p_intra ∈ {list(P_INTRA_RANGE)}`, "
            f"`p_inter ∈ {list(P_INTER_RANGE)}` scaled by diversity."
        ),
        (
            f"- Fixed-mode reference (diversity=0) uses the midpoints: "
            f"`p_intra = {_midpoint_float(*P_INTRA_RANGE):.3f}`, "
            f"`p_inter = {_midpoint_float(*P_INTER_RANGE):.3f}`"
        ),
        f"- Noise: gaussian @ ε = {NOISE_LEVEL}",
        "- Frame modes: `frechet` (default, dataset-wide common frame via extrinsic "
        "Grassmannian mean) and `per_graph` (align each noisy V̂ to that graph's clean V; "
        "retained as diagnostic — does not satisfy the common-frame requirement of "
        "eq. (18) across heterogeneous datasets).",
        "- Estimators: "
        "`knn_top_k` (kNN with k-dim top-k eigenvalues on raw B), "
        "`knn_1d` and `bin_1d` (same 1-D spectral-gap feature, comparable), "
        "`invariants_knn` (kNN on frame-invariant summaries of B — frame-free cross-check).",
        "- Permutation null: every cell is also computed with conditioning features "
        "shuffled; a calibrated estimator returns `FVE ≈ 0` under the null.",
        "",
        "## Monotonicity verdicts",
        "",
        (
            "Success criterion per cell: **mean FVE** across seeds is "
            "monotone non-decreasing in diversity (0 → 0.33 → 0.67 → 1.0). "
            "Absolute ĝ is also reported but is not the gate — it tracks "
            "`trace(Cov B)` which mechanically grows with diversity."
        ),
        "",
        "### Real features (permutation off)",
        "",
        "| frame | estimator | k | FVE series (mean±std) | ĝ monotone? | FVE monotone? |",
        "|---|---|---|---|---|---|",
    ]

    def _fmt_series(series: list[tuple[float, float]]) -> str:
        return ", ".join(f"{m:.3f}±{s:.3f}" for m, s in series)

    any_real_fve_non_monotone = False
    for frame_mode in FRAME_MODES:
        for spec in ESTIMATORS:
            for k in K_VALUES:
                g_series = _series_mean_std(
                    records,
                    frame_mode=frame_mode,
                    estimator_label=spec.label,
                    permuted=False,
                    k=k,
                    field="g_hat",
                )
                r_series = _series_mean_std(
                    records,
                    frame_mode=frame_mode,
                    estimator_label=spec.label,
                    permuted=False,
                    k=k,
                    field="fve",
                )
                g_mon = _is_monotone_increasing(g_series)
                r_mon = _is_monotone_increasing(r_series)
                if not r_mon:
                    any_real_fve_non_monotone = True
                lines.append(
                    f"| {frame_mode} | {spec.label} | {k} | {_fmt_series(r_series)} "
                    f"| {'✓' if g_mon else '✗'} | {'✓' if r_mon else '✗'} |"
                )
    lines.append("")

    lines.extend(
        [
            "### Permutation null (features shuffled; should be ≈0 across diversity)",
            "",
            "| frame | estimator | k | null FVE series (mean±std) | null FVE max |",
            "|---|---|---|---|---|",
        ]
    )
    any_null_fve_elevated = False
    for frame_mode in FRAME_MODES:
        for spec in ESTIMATORS:
            for k in K_VALUES:
                null_fve_series = _series_mean_std(
                    records,
                    frame_mode=frame_mode,
                    estimator_label=spec.label,
                    permuted=True,
                    k=k,
                    field="fve",
                )
                null_max = max(m for m, _ in null_fve_series)
                if null_max > 0.30:
                    any_null_fve_elevated = True
                lines.append(
                    f"| {frame_mode} | {spec.label} | {k} | "
                    f"{_fmt_series(null_fve_series)} | {null_max:.3f} |"
                )
    lines.append("")

    lines.append("## Conclusion")
    lines.append("")
    if any_real_fve_non_monotone:
        lines.append(
            "At least one (frame, estimator, k) cell has a non-monotone mean FVE "
            "across diversity. Phase 4 still blocked — investigate the failing cell."
        )
    elif any_null_fve_elevated:
        lines.append(
            "All real-feature FVEs are monotone, but at least one permutation-null "
            "FVE exceeds 0.30 — this signals material finite-sample estimator bias. "
            "Report the real-minus-null difference rather than the raw FVE if "
            "proceeding to Phase 4."
        )
    else:
        lines.append(
            "Mean FVEs monotone across every cell AND permutation-null FVEs "
            "bounded below 0.30 across every cell. Phase 4 unblocked."
        )
    lines.append("")
    lines.append(f"Raw data: `{csv_path.name}` ({len(records)} rows).")
    lines.append("")

    md_path.write_text("\n".join(lines))
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Markdown: {md_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    work_root = repo_root / ".local-storage" / "diversity_sweep_2026-04-19"
    report_dir = repo_root / "docs" / "reports" / "2026-04-19-diversity-knob-validation"
    records = sweep(work_root)
    write_report(records, report_dir)


if __name__ == "__main__":
    main()
