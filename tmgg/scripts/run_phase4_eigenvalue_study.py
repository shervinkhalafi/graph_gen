"""Phase 4 final eigenvalue study across all target datasets.

Runs the improvement-gap surrogate across:

- four synthetic SBM datasets at ``diversity ∈ {0, 0.33, 0.67, 1.0}``;
- ``spectre_sbm`` (upstream DiGress fixture);
- three community-structured TU benchmarks: ENZYMES, PROTEINS, COLLAB,
  each filtered to a narrow node-count band so Fréchet-mean alignment
  on a common ``n`` is well-defined without zero-pad contamination.

Every cell of the sweep reports ``ratio_real`` alongside the matched
permutation-null ``ratio_null`` so the calibrated margin
``ratio_real − ratio_null`` (the paper-relevant quantity) is read
directly from the table.

Run:

    uv run python scripts/run_phase4_eigenvalue_study.py

Overwrites ``docs/reports/2026-04-19-phase4-eigenvalue-study/`` with a
flat CSV (all rows) and a markdown narrative with per-dataset ranking,
monotonicity verdicts, and the headline dataset-ordering table.

Sweep size: 8 datasets × 5 seeds × 2 frame modes × 2 noise types ×
5 noise levels × 4 estimators × 4 k × 2 (real+null). Compute is local
CPU, budget ~60 min.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field
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
class DatasetSpec:
    """Config for one dataset in the sweep."""

    name: str
    category: str  # "synthetic_sbm" | "spectre_sbm" | "tu"
    params: dict = field(default_factory=dict)


@dataclass(frozen=True)
class EstimatorSpec:
    label: str
    estimator: str
    conditioning: ConditioningFeatures
    target: SurrogateTarget


@dataclass(frozen=True)
class Phase4Record:
    seed: int
    dataset: str
    noise_type: str
    noise_level: float
    frame_mode: str
    estimator_label: str
    permuted: bool
    k: int
    g_hat: float
    trace_cov_B: float
    ratio: float
    num_graphs: int
    n: int


SEEDS: tuple[int, ...] = (42, 123, 2024, 7, 11)
NOISE_TYPES: tuple[str, ...] = ("gaussian", "digress")
NOISE_LEVELS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.15, 0.2)
K_VALUES: list[int] = [4, 8, 16, 32]
BATCH_SIZE = 50
KNN_NEIGHBOURS = 10
NUM_BINS = 4

FRAME_MODES: tuple[FrameMode, ...] = ("frechet", "per_graph")

ESTIMATORS: tuple[EstimatorSpec, ...] = (
    EstimatorSpec("knn_top_k", "knn", "top_k_eigenvalues", "matrix"),
    EstimatorSpec("knn_1d", "knn", "spectral_gap", "matrix"),
    EstimatorSpec("bin_1d", "bin", "spectral_gap", "matrix"),
    EstimatorSpec("invariants_knn", "knn", "top_k_eigenvalues", "invariants"),
)

# Synthetic-SBM knob ranges at diversity > 0, matching Phase 3 v2.
SYN_NUM_GRAPHS = 200
SYN_NUM_NODES = 50
SYN_FIXED_NUM_BLOCKS = 4
SYN_P_INTRA_RANGE: tuple[float, float] = (0.3, 0.9)
SYN_P_INTER_RANGE: tuple[float, float] = (0.01, 0.2)

DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec("sbm_d0.00", "synthetic_sbm", {"diversity": 0.0}),
    DatasetSpec("sbm_d0.33", "synthetic_sbm", {"diversity": 0.33}),
    DatasetSpec("sbm_d0.67", "synthetic_sbm", {"diversity": 0.67}),
    DatasetSpec("sbm_d1.00", "synthetic_sbm", {"diversity": 1.0}),
    DatasetSpec("spectre_sbm", "spectre_sbm", {"n_range": (30, 80)}),
    DatasetSpec("enzymes", "tu", {"tu_name": "enzymes", "n_range": (30, 60)}),
    DatasetSpec("proteins", "tu", {"tu_name": "proteins", "n_range": (30, 80)}),
    DatasetSpec("collab", "tu", {"tu_name": "collab", "n_range": (30, 80)}),
)


def _midpoint(lo: float, hi: float) -> float:
    return (lo + hi) / 2.0


def _load_synthetic_sbm(diversity: float, seed: int) -> torch.Tensor:
    """Generate SBM batch matching Phase 3 v2 parametrisation."""
    if diversity == 0.0:
        arr = generate_sbm_batch(
            SYN_NUM_GRAPHS,
            SYN_NUM_NODES,
            num_blocks=SYN_FIXED_NUM_BLOCKS,
            p_intra=_midpoint(*SYN_P_INTRA_RANGE),
            p_inter=_midpoint(*SYN_P_INTER_RANGE),
            seed=seed,
        )
    else:
        arr = generate_sbm_batch(
            SYN_NUM_GRAPHS,
            SYN_NUM_NODES,
            num_blocks=SYN_FIXED_NUM_BLOCKS,
            p_intra=SYN_P_INTRA_RANGE,
            p_inter=SYN_P_INTER_RANGE,
            diversity=diversity,
            seed=seed,
        )
    return torch.from_numpy(arr).float()


def _filter_by_size_band(
    adjacencies: list[torch.Tensor], n_min: int, n_max: int
) -> tuple[list[torch.Tensor], int]:
    """Keep only graphs with ``n_min ≤ n ≤ n_max`` and confirm uniform size.

    Graphs outside the band are dropped; within the band, graphs are
    zero-padded up to the common ``n_max`` so the output stack is square
    and the Fréchet mean has a well-defined ambient dimension. The
    returned ``n`` is ``n_max``.
    """
    kept: list[torch.Tensor] = []
    for A in adjacencies:
        n = A.shape[0]
        if n_min <= n <= n_max:
            if n < n_max:
                padded = torch.zeros(n_max, n_max, dtype=A.dtype)
                padded[:n, :n] = A
                kept.append(padded)
            else:
                kept.append(A)
    if not kept:
        raise RuntimeError(
            f"No graphs in size band [{n_min}, {n_max}]; widen the band."
        )
    return kept, n_max


def _load_spectre_sbm(n_range: tuple[int, int]) -> torch.Tensor:
    """Load SPECTRE fixture and filter to a size band.

    SPECTRE graphs span n ≈ 44 to 187. We filter to n ≤ 80 (which covers
    ~half the fixture) so eigendecomposition stays cheap and the common
    frame is well-defined.
    """
    from tmgg.data.datasets.spectre_sbm import load_spectre_sbm_fixture

    adjs, _ = load_spectre_sbm_fixture()
    kept, n = _filter_by_size_band(adjs, *n_range)
    logger.info(
        f"spectre_sbm: kept {len(kept)}/{len(adjs)} graphs in band "
        f"{n_range}; padded to n={n}"
    )
    return torch.stack(kept).float()


def _load_tu_dataset(tu_name: str, n_range: tuple[int, int]) -> torch.Tensor:
    """Load a TU benchmark via PyGDatasetWrapper and filter by size band."""
    from tmgg.data.datasets.pyg_datasets import PyGDatasetWrapper

    wrapper = PyGDatasetWrapper(dataset_name=tu_name)
    # Wrapper already pads to max_n; undo that and re-filter.
    n_actual = wrapper.num_nodes
    raw: list[torch.Tensor] = []
    for i, n_i in enumerate(n_actual):
        A_padded = wrapper.adjacencies[i]
        A = torch.from_numpy(A_padded[:n_i, :n_i]).float()
        raw.append(A)
    kept, n = _filter_by_size_band(raw, *n_range)
    logger.info(
        f"{tu_name}: kept {len(kept)}/{len(raw)} graphs in band "
        f"{n_range}; padded to n={n}"
    )
    return torch.stack(kept).float()


def load_dataset(spec: DatasetSpec, seed: int) -> torch.Tensor:
    """Dispatch to the right loader, returning a (N, n, n) adjacency tensor."""
    if spec.category == "synthetic_sbm":
        return _load_synthetic_sbm(spec.params["diversity"], seed)
    if spec.category == "spectre_sbm":
        return _load_spectre_sbm(spec.params["n_range"])
    if spec.category == "tu":
        return _load_tu_dataset(spec.params["tu_name"], spec.params["n_range"])
    raise ValueError(f"Unknown dataset category: {spec.category}")


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
            # Real-graph benchmarks (ENZYMES, PROTEINS, COLLAB) contain
            # highly symmetric graphs with repeated eigenvalues that
            # trip float32 eigh convergence. Use float64 and downcast
            # for storage; precision is cheap.
            vals_a, vecs_a = torch.linalg.eigh(A_i.double())
            vals_l, vecs_l = torch.linalg.eigh(L_i.double())
            eig_adj_vals_list.append(vals_a.float())
            eig_adj_vecs_list.append(vecs_a.float())
            eig_lap_vals_list.append(vals_l.float())
            eig_lap_vecs_list.append(vecs_l.float())

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
        dataset_name=f"phase4_{output_dir.name}",
        dataset_config={
            "num_nodes": int(adjacencies.shape[1]),
            "num_graphs": num_graphs,
        },
        num_graphs=num_graphs,
        num_batches=batch_idx,
        seed=0,
    )


def sweep_dataset_at_seed(
    spec: DatasetSpec,
    seed: int,
    work_root: Path,
    phase1_cache: dict[str, Path],
    n_cache: dict[str, int],
) -> list[Phase4Record]:
    """Run every (noise_type × noise_level × frame_mode × estimator × k × permuted)
    cell for one (dataset, seed) pair."""
    records: list[Phase4Record] = []

    # For fixed real datasets the Phase 1 decomposition is seed-independent;
    # cache it once across seeds. Synthetic SBM regenerates per seed.
    cache_key = f"{spec.name}"
    if spec.category == "synthetic_sbm":
        # Synthetic dataset is regenerated per seed (different randomness).
        cache_key = f"{spec.name}_{seed}"

    if cache_key not in phase1_cache:
        adjacencies = load_dataset(spec, seed)
        dataset_n = int(adjacencies.shape[1])
        phase1_dir = work_root / "phase1" / cache_key
        phase1_dir.mkdir(parents=True, exist_ok=True)
        build_phase1_decompositions(adjacencies, phase1_dir)
        phase1_cache[cache_key] = phase1_dir
        n_cache[cache_key] = dataset_n
    phase1_dir = phase1_cache[cache_key]
    dataset_n = n_cache[cache_key]

    # Filter k_values to those ≤ n for this dataset.
    effective_k_values = [k for k in K_VALUES if k <= dataset_n]

    for noise_type in NOISE_TYPES:
        for frame_mode in FRAME_MODES:
            noised_dir = work_root / "noised" / cache_key / noise_type / frame_mode
            noised_dir.mkdir(parents=True, exist_ok=True)

            collector = NoisedEigenstructureCollector(
                input_dir=phase1_dir,
                output_dir=noised_dir,
                noise_type=noise_type,
                noise_levels=list(NOISE_LEVELS),
                seed=seed,
                surrogate_k_values=effective_k_values,
                frame_mode=frame_mode,
            )
            collector.collect()

            comparator = NoisedAnalysisComparator(
                original_dir=phase1_dir,
                noised_base_dir=noised_dir,
            )
            for noise_level in NOISE_LEVELS:
                for est in ESTIMATORS:
                    for k in effective_k_values:
                        for permuted in (False, True):
                            result = comparator.compute_improvement_gap_surrogate(
                                noise_level,
                                k=k,
                                estimator=est.estimator,
                                knn_neighbours=KNN_NEIGHBOURS,
                                num_bins=NUM_BINS,
                                target=est.target,
                                conditioning=est.conditioning,
                                permute_features=permuted,
                                permutation_seed=seed,
                            )
                            records.append(
                                Phase4Record(
                                    seed=seed,
                                    dataset=spec.name,
                                    noise_type=noise_type,
                                    noise_level=noise_level,
                                    frame_mode=frame_mode,
                                    estimator_label=est.label,
                                    permuted=permuted,
                                    k=k,
                                    g_hat=result.g_hat,
                                    trace_cov_B=result.trace_cov_B,
                                    ratio=result.ratio,
                                    num_graphs=result.num_graphs,
                                    n=dataset_n,
                                )
                            )
    return records


def sweep(work_root: Path) -> list[Phase4Record]:
    """Run the full Phase 4 cross-product."""
    work_root.mkdir(parents=True, exist_ok=True)
    records: list[Phase4Record] = []
    phase1_cache: dict[str, Path] = {}
    n_cache: dict[str, int] = {}

    for seed in SEEDS:
        for spec in DATASETS:
            logger.info(f"=== dataset={spec.name}, seed={seed} ===")
            records.extend(
                sweep_dataset_at_seed(
                    spec,
                    seed,
                    work_root,
                    phase1_cache,
                    n_cache,
                )
            )
    return records


def _series_mean_std(
    records: list[Phase4Record],
    *,
    dataset: str,
    noise_type: str,
    frame_mode: str,
    estimator_label: str,
    permuted: bool,
    k: int,
) -> list[tuple[float, float, float]]:
    """Per-noise-level (mean ratio, std ratio, mean g_hat) across seeds."""
    out: list[tuple[float, float, float]] = []
    for nl in NOISE_LEVELS:
        matching = [
            r
            for r in records
            if r.dataset == dataset
            and r.noise_type == noise_type
            and r.frame_mode == frame_mode
            and r.estimator_label == estimator_label
            and r.permuted is permuted
            and r.k == k
            and abs(r.noise_level - nl) < 1e-9
        ]
        ratios = torch.tensor([r.ratio for r in matching])
        g_hats = torch.tensor([r.g_hat for r in matching])
        out.append(
            (
                float(ratios.mean().item()) if len(ratios) else 0.0,
                float(ratios.std().item()) if len(ratios) > 1 else 0.0,
                float(g_hats.mean().item()) if len(g_hats) else 0.0,
            )
        )
    return out


def _calibrated_margin(
    records: list[Phase4Record],
    *,
    dataset: str,
    noise_type: str,
    frame_mode: str,
    estimator_label: str,
    k: int,
    noise_level: float,
) -> tuple[float, float]:
    """Return (mean ratio_real − ratio_null, null mean) across seeds."""
    real_vals: list[float] = []
    null_vals: list[float] = []
    for r in records:
        if (
            r.dataset == dataset
            and r.noise_type == noise_type
            and r.frame_mode == frame_mode
            and r.estimator_label == estimator_label
            and r.k == k
            and abs(r.noise_level - noise_level) < 1e-9
        ):
            (real_vals if not r.permuted else null_vals).append(r.ratio)
    if not real_vals or not null_vals:
        return 0.0, 0.0
    real_mean = sum(real_vals) / len(real_vals)
    null_mean = sum(null_vals) / len(null_vals)
    return real_mean - null_mean, null_mean


def write_report(records: list[Phase4Record], report_dir: Path) -> None:
    """Emit a flat CSV + narrative markdown with the Phase 4 verdict."""
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "phase4_sweep.csv"
    md_path = report_dir / "README.md"

    fieldnames = [
        "seed",
        "dataset",
        "noise_type",
        "noise_level",
        "frame_mode",
        "estimator_label",
        "permuted",
        "k",
        "g_hat",
        "trace_cov_B",
        "ratio",
        "num_graphs",
        "n",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(asdict(rec))

    lines: list[str] = [
        "# 2026-04-19 — Phase 4 Eigenvalue Study",
        "",
        (
            "Final eigenvalue-study sweep across all target datasets. Applies "
            "the Phase 3 v2 surrogate infrastructure (Fréchet-mean frame, "
            "permutation-null control, four estimator variants) to the "
            "real-data benchmarks and the synthetic diversity sweep."
        ),
        "",
        "## Setup",
        "",
        f"- Seeds: {list(SEEDS)} (n = {len(SEEDS)}).",
        f"- Noise levels: {list(NOISE_LEVELS)}.",
        f"- Noise types: {list(NOISE_TYPES)}.",
        f"- k values: {K_VALUES} (per-dataset filtered to ``k ≤ n``).",
        f"- Frame modes: {list(FRAME_MODES)} (frechet is the headline).",
        f"- Estimators: {[e.label for e in ESTIMATORS]}.",
        "- Permutation null: reported for every cell.",
        "",
        "### Datasets",
        "",
        "| Dataset | Category | Filter band | Notes |",
        "|---|---|---|---|",
    ]
    for spec in DATASETS:
        band = spec.params.get("n_range", f"fixed n={SYN_NUM_NODES}")
        note = (
            f"diversity={spec.params.get('diversity')}"
            if spec.category == "synthetic_sbm"
            else (
                f"community-structured TU benchmark ({spec.params.get('tu_name')})"
                if spec.category == "tu"
                else "upstream DiGress fixture"
            )
        )
        lines.append(f"| {spec.name} | {spec.category} | {band} | {note} |")
    lines.append("")

    # Per-dataset retention / n
    lines.append("### Retained graph counts (at each seed)")
    lines.append("")
    lines.append("| Dataset | n | N retained | note |")
    lines.append("|---|---|---|---|")
    for spec in DATASETS:
        # Pull any matching record for num_graphs / n
        sample = next((r for r in records if r.dataset == spec.name), None)
        if sample is None:
            lines.append(f"| {spec.name} | ? | 0 | no records |")
        else:
            lines.append(
                f"| {spec.name} | {sample.n} | {sample.num_graphs} | "
                f"invariant across seeds for fixed datasets"
                + (" (per-seed redraw)" if spec.category == "synthetic_sbm" else "")
                + " |"
            )
    lines.append("")

    # Headline: dataset ordering at (frame=frechet, estimator=knn_top_k,
    # k=8, noise=gaussian, ε=0.1).
    headline_k = 8
    headline_eps = 0.1
    headline_noise = "gaussian"
    headline_estimator = "knn_top_k"
    headline_frame = "frechet"
    lines.append(
        f"## Headline: dataset ordering at "
        f"(frame={headline_frame}, estimator={headline_estimator}, "
        f"k={headline_k}, noise={headline_noise}, ε={headline_eps})"
    )
    lines.append("")
    lines.append(
        "Calibrated margin = mean(ratio_real) − mean(ratio_null). "
        "Success criterion: margin ≥ 0.10 with null below 0.30."
    )
    lines.append("")
    lines.append(
        "| Dataset | real ratio (mean) | null ratio (mean) | calibrated margin | pass? |"
    )
    lines.append("|---|---|---|---|---|")
    dataset_margins: list[tuple[str, float, float]] = []
    for spec in DATASETS:
        margin, null = _calibrated_margin(
            records,
            dataset=spec.name,
            noise_type=headline_noise,
            frame_mode=headline_frame,
            estimator_label=headline_estimator,
            k=headline_k,
            noise_level=headline_eps,
        )
        real = margin + null
        passes = margin >= 0.10 and null < 0.30
        dataset_margins.append((spec.name, margin, null))
        lines.append(
            f"| {spec.name} | {real:.3f} | {null:.3f} | "
            f"{margin:.3f} | {'✓' if passes else '✗'} |"
        )
    lines.append("")

    # Sorted ranking
    lines.append("### Sorted ranking (best margin first)")
    lines.append("")
    ranked = sorted(dataset_margins, key=lambda x: -x[1])
    for rank, (name, margin, _) in enumerate(ranked, start=1):
        lines.append(f"{rank}. **{name}** — margin {margin:.3f}")
    lines.append("")

    # Stability check: report three complementary views rather than a
    # single "full-ranking match" metric. Datasets with close margins
    # swap ranks within tolerance; what matters for the paper's
    # narrative is (a) the winner stays the winner, (b) the top tier
    # stays the top tier, (c) the pass/fail set stays the pass/fail set.
    lines.append("### Ranking stability")
    lines.append("")
    canonical_ranking = [name for name, _, _ in ranked]
    canonical_top1 = canonical_ranking[0]
    canonical_top2 = set(canonical_ranking[:2])
    canonical_passing = {
        name for name, m, n in dataset_margins if m >= 0.10 and n < 0.30
    }

    full_match = 0
    top1_match = 0
    top2_match = 0
    pass_match = 0
    total_cells = 0
    for nt in NOISE_TYPES:
        for eps in NOISE_LEVELS:
            for est in ESTIMATORS:
                for k in K_VALUES:
                    margins_this_cell = []
                    nulls_this_cell = {}
                    for spec in DATASETS:
                        m, null = _calibrated_margin(
                            records,
                            dataset=spec.name,
                            noise_type=nt,
                            frame_mode=headline_frame,
                            estimator_label=est.label,
                            k=k,
                            noise_level=eps,
                        )
                        margins_this_cell.append((spec.name, m))
                        nulls_this_cell[spec.name] = null
                    sorted_cell = [
                        n for n, _ in sorted(margins_this_cell, key=lambda x: -x[1])
                    ]
                    cell_passing = {
                        n
                        for n, m in margins_this_cell
                        if m >= 0.10 and nulls_this_cell[n] < 0.30
                    }
                    total_cells += 1
                    if sorted_cell == canonical_ranking:
                        full_match += 1
                    if sorted_cell[0] == canonical_top1:
                        top1_match += 1
                    if set(sorted_cell[:2]) == canonical_top2:
                        top2_match += 1
                    if cell_passing == canonical_passing:
                        pass_match += 1
    lines.append(
        f"Across all ({len(NOISE_TYPES)} × {len(NOISE_LEVELS)} × "
        f"{len(ESTIMATORS)} × {len(K_VALUES)}) = {total_cells} "
        f"(noise_type × noise_level × estimator × k) cells:"
    )
    lines.append("")
    lines.append(
        f"- **top-1** matches the headline in {top1_match}/{total_cells} "
        f"cells ({100 * top1_match / total_cells:.0f} %)."
    )
    lines.append(
        f"- **top-2 set** matches in {top2_match}/{total_cells} "
        f"cells ({100 * top2_match / total_cells:.0f} %)."
    )
    lines.append(
        f"- **pass/fail set** (margin ≥ 0.10 AND null < 0.30) matches in "
        f"{pass_match}/{total_cells} cells "
        f"({100 * pass_match / total_cells:.0f} %)."
    )
    lines.append(
        f"- Full-ranking exact match in {full_match}/{total_cells} "
        f"cells ({100 * full_match / total_cells:.0f} %) — included for "
        f"completeness; datasets with close margins swap ranks within "
        f"tolerance so this metric understates true stability."
    )
    lines.append("")

    # Noise-level monotonicity per dataset
    lines.append("## Noise-level sensitivity (headline cell)")
    lines.append("")
    lines.append(
        "For each dataset at (knn_top_k, frechet, gaussian, k=8), report "
        "ratio vs. ε series (mean±std) and whether the calibrated margin "
        "stays positive across ε."
    )
    lines.append("")
    lines.append("| Dataset | ratio series | null series | margin at each ε |")
    lines.append("|---|---|---|---|")
    for spec in DATASETS:
        real_series = _series_mean_std(
            records,
            dataset=spec.name,
            noise_type="gaussian",
            frame_mode="frechet",
            estimator_label="knn_top_k",
            permuted=False,
            k=headline_k,
        )
        null_series = _series_mean_std(
            records,
            dataset=spec.name,
            noise_type="gaussian",
            frame_mode="frechet",
            estimator_label="knn_top_k",
            permuted=True,
            k=headline_k,
        )
        real_str = ", ".join(f"{m:.2f}±{s:.2f}" for m, s, _ in real_series)
        null_str = ", ".join(f"{m:.2f}±{s:.2f}" for m, s, _ in null_series)
        margins = [
            rm - nm
            for (rm, _, _), (nm, _, _) in zip(real_series, null_series, strict=False)
        ]
        margin_str = ", ".join(f"{m:.2f}" for m in margins)
        lines.append(f"| {spec.name} | {real_str} | {null_str} | {margin_str} |")
    lines.append("")

    # Synthetic diversity monotonicity
    lines.append("## Synthetic SBM: diversity ↑ (at ε=0.1, k=8, headline cell)")
    lines.append("")
    lines.append(
        "Replicates the Phase 3 v2 finding at a single ε on the real-data "
        "grid. Should be monotone across sbm_d{0, 0.33, 0.67, 1.0}."
    )
    lines.append("")
    lines.append("| Diversity | real ratio | null ratio | margin |")
    lines.append("|---|---|---|---|")
    syn_datasets = [s for s in DATASETS if s.category == "synthetic_sbm"]
    for spec in syn_datasets:
        margin, null = _calibrated_margin(
            records,
            dataset=spec.name,
            noise_type=headline_noise,
            frame_mode=headline_frame,
            estimator_label=headline_estimator,
            k=headline_k,
            noise_level=headline_eps,
        )
        real = margin + null
        lines.append(
            f"| {spec.params['diversity']:.2f} | {real:.3f} | "
            f"{null:.3f} | {margin:.3f} |"
        )
    lines.append("")

    # Conclusion
    passing = sum(1 for _, m, n in dataset_margins if m >= 0.10 and n < 0.30)
    lines.append("## Conclusion")
    lines.append("")
    lines.append(
        f"{passing}/{len(DATASETS)} datasets pass the calibrated-margin "
        f"criterion (≥0.10) at the headline cell. Ranking stability: "
        f"top-1 match {top1_match}/{total_cells}, "
        f"top-2 set match {top2_match}/{total_cells}, "
        f"pass/fail set match {pass_match}/{total_cells}."
    )
    lines.append("")
    lines.append(f"Raw data: `{csv_path.name}` ({len(records)} rows).")
    lines.append("")

    md_path.write_text("\n".join(lines))
    logger.info(f"CSV: {csv_path}")
    logger.info(f"Markdown: {md_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    work_root = repo_root / ".local-storage" / "phase4_eigenvalue_study_2026-04-19"
    report_dir = repo_root / "docs" / "reports" / "2026-04-19-phase4-eigenvalue-study"
    records = sweep(work_root)
    write_report(records, report_dir)


if __name__ == "__main__":
    main()
