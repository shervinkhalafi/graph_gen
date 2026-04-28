"""Diagnose why orbit_mmd collapses to ~0 in the GDPO SBM sanity check.

Hypotheses (in order of investigation):
    1. orbit-count vectors degenerate (all zeros / identical between gen and ref)
    2. ORCA binary missing or producing silent fallback zeros
    3. kernel / sigma mismatch vs upstream DiGress reference
    4. accidentally comparing generated to itself

Run from the repo root:
    uv run python analysis/digress-loss-check/validate-gdpo-sbm/diagnose_orbit_mmd.py

Fail loudly: no graceful fallbacks, no exception swallowing.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

# Resolve repo root (this file lives at <repo>/analysis/digress-loss-check/validate-gdpo-sbm/)
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from tmgg.data.data_modules.spectre_sbm import SpectreSBMDataModule  # noqa: E402
from tmgg.evaluation.graph_evaluator import compute_orbit_mmd  # noqa: E402
from tmgg.evaluation.mmd_metrics import (  # noqa: E402
    compute_mmd,
    gaussian_tv_kernel,
)
from tmgg.evaluation.orca import _get_binary_path, is_available, run_orca  # noqa: E402

SAMPLES_PATH = (
    REPO_ROOT
    / "analysis/digress-loss-check/validate-gdpo-sbm/outputs/modal-20260424T163008/samples.jsonl"
)


def load_generated_graphs(path: Path) -> list[nx.Graph[Any]]:
    """Load the JSONL of generated graphs into NetworkX objects."""
    graphs: list[nx.Graph[Any]] = []
    with path.open() as f:
        for line in f:
            obj = json.loads(line)
            g: nx.Graph[Any] = nx.empty_graph(obj["num_nodes"])
            g.add_edges_from((u, v) for u, v in obj["edges"])
            graphs.append(g)
    return graphs


def orbit_count_vector(graph: nx.Graph[Any]) -> np.ndarray:
    """Mean-per-node orbit count vector (matches _orca_normalized_counts)."""
    g_clean = graph.copy()
    g_clean.remove_edges_from(nx.selfloop_edges(g_clean))
    counts = run_orca(g_clean)
    return np.sum(counts, axis=0) / g_clean.number_of_nodes()


def summarize_matrix(name: str, mat: np.ndarray) -> None:
    print(f"  {name}: shape={mat.shape}")
    print(
        f"    min={mat.min():.4f}  max={mat.max():.4f}  mean={mat.mean():.4f}  std={mat.std():.4f}"
    )
    print(f"    per-orbit max counts (15 orbits): {mat.max(axis=0).round(3).tolist()}")
    print(f"    per-orbit mean counts:            {mat.mean(axis=0).round(3).tolist()}")
    print(f"    fraction strictly zero entries:   {(mat == 0).mean():.4f}")


def main() -> int:
    print("=" * 70)
    print("Orbit MMD diagnostic for GDPO SBM sanity check")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Hypothesis 2: ORCA binary present and runnable?
    # ------------------------------------------------------------------
    print("\n[H2] ORCA binary availability")
    if not is_available():
        raise RuntimeError(
            "ORCA binary/source is unavailable: this would silently disable "
            "orbit_mmd in GraphEvaluator (set to None, but here orbit_mmd "
            "came back as 1.3e-7, so this is NOT the cause). Investigate "
            "why is_available() returns False."
        )
    print("  is_available(): True")
    binary = _get_binary_path()
    print(f"  binary path: {binary}")
    print(
        f"  binary exists: {binary.exists()}, executable: {binary.stat().st_mode & 0o111 != 0}"
    )
    which_orca = shutil.which("orca")
    print(
        f"  shutil.which('orca') (system PATH): {which_orca!r}  (expected None — we ship our own)"
    )

    # ------------------------------------------------------------------
    # Load 40 generated graphs from the JSONL
    # ------------------------------------------------------------------
    print(f"\n[Load] Generated graphs from {SAMPLES_PATH.name}")
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"Generated samples not found at {SAMPLES_PATH}")
    gen_graphs = load_generated_graphs(SAMPLES_PATH)
    print(f"  Loaded {len(gen_graphs)} generated graphs")
    print(
        f"  Node-count range: "
        f"{min(g.number_of_nodes() for g in gen_graphs)}–"
        f"{max(g.number_of_nodes() for g in gen_graphs)}"
    )
    print(
        f"  Edge-count range: "
        f"{min(g.number_of_edges() for g in gen_graphs)}–"
        f"{max(g.number_of_edges() for g in gen_graphs)}"
    )

    # ------------------------------------------------------------------
    # Load 40 SPECTRE-test SBM reference graphs the same way validate.py does
    # ------------------------------------------------------------------
    print("\n[Load] SPECTRE test SBM reference graphs (via SpectreSBMDataModule)")
    datamodule = SpectreSBMDataModule(batch_size=12, num_workers=0)
    datamodule.setup()
    ref_graphs = datamodule.get_reference_graphs("test", 40)
    print(f"  Loaded {len(ref_graphs)} reference graphs")
    print(
        f"  Node-count range: "
        f"{min(g.number_of_nodes() for g in ref_graphs)}–"
        f"{max(g.number_of_nodes() for g in ref_graphs)}"
    )
    print(
        f"  Edge-count range: "
        f"{min(g.number_of_edges() for g in ref_graphs)}–"
        f"{max(g.number_of_edges() for g in ref_graphs)}"
    )

    # Sanity check H4: are gen and ref the same Python objects?
    print("\n[H4] Are gen and ref the same set of graphs?")
    same_object = gen_graphs is ref_graphs
    print(f"  gen_graphs is ref_graphs: {same_object}  (must be False)")

    # ------------------------------------------------------------------
    # Hypothesis 1: orbit-count vectors per graph
    # ------------------------------------------------------------------
    print("\n[H1] Orbit-count matrices for first 5 graphs of each set")
    n_probe = 5
    gen_orb_full = np.array([orbit_count_vector(g) for g in gen_graphs])
    ref_orb_full = np.array([orbit_count_vector(g) for g in ref_graphs])

    print("  -- Generated set (first 5 graphs, mean per-node orbit counts):")
    for i in range(n_probe):
        print(
            f"    gen[{i}] (n={gen_graphs[i].number_of_nodes()}): {gen_orb_full[i].round(3).tolist()}"
        )
    print("  -- Reference set (first 5 graphs):")
    for i in range(n_probe):
        print(
            f"    ref[{i}] (n={ref_graphs[i].number_of_nodes()}): {ref_orb_full[i].round(3).tolist()}"
        )

    print("\n  Summary stats for FULL orbit-count matrices:")
    summarize_matrix("generated", gen_orb_full)
    summarize_matrix("reference", ref_orb_full)

    # ------------------------------------------------------------------
    # Reproduce the orbit MMD computation as evaluator does it
    # ------------------------------------------------------------------
    print("\n[Reproduce] compute_orbit_mmd(refs, generated, kernel='gaussian_tv')")
    orbit_mmd = compute_orbit_mmd(ref_graphs, gen_graphs, kernel="gaussian_tv")
    print(f"  orbit_mmd (sigma=30.0, default): {orbit_mmd:.6e}")

    # Also the upstream-DiGress kernel sigma value used in spectre_utils orbit_stats_all
    orbit_mmd_s30 = compute_orbit_mmd(
        ref_graphs, gen_graphs, kernel="gaussian_tv", sigma=30.0
    )
    print(f"  orbit_mmd (sigma=30.0, explicit): {orbit_mmd_s30:.6e}")

    # ------------------------------------------------------------------
    # Hypothesis 3: kernel sigma + PMF-normalisation mismatch
    # ------------------------------------------------------------------
    print("\n[H3] Kernel-call diagnostic")
    print("  Inspect the gaussian_tv_kernel call on a (gen, ref) pair")
    x = gen_orb_full[0]
    y = ref_orb_full[0]
    print(f"    raw x sum = {x.sum():.3f}, raw y sum = {y.sum():.3f}")
    print(
        f"    raw |x-y|.sum()/2 (upstream TV on raw counts) = {(np.abs(x - y).sum() / 2.0):.3f}"
    )
    x_pmf = x / x.sum()
    y_pmf = y / y.sum()
    tv_pmf = 0.5 * np.abs(x_pmf - y_pmf).sum()
    print(
        f"    PMF-normalised TV (our gaussian_tv_kernel impl) = {tv_pmf:.4f}  (always in [0,1])"
    )
    sigma = 30.0
    k_pmf = gaussian_tv_kernel(x, y, sigma=sigma)
    print(f"    gaussian_tv_kernel(x, y, sigma={sigma}) = {k_pmf:.6f}")
    print(
        "    NOTE: our kernel internally divides by sums (PMF-normalises) before "
        "computing TV. With TV ∈ [0,1] and sigma=30, kernel ≈ exp(-1/1800) ≈ 0.99944, "
        "so MMD = k11 + k22 - 2*k12 collapses to ~0."
    )

    # Compare with what UPSTREAM does: TV on raw orbit-count vectors, no PMF normalisation.
    def upstream_gaussian_tv(
        x: np.ndarray, y: np.ndarray, sigma: float = 30.0
    ) -> float:
        x = x.astype(float)
        y = y.astype(float)
        m = max(len(x), len(y))
        if len(x) < m:
            x = np.concatenate([x, np.zeros(m - len(x))])
        if len(y) < m:
            y = np.concatenate([y, np.zeros(m - len(y))])
        d = np.abs(x - y).sum() / 2.0
        return float(np.exp(-d * d / (2 * sigma * sigma)))

    def upstream_mmd(
        samples1: list[np.ndarray], samples2: list[np.ndarray], sigma: float = 30.0
    ) -> float:
        # is_hist=False path of upstream compute_mmd: NO PMF normalisation
        def disc(a: list[np.ndarray], b: list[np.ndarray]) -> float:
            tot = 0.0
            for s1 in a:
                for s2 in b:
                    tot += upstream_gaussian_tv(s1, s2, sigma=sigma)
            return tot / max(1, len(a) * len(b))

        return (
            disc(samples1, samples1)
            + disc(samples2, samples2)
            - 2 * disc(samples1, samples2)
        )

    upstream_value = upstream_mmd(list(ref_orb_full), list(gen_orb_full), sigma=30.0)
    print(
        f"\n  Upstream-faithful gaussian_tv MMD on RAW orbit counts (sigma=30): {upstream_value:.6f}"
    )
    print("  Vignac README pinned reference value for orbit_mmd:                0.0462")

    # Sanity: confirm our compute_mmd would produce the same near-zero number
    sanity = compute_mmd(
        list(ref_orb_full), list(gen_orb_full), kernel="gaussian_tv", sigma=30.0
    )
    print(
        f"  Our compute_mmd via PMF-normalising kernel (sigma=30):           {sanity:.6e}"
    )

    # ------------------------------------------------------------------
    # Diagnosis
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)

    # H2 already checked — ORCA binary OK if we got this far.
    h1_zero_gen = (gen_orb_full == 0).all()
    h1_zero_ref = (ref_orb_full == 0).all()
    h1_identical = np.array_equal(gen_orb_full, ref_orb_full)

    if h1_zero_gen or h1_zero_ref:
        msg = "orbit vectors all zero (extractor broken / silent fallback)"
    elif h1_identical:
        msg = "generated and reference orbit vectors are identical (H4: comparing to self)"
    elif abs(upstream_value - 0.0462) < 0.1 and orbit_mmd < 1e-3:
        msg = (
            "kernel sigma + PMF-normalisation mismatch vs upstream DiGress: "
            "our gaussian_tv_kernel PMF-normalises raw orbit counts before TV, "
            "collapsing kernel to ≈1 and MMD to ≈0. Fix: stop PMF-normalising "
            "inside gaussian_tv_kernel (upstream uses raw counts, divides by 2 "
            "for the TV scaling, exp(-d²/(2σ²)))."
        )
    else:
        msg = (
            f"no anomaly found — orbit_mmd={orbit_mmd:.4e}, "
            f"upstream-faithful={upstream_value:.4f}, ref_target=0.0462. "
            "Investigate further (sample-distribution shift?)."
        )
    print(f"  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
