#!/usr/bin/env -S uv run --quiet --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "tmgg",
# ]
# [tool.uv.sources]
# tmgg = { path = "..", editable = true }
# ///
"""Compute and cache the per-dataset ``MMD²(train, test)`` baseline.

DiGress (Vignac et al. 2023) Appendix F.1 reports MMD as a ratio
``r = MMD²(generated, test) / MMD²(training, test)``. The denominator is
dataset-determined and kernel-determined — once per dataset, then reused
by every training run. This script materialises the train / test splits
through the project's datamodules, computes ``MMD²(train, test)`` for the
four metrics our pipeline emits (degree, clustering, spectral, orbit),
and writes one JSON per dataset to ``data/eval/mmd_baselines/``.

Schema and loader live in ``src/tmgg/evaluation/mmd_baselines.py``.

Usage
-----
    # All built-in datasets:
    uv run scripts/compute_mmd_baselines.py --all

    # One dataset:
    uv run scripts/compute_mmd_baselines.py --dataset spectre_sbm

    # Custom output root:
    uv run scripts/compute_mmd_baselines.py --all --out path/to/baselines

The script is **not** idempotent on existing files by default — pass
``--skip-existing`` to keep a previously computed baseline rather than
recomputing it. Parameter overrides (kernel/sigma/bins) get embedded in
the file's fingerprint, so they can never silently overwrite a baseline
computed under a different configuration.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import logging
import math
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import networkx as nx  # pyright: ignore[reportMissingModuleSource]
from torch_geometric.data import Data  # pyright: ignore[reportMissingImports]
from torch_geometric.utils import to_networkx  # pyright: ignore[reportMissingImports]

from tmgg.evaluation.graph_evaluator import compute_orbit_mmd
from tmgg.evaluation.mmd_baselines import (
    MMDBaseline,
    MMDBaselineParams,
    baseline_path,
    save_baseline,
)
from tmgg.evaluation.mmd_metrics import compute_mmd_metrics

logger = logging.getLogger("compute_mmd_baselines")


# Production sigma defaults: these match
# ``configs/models/discrete/discrete_sbm_official.yaml`` and the
# ``compute_orbit_mmd`` default in ``graph_evaluator.py``. Changing any
# of these changes the meaning of ``gen-val/*_mmd`` numbers and MUST be
# reflected in the produced baseline's fingerprint.
DEFAULT_KERNEL = "gaussian_tv"
DEFAULT_DEGREE_SIGMA = 1.0
DEFAULT_CLUSTERING_SIGMA = 0.1  # DiGress uses 0.1 for clustering
DEFAULT_SPECTRAL_SIGMA = 1.0
DEFAULT_ORBIT_SIGMA = 30.0
DEFAULT_CLUSTERING_BINS = 100
DEFAULT_SPECTRAL_BINS = 200


# ---------------------------------------------------------------------------
# Datamodule construction
# ---------------------------------------------------------------------------


def _build_spectre_sbm() -> Any:
    from tmgg.data.data_modules.spectre_sbm import SpectreSBMDataModule

    return SpectreSBMDataModule()


def _build_spectre_planar() -> Any:
    from tmgg.data.data_modules.spectre_planar import SpectrePlanarDataModule

    return SpectrePlanarDataModule()


def _build_pyg_enzymes() -> Any:
    from tmgg.data import GraphDataModule

    # Mirrors ``configs/data/pyg_enzymes.yaml``. Splits are 70 / 10 / 20.
    return GraphDataModule(
        graph_type="enzymes",
        graph_config={
            "num_nodes": 126,  # published ENZYMES max; PyG datasets ignore at runtime
            "root": None,
            "max_graphs": None,
            "seed": 42,
        },
        samples_per_graph=1,  # baseline uses the raw set, not repeated samples
        train_ratio=0.7,
        val_ratio=0.1,
        seed=42,
    )


# Registry: dataset name → zero-arg builder. Add new entries here when a
# new dataset config lands.
DATASET_BUILDERS: dict[str, Callable[[], Any]] = {
    "spectre_sbm": _build_spectre_sbm,
    "spectre_planar": _build_spectre_planar,
    "pyg_enzymes": _build_pyg_enzymes,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    """Return the repo HEAD's short SHA, or ``"unknown"`` if not in a checkout."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _data_to_nx(d: Data) -> nx.Graph[Any]:
    """Convert a single PyG ``Data`` to an undirected NetworkX graph.

    Drops node / edge attributes — the MMD metrics in this codebase
    operate on plain undirected simple graphs.
    """
    return to_networkx(d, to_undirected=True)


def _split_graphs(dm: Any, split: str) -> list[nx.Graph[Any]]:
    """Pull the materialised ``list[Data]`` for a split and convert to NX."""
    attr = f"_{split}_data"
    payload = getattr(dm, attr)
    if payload is None:
        raise RuntimeError(
            f"{type(dm).__name__}.setup() did not populate {attr}; cannot "
            "compute baseline."
        )
    return [_data_to_nx(d) for d in payload]


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def compute_baseline(
    dataset: str,
    params: MMDBaselineParams,
    *,
    max_workers: int | None = None,
) -> MMDBaseline:
    """Materialise splits and compute the four MMD² values.

    Parameters
    ----------
    dataset
        Dataset name. Must be a key of :data:`DATASET_BUILDERS`.
    params
        Kernel / bandwidth / bin configuration. Embedded in the returned
        baseline's fingerprint.
    max_workers
        Forwarded to ``compute_graph_statistics`` for parallel histogram
        computation.

    Returns
    -------
    MMDBaseline
        Ready-to-persist baseline record.
    """
    if dataset not in DATASET_BUILDERS:
        known = ", ".join(sorted(DATASET_BUILDERS))
        raise ValueError(f"Unknown dataset {dataset!r}. Known: {known}.")

    logger.info("[%s] instantiating datamodule…", dataset)
    dm = DATASET_BUILDERS[dataset]()

    logger.info("[%s] running setup()…", dataset)
    t0 = time.time()
    dm.setup()
    logger.info("[%s] setup() finished in %.1fs", dataset, time.time() - t0)

    train_g = _split_graphs(dm, "train")
    test_g = _split_graphs(dm, "test")
    logger.info(
        "[%s] split sizes: train=%d, test=%d", dataset, len(train_g), len(test_g)
    )
    if len(train_g) < 2 or len(test_g) < 2:
        raise RuntimeError(
            f"Dataset {dataset!r} has insufficient graphs for MMD: "
            f"train={len(train_g)}, test={len(test_g)} (need ≥ 2 each)."
        )

    logger.info("[%s] computing degree/clustering/spectral MMD²…", dataset)
    t1 = time.time()
    base_three = compute_mmd_metrics(
        ref_graphs=train_g,
        gen_graphs=test_g,
        kernel=params.kernel,  # type: ignore[arg-type]
        sigma=1.0,
        max_workers=max_workers,
        degree_sigma=params.degree_sigma,
        clustering_sigma=params.clustering_sigma,
        spectral_sigma=params.spectral_sigma,
    )
    logger.info("[%s] three-stat MMD² done in %.1fs", dataset, time.time() - t1)

    logger.info("[%s] computing orbit MMD² (ORCA)…", dataset)
    t2 = time.time()
    orbit_value = compute_orbit_mmd(
        ref_graphs=train_g,
        gen_graphs=test_g,
        kernel=params.kernel,  # type: ignore[arg-type]
        sigma=params.orbit_sigma,
    )
    logger.info("[%s] orbit MMD² done in %.1fs", dataset, time.time() - t2)

    mmd_squared = {
        "degree_mmd": base_three.degree_mmd,
        "clustering_mmd": base_three.clustering_mmd,
        "spectral_mmd": base_three.spectral_mmd,
        "orbit_mmd": orbit_value,
    }
    # Raw (un-squared) MMD — the GraphRNN/GRAN reporting convention used
    # by some reproductions (e.g. HiGen Table 1). ``compute_mmd`` clamps
    # the V-statistic to ``[0, ∞)``, so the sqrt is always real.
    mmd = {k: math.sqrt(v) for k, v in mmd_squared.items()}

    return MMDBaseline(
        dataset=dataset,
        n_train=len(train_g),
        n_test=len(test_g),
        params=params,
        mmd_squared=mmd_squared,
        mmd=mmd,
        computed_at=_dt.datetime.now(_dt.UTC).isoformat(timespec="seconds"),
        git_sha=_git_sha(),
        extra={
            "datamodule_class": type(dm).__name__,
        },
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    doc = __doc__ or ""
    description, _, epilog = doc.partition("\n")
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=epilog or None,
    )
    target = p.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--dataset",
        action="append",
        choices=sorted(DATASET_BUILDERS),
        help="Dataset to compute. Repeat for multiple.",
    )
    target.add_argument(
        "--all",
        action="store_true",
        help="Compute baselines for every registered dataset.",
    )

    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default: data/eval/mmd_baselines).",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Do not recompute when an output JSON already exists.",
    )
    p.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Parallel workers for histogram computation.",
    )

    g = p.add_argument_group("MMD parameters (defaults match production eval)")
    g.add_argument(
        "--kernel", default=DEFAULT_KERNEL, choices=["gaussian", "gaussian_tv"]
    )
    g.add_argument("--degree-sigma", type=float, default=DEFAULT_DEGREE_SIGMA)
    g.add_argument("--clustering-sigma", type=float, default=DEFAULT_CLUSTERING_SIGMA)
    g.add_argument("--spectral-sigma", type=float, default=DEFAULT_SPECTRAL_SIGMA)
    g.add_argument("--orbit-sigma", type=float, default=DEFAULT_ORBIT_SIGMA)
    g.add_argument("--clustering-bins", type=int, default=DEFAULT_CLUSTERING_BINS)
    g.add_argument("--spectral-bins", type=int, default=DEFAULT_SPECTRAL_BINS)

    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    datasets: list[str]
    if args.all:
        datasets = sorted(DATASET_BUILDERS)
    else:
        # ``--dataset`` is ``action="append"``; one or more entries.
        datasets = list(dict.fromkeys(args.dataset))  # de-dup, keep order

    params = MMDBaselineParams(
        kernel=args.kernel,
        degree_sigma=args.degree_sigma,
        clustering_sigma=args.clustering_sigma,
        spectral_sigma=args.spectral_sigma,
        orbit_sigma=args.orbit_sigma,
        clustering_bins=args.clustering_bins,
        spectral_bins=args.spectral_bins,
    )
    logger.info("Parameters fingerprint: %s", params.fingerprint())

    failures: list[str] = []
    for ds in datasets:
        path = baseline_path(ds, root=args.out)
        if args.skip_existing and path.exists():
            logger.info("[%s] %s exists; skipping (--skip-existing)", ds, path)
            continue
        try:
            baseline = compute_baseline(ds, params, max_workers=args.max_workers)
        except Exception as exc:
            # Fail loud per CLAUDE.md (no graceful fallback) but still
            # try the remaining datasets so one bad fixture does not
            # block the whole batch. Per-dataset error printed; non-zero
            # exit at the end.
            logger.exception("[%s] baseline computation failed: %s", ds, exc)
            failures.append(ds)
            continue
        out_path = save_baseline(baseline, root=args.out)
        logger.info("[%s] wrote %s", ds, out_path)
        for k in sorted(baseline.mmd_squared):
            logger.info(
                "[%s]   %-16s mmd²=%.6e  mmd=%.6e",
                ds,
                k,
                baseline.mmd_squared[k],
                baseline.mmd[k],
            )

    if failures:
        logger.error(
            "Completed with failures: %s",
            ", ".join(failures),
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
