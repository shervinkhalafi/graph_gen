"""Eigenstructure analysis report.

Wraps the ``SpectralAnalyzer`` from ``tmgg.experiment_utils.eigenstructure_study``
into the ``ReportGenerator`` framework.  When real eigenstructure data has been
collected, the report analyses spectral gaps, algebraic connectivity, eigenvalue
entropy, and related metrics across datasets.  When data is absent it produces
placeholder tables and figures so that the pipeline remains testable end to end.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig

from tmgg.analysis.report_base import ReportGenerator, register_report

# Columns in the raw data DataFrame, matching SpectralAnalysisResult fields.
_METRIC_COLUMNS: list[str] = [
    "dataset",
    "num_graphs",
    "spectral_gap_mean",
    "spectral_gap_std",
    "spectral_gap_min",
    "spectral_gap_max",
    "algebraic_connectivity_mean",
    "algebraic_connectivity_std",
    "algebraic_connectivity_min",
    "algebraic_connectivity_max",
    "eigenvalue_entropy_adj",
    "eigenvalue_entropy_lap",
    "coherence_mean",
    "coherence_std",
    "effective_rank_adj_mean",
    "effective_rank_lap_mean",
]

_DEFAULT_DATASETS: list[str] = ["sbm", "er", "tree", "regular", "enzymes"]


def _resolve_results_dir(config: DictConfig | dict[str, Any]) -> Path | None:
    """Extract the results directory from the report config.

    Returns None when the path is unresolvable or does not exist,
    which triggers placeholder-data mode.
    """
    try:
        if isinstance(config, DictConfig):
            raw = str(config.report.data.results_dir)
        else:
            raw = str(config["report"]["data"]["results_dir"])
    except (KeyError, AttributeError) as exc:
        warnings.warn(
            f"Could not resolve results_dir from config: {exc}. "
            f"Falling back to placeholder data.",
            UserWarning,
            stacklevel=2,
        )
        return None

    p = Path(str(raw))
    if not p.is_dir():
        warnings.warn(
            f"Results directory does not exist: {p}. Falling back to placeholder data.",
            UserWarning,
            stacklevel=2,
        )
        return None
    return p


def _resolve_datasets(config: DictConfig | dict[str, Any]) -> list[str]:
    """Extract the list of dataset names from the config."""
    try:
        if isinstance(config, DictConfig):
            ds = config.report.data.datasets
        else:
            ds = config["report"]["data"]["datasets"]
        return list(ds)
    except (KeyError, AttributeError) as exc:
        warnings.warn(
            f"Could not resolve dataset list from config: {exc}. "
            f"Using default datasets: {_DEFAULT_DATASETS}.",
            UserWarning,
            stacklevel=2,
        )
        return list(_DEFAULT_DATASETS)


def _build_placeholder_data(datasets: list[str]) -> pd.DataFrame:
    """Return synthetic rows so the pipeline can run without real data."""
    import numpy as np

    warnings.warn(
        "Generating placeholder eigenstructure data because no real "
        "results were found. Tables and figures will contain synthetic values.",
        UserWarning,
        stacklevel=2,
    )

    rng = np.random.default_rng(42)
    rows: list[dict[str, Any]] = []
    for ds in datasets:
        rows.append(
            {
                "dataset": ds,
                "num_graphs": 0,
                "spectral_gap_mean": rng.uniform(0.1, 2.0),
                "spectral_gap_std": rng.uniform(0.01, 0.5),
                "spectral_gap_min": rng.uniform(0.0, 0.1),
                "spectral_gap_max": rng.uniform(2.0, 4.0),
                "algebraic_connectivity_mean": rng.uniform(0.05, 1.0),
                "algebraic_connectivity_std": rng.uniform(0.01, 0.3),
                "algebraic_connectivity_min": rng.uniform(0.0, 0.05),
                "algebraic_connectivity_max": rng.uniform(1.0, 3.0),
                "eigenvalue_entropy_adj": rng.uniform(1.0, 4.0),
                "eigenvalue_entropy_lap": rng.uniform(1.0, 4.0),
                "coherence_mean": rng.uniform(0.1, 0.9),
                "coherence_std": rng.uniform(0.01, 0.2),
                "effective_rank_adj_mean": rng.uniform(2.0, 10.0),
                "effective_rank_lap_mean": rng.uniform(2.0, 10.0),
            }
        )
    return pd.DataFrame(rows, columns=_METRIC_COLUMNS)  # pyright: ignore[reportArgumentType]


@register_report("eigenstructure")
class EigenstructureReport(ReportGenerator):
    """Report on spectral properties of graph datasets.

    Loads eigenstructure analysis results produced by
    ``tmgg.experiment_utils.eigenstructure_study.SpectralAnalyzer`` and
    summarises them as tables and bar charts.  Falls back to placeholder
    data when no collected results are available.
    """

    def load_data(self, config: DictConfig | dict[str, Any]) -> pd.DataFrame:
        """Load spectral analysis results for each dataset.

        Parameters
        ----------
        config : DictConfig or dict
            Must contain ``report.data.results_dir`` and
            ``report.data.datasets`` (see ``eigenstructure.yaml``).

        Returns
        -------
        pd.DataFrame
            One row per dataset with spectral metric columns.
        """
        datasets = _resolve_datasets(config)
        results_dir = _resolve_results_dir(config)

        if results_dir is None:
            return _build_placeholder_data(datasets)

        # Lazy import to avoid pulling torch when data is absent.
        from tmgg.experiment_utils.eigenstructure_study import (
            SpectralAnalyzer,
        )

        rows: list[dict[str, Any]] = []
        for ds_name in datasets:
            ds_dir = results_dir / ds_name
            if not ds_dir.is_dir():
                warnings.warn(
                    f"Skipping dataset '{ds_name}': directory {ds_dir} does not exist.",
                    UserWarning,
                    stacklevel=2,
                )
                continue
            analyzer = SpectralAnalyzer(ds_dir)
            result = analyzer.analyze()
            row = asdict(result)
            row["dataset"] = row.pop("dataset_name")
            rows.append(row)

        if not rows:
            return _build_placeholder_data(datasets)

        return pd.DataFrame(rows, columns=_METRIC_COLUMNS)  # pyright: ignore[reportArgumentType]

    def compute_tables(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Derive summary tables from loaded spectral data.

        Parameters
        ----------
        data : pd.DataFrame
            Output of ``load_data``.

        Returns
        -------
        dict[str, pd.DataFrame]
            ``"spectral_summary"`` -- key metrics per dataset.
            ``"connectivity_comparison"`` -- algebraic connectivity
            breakdown across datasets.
        """
        summary = data[
            [
                "dataset",
                "spectral_gap_mean",
                "spectral_gap_std",
                "algebraic_connectivity_mean",
                "eigenvalue_entropy_adj",
                "eigenvalue_entropy_lap",
                "effective_rank_adj_mean",
                "effective_rank_lap_mean",
            ]
        ].copy()
        summary = summary.set_index("dataset")

        connectivity = data[
            [
                "dataset",
                "algebraic_connectivity_mean",
                "algebraic_connectivity_std",
                "algebraic_connectivity_min",
                "algebraic_connectivity_max",
            ]
        ].copy()
        connectivity = connectivity.set_index("dataset")

        return {
            "spectral_summary": summary.reset_index(),
            "connectivity_comparison": connectivity.reset_index(),
        }

    def generate_figures(
        self,
        data: pd.DataFrame,
        tables: dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> list[Path]:
        """Create bar charts of spectral gap and algebraic connectivity.

        Parameters
        ----------
        data : pd.DataFrame
            Output of ``load_data``.
        tables : dict[str, pd.DataFrame]
            Output of ``compute_tables`` (unused here but part of the
            protocol).
        output_dir : Path
            Directory in which to write figure PNGs.

        Returns
        -------
        list[Path]
            Paths to saved figure files.
        """
        # matplotlib imported lazily to keep the module lightweight
        import matplotlib.pyplot as plt

        from tmgg.analysis.figures import save_figure, setup_style

        setup_style()
        saved: list[Path] = []

        # -- Spectral gap bar chart ------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            data["dataset"],
            data["spectral_gap_mean"],
            yerr=data["spectral_gap_std"],
            capsize=4,
            color="steelblue",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Spectral gap (mean \u00b1 std)")
        ax.set_title("Spectral gap across graph datasets")
        ax.tick_params(axis="x", rotation=45)
        saved.append(save_figure(fig, output_dir / "spectral_gap.png"))

        # -- Algebraic connectivity bar chart --------------------------------
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.bar(
            data["dataset"],
            data["algebraic_connectivity_mean"],
            yerr=data["algebraic_connectivity_std"],
            capsize=4,
            color="coral",
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Algebraic connectivity (mean \u00b1 std)")
        ax.set_title("Algebraic connectivity (Fiedler value) across graph datasets")
        ax.tick_params(axis="x", rotation=45)
        saved.append(save_figure(fig, output_dir / "algebraic_connectivity.png"))

        return saved
