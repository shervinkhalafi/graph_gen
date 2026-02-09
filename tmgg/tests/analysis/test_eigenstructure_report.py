"""Tests for tmgg.analysis.reports.eigenstructure.

Validates the EigenstructureReport class and its helper functions: config
resolution, placeholder data generation, table computation, figure output,
and the registry integration.  Each silent fallback path must emit a
UserWarning (CLAUDE.md: fail loudly, no silent fallbacks).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest

from tmgg.analysis.figures import setup_style
from tmgg.analysis.report_base import REPORT_REGISTRY
from tmgg.analysis.reports.eigenstructure import (
    _DEFAULT_DATASETS,
    _METRIC_COLUMNS,
    EigenstructureReport,
    _build_placeholder_data,
    _resolve_datasets,
    _resolve_results_dir,
)

# Ensure non-interactive backend for all figure tests.
setup_style()


@pytest.fixture(autouse=True)
def _close_figures():
    """Close all matplotlib figures after each test to avoid resource leaks."""
    yield
    plt.close("all")


# ---------------------------------------------------------------------------
# TestResolveResultsDir
# ---------------------------------------------------------------------------


class TestResolveResultsDir:
    """_resolve_results_dir should warn and return None when the config key
    is missing or the directory doesn't exist, and return a Path when valid.

    Rationale: prior to B3 these paths returned None silently, violating
    the project's fail-loud policy.
    """

    def test_missing_key_warns_and_returns_none(self) -> None:
        """An empty config has no 'report.data.results_dir' key, so the
        function must warn about the KeyError and return None."""
        with pytest.warns(UserWarning, match="Could not resolve results_dir"):
            result = _resolve_results_dir({})
        assert result is None

    def test_nonexistent_dir_warns_and_returns_none(self) -> None:
        """When the config points to a path that does not exist on disk,
        the function must warn and return None rather than silently
        falling back."""
        config = {"report": {"data": {"results_dir": "/nonexistent/path/xyz"}}}
        with pytest.warns(UserWarning, match="Results directory does not exist"):
            result = _resolve_results_dir(config)
        assert result is None

    def test_valid_dir_returns_path(self, tmp_path: Path) -> None:
        """A config pointing to an existing directory should return the
        Path without any warnings."""
        config = {"report": {"data": {"results_dir": str(tmp_path)}}}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _resolve_results_dir(config)
        assert result == tmp_path


# ---------------------------------------------------------------------------
# TestResolveDatasets
# ---------------------------------------------------------------------------


class TestResolveDatasets:
    """_resolve_datasets should fall back to _DEFAULT_DATASETS with a warning
    when config keys are absent, and return the configured list otherwise.

    Rationale: the fallback was previously silent, masking configuration
    errors from the user.
    """

    def test_missing_key_warns_and_returns_defaults(self) -> None:
        """Empty config triggers the KeyError path, which must warn and
        return a copy of _DEFAULT_DATASETS."""
        with pytest.warns(UserWarning, match="Could not resolve dataset list"):
            result = _resolve_datasets({})
        assert result == _DEFAULT_DATASETS
        # Must be a copy, not the same object.
        assert result is not _DEFAULT_DATASETS

    def test_valid_config_returns_configured_list(self) -> None:
        """When the config provides an explicit list of datasets, the
        function returns that list without warnings."""
        custom = ["graphA", "graphB"]
        config = {"report": {"data": {"datasets": custom}}}
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = _resolve_datasets(config)
        assert result == custom


# ---------------------------------------------------------------------------
# TestBuildPlaceholderData
# ---------------------------------------------------------------------------


class TestBuildPlaceholderData:
    """_build_placeholder_data must produce a DataFrame of the correct shape,
    warn the user that synthetic data is being used, and yield deterministic
    output via its fixed seed.

    Rationale: placeholder generation was previously silent. Determinism
    is tested to guard against accidental seed changes that would break
    snapshot-style assertions downstream.
    """

    def test_correct_shape(self) -> None:
        """Output should have one row per dataset and one column per metric."""
        datasets = ["a", "b", "c"]
        with pytest.warns(UserWarning, match="placeholder"):
            df = _build_placeholder_data(datasets)
        assert df.shape == (len(datasets), len(_METRIC_COLUMNS))

    def test_warns_on_invocation(self) -> None:
        """Every call must emit a UserWarning about placeholder data."""
        with pytest.warns(UserWarning, match="Generating placeholder"):
            _build_placeholder_data(["x"])

    def test_deterministic_seed_42(self) -> None:
        """Two calls with the same datasets must produce identical DataFrames,
        since the RNG seed is hardcoded to 42."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df1 = _build_placeholder_data(["a", "b"])
            df2 = _build_placeholder_data(["a", "b"])
        pd.testing.assert_frame_equal(df1, df2)


# ---------------------------------------------------------------------------
# TestEigenstructureReport
# ---------------------------------------------------------------------------


class TestEigenstructureReport:
    """Integration tests for the EigenstructureReport class: load_data with
    an empty config should produce placeholder data, compute_tables should
    return the expected keys, and generate_figures should write two PNGs.

    Rationale: the eigenstructure report was added without tests; these
    cover the primary code paths through the ReportGenerator ABC.
    """

    def test_load_data_empty_config_returns_placeholder(self) -> None:
        """An empty config triggers placeholder mode. The resulting
        DataFrame must have one row per default dataset."""
        report = EigenstructureReport(name="eigenstructure")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = report.load_data({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == len(_DEFAULT_DATASETS)
        assert list(df.columns) == _METRIC_COLUMNS

    def test_compute_tables_returns_correct_keys(self) -> None:
        """compute_tables must produce exactly 'spectral_summary' and
        'connectivity_comparison' DataFrames."""
        report = EigenstructureReport(name="eigenstructure")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = report.load_data({})
        tables = report.compute_tables(df)
        assert set(tables.keys()) == {"spectral_summary", "connectivity_comparison"}
        for v in tables.values():
            assert isinstance(v, pd.DataFrame)

    def test_generate_figures_creates_two_pngs(self, tmp_path: Path) -> None:
        """generate_figures should write spectral_gap.png and
        algebraic_connectivity.png into the output directory."""
        report = EigenstructureReport(name="eigenstructure")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            df = report.load_data({})
        tables = report.compute_tables(df)
        figures_dir = tmp_path / "figures"
        figures_dir.mkdir()
        paths = report.generate_figures(df, tables, figures_dir)
        assert len(paths) == 2
        for p in paths:
            assert p.exists()
            assert p.suffix == ".png"
        filenames = {p.name for p in paths}
        assert "spectral_gap.png" in filenames
        assert "algebraic_connectivity.png" in filenames

    def test_registered_in_report_registry(self) -> None:
        """The @register_report('eigenstructure') decorator must have
        placed the class in REPORT_REGISTRY."""
        assert "eigenstructure" in REPORT_REGISTRY
        assert REPORT_REGISTRY["eigenstructure"] is EigenstructureReport


# ---------------------------------------------------------------------------
# TestDefaultDatasets
# ---------------------------------------------------------------------------


class TestDefaultDatasets:
    """Safety-net test ensuring _DEFAULT_DATASETS matches the canonical
    list of datasets used throughout the eigenstructure module.

    Rationale: if someone adds or removes a dataset from the constant,
    this test forces them to consciously update the expected value here.
    """

    def test_default_datasets_match_expected(self) -> None:
        assert _DEFAULT_DATASETS == ["sbm", "er", "tree", "regular", "enzymes"]
