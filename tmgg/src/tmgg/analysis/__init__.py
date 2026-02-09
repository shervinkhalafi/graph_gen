"""Analysis utilities for TMGG experiment data.

Provides four submodules:

- ``parsing``: functions that extract structured fields (stage, architecture,
  model type, dataset, protocol, hyperparameters) from W&B run names and
  config columns.  Single source of truth for all run-name parsing.

- ``statistics``: statistical comparison helpers (Cohen's d, group-level
  summaries, pairwise significance tests) and RF-based hyperparameter
  importance analysis.

- ``report_base``: abstract ``ReportGenerator`` base class with a
  registry-based discovery mechanism for pluggable analysis reports.

- ``cli``: Click CLI entry point for running registered reports.

- ``figures``: shared matplotlib/seaborn figure utilities for
  publication-quality plots.
"""

# Import reports subpackage so @register_report decorators fire at import time.
import tmgg.analysis.reports  # noqa: F401, E402
from tmgg.analysis.cli import main
from tmgg.analysis.parsing import (
    enrich_dataframe,
    parse_architecture,
    parse_dataset,
    parse_model_type,
    parse_protocol,
    parse_run_name_fields,
    parse_stage,
)
from tmgg.analysis.report_base import REPORT_REGISTRY, ReportGenerator, register_report
from tmgg.analysis.statistics import (
    ComparisonResult,
    GroupStats,
    analyze_grouping,
    compute_cohens_d,
    compute_importance,
    interpret_effect_size,
)

__all__ = [
    # parsing
    "parse_stage",
    "parse_architecture",
    "parse_model_type",
    "parse_run_name_fields",
    "parse_protocol",
    "parse_dataset",
    "enrich_dataframe",
    # statistics
    "GroupStats",
    "ComparisonResult",
    "compute_cohens_d",
    "interpret_effect_size",
    "analyze_grouping",
    "compute_importance",
    # report framework
    "ReportGenerator",
    "REPORT_REGISTRY",
    "register_report",
    "main",
]
