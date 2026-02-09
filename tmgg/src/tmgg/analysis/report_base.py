"""Report generator framework for TMGG experiment analysis.

Defines the abstract ``ReportGenerator`` base class and a registry-based
discovery mechanism.  Concrete report classes register themselves via
the ``@register_report`` decorator; the CLI (``tmgg.analysis.cli``)
then discovers and invokes them by name.

Typical usage::

    from tmgg.analysis.report_base import ReportGenerator, register_report

    @register_report("my_report")
    class MyReport(ReportGenerator):
        def load_data(self, config):
            ...
        def compute_tables(self, data):
            ...
        def generate_figures(self, data, tables, output_dir):
            ...
"""

from __future__ import annotations

import abc
import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from omegaconf import DictConfig, OmegaConf

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REPORT_REGISTRY: dict[str, type[ReportGenerator]] = {}
"""Module-level registry mapping report names to their generator classes."""


def register_report(name: str):  # noqa: ANN201 — returns decorator
    """Class decorator that registers a ``ReportGenerator`` subclass.

    Parameters
    ----------
    name : str
        Key under which the report will be discoverable in
        ``REPORT_REGISTRY``.

    Returns
    -------
    Callable
        Decorator that registers and returns the class unchanged.

    Raises
    ------
    ValueError
        If ``name`` is already registered (duplicate names are a
        programming error and should surface immediately).
    TypeError
        If the decorated class is not a ``ReportGenerator`` subclass.
    """

    def _decorator(cls: type) -> type[ReportGenerator]:
        if not issubclass(cls, ReportGenerator):
            raise TypeError(
                f"@register_report('{name}') applied to {cls!r}, "
                "which is not a ReportGenerator subclass"
            )
        if name in REPORT_REGISTRY:
            raise ValueError(
                f"Duplicate report name '{name}': "
                f"{REPORT_REGISTRY[name].__qualname__} already registered"
            )
        REPORT_REGISTRY[name] = cls
        return cls

    return _decorator


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class ReportGenerator(abc.ABC):
    """Abstract base for analysis report generators.

    Subclasses implement the three abstract hooks — ``load_data``,
    ``compute_tables``, ``generate_figures`` — and the concrete
    ``generate`` method orchestrates them into a reproducible pipeline
    that writes artefacts (figures, tables, markdown summary) to an
    output directory.

    Parameters
    ----------
    name : str
        Human-readable report name (used in the summary header).
    """

    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    # -- Abstract hooks ------------------------------------------------------

    @abc.abstractmethod
    def load_data(self, config: DictConfig | dict[str, Any]) -> pd.DataFrame:
        """Load and return the raw data needed for this report.

        Parameters
        ----------
        config : DictConfig or dict
            Report-specific configuration (data paths, filters, etc.).

        Returns
        -------
        pd.DataFrame
            Raw experiment data ready for ``compute_tables``.
        """
        ...

    @abc.abstractmethod
    def compute_tables(self, data: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Derive summary tables from raw data.

        Parameters
        ----------
        data : pd.DataFrame
            Output of ``load_data``.

        Returns
        -------
        dict[str, pd.DataFrame]
            Named summary tables.  Keys become file stems when saved.
        """
        ...

    @abc.abstractmethod
    def generate_figures(
        self,
        data: pd.DataFrame,
        tables: dict[str, pd.DataFrame],
        output_dir: Path,
    ) -> list[Path]:
        """Create and save figures to *output_dir*.

        Parameters
        ----------
        data : pd.DataFrame
            Output of ``load_data``.
        tables : dict[str, pd.DataFrame]
            Output of ``compute_tables``.
        output_dir : Path
            Directory in which to write figure files.

        Returns
        -------
        list[Path]
            Paths to all saved figure files (used in the summary).
        """
        ...

    # -- Orchestrator --------------------------------------------------------

    def generate(
        self,
        config: DictConfig | dict[str, Any],
        output_dir: Path,
    ) -> Path:
        """Run the full report pipeline and write a markdown summary.

        Calls ``load_data``, ``compute_tables``, ``generate_figures``
        in sequence, then writes a ``summary.md`` that links to all
        produced artefacts.

        Parameters
        ----------
        config : DictConfig or dict
            Report-specific configuration.
        output_dir : Path
            Root directory for all report artefacts.

        Returns
        -------
        Path
            Path to the generated ``summary.md``.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        tables_dir = output_dir / "tables"
        tables_dir.mkdir(exist_ok=True)
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(exist_ok=True)

        # 1. Load
        data = self.load_data(config)

        # 2. Tables
        tables = self.compute_tables(data)
        table_paths: list[Path] = []
        for table_name, table_df in tables.items():
            csv_path = tables_dir / f"{table_name}.csv"
            table_df.to_csv(csv_path, index=False)
            table_paths.append(csv_path)

        # 3. Figures
        figure_paths = self.generate_figures(data, tables, figures_dir)

        # 4. Summary
        summary_path = output_dir / "summary.md"
        summary_path.write_text(self._render_summary(config, table_paths, figure_paths))

        return summary_path

    # -- Internal helpers ----------------------------------------------------

    def _render_summary(
        self,
        config: DictConfig | dict[str, Any],
        table_paths: list[Path],
        figure_paths: list[Path],
    ) -> str:
        """Render a markdown summary linking to all artefacts.

        Parameters
        ----------
        config : DictConfig or dict
            The config used for this run (serialised into the summary).
        table_paths : list[Path]
            Paths to saved CSV tables.
        figure_paths : list[Path]
            Paths to saved figure files.

        Returns
        -------
        str
            Markdown text.
        """
        timestamp = datetime.datetime.now(tz=datetime.UTC).isoformat(timespec="seconds")
        config_str = (
            OmegaConf.to_yaml(config) if isinstance(config, DictConfig) else str(config)
        )

        lines = [
            f"# Report: {self.name}",
            "",
            f"Generated: {timestamp}",
            "",
            "## Configuration",
            "",
            "```yaml",
            config_str.rstrip(),
            "```",
            "",
        ]

        if table_paths:
            lines.append("## Tables")
            lines.append("")
            for p in table_paths:
                lines.append(f"- [{p.stem}](tables/{p.name})")
            lines.append("")

        if figure_paths:
            lines.append("## Figures")
            lines.append("")
            for p in figure_paths:
                lines.append(f"![{p.stem}](figures/{p.name})")
                lines.append("")

        return "\n".join(lines)
