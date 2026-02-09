"""Click CLI for running registered analysis reports.

Provides three commands:

- ``generate`` — run a single named report.
- ``generate-all`` — run every registered report.
- ``list`` — print available report names.

Entry point::

    uv run python -m tmgg.analysis.cli generate --report-name my_report --output-dir out/
    uv run python -m tmgg.analysis.cli list
"""

from __future__ import annotations

from pathlib import Path

import click
from omegaconf import DictConfig, OmegaConf

from tmgg.analysis.report_base import REPORT_REGISTRY


@click.group()
def main() -> None:
    """TMGG analysis report generator."""


@main.command("list")
def list_reports() -> None:
    """Print all registered report names."""
    if not REPORT_REGISTRY:
        click.echo("No reports registered.")
        return
    click.echo("Available reports:")
    for name in sorted(REPORT_REGISTRY):
        click.echo(f"  {name}")


@main.command()
@click.option(
    "--report-name",
    required=True,
    help="Name of the registered report to generate.",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Directory for report artefacts.",
)
@click.option(
    "--config-path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Optional YAML config file for the report.",
)
def generate(report_name: str, output_dir: Path, config_path: Path | None) -> None:
    """Generate a single named report."""
    if report_name not in REPORT_REGISTRY:
        raise click.ClickException(
            f"Unknown report '{report_name}'. "
            f"Available: {', '.join(sorted(REPORT_REGISTRY))}"
        )

    config: DictConfig | dict[str, object] = (
        OmegaConf.load(config_path)  # pyright: ignore[reportAssignmentType]
        if config_path is not None
        else {}
    )

    report_cls = REPORT_REGISTRY[report_name]
    report = report_cls(name=report_name)
    summary = report.generate(config, output_dir)
    click.echo(f"Report '{report_name}' written to {summary}")


@main.command("generate-all")
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Root directory; each report gets a subdirectory.",
)
@click.option(
    "--config-path",
    default=None,
    type=click.Path(exists=True, path_type=Path),
    help="Optional YAML config file shared across all reports.",
)
def generate_all(output_dir: Path, config_path: Path | None) -> None:
    """Generate all registered reports."""
    if not REPORT_REGISTRY:
        click.echo("No reports registered.")
        return

    config: DictConfig | dict[str, object] = (
        OmegaConf.load(config_path)  # pyright: ignore[reportAssignmentType]
        if config_path is not None
        else {}
    )

    for name in sorted(REPORT_REGISTRY):
        report_cls = REPORT_REGISTRY[name]
        report = report_cls(name=name)
        report_dir = output_dir / name
        summary = report.generate(config, report_dir)
        click.echo(f"Report '{name}' written to {summary}")


if __name__ == "__main__":
    main()
