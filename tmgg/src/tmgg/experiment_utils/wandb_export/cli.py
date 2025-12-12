"""Click CLI for W&B data export.

Provides command-line interface for exporting W&B project data to local files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from loguru import logger

from .exporter import ExportConfig, WandbExporter


def setup_logging(verbose: bool) -> None:
    """Configure loguru logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr,
        format="<level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True,
    )


@click.command()
@click.option(
    "--project",
    required=True,
    help="W&B project name (e.g., 'my-project' or 'entity/my-project')",
)
@click.option(
    "--entity",
    default=None,
    help="W&B entity (team/user). Optional if included in project path.",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=Path("./wandb_export"),
    help="Output directory for exported data.",
)
@click.option(
    "--runs",
    multiple=True,
    help="Specific run IDs to export. Can be specified multiple times. Default: all runs.",
)
@click.option(
    "--skip-media",
    is_flag=True,
    help="Skip downloading media files (images, etc.).",
)
@click.option(
    "--skip-artifacts",
    is_flag=True,
    help="Skip downloading artifacts.",
)
@click.option(
    "--force",
    is_flag=True,
    help="Re-export runs even if already completed.",
)
@click.option(
    "--max-retries",
    default=5,
    show_default=True,
    help="Maximum retries for rate-limited API requests.",
)
@click.option(
    "--page-size",
    default=1000,
    show_default=True,
    help="Page size for history pagination (larger = fewer API calls, more memory).",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose (debug) logging.",
)
def main(
    project: str,
    entity: str | None,
    output_dir: Path,
    runs: tuple[str, ...],
    skip_media: bool,
    skip_artifacts: bool,
    force: bool,
    max_retries: int,
    page_size: int,
    verbose: bool,
) -> None:
    """Export W&B project data to local Parquet/JSON files.

    Exports full metrics history (without subsampling), configurations,
    summaries, and optionally media files and artifacts.

    State is tracked in a JSONL file to allow resumable exports. Incomplete
    exports are automatically cleaned up and restarted.

    \b
    Examples:
        # Export all runs from a project
        tmgg-wandb-export --project my-project

        # Export with entity specified
        tmgg-wandb-export --project my-project --entity my-team

        # Export specific runs only
        tmgg-wandb-export --project my-project --runs abc123 --runs def456

        # Skip large files, force re-export
        tmgg-wandb-export --project my-project --skip-media --skip-artifacts --force
    """
    setup_logging(verbose)

    # Parse entity from project path if provided
    if "/" in project and entity is None:
        entity, project = project.split("/", 1)
        logger.debug(f"Parsed entity={entity} from project path")

    logger.info(f"Starting W&B export for project: {project}")
    if entity:
        logger.info(f"Entity: {entity}")
    logger.info(f"Output directory: {output_dir.absolute()}")

    config = ExportConfig(
        project=project,
        entity=entity,
        output_dir=output_dir,
        skip_media=skip_media,
        skip_artifacts=skip_artifacts,
        page_size=page_size,
        max_retries=max_retries,
    )

    exporter = WandbExporter(config)

    run_ids = list(runs) if runs else None
    if run_ids:
        logger.info(f"Exporting specific runs: {run_ids}")

    summary = exporter.export_project(run_ids=run_ids, force=force)

    # Print summary
    click.echo("\n" + "=" * 50)
    click.echo("Export Summary")
    click.echo("=" * 50)
    click.echo(f"Exported: {summary['exported']}")
    click.echo(f"Skipped:  {summary['skipped']}")
    click.echo(f"Failed:   {summary['failed']}")

    if summary["errors"]:
        click.echo("\nErrors:")
        for err in summary["errors"]:
            click.echo(f"  - {err['run_id']}: {err['error']}")

    click.echo(f"\nOutput: {output_dir.absolute() / project}")

    # Exit with error code if any failures
    if summary["failed"] > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
