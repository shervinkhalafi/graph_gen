"""CLI interface for TensorBoard export tool."""

from pathlib import Path

import click
from loguru import logger

from .exporter import TensorBoardExporter


@click.command()
@click.option(
    "--input-dir",
    "-i",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    help="Root directory containing TensorBoard runs.",
)
@click.option(
    "--output-dir",
    "-o",
    required=True,
    type=click.Path(file_okay=False, path_type=Path),
    help="Output directory for parquet files.",
)
@click.option(
    "--project-id",
    "-p",
    default=None,
    help="Project identifier. Defaults to input directory name.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    help="Enable verbose logging.",
)
def main(
    input_dir: Path,
    output_dir: Path,
    project_id: str | None,
    verbose: bool,
) -> None:
    """Export TensorBoard event files to parquet DataFrames.

    Discovers all TensorBoard runs in INPUT_DIR, loads events and configs,
    and saves unified DataFrames to OUTPUT_DIR.

    Output files:
    - events.parquet: All scalar events (project_id, run_id, tag, step, wall_time, value)
    - hparams.parquet: Hyperparameters/config for each run
    """
    if verbose:
        logger.enable("tmgg")
    else:
        logger.disable("tmgg")

    click.echo(f"Exporting TensorBoard runs from: {input_dir}")

    exporter = TensorBoardExporter()
    result = exporter.export_directory(input_dir, project_id)

    click.echo(result.summary())
    click.echo()

    result.save(output_dir)
    click.echo(f"\nExport complete. Output saved to: {output_dir}")


if __name__ == "__main__":
    main()
