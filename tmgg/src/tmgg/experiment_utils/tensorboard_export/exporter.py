"""Main exporter class for TensorBoard event files."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from loguru import logger

from .discovery import DiscoveredRun, discover_runs
from .loader import load_run


@dataclass
class ExportResult:
    """Result of a TensorBoard export operation.

    Attributes
    ----------
    events_df
        DataFrame with all scalar events across runs.
        Columns: project_id, run_id, tag, step, wall_time, value
    hparams_df
        DataFrame with hyperparameters/config for each run.
        Columns: project_id, run_id, plus flattened config fields
    runs_processed
        Number of runs successfully processed.
    runs_failed
        Number of runs that failed to process.
    failed_runs
        List of run_ids that failed.
    """

    events_df: pd.DataFrame
    hparams_df: pd.DataFrame
    runs_processed: int = 0
    runs_failed: int = 0
    failed_runs: list[str] = field(default_factory=list)

    def save(self, output_dir: str | Path) -> None:
        """Save DataFrames to parquet files.

        Parameters
        ----------
        output_dir
            Directory to save output files.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        events_path = output_dir / "events.parquet"
        hparams_path = output_dir / "hparams.parquet"

        self.events_df.to_parquet(events_path, index=False)
        self.hparams_df.to_parquet(hparams_path, index=False)

        logger.info(f"Saved events to {events_path} ({len(self.events_df)} rows)")
        logger.info(f"Saved hparams to {hparams_path} ({len(self.hparams_df)} rows)")

    def summary(self) -> str:
        """Return a summary of the export result."""
        lines = [
            "TensorBoard Export Summary",
            f"{'=' * 40}",
            f"Runs processed: {self.runs_processed}",
            f"Runs failed: {self.runs_failed}",
            f"Total events: {len(self.events_df)}",
            f"Unique tags: {self.events_df['tag'].nunique() if len(self.events_df) > 0 else 0}",
        ]
        if self.failed_runs:
            lines.append(f"Failed runs: {', '.join(self.failed_runs[:5])}")
            if len(self.failed_runs) > 5:
                lines.append(f"  ... and {len(self.failed_runs) - 5} more")
        return "\n".join(lines)


class TensorBoardExporter:
    """Export TensorBoard event files to DataFrames.

    Discovers runs in a directory tree, loads events and configs,
    and produces unified DataFrames for analysis.

    Examples
    --------
    >>> exporter = TensorBoardExporter()
    >>> result = exporter.export_directory("outputs/sweeps/stage1_poc")
    >>> result.events_df.head()
    >>> result.save("analysis_output/")
    """

    def export_directory(
        self,
        root_dir: str | Path,
        project_id: str | None = None,
    ) -> ExportResult:
        """Export all TensorBoard runs from a directory.

        Parameters
        ----------
        root_dir
            Root directory containing run subdirectories.
        project_id
            Project identifier. If None, uses directory name.

        Returns
        -------
        ExportResult
            Contains events_df and hparams_df with all runs.
        """
        root_dir = Path(root_dir)
        logger.info(f"Discovering runs in {root_dir}")

        runs = discover_runs(root_dir, project_id)
        logger.info(f"Found {len(runs)} runs")

        return self._export_runs(runs)

    def export_runs(self, runs: list[DiscoveredRun]) -> ExportResult:
        """Export specific discovered runs.

        Parameters
        ----------
        runs
            List of discovered runs to export.

        Returns
        -------
        ExportResult
            Contains events_df and hparams_df for the runs.
        """
        return self._export_runs(runs)

    def _export_runs(self, runs: list[DiscoveredRun]) -> ExportResult:
        """Internal method to export a list of runs."""
        all_events: list[pd.DataFrame] = []
        all_configs: list[dict] = []
        failed_runs: list[str] = []

        for run in runs:
            try:
                logger.debug(f"Processing run: {run.run_id}")
                events_df, config = load_run(run)

                if len(events_df) > 0:
                    all_events.append(events_df)
                all_configs.append(config)

            except Exception as e:
                logger.warning(f"Failed to process run {run.run_id}: {e}")
                failed_runs.append(run.run_id)
                continue

        # Combine events
        if all_events:
            events_df = pd.concat(all_events, ignore_index=True)
        else:
            events_df = pd.DataFrame(
                columns=["project_id", "run_id", "tag", "step", "wall_time", "value"]  # pyright: ignore[reportArgumentType]
            )

        # Combine configs into DataFrame
        if all_configs:
            hparams_df = pd.DataFrame(all_configs)
        else:
            hparams_df = pd.DataFrame(columns=["project_id", "run_id"])  # pyright: ignore[reportArgumentType]

        return ExportResult(
            events_df=events_df,
            hparams_df=hparams_df,
            runs_processed=len(runs) - len(failed_runs),
            runs_failed=len(failed_runs),
            failed_runs=failed_runs,
        )
