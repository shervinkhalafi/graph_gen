"""Core W&B data export logic.

Handles exporting runs from a W&B project to local Parquet/JSON files with
full history (no subsampling), media files, and artifacts.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pandas as pd
import wandb
from loguru import logger
from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
)

from .rate_limiter import RateLimiter
from .state import ExportState

if TYPE_CHECKING:
    from wandb.apis.public.runs import Run


@dataclass
class ExportConfig:
    """Configuration for W&B export operation.

    Parameters
    ----------
    project
        W&B project name.
    entity
        W&B entity (team/user). None uses default from wandb config.
    output_dir
        Base directory for exported data.
    skip_media
        Skip downloading media files.
    skip_artifacts
        Skip downloading artifacts.
    page_size
        Page size for scan_history pagination.
    max_retries
        Maximum retries for rate-limited API calls.
    """

    project: str
    entity: str | None
    output_dir: Path
    skip_media: bool = False
    skip_artifacts: bool = False
    page_size: int = 1000
    max_retries: int = 5


class WandbExporter:
    """Handles W&B data export with state tracking and rate limiting.

    Exports run data including:
    - Configuration (JSON)
    - Summary metrics (JSON)
    - Metadata: tags, state, timestamps (JSON)
    - Full metrics history (Parquet, no subsampling)
    - System metrics (Parquet)
    - Media files (optional)
    - Artifacts (optional)

    Parameters
    ----------
    config
        Export configuration.
    console
        Rich console for progress display. If None, creates a new one.
    """

    def __init__(self, config: ExportConfig, console: Console | None = None) -> None:
        self.config = config
        self.rate_limiter = RateLimiter(max_retries=config.max_retries)
        self.api = wandb.Api()
        self.console = console or Console()

        project_dir = config.output_dir / config.project
        project_dir.mkdir(parents=True, exist_ok=True)
        self.state = ExportState(project_dir / "_export_state.jsonl")

    def export_project(
        self,
        run_ids: list[str] | None = None,
        force: bool = False,
    ) -> dict[str, Any]:
        """Export all runs in project, or specific runs if provided.

        Parameters
        ----------
        run_ids
            Specific run IDs to export. If None, exports all runs.
        force
            Re-export runs even if already completed.

        Returns
        -------
        dict
            Summary with counts of exported, skipped, and failed runs.
        """
        project_path: str = (
            f"{self.config.entity}/{self.config.project}"
            if self.config.entity
            else self.config.project
        )

        _ = logger.info(f"Fetching runs from project: {project_path}")
        try:
            runs: list[Run] = self.rate_limiter.call_with_retry(
                lambda: list(self.api.runs(project_path))
            )
        except ValueError as e:
            # wandb raises ValueError when project is not found
            _ = logger.error(f"Project not found: {project_path}")
            _ = logger.error("Check that the project name and entity are correct.")
            # Try to list available projects for helpful error message
            try:
                entity: str | None = self.config.entity or self.api.default_entity
                available: list[str] = [p.name for p in self.api.projects(entity)]
                if available:
                    _ = logger.info(f"Available projects for entity '{entity}':")
                    for proj in available[:10]:
                        _ = logger.info(f"  - {proj}")
                    if len(available) > 10:
                        _ = logger.info(f"  ... and {len(available) - 10} more")
            except Exception:
                pass
            raise SystemExit(1) from e
        _ = logger.info(f"Found {len(runs)} runs in project")

        if run_ids:
            run_ids_set: set[str] = set(run_ids)
            runs = [r for r in runs if r.id in run_ids_set]
            _ = logger.info(f"Filtered to {len(runs)} specified runs")

        summary: dict[str, Any] = {
            "exported": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }

        # Determine which runs to process
        runs_to_export: list[Run] = []
        for run in runs:
            if not force and self.state.is_completed(run.id):
                _ = logger.info(f"Skipping completed run: {run.id} ({run.name})")
                summary["skipped"] += 1
            else:
                _ = runs_to_export.append(run)

        if not runs_to_export:
            _ = logger.info("No runs to export")
            return summary

        # Export with progress bar
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            # Overall progress
            overall_task = progress.add_task(
                f"Exporting {len(runs_to_export)} runs",
                total=len(runs_to_export),
            )

            for run in runs_to_export:
                if self.state.is_in_progress(run.id):
                    _ = logger.warning(f"Cleaning up incomplete export for: {run.id}")
                    _ = self._cleanup_partial(run.id)

                try:
                    _ = self._export_run_with_progress(run, progress)
                    summary["exported"] += 1
                except Exception as e:
                    _ = logger.error(f"Failed to export run {run.id}: {e}")
                    _ = self.state.mark_failed(run.id, str(e))
                    summary["failed"] += 1
                    _ = summary["errors"].append({"run_id": run.id, "error": str(e)})

                _ = progress.advance(overall_task)

        _ = logger.info(
            f"Export complete: {summary['exported']} exported, "
            f"{summary['skipped']} skipped, {summary['failed']} failed"
        )

        # Log output location
        output_path: Path = self.config.output_dir / self.config.project
        _ = logger.info(f"Output saved to: {output_path.absolute()}")

        return summary

    def _export_run_with_progress(self, run: Run, progress: Progress) -> None:
        """Export a single run with progress tracking.

        Parameters
        ----------
        run
            W&B Run object to export.
        progress
            Rich Progress instance for display.
        """
        run_dir: Path = self.config.output_dir / self.config.project / run.id
        _ = run_dir.mkdir(parents=True, exist_ok=True)

        # Determine components to export
        components_to_export: list[str] = [
            "config",
            "summary",
            "metadata",
            "history",
            "system_metrics",
        ]
        if not self.config.skip_media:
            _ = components_to_export.append("media")
        if not self.config.skip_artifacts:
            _ = components_to_export.append("artifacts")

        # Per-run progress
        run_task: TaskID = progress.add_task(
            f"  {run.id} ({run.name[:30]}...)"
            if len(run.name) > 30
            else f"  {run.id} ({run.name})",
            total=len(components_to_export),
        )

        # run.path is a list like ["entity", "project", "run_id"], convert to string
        run_path_str = "/".join(str(p) for p in run.path)
        _ = self.state.mark_started(run.id, run_path_str)
        components: dict[str, bool] = {}

        for component in components_to_export:
            _ = progress.update(run_task, description=f"  {run.id}: {component}")

            if component == "config":
                components["config"] = self._export_config(run, run_dir)
            elif component == "summary":
                components["summary"] = self._export_summary(run, run_dir)
            elif component == "metadata":
                components["metadata"] = self._export_metadata(run, run_dir)
            elif component == "history":
                components["history"] = self._export_history(
                    run, run_dir, progress, run_task
                )
            elif component == "system_metrics":
                components["system_metrics"] = self._export_system_metrics(run, run_dir)
            elif component == "media":
                components["media"] = self._export_media(run, run_dir)
            elif component == "artifacts":
                components["artifacts"] = self._export_artifacts(run, run_dir)

            _ = progress.advance(run_task)

        # Write completion marker
        marker_path: Path = run_dir / "_export_complete.marker"
        _ = marker_path.write_text(
            json.dumps({"run_id": run.id, "components": components})
        )
        _ = logger.info(f"Saved marker: {marker_path}")

        _ = self.state.mark_completed(run.id, components)
        _ = progress.update(run_task, description=f"  {run.id}: done", visible=False)
        _ = logger.info(f"Completed export for run: {run.id} -> {run_dir}")

    def export_run(self, run: Run) -> None:
        """Export a single run (without progress bar).

        Parameters
        ----------
        run
            W&B Run object to export.
        """
        with Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            console=self.console,
            transient=True,
        ) as progress:
            self._export_run_with_progress(run, progress)

    def _export_config(self, run: Run, run_dir: Path) -> bool:
        """Export run configuration to JSON."""
        config_path: Path = run_dir / "config.json"
        try:
            config_data: dict[str, Any] = self.rate_limiter.call_with_retry(
                lambda: dict(run.config)
            )
            _ = config_path.write_text(json.dumps(config_data, indent=2, default=str))
            _ = logger.info(f"Saved config: {config_path}")
            return True
        except Exception as e:
            _ = logger.error(f"Failed to export config for {run.id}: {e}")
            raise

    def _export_summary(self, run: Run, run_dir: Path) -> bool:
        """Export run summary metrics to JSON."""
        summary_path: Path = run_dir / "summary.json"
        try:
            # run.summary is a special wandb object, iterate carefully
            summary_data: dict[str, Any] = {}
            raw_summary: Any = self.rate_limiter.call_with_retry(lambda: run.summary)
            for key in raw_summary:
                try:
                    summary_data[key] = raw_summary[key]
                except Exception as key_err:
                    _ = logger.warning(
                        f"Failed to access summary key '{key}': {key_err}"
                    )
                    summary_data[key] = f"<error: {key_err}>"
            # Convert any non-serializable values
            serializable: Any = _make_json_serializable(summary_data)
            _ = summary_path.write_text(json.dumps(serializable, indent=2, default=str))
            _ = logger.info(f"Saved summary: {summary_path}")
            return True
        except Exception as e:
            _ = logger.error(f"Failed to export summary for {run.id}: {e}")
            raise

    def _export_metadata(self, run: Run, run_dir: Path) -> bool:
        """Export run metadata (tags, state, timestamps) to JSON."""
        metadata_path: Path = run_dir / "metadata.json"
        try:
            metadata: dict[str, Any] = {
                "id": run.id,
                "name": run.name,
                "path": run.path,
                "state": run.state,
                "tags": list(run.tags) if run.tags else [],
                "created_at": run.created_at,
                "updated_at": getattr(run, "updated_at", None),
                "url": run.url,
                "notes": getattr(run, "notes", None),
                "group": getattr(run, "group", None),
                "job_type": getattr(run, "job_type", None),
            }
            _ = metadata_path.write_text(json.dumps(metadata, indent=2, default=str))
            _ = logger.info(f"Saved metadata: {metadata_path}")
            return True
        except Exception as e:
            _ = logger.error(f"Failed to export metadata for {run.id}: {e}")
            raise

    def _export_history(
        self,
        run: Run,
        run_dir: Path,
        progress: Progress | None = None,
        task_id: TaskID | None = None,
    ) -> bool:
        """Export full metrics history to Parquet using scan_history.

        Uses scan_history instead of history() to get full data without
        W&B's default subsampling.
        """
        metrics_dir: Path = run_dir / "metrics"
        _ = metrics_dir.mkdir(exist_ok=True)
        history_path: Path = metrics_dir / "history.parquet"

        try:
            # scan_history returns an iterator, collect all records
            # This avoids subsampling that history() does by default
            records: list[dict[str, Any]] = []
            scanner: Any = self.rate_limiter.call_with_retry(
                lambda: run.scan_history(page_size=self.config.page_size)
            )

            # Collect with progress updates
            for i, record in enumerate(scanner):
                _ = records.append(record)
                if progress and task_id and i % 500 == 0:
                    _ = progress.update(
                        task_id, description=f"  {run.id}: history ({i} rows)"
                    )

            if not records:
                _ = logger.info(f"No history records for {run.id}")
                return True

            df: pd.DataFrame = pd.DataFrame(records)
            _ = df.to_parquet(history_path, index=False)
            _ = logger.info(f"Saved history ({len(records)} rows): {history_path}")
            return True
        except Exception as e:
            _ = logger.error(f"Failed to export history for {run.id}: {e}")
            raise

    def _export_system_metrics(self, run: Run, run_dir: Path) -> bool:
        """Export system metrics (GPU, CPU, memory) to Parquet.

        System metrics are stored separately from user metrics in W&B.
        We try multiple approaches to get complete data:
        1. Download raw wandb-events.jsonl file (contains full system metrics)
        2. Fall back to history(stream="system") API with progressively smaller sample sizes

        System metrics export is best-effort - failures don't fail the run export.
        """
        metrics_dir = run_dir / "metrics"
        metrics_dir.mkdir(exist_ok=True)
        system_path = metrics_dir / "system_metrics.parquet"

        # Approach 1: Try to get system metrics from files (full data)
        try:
            files = self.rate_limiter.call_with_retry(lambda: list(run.files()))
            system_files = [
                f
                for f in files
                if "events" in f.name.lower() or "system" in f.name.lower()
            ]

            if system_files:
                # Download system metrics files
                system_files_dir = metrics_dir / "system_files"
                system_files_dir.mkdir(exist_ok=True)

                for f in system_files:
                    try:
                        self.rate_limiter.call_with_retry(
                            lambda file=f: file.download(
                                root=str(system_files_dir), replace=True
                            )
                        )
                        logger.info(
                            f"Downloaded system file: {system_files_dir / f.name}"
                        )
                    except Exception as e:
                        logger.debug(f"Could not download {f.name}: {e}")

        except Exception as e:
            logger.debug(f"Could not list files for system metrics: {e}")

        # Approach 2: Try history API with different sample sizes
        sample_sizes = [10000, 5000, 1000, 500]

        for samples in sample_sizes:
            try:
                logger.debug(f"Trying system metrics with samples={samples}")
                # Cast to DataFrame - run.history() with pandas=True returns DataFrame
                # but wandb's type hints don't capture this
                system_df = cast(
                    pd.DataFrame | None,
                    self.rate_limiter.call_with_retry(
                        lambda s=samples: run.history(
                            samples=s, stream="system", pandas=True
                        )
                    ),
                )

                if system_df is not None and not system_df.empty:
                    system_df.to_parquet(system_path, index=False)
                    logger.info(
                        f"Saved system metrics ({len(system_df)} rows): {system_path}"
                    )
                    return True
                else:
                    logger.info(f"No system metrics for {run.id}")
                    return True

            except Exception as e:
                if samples == sample_sizes[-1]:
                    # Last attempt failed
                    logger.warning(f"Failed to export system metrics for {run.id}: {e}")
                    logger.warning("Continuing without system metrics (best-effort)")
                    return False
                else:
                    logger.debug(f"samples={samples} failed, trying smaller: {e}")
                    continue

        return False

    def _export_media(self, run: Run, run_dir: Path) -> bool:
        """Export media files (images, etc.)."""
        media_dir = run_dir / "media"

        try:
            files = self.rate_limiter.call_with_retry(lambda: list(run.files()))

            if not files:
                logger.debug(f"No files for {run.id}")
                return True

            # Filter out standard wandb files
            skip_files = {
                "config.yaml",
                "wandb-metadata.json",
                "wandb-summary.json",
                "requirements.txt",
                "output.log",
            }
            # Also skip system metric files (handled separately)
            skip_patterns = ["events", "system"]

            media_files = [
                f
                for f in files
                if f.name not in skip_files
                and not any(p in f.name.lower() for p in skip_patterns)
            ]

            if not media_files:
                logger.debug(f"No media files for {run.id}")
                return True

            media_dir.mkdir(exist_ok=True)

            for file in media_files:
                try:
                    self.rate_limiter.call_with_retry(
                        lambda f=file: f.download(root=str(media_dir), replace=True)
                    )
                    logger.info(f"Saved media: {media_dir / file.name}")
                except Exception as e:
                    logger.warning(f"Failed to download {file.name}: {e}")

            logger.info(f"Exported {len(media_files)} media files to {media_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to export media for {run.id}: {e}")
            raise

    def _export_artifacts(self, run: Run, run_dir: Path) -> bool:
        """Export logged artifacts."""
        artifacts_dir = run_dir / "artifacts"

        try:
            artifacts = self.rate_limiter.call_with_retry(
                lambda: list(run.logged_artifacts())
            )

            if not artifacts:
                logger.debug(f"No artifacts for {run.id}")
                return True

            artifacts_dir.mkdir(exist_ok=True)

            for artifact in artifacts:
                artifact_name = artifact.name.replace("/", "_").replace(":", "_")
                artifact_path = artifacts_dir / artifact_name

                try:
                    self.rate_limiter.call_with_retry(
                        lambda a=artifact, p=artifact_path: a.download(root=str(p))
                    )
                    logger.info(f"Saved artifact: {artifact_path}")
                except Exception as e:
                    logger.warning(f"Failed to download artifact {artifact.name}: {e}")

            logger.info(f"Exported {len(artifacts)} artifacts to {artifacts_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to export artifacts for {run.id}: {e}")
            raise

    def _cleanup_partial(self, run_id: str) -> None:
        """Remove partial export directory for clean re-export."""
        run_dir = self.config.output_dir / self.config.project / run_id
        if run_dir.exists():
            logger.warning(f"Removing incomplete export directory: {run_dir}")
            shutil.rmtree(run_dir)


def _make_json_serializable(obj: Any) -> Any:
    """Convert object to JSON-serializable form.

    Handles wandb-specific types and numpy arrays.
    """
    # Handle None and basic JSON types first
    if obj is None or isinstance(obj, str | int | float | bool):
        return obj
    if isinstance(obj, dict):
        return {str(k): _make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_make_json_serializable(v) for v in obj]

    # Check type name to detect wandb objects (they override __getattr__ badly)
    type_name = type(obj).__name__

    # Handle wandb SummarySubDict - convert to regular dict
    if "SummarySubDict" in type_name or "Summary" in type_name:
        try:
            if hasattr(type(obj), "keys"):
                return {str(k): _make_json_serializable(obj[k]) for k in obj}
        except Exception:
            pass

    # numpy arrays - check via type, not hasattr (wandb objects break hasattr)
    if type_name == "ndarray" or "numpy" in str(type(obj)):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)

    # wandb Image, Histogram, etc.
    if "wandb" in str(type(obj).__module__):
        try:
            if hasattr(type(obj), "_json_dict"):
                return obj._json_dict
        except Exception:
            pass
        return f"<{type_name}>"

    # Fallback to string representation
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"
