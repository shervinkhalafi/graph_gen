"""W&B data export utilities.

Provides tools for exporting W&B project data to local Parquet/JSON files
with full history (no subsampling), state tracking for resumable exports,
and proper rate limiting.

Example usage:

    from tmgg.experiment_utils.wandb_export import WandbExporter, ExportConfig

    config = ExportConfig(
        project="my-project",
        entity="my-team",
        output_dir=Path("./exports"),
    )
    exporter = WandbExporter(config)
    summary = exporter.export_project()

CLI usage:

    tmgg-wandb-export --project my-project --output-dir ./exports
"""

from .exporter import ExportConfig, WandbExporter
from .rate_limiter import RateLimiter
from .state import ExportState, RunExportStatus

__all__ = [
    "ExportConfig",
    "ExportState",
    "RateLimiter",
    "RunExportStatus",
    "WandbExporter",
]
