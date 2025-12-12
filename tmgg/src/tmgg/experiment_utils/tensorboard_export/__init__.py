"""TensorBoard export utilities.

Export TensorBoard event files to pandas DataFrames for analysis.

Examples
--------
>>> from tmgg.experiment_utils.tensorboard_export import TensorBoardExporter
>>> exporter = TensorBoardExporter()
>>> result = exporter.export_directory("outputs/sweeps/stage1_poc")
>>> result.events_df.head()
>>> result.hparams_df.head()
>>> result.save("analysis_output/")
"""

from .discovery import DiscoveredRun, discover_runs
from .exporter import ExportResult, TensorBoardExporter
from .loader import load_config, load_events, load_run

__all__ = [
    "TensorBoardExporter",
    "ExportResult",
    "DiscoveredRun",
    "discover_runs",
    "load_events",
    "load_config",
    "load_run",
]
