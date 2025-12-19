"""JSONL state management for W&B export progress tracking.

Tracks which runs have been exported, allowing resumable exports and detection
of partial/incomplete exports that need cleanup.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from typing import Any

EXPORT_VERSION = 1


@dataclass
class RunExportStatus:
    """Status of a single run's export.

    Parameters
    ----------
    run_id
        W&B run ID (short form, e.g. "abc123").
    run_path
        Full W&B path (e.g. "entity/project/abc123").
    started_at
        ISO timestamp when export started.
    completed_at
        ISO timestamp when export completed, None if incomplete.
    status
        One of "in_progress", "completed", "failed".
    error_message
        Error details if status is "failed".
    export_version
        Schema version for future migrations.
    components
        Dict mapping component names to export success status.
    """

    run_id: str
    run_path: str
    started_at: str
    completed_at: str | None = None
    status: str = "in_progress"
    error_message: str | None = None
    export_version: int = EXPORT_VERSION
    components: dict[str, bool] = field(default_factory=dict)

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RunExportStatus:
        """Deserialize from dict."""
        return cls(
            run_id=data["run_id"],
            run_path=data["run_path"],
            started_at=data["started_at"],
            completed_at=data.get("completed_at"),
            status=data.get("status", "in_progress"),
            error_message=data.get("error_message"),
            export_version=data.get("export_version", 1),
            components=data.get("components", {}),
        )


class ExportState:
    """Manages JSONL state file for tracking export progress.

    The state file contains one JSON object per line, with the most recent
    status for each run taking precedence. This append-only design ensures
    crash safety - we never modify existing lines.

    Parameters
    ----------
    state_file
        Path to the JSONL state file. Will be created if it doesn't exist.
    """

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self._cache: dict[str, RunExportStatus] = {}
        self._load()

    def _load(self) -> None:
        """Load existing state from JSONL file.

        Reads all lines and keeps the most recent status for each run_id.
        """
        if not self.state_file.exists():
            logger.debug(f"State file does not exist: {self.state_file}")
            return

        with self.state_file.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    status = RunExportStatus.from_dict(data)
                    self._cache[status.run_id] = status
                except (json.JSONDecodeError, KeyError) as e:
                    logger.warning(
                        f"Skipping malformed line {line_num} in state file: {e}"
                    )

        logger.debug(f"Loaded {len(self._cache)} run statuses from {self.state_file}")

    def is_completed(self, run_id: str) -> bool:
        """Check if run has been fully exported."""
        status = self._cache.get(run_id)
        return status is not None and status.status == "completed"

    def is_in_progress(self, run_id: str) -> bool:
        """Check if run export started but didn't complete (partial export)."""
        status = self._cache.get(run_id)
        return status is not None and status.status == "in_progress"

    def is_failed(self, run_id: str) -> bool:
        """Check if run export previously failed."""
        status = self._cache.get(run_id)
        return status is not None and status.status == "failed"

    def get_status(self, run_id: str) -> RunExportStatus | None:
        """Get the current status for a run."""
        return self._cache.get(run_id)

    def mark_started(self, run_id: str, run_path: str) -> None:
        """Mark run export as started."""
        status = RunExportStatus(
            run_id=run_id,
            run_path=run_path,
            started_at=datetime.now(UTC).isoformat(),
            status="in_progress",
        )
        self._append(status)

    def mark_completed(self, run_id: str, components: dict[str, bool]) -> None:
        """Mark run export as completed with component status."""
        existing = self._cache.get(run_id)
        if existing is None:
            raise ValueError(
                f"Cannot mark {run_id} as completed - no started status found"
            )

        status = RunExportStatus(
            run_id=run_id,
            run_path=existing.run_path,
            started_at=existing.started_at,
            completed_at=datetime.now(UTC).isoformat(),
            status="completed",
            components=components,
        )
        self._append(status)

    def mark_failed(self, run_id: str, error: str) -> None:
        """Mark run export as failed with error message."""
        existing = self._cache.get(run_id)
        if existing is None:
            raise ValueError(
                f"Cannot mark {run_id} as failed - no started status found"
            )

        status = RunExportStatus(
            run_id=run_id,
            run_path=existing.run_path,
            started_at=existing.started_at,
            completed_at=datetime.now(UTC).isoformat(),
            status="failed",
            error_message=error,
        )
        self._append(status)

    def get_incomplete_runs(self) -> list[str]:
        """Get list of run IDs that started but didn't complete."""
        return [
            run_id
            for run_id, status in self._cache.items()
            if status.status == "in_progress"
        ]

    def get_completed_runs(self) -> list[str]:
        """Get list of run IDs that have been fully exported."""
        return [
            run_id
            for run_id, status in self._cache.items()
            if status.status == "completed"
        ]

    def _append(self, status: RunExportStatus) -> None:
        """Append new status line to JSONL file and update cache."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with self.state_file.open("a", encoding="utf-8") as f:
            f.write(status.to_json() + "\n")
        self._cache[status.run_id] = status
        logger.debug(f"Updated state for {status.run_id}: {status.status}")
