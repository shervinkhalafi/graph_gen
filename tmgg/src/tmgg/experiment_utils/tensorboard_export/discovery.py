"""Discovery module for finding TensorBoard event files and configs."""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DiscoveredRun:
    """A discovered TensorBoard run with its associated files."""

    run_id: str
    project_id: str
    event_files: list[Path]
    config_file: Path | None
    run_dir: Path


def discover_runs(root_dir: Path, project_id: str | None = None) -> list[DiscoveredRun]:
    """Discover all TensorBoard runs in a directory tree.

    Searches for tfevents files and associated config.yaml files.
    The directory structure is expected to be:
        root_dir/
        ├── run_id_1/
        │   ├── config.yaml
        │   └── tensorboard*/.../*.tfevents.*
        ├── run_id_2/
        │   └── ...

    Parameters
    ----------
    root_dir
        Root directory to search for runs.
    project_id
        Project identifier. If None, uses root_dir.name.

    Returns
    -------
    list[DiscoveredRun]
        List of discovered runs with their event files and configs.
    """
    root_dir = Path(root_dir)
    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    if project_id is None:
        project_id = root_dir.name

    runs = []

    # Find all tfevents files
    event_files = list(root_dir.rglob("events.out.tfevents.*"))

    # Group by run directory (parent of tensorboard* dir)
    run_dirs: dict[Path, list[Path]] = {}
    for event_file in event_files:
        # Walk up to find the run directory (contains config.yaml or is direct child of root)
        run_dir = _find_run_dir(event_file, root_dir)
        if run_dir:
            if run_dir not in run_dirs:
                run_dirs[run_dir] = []
            run_dirs[run_dir].append(event_file)

    # Create DiscoveredRun objects
    for run_dir, events in run_dirs.items():
        config_file = run_dir / "config.yaml"
        if not config_file.exists():
            config_file = None

        run_id = run_dir.name
        runs.append(
            DiscoveredRun(
                run_id=run_id,
                project_id=project_id,
                event_files=sorted(events),
                config_file=config_file,
                run_dir=run_dir,
            )
        )

    return sorted(runs, key=lambda r: r.run_id)


def _find_run_dir(event_file: Path, root_dir: Path) -> Path | None:
    """Find the run directory for an event file.

    Walks up the directory tree looking for a directory that:
    1. Contains config.yaml, or
    2. Is a direct child of root_dir
    """
    current = event_file.parent
    while current != root_dir and current != current.parent:
        # Check if this is the run dir (has config.yaml)
        if (current / "config.yaml").exists():
            return current
        # Check if parent is root_dir (this is a direct child)
        if current.parent == root_dir:
            return current
        current = current.parent

    # If we reached root_dir, check if the first child is the run dir
    relative = event_file.relative_to(root_dir)
    if len(relative.parts) > 0:
        return root_dir / relative.parts[0]

    return None
