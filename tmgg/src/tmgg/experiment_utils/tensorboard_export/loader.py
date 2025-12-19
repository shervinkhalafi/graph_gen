"""Loader module for parsing TensorBoard events and config files."""

import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pandas as pd
from omegaconf import OmegaConf
from tensorboard.compat.proto import event_pb2

from .discovery import DiscoveredRun


def _read_events(path: Path) -> Iterator[Any]:
    """Read events from a TensorBoard event file using raw protobuf.

    TensorBoard event files use a record format:
    - 8 bytes: length of data (uint64, little endian)
    - 4 bytes: CRC32 of length
    - N bytes: data (protobuf Event message)
    - 4 bytes: CRC32 of data
    """
    with open(path, "rb") as f:
        while True:
            # Read length header
            header = f.read(8)
            if len(header) < 8:
                break

            length = struct.unpack("<Q", header)[0]

            # Skip length CRC
            f.read(4)

            # Read data
            data = f.read(length)
            if len(data) < length:
                break

            # Skip data CRC
            f.read(4)

            # Parse event
            event = event_pb2.Event()
            event.ParseFromString(data)  # pyright: ignore[reportAttributeAccessIssue]
            yield event


def load_events(run: DiscoveredRun) -> pd.DataFrame:
    """Load all scalar events from a run's TensorBoard files.

    Reads event files directly using protobuf without TensorFlow dependency.

    Parameters
    ----------
    run
        Discovered run with event file paths.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: project_id, run_id, tag, step, wall_time, value
    """
    all_events = []

    for event_file in run.event_files:
        try:
            for event in _read_events(event_file):
                # Only process summary events with scalar values
                if event.HasField("summary"):
                    for value in event.summary.value:
                        # Check for simple_value (scalar)
                        if value.HasField("simple_value"):
                            all_events.append(
                                {
                                    "project_id": run.project_id,
                                    "run_id": run.run_id,
                                    "tag": value.tag,
                                    "step": event.step,
                                    "wall_time": event.wall_time,
                                    "value": value.simple_value,
                                }
                            )

        except Exception as e:
            print(f"Warning: Failed to load {event_file}: {e}")
            continue

    if not all_events:
        return pd.DataFrame(
            columns=["project_id", "run_id", "tag", "step", "wall_time", "value"]  # pyright: ignore[reportArgumentType]
        )

    return pd.DataFrame(all_events)


def load_config(run: DiscoveredRun) -> dict[str, Any]:
    """Load and flatten config from a run's config.yaml.

    Parameters
    ----------
    run
        Discovered run with config file path.

    Returns
    -------
    dict[str, Any]
        Flattened config dictionary with project_id and run_id.
    """
    result = {
        "project_id": run.project_id,
        "run_id": run.run_id,
    }

    if run.config_file is None or not run.config_file.exists():
        return result

    try:
        config = OmegaConf.load(run.config_file)
        config_dict = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config_dict, dict):
            return result
        # Cast to satisfy pyright - we've verified it's a dict above
        flattened = _flatten_dict(cast(dict[str, Any], config_dict))
        result.update(flattened)
    except Exception as e:
        print(f"Warning: Failed to load config {run.config_file}: {e}")

    return result


def _flatten_dict(
    d: dict[str, Any], parent_key: str = "", sep: str = "."
) -> dict[str, Any]:
    """Flatten a nested dictionary.

    Parameters
    ----------
    d
        Dictionary to flatten.
    parent_key
        Prefix for keys (used in recursion).
    sep
        Separator between nested keys.

    Returns
    -------
    dict[str, Any]
        Flattened dictionary with dot-separated keys.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        elif isinstance(v, list):
            # Convert lists to string representation for DataFrame compatibility
            items.append((new_key, str(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def load_run(run: DiscoveredRun) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load both events and config for a run.

    Parameters
    ----------
    run
        Discovered run to load.

    Returns
    -------
    tuple[pd.DataFrame, dict[str, Any]]
        Events DataFrame and config dictionary.
    """
    events = load_events(run)
    config = load_config(run)
    return events, config
