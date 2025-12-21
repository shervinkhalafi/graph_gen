"""Utility functions for integration tests.

This module provides helpers for running CLI commands as subprocesses with
timeout handling and output validation. Designed for testing experiment
runners without risking Hydra state conflicts.
"""

import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path


def run_cli_command(
    cmd: Sequence[str],
    timeout: int = 120,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a CLI command with timeout, returning captured output.

    Parameters
    ----------
    cmd
        Command and arguments to execute.
    timeout
        Maximum execution time in seconds. Defaults to 120s.
    cwd
        Working directory for the command. Defaults to current directory.

    Returns
    -------
    subprocess.CompletedProcess
        Result with returncode, stdout, and stderr.

    Raises
    ------
    subprocess.TimeoutExpired
        If command exceeds timeout.
    """
    result = subprocess.run(
        list(cmd),
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=cwd,
        env=None,  # Inherit environment
    )
    return result


def assert_training_success(result: subprocess.CompletedProcess[str]) -> None:
    """Assert that a training run completed without Python exceptions.

    Parameters
    ----------
    result
        Completed subprocess result from run_cli_command.

    Raises
    ------
    AssertionError
        If the command failed or stderr contains exception traces.
    """
    # Check exit code
    if result.returncode != 0:
        raise AssertionError(
            f"Command failed with exit code {result.returncode}.\n"
            f"stdout:\n{result.stdout}\n"
            f"stderr:\n{result.stderr}"
        )

    # Check for Python exceptions in stderr
    exception_markers = [
        "Traceback (most recent call last)",
        "Error:",
        "Exception:",
        "raise ",
    ]
    stderr_lower = result.stderr.lower()
    for marker in exception_markers:
        if marker.lower() in stderr_lower:
            # Allow certain expected log messages
            if "error" in marker.lower() and (
                "error_rate" in stderr_lower or "no error" in stderr_lower
            ):
                continue
            raise AssertionError(
                f"Found exception marker '{marker}' in stderr.\n"
                f"stderr:\n{result.stderr}"
            )


def get_quick_training_overrides(output_dir: Path) -> list[str]:
    """Generate Hydra overrides for a minimal training run.

    These overrides configure the experiment to complete in seconds by:
    - Limiting to 2 training steps
    - Disabling checkpointing and logging
    - Using minimal data
    - Running on CPU

    Parameters
    ----------
    output_dir
        Directory for experiment outputs.

    Returns
    -------
    list[str]
        Hydra override strings.
    """
    return [
        f"paths.output_dir={output_dir}",
        f"paths.results_dir={output_dir}/results",
        "trainer.max_steps=2",
        "trainer.val_check_interval=1",
        "trainer.accelerator=cpu",
        # Note: enable_checkpointing and enable_progress_bar must stay true
        # because create_callbacks unconditionally adds ModelCheckpoint and
        # StepProgressBar, which conflict with False settings
        # Disable logger config (tensorboard/wandb) - falls back to CSVLogger
        "~logger",
        # Data reduction - common keys that exist in base_dataloader
        "data.batch_size=2",
        "data.num_workers=0",
        # Reduce samples/partitions - use ++ to add if not present
        "++data.num_samples_per_graph=4",
        "++data.dataset_config.num_train_partitions=2",
        "++data.dataset_config.num_test_partitions=2",
        # For SingleGraphDataModule (stage configs)
        "++data.num_train_samples=8",
        "++data.num_val_samples=4",
        "++data.num_test_samples=4",
        # For GraphDataModule pattern
        "++data.dataset_config.num_graphs=4",
        # Set hydra run dir to output dir
        f"hydra.run.dir={output_dir}",
    ]


def build_runner_command(
    runner: str,
    overrides: list[str],
) -> list[str]:
    """Build a command to run an experiment runner with overrides.

    Parameters
    ----------
    runner
        Name of the CLI runner (e.g., 'tmgg-spectral').
    overrides
        List of Hydra override strings.

    Returns
    -------
    list[str]
        Full command including 'uv run' prefix.
    """
    return [sys.executable, "-m", "uv", "run", runner, *overrides]
