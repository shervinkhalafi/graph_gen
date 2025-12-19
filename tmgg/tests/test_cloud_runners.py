"""Tests for cloud runners and coordinator.

Test rationale
--------------
Verifies that both LocalRunner and ModalRunner can be imported at the top level
without circular dependencies, and that ExperimentCoordinator correctly accepts
runner instances directly.
"""

from pathlib import Path

import pytest

from tmgg.experiment_utils.cloud import (
    CloudRunner,
    ExperimentCoordinator,
    LocalRunner,
)
from tmgg.modal.runner import ModalRunner


class TestRunnerImports:
    """Verify that runners can be imported without circular dependencies."""

    def test_local_runner_importable(self):
        """LocalRunner imports from cloud package."""
        assert LocalRunner is not None
        assert issubclass(LocalRunner, CloudRunner)

    def test_modal_runner_importable(self):
        """ModalRunner imports from modal package."""
        assert ModalRunner is not None
        assert issubclass(ModalRunner, CloudRunner)

    def test_both_runners_importable_together(self):
        """Both runners can be imported in the same module."""
        # This test passes by virtue of the imports at module level
        assert LocalRunner is not None
        assert ModalRunner is not None


class TestLocalRunner:
    """Tests for LocalRunner."""

    def test_create_local_runner(self):
        """LocalRunner can be instantiated."""
        runner = LocalRunner()
        assert isinstance(runner, LocalRunner)

    def test_local_runner_with_output_dir(self):
        """LocalRunner accepts custom output_dir."""
        runner = LocalRunner(output_dir=Path("/tmp/test"))
        assert runner.output_dir == Path("/tmp/test")


class TestCoordinatorWithRunner:
    """Tests for ExperimentCoordinator with direct runner injection."""

    def test_coordinator_default_local(self):
        """Coordinator defaults to LocalRunner when no runner provided."""
        coordinator = ExperimentCoordinator()
        assert isinstance(coordinator.runner, LocalRunner)

    def test_coordinator_with_local_runner(self):
        """Coordinator accepts LocalRunner directly."""
        runner = LocalRunner()
        coordinator = ExperimentCoordinator(runner=runner)
        assert coordinator.runner is runner

    def test_coordinator_with_modal_runner(self):
        """Coordinator accepts ModalRunner directly."""
        runner = ModalRunner()
        coordinator = ExperimentCoordinator(runner=runner)  # pyright: ignore[reportArgumentType]
        assert coordinator.runner is runner
        assert isinstance(coordinator.runner, ModalRunner)

    def test_coordinator_runner_has_precedence(self):
        """Explicit runner instance is used as provided."""
        custom_runner = LocalRunner(output_dir=Path("/custom"))
        coordinator = ExperimentCoordinator(runner=custom_runner)
        assert coordinator.runner is custom_runner
        assert coordinator.runner.output_dir == Path("/custom")  # pyright: ignore[reportAttributeAccessIssue]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
