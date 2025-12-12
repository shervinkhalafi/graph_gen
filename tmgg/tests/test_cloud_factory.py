"""Tests for CloudRunner factory."""

import pytest
from pathlib import Path

from tmgg.experiment_utils.cloud import (
    CloudRunner,
    CloudRunnerFactory,
    LocalRunner,
    ExperimentCoordinator,
)


class TestCloudRunnerFactory:
    """Tests for CloudRunnerFactory class."""

    def test_local_runner_registered(self):
        """Test that LocalRunner is registered by default."""
        assert CloudRunnerFactory.is_registered("local")

    def test_create_local_runner(self):
        """Test creating a LocalRunner via factory."""
        runner = CloudRunnerFactory.create("local")
        assert isinstance(runner, LocalRunner)

    def test_create_local_runner_with_kwargs(self):
        """Test creating LocalRunner with custom output_dir."""
        runner = CloudRunnerFactory.create("local", output_dir=Path("/tmp/test"))
        assert isinstance(runner, LocalRunner)
        assert runner.output_dir == Path("/tmp/test")

    def test_unknown_backend_raises(self):
        """Test that unknown backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            CloudRunnerFactory.create("nonexistent_backend")

    def test_available_backends_includes_local(self):
        """Test that available_backends includes local."""
        backends = CloudRunnerFactory.available_backends()
        assert "local" in backends

    def test_register_custom_runner(self):
        """Test registering a custom runner."""

        class CustomRunner(CloudRunner):
            def run_experiment(self, config, gpu_type="standard", timeout_seconds=3600):
                pass

            def run_sweep(self, configs, gpu_type="standard", parallelism=4, timeout_seconds=3600):
                return []

            def get_status(self, run_id):
                return "unknown"

            def cancel(self, run_id):
                return False

        CloudRunnerFactory.register("custom_test", CustomRunner)
        assert CloudRunnerFactory.is_registered("custom_test")

        runner = CloudRunnerFactory.create("custom_test")
        assert isinstance(runner, CustomRunner)

        # Cleanup
        del CloudRunnerFactory._runners["custom_test"]


class TestCoordinatorWithFactory:
    """Tests for ExperimentCoordinator using factory."""

    def test_coordinator_default_local(self):
        """Test coordinator defaults to local backend."""
        coordinator = ExperimentCoordinator()
        assert isinstance(coordinator.runner, LocalRunner)

    def test_coordinator_explicit_backend(self):
        """Test coordinator with explicit backend parameter."""
        coordinator = ExperimentCoordinator(backend="local")
        assert isinstance(coordinator.runner, LocalRunner)

    def test_coordinator_runner_takes_precedence(self):
        """Test that explicit runner takes precedence over backend."""
        custom_runner = LocalRunner(output_dir=Path("/custom"))
        coordinator = ExperimentCoordinator(
            runner=custom_runner,
            backend="local"  # Should be ignored
        )
        assert coordinator.runner is custom_runner
        assert coordinator.runner.output_dir == Path("/custom")

    def test_coordinator_invalid_backend_raises(self):
        """Test that invalid backend raises ValueError."""
        with pytest.raises(ValueError, match="Unknown backend"):
            ExperimentCoordinator(backend="invalid_backend")


class TestModalAutoRegistration:
    """Tests for Modal runner auto-registration."""

    def test_modal_registration_status(self):
        """Test Modal registration depends on tmgg_modal availability.

        If tmgg_modal is installed, 'modal' should be registered.
        If not, it should not be registered.
        """
        try:
            from tmgg_modal.runner import ModalRunner
            modal_available = True
        except ImportError:
            modal_available = False

        if modal_available:
            assert CloudRunnerFactory.is_registered("modal")
        else:
            # Modal not available is fine - just check factory works
            assert "local" in CloudRunnerFactory.available_backends()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
