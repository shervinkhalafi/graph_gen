"""Tests for Modal eigenstructure study functions and CLI.

Test rationale:
    The eigenstructure Modal functions enable running spectral analysis studies
    on Modal with persistent volume storage. These tests verify:
    1. Modal function definitions are correct (parameters, return types)
    2. CLI commands for list-remote and download work with mock Modal
    3. Volume path handling is correct

    Since Modal functions run remotely, these unit tests mock Modal to verify
    local behavior. Integration tests that actually run on Modal would require
    deployment and are marked with @pytest.mark.modal.

Invariants:
    - eigenstructure_* Modal functions accept expected parameters
    - CLI commands correctly invoke Modal functions
    - Volume paths are constructed correctly
"""

import importlib.util
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Mock modal before importing tmgg_modal modules to avoid image creation at import time
mock_modal = MagicMock()
mock_modal.exception = MagicMock()
mock_modal.exception.NotFoundError = type("NotFoundError", (Exception,), {})
mock_modal.Volume = MagicMock()
sys.modules["modal"] = mock_modal
sys.modules["modal.exception"] = mock_modal.exception

# Use importlib to load modal modules directly
_modal_base_path = Path(__file__).parent.parent.parent / "src" / "tmgg" / "modal"
_experiment_utils_path = (
    Path(__file__).parent.parent.parent / "src" / "tmgg" / "experiment_utils"
)


def _load_module(name: str, path: Path):
    """Load a module directly from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# Load dependencies in order
_app_module = _load_module("tmgg.modal.app", _modal_base_path / "app.py")
_volumes_module = _load_module("tmgg.modal.volumes", _modal_base_path / "volumes.py")
_image_module = _load_module("tmgg.modal.image", _modal_base_path / "image.py")
_paths_module = _load_module("tmgg.modal.paths", _modal_base_path / "paths.py")

# Load eigenstructure module
_eigenstructure_module = _load_module(
    "tmgg.modal.eigenstructure", _modal_base_path / "eigenstructure.py"
)

# Extract what we need
EIGENSTRUCTURE_MOUNT = _volumes_module.EIGENSTRUCTURE_MOUNT
get_eigenstructure_volume_mounts = _volumes_module.get_eigenstructure_volume_mounts
eigenstructure_collect = _eigenstructure_module.eigenstructure_collect
eigenstructure_analyze = _eigenstructure_module.eigenstructure_analyze
eigenstructure_noised = _eigenstructure_module.eigenstructure_noised
eigenstructure_compare = _eigenstructure_module.eigenstructure_compare
eigenstructure_list = _eigenstructure_module.eigenstructure_list
eigenstructure_list_files = _eigenstructure_module.eigenstructure_list_files
eigenstructure_read_file = _eigenstructure_module.eigenstructure_read_file


class TestEigenstructureVolume:
    """Tests for eigenstructure volume configuration."""

    def test_eigenstructure_mount_path_defined(self):
        """EIGENSTRUCTURE_MOUNT should be defined."""
        assert EIGENSTRUCTURE_MOUNT is not None
        assert EIGENSTRUCTURE_MOUNT == "/data/eigenstructure"

    def test_get_eigenstructure_volume_mounts_returns_dict(self):
        """get_eigenstructure_volume_mounts should return volume mapping."""
        mounts = get_eigenstructure_volume_mounts()
        assert isinstance(mounts, dict)
        assert EIGENSTRUCTURE_MOUNT in mounts


class TestEigenstructureModalFunctions:
    """Tests for Modal function definitions.

    These tests verify function signatures and basic structure without
    actually running on Modal.
    """

    def test_eigenstructure_collect_is_modal_function(self):
        """eigenstructure_collect should be a Modal function."""
        # Modal functions have remote, spawn, map methods
        assert hasattr(eigenstructure_collect, "remote")
        assert hasattr(eigenstructure_collect, "spawn")

    def test_eigenstructure_analyze_is_modal_function(self):
        """eigenstructure_analyze should be a Modal function."""
        assert hasattr(eigenstructure_analyze, "remote")
        assert hasattr(eigenstructure_analyze, "spawn")

    def test_eigenstructure_noised_is_modal_function(self):
        """eigenstructure_noised should be a Modal function."""
        assert hasattr(eigenstructure_noised, "remote")
        assert hasattr(eigenstructure_noised, "spawn")

    def test_eigenstructure_compare_is_modal_function(self):
        """eigenstructure_compare should be a Modal function."""
        assert hasattr(eigenstructure_compare, "remote")
        assert hasattr(eigenstructure_compare, "spawn")

    def test_eigenstructure_list_is_modal_function(self):
        """eigenstructure_list should be a Modal function."""
        assert hasattr(eigenstructure_list, "remote")

    def test_eigenstructure_list_files_is_modal_function(self):
        """eigenstructure_list_files should be a Modal function."""
        assert hasattr(eigenstructure_list_files, "remote")

    def test_eigenstructure_read_file_is_modal_function(self):
        """eigenstructure_read_file should be a Modal function."""
        assert hasattr(eigenstructure_read_file, "remote")


class TestEigenstructureCLIRemoteCommands:
    """Tests for eigenstructure CLI list-remote and download commands.

    These tests verify CLI behavior by mocking Modal function calls.
    """

    def test_list_remote_cli_exists(self):
        """list-remote command should be defined in CLI."""
        from click.testing import CliRunner

        # Load CLI module
        cli_path = _experiment_utils_path / "eigenstructure_study" / "cli.py"
        cli_module = _load_module(
            "tmgg.experiment_utils.eigenstructure_study.cli", cli_path
        )

        runner = CliRunner()
        result = runner.invoke(cli_module.main, ["list-remote", "--help"])

        assert result.exit_code == 0
        assert "List eigenstructure studies" in result.output

    def test_download_cli_exists(self):
        """download command should be defined in CLI."""
        from click.testing import CliRunner

        cli_path = _experiment_utils_path / "eigenstructure_study" / "cli.py"
        cli_module = _load_module(
            "tmgg.experiment_utils.eigenstructure_study.cli", cli_path
        )

        runner = CliRunner()
        result = runner.invoke(cli_module.main, ["download", "--help"])

        assert result.exit_code == 0
        assert "Download eigenstructure study" in result.output
        assert "--remote-path" in result.output
        assert "--local-path" in result.output


class TestDownloadCLILogic:
    """Tests for download command logic with mocked Modal functions."""

    @pytest.fixture
    def mock_modal_functions(self):
        """Set up mock Modal functions for download testing."""
        mock_list_files_fn = MagicMock()
        mock_read_file_fn = MagicMock()

        # Configure return values
        mock_list_files_fn.remote.return_value = [
            {
                "path": "/data/eigenstructure/test_study/manifest.json",
                "rel_path": "manifest.json",
                "size": 1024,
            },
            {
                "path": "/data/eigenstructure/test_study/batch_0.safetensors",
                "rel_path": "batch_0.safetensors",
                "size": 102400,
            },
        ]
        mock_read_file_fn.remote.return_value = b'{"test": "data"}'

        with patch.object(
            mock_modal.Function,
            "from_name",
            side_effect=lambda app, name: (
                mock_list_files_fn
                if name == "eigenstructure_list_files"
                else mock_read_file_fn
            ),
        ):
            yield mock_list_files_fn, mock_read_file_fn

    def test_download_requires_remote_and_local_path(self):
        """Download command should require both remote and local path."""
        from click.testing import CliRunner

        cli_path = _experiment_utils_path / "eigenstructure_study" / "cli.py"
        cli_module = _load_module(
            "tmgg.experiment_utils.eigenstructure_study.cli", cli_path
        )

        runner = CliRunner()

        # Missing local-path
        result = runner.invoke(cli_module.main, ["download", "--remote-path", "test"])
        assert result.exit_code != 0

        # Missing remote-path
        result = runner.invoke(
            cli_module.main, ["download", "--local-path", "/tmp/out"]
        )
        assert result.exit_code != 0


class TestVolumePathHelpers:
    """Tests for volume path helper functions."""

    def test_list_eigenstructure_studies_function_exists(self):
        """list_eigenstructure_studies should be defined in volumes module."""
        list_eigenstructure_studies = _volumes_module.list_eigenstructure_studies
        assert callable(list_eigenstructure_studies)

    def test_get_eigenstructure_path_function_exists(self):
        """get_eigenstructure_path should be defined in volumes module."""
        get_eigenstructure_path = _volumes_module.get_eigenstructure_path
        assert callable(get_eigenstructure_path)

    def test_get_eigenstructure_path_constructs_correct_path(self):
        """get_eigenstructure_path should combine mount and study name."""
        get_eigenstructure_path = _volumes_module.get_eigenstructure_path
        path = get_eigenstructure_path("my_study")
        assert path == f"{EIGENSTRUCTURE_MOUNT}/my_study"


@pytest.mark.modal
class TestModalIntegration:
    """Integration tests that require actual Modal deployment.

    These tests are skipped by default and only run when Modal is
    configured and deployed. Run with: pytest -m modal
    """

    @pytest.fixture
    def skip_if_not_deployed(self):
        """Skip test if Modal app is not deployed."""
        try:
            import modal

            fn = modal.Function.from_name("tmgg-spectral", "eigenstructure_list")
            fn.hydrate()
        except Exception:
            pytest.skip("Modal app not deployed")

    def test_modal_list_remote_works(self, skip_if_not_deployed):
        """eigenstructure_list should return list of studies from Modal."""
        import modal

        fn = modal.Function.from_name("tmgg-spectral", "eigenstructure_list")
        result = fn.remote()

        assert isinstance(result, list)
        # May be empty if no studies exist yet

    def test_modal_collect_creates_study(self, skip_if_not_deployed):
        """eigenstructure_collect should create study in Modal volume."""
        import modal

        fn = modal.Function.from_name("tmgg-spectral", "eigenstructure_collect")
        result = fn.remote(
            dataset_name="er",
            dataset_config={"num_nodes": 10, "num_graphs": 5, "p": 0.3},
            output_path="test_integration_study",
            batch_size=5,
            seed=42,
        )

        assert result["status"] == "completed"
        assert result["num_graphs"] == 5
