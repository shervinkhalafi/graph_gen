"""Tests for Modal eigenstructure volume configuration and CLI.

Test rationale:
    The eigenstructure Modal functions enable running spectral analysis studies
    on Modal with persistent volume storage. These tests verify:
    1. Volume path constants are correct
    2. CLI commands for list-remote and download are properly defined
    3. Volume path helper functions work correctly

    Integration tests that actually run on Modal are marked with
    @pytest.mark.modal and skipped by default.

Invariants:
    - CLI commands correctly accept expected parameters
    - Volume paths are constructed correctly
"""

from unittest.mock import MagicMock, patch

import pytest

from tmgg.modal.volumes import (
    EIGENSTRUCTURE_MOUNT,
)


class TestEigenstructureVolume:
    """Tests for eigenstructure volume configuration."""

    def test_eigenstructure_mount_path_defined(self):
        """EIGENSTRUCTURE_MOUNT should be defined."""
        assert EIGENSTRUCTURE_MOUNT is not None
        assert EIGENSTRUCTURE_MOUNT == "/data/eigenstructure"

    def test_get_eigenstructure_volume_mounts_returns_dict(self):
        """get_eigenstructure_volume_mounts should return volume mapping."""
        with patch("modal.Volume") as mock_Volume:
            mock_Volume.from_name.return_value = MagicMock()
            from tmgg.modal.volumes import get_eigenstructure_volume_mounts

            mounts = get_eigenstructure_volume_mounts()
            assert isinstance(mounts, dict)
            assert EIGENSTRUCTURE_MOUNT in mounts


class TestEigenstructureCLIRemoteCommands:
    """Tests for eigenstructure CLI list-remote and download commands.

    These tests verify CLI behavior by mocking Modal function calls.
    """

    def test_list_remote_cli_exists(self):
        """list-remote command should be defined in CLI."""
        from click.testing import CliRunner

        from tmgg.experiment_utils.eigenstructure_study.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["list-remote", "--help"])

        assert result.exit_code == 0
        assert "List eigenstructure studies" in result.output

    def test_download_cli_exists(self):
        """download command should be defined in CLI."""
        from click.testing import CliRunner

        from tmgg.experiment_utils.eigenstructure_study.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["download", "--help"])

        assert result.exit_code == 0
        assert "Download eigenstructure study" in result.output
        assert "--remote-path" in result.output
        assert "--local-path" in result.output


class TestDownloadCLILogic:
    """Tests for download command logic with mocked Modal functions."""

    def test_download_requires_remote_and_local_path(self):
        """Download command should require both remote and local path."""
        from click.testing import CliRunner

        from tmgg.experiment_utils.eigenstructure_study.cli import main

        runner = CliRunner()

        # Missing local-path
        result = runner.invoke(main, ["download", "--remote-path", "test"])
        assert result.exit_code != 0

        # Missing remote-path
        result = runner.invoke(main, ["download", "--local-path", "/tmp/out"])
        assert result.exit_code != 0


class TestVolumePathHelpers:
    """Tests for volume path helper functions."""

    def test_list_eigenstructure_studies_function_exists(self):
        """list_eigenstructure_studies should be defined in volumes module."""
        from tmgg.modal.volumes import list_eigenstructure_studies

        assert callable(list_eigenstructure_studies)

    def test_get_eigenstructure_path_function_exists(self):
        """get_eigenstructure_path should be defined in volumes module."""
        from tmgg.modal.volumes import get_eigenstructure_path

        assert callable(get_eigenstructure_path)

    def test_get_eigenstructure_path_constructs_correct_path(self):
        """get_eigenstructure_path should combine mount and study name."""
        from tmgg.modal.volumes import get_eigenstructure_path

        path = get_eigenstructure_path("my_study")
        assert path == f"{EIGENSTRUCTURE_MOUNT}/my_study"


@pytest.mark.modal
class TestModalIntegration:
    """Integration tests that require actual Modal deployment.

    Skipped by default. Run with ``pytest -m modal``. Requires the
    ``igor-26028`` Modal profile with ``tmgg-spectral`` deployed.
    """

    @pytest.fixture(autouse=True)
    def _require_modal(self, require_modal_profile):
        pass

    def test_modal_list_remote_works(self):
        """eigenstructure_list should return list of studies from Modal."""
        import modal

        fn = modal.Function.from_name("tmgg-spectral", "eigenstructure_list")
        result = fn.remote()

        assert isinstance(result, list)

    def test_modal_collect_creates_study(self):
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
