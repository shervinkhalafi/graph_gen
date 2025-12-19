"""Tests for TigrisStorage and S3Storage cloud storage backends.

Test rationale:
    The storage layer handles checkpoint persistence and metrics upload/download
    for distributed experiments on Modal. Correct behavior is critical because:
    1. Lost checkpoints waste GPU-hours (166.5 hours for Stage 2)
    2. Corrupted metrics invalidate experiment results
    3. Environment misconfiguration should fail fast with clear errors

Invariants:
    - upload_file returns a valid S3 URI
    - download_file returns the exact bytes that were uploaded
    - metrics round-trip through JSON without loss
    - Missing environment variables raise ValueError (not AttributeError)
"""

import os
from unittest.mock import MagicMock, patch

import pytest


class TestTigrisStorageConfiguration:
    """Test TigrisStorage environment variable handling."""

    def test_missing_bucket_raises_valueerror(self):
        """Missing TMGG_TIGRIS_BUCKET should raise clear error."""
        with patch.dict(os.environ, {}, clear=True):
            from tmgg_modal.storage import TigrisStorage  # pyright: ignore[reportImplicitRelativeImport]

            with pytest.raises(ValueError, match="TMGG_TIGRIS_BUCKET"):
                TigrisStorage()

    def test_missing_access_key_raises_valueerror(self):
        """Missing access credentials should raise clear error."""
        env = {"TMGG_TIGRIS_BUCKET": "test-bucket"}
        with patch.dict(os.environ, env, clear=True):
            from tmgg_modal.storage import TigrisStorage  # pyright: ignore[reportImplicitRelativeImport]

            with pytest.raises(ValueError, match="ACCESS_KEY.*SECRET_KEY"):
                TigrisStorage()

    def test_valid_config_creates_storage(self):
        """Valid environment should create storage instance."""
        env = {
            "TMGG_TIGRIS_BUCKET": "test-bucket",
            "TMGG_TIGRIS_ACCESS_KEY": "test-key",
            "TMGG_TIGRIS_SECRET_KEY": "test-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            from tmgg_modal.storage import TigrisStorage  # pyright: ignore[reportImplicitRelativeImport]

            storage = TigrisStorage()

            assert storage.bucket == "test-bucket"
            assert storage.access_key == "test-key"
            assert storage.secret_key == "test-secret"

    def test_uses_default_endpoint(self):
        """Should use Tigris default endpoint when not specified."""
        env = {
            "TMGG_TIGRIS_BUCKET": "test-bucket",
            "TMGG_TIGRIS_ACCESS_KEY": "test-key",
            "TMGG_TIGRIS_SECRET_KEY": "test-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            from tmgg_modal.storage import TigrisStorage  # pyright: ignore[reportImplicitRelativeImport]

            storage = TigrisStorage()

            assert storage.endpoint_url == TigrisStorage.DEFAULT_ENDPOINT

    def test_custom_endpoint_override(self):
        """Custom endpoint should override default."""
        env = {
            "TMGG_TIGRIS_BUCKET": "test-bucket",
            "TMGG_TIGRIS_ACCESS_KEY": "test-key",
            "TMGG_TIGRIS_SECRET_KEY": "test-secret",
            "TMGG_TIGRIS_ENDPOINT": "https://custom.endpoint.com",
        }
        with patch.dict(os.environ, env, clear=True):
            from tmgg_modal.storage import TigrisStorage  # pyright: ignore[reportImplicitRelativeImport]

            storage = TigrisStorage()

            assert storage.endpoint_url == "https://custom.endpoint.com"


class TestGetStorageFromEnv:
    """Test the get_storage_from_env factory function."""

    def test_returns_none_when_unconfigured(self):
        """Should return None when environment is not configured."""
        with patch.dict(os.environ, {}, clear=True):
            from tmgg_modal.storage import get_storage_from_env  # pyright: ignore[reportImplicitRelativeImport]

            result = get_storage_from_env()

            assert result is None

    def test_returns_storage_when_configured(self):
        """Should return TigrisStorage when environment is configured."""
        env = {
            "TMGG_TIGRIS_BUCKET": "test-bucket",
            "TMGG_TIGRIS_ACCESS_KEY": "test-key",
            "TMGG_TIGRIS_SECRET_KEY": "test-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            from tmgg_modal.storage import TigrisStorage, get_storage_from_env  # pyright: ignore[reportImplicitRelativeImport]

            result = get_storage_from_env()

            assert isinstance(result, TigrisStorage)


class TestS3StorageOperations:
    """Test S3Storage file operations with mocked boto3."""

    @pytest.fixture
    def mock_boto3(self):
        """Mock boto3 client for S3 operations."""
        # boto3 is imported lazily in the client property, so patch it there
        with patch("boto3.client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client_factory.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def storage(self, mock_boto3):
        """Create S3Storage with mocked boto3."""
        from tmgg.experiment_utils.cloud.storage import S3Storage

        storage = S3Storage(
            bucket="test-bucket",
            access_key="test-key",
            secret_key="test-secret",
            prefix="test-prefix",
        )
        # Force client creation to use the mock
        _ = storage.client
        return storage

    def test_upload_file_returns_s3_uri(self, storage, mock_boto3, tmp_path):
        """upload_file should return valid S3 URI."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("test content")

        uri = storage.upload_file(test_file, "subdir/test.txt")

        assert uri == "s3://test-bucket/test-prefix/subdir/test.txt"
        mock_boto3.upload_file.assert_called_once()

    def test_upload_metrics_creates_json(self, storage, mock_boto3):
        """upload_metrics should serialize dict to JSON and upload."""
        metrics = {"loss": 0.5, "accuracy": 0.95}

        uri = storage.upload_metrics(metrics, "run-123")

        assert "metrics/run-123.json" in uri
        mock_boto3.upload_file.assert_called_once()

    def test_list_checkpoints_with_prefix(self, storage, mock_boto3):
        """list_checkpoints should use correct prefix."""
        mock_boto3.list_objects_v2.return_value = {
            "Contents": [
                {"Key": "test-prefix/checkpoints/run-1/model.ckpt"},
                {"Key": "test-prefix/checkpoints/run-2/model.ckpt"},
            ]
        }

        checkpoints = storage.list_checkpoints("run-")

        mock_boto3.list_objects_v2.assert_called_once()
        assert len(checkpoints) == 2

    def test_exists_returns_true_for_existing_key(self, storage, mock_boto3):
        """exists should return True when head_object succeeds."""
        mock_boto3.head_object.return_value = {}

        result = storage.exists("some/key.txt")

        assert result is True

    def test_exists_returns_false_for_missing_key(self, storage, mock_boto3):
        """exists should return False when head_object raises ClientError."""
        from botocore.exceptions import ClientError

        mock_boto3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404"}}, "HeadObject"
        )

        result = storage.exists("missing/key.txt")

        assert result is False


class TestLocalStorageOperations:
    """Test LocalStorage file operations."""

    @pytest.fixture
    def storage(self, tmp_path):
        """Create LocalStorage in temp directory."""
        from tmgg.experiment_utils.cloud.storage import LocalStorage

        return LocalStorage(base_dir=tmp_path)

    def test_upload_and_download_file_roundtrip(self, storage, tmp_path):
        """Files should round-trip through local storage unchanged."""
        # Create source file
        src_file = tmp_path / "source" / "data.bin"
        src_file.parent.mkdir()
        original_content = b"binary content \x00\xff"
        src_file.write_bytes(original_content)

        # Upload
        storage.upload_file(src_file, "uploads/data.bin")

        # Download to different location
        dest_file = tmp_path / "download" / "data.bin"
        storage.download_file("uploads/data.bin", dest_file)

        assert dest_file.read_bytes() == original_content

    def test_upload_metrics_creates_valid_json(self, storage):
        """Metrics should be stored as valid JSON."""
        metrics = {
            "best_val_loss": 0.123,
            "config": {"lr": 1e-4, "batch_size": 32},
            "nested": {"a": [1, 2, 3]},
        }

        storage.upload_metrics(metrics, "test-run")

        # Verify file exists and is valid JSON
        loaded = storage.download_metrics("test-run")
        assert loaded == metrics

    def test_exists_returns_correct_status(self, storage, tmp_path):
        """exists should accurately reflect file presence."""
        # File doesn't exist yet
        assert storage.exists("nonexistent.txt") is False

        # Create file
        test_file = tmp_path / "source.txt"
        test_file.write_text("content")
        storage.upload_file(test_file, "exists.txt")

        # Now it should exist
        assert storage.exists("exists.txt") is True


class TestTigrisStorageCheckpoints:
    """Test TigrisStorage checkpoint-specific methods."""

    @pytest.fixture
    def mock_boto3(self):
        """Mock boto3 for Tigris operations."""
        with patch("boto3.client") as mock_client_factory:
            mock_client = MagicMock()
            mock_client_factory.return_value = mock_client
            yield mock_client

    @pytest.fixture
    def storage(self, mock_boto3):
        """Create TigrisStorage with mocked boto3."""
        env = {
            "TMGG_TIGRIS_BUCKET": "test-bucket",
            "TMGG_TIGRIS_ACCESS_KEY": "test-key",
            "TMGG_TIGRIS_SECRET_KEY": "test-secret",
        }
        with patch.dict(os.environ, env, clear=True):
            from tmgg_modal.storage import TigrisStorage  # pyright: ignore[reportImplicitRelativeImport]

            storage = TigrisStorage(prefix="experiments")
            _ = storage.client  # Force client creation
            return storage

    def test_upload_checkpoint_uses_correct_path(self, storage, mock_boto3, tmp_path):
        """upload_checkpoint should organize by run_id."""
        ckpt_file = tmp_path / "model.ckpt"
        ckpt_file.write_text("checkpoint data")

        uri = storage.upload_checkpoint(ckpt_file, "run-abc123")

        # Should be uploaded to checkpoints/{run_id}/{filename}
        assert "checkpoints/run-abc123/model.ckpt" in uri

    def test_sync_results_uploads_all_files(self, storage, mock_boto3, tmp_path):
        """sync_results should upload all files in directory."""
        # Create result files
        results_dir = tmp_path / "results"
        results_dir.mkdir()
        (results_dir / "metrics.json").write_text('{"loss": 0.5}')
        (results_dir / "plots").mkdir()
        (results_dir / "plots" / "loss.png").write_bytes(b"PNG data")

        uris = storage.sync_results(results_dir, "run-xyz")

        # Should upload both files
        assert len(uris) == 2
        mock_boto3.upload_file.call_count == 2
