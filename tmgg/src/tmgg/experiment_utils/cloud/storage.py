"""Cloud storage abstractions for checkpoints and metrics.

Provides S3-compatible storage that works with Tigris, AWS S3, and MinIO.
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class CloudStorage(ABC):
    """Abstract base for cloud storage backends.

    Handles checkpoint upload/download and metrics persistence for
    distributed experiment runs.
    """

    @abstractmethod
    def upload_file(self, local_path: Path, remote_key: str) -> str:
        """Upload a local file to cloud storage.

        Parameters
        ----------
        local_path
            Path to the local file.
        remote_key
            Key (path) in the storage bucket.

        Returns
        -------
        str
            The full remote URI of the uploaded file.
        """
        ...

    @abstractmethod
    def download_file(self, remote_key: str, local_path: Path) -> Path:
        """Download a file from cloud storage.

        Parameters
        ----------
        remote_key
            Key (path) in the storage bucket.
        local_path
            Local path to save the file.

        Returns
        -------
        Path
            The local path where the file was saved.
        """
        ...

    @abstractmethod
    def upload_metrics(self, metrics: dict[str, Any], run_id: str) -> str:
        """Upload metrics JSON for an experiment run.

        Parameters
        ----------
        metrics
            Dictionary of metrics to store.
        run_id
            Unique identifier for the run.

        Returns
        -------
        str
            The remote URI of the metrics file.
        """
        ...

    @abstractmethod
    def download_metrics(self, run_id: str) -> dict[str, Any]:
        """Download metrics for an experiment run.

        Parameters
        ----------
        run_id
            Unique identifier for the run.

        Returns
        -------
        dict
            The stored metrics dictionary.
        """
        ...

    @abstractmethod
    def list_checkpoints(self, prefix: str = "") -> list[str]:
        """List checkpoint files in storage.

        Parameters
        ----------
        prefix
            Optional prefix to filter results.

        Returns
        -------
        list[str]
            List of checkpoint keys matching the prefix.
        """
        ...

    @abstractmethod
    def exists(self, remote_key: str) -> bool:
        """Check if a key exists in storage.

        Parameters
        ----------
        remote_key
            Key to check.

        Returns
        -------
        bool
            True if the key exists.
        """
        ...


class S3Storage(CloudStorage):
    """S3-compatible storage backend.

    Works with AWS S3, Tigris, MinIO, and other S3-compatible services.
    Configure via environment variables or constructor parameters.

    Environment Variables
    ---------------------
    TMGG_S3_BUCKET : str
        Bucket name for storage.
    TMGG_S3_ENDPOINT : str, optional
        Custom endpoint URL (required for Tigris/MinIO).
    TMGG_S3_ACCESS_KEY : str
        Access key ID.
    TMGG_S3_SECRET_KEY : str
        Secret access key.
    TMGG_S3_REGION : str, optional
        AWS region (default: us-east-1).
    """

    def __init__(
        self,
        bucket: str | None = None,
        endpoint_url: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str | None = None,
        prefix: str = "tmgg",
    ):
        """Initialize S3 storage client.

        Parameters
        ----------
        bucket
            S3 bucket name. Falls back to TMGG_S3_BUCKET env var.
        endpoint_url
            Custom S3 endpoint for Tigris/MinIO. Falls back to TMGG_S3_ENDPOINT.
        access_key
            Access key ID. Falls back to TMGG_S3_ACCESS_KEY.
        secret_key
            Secret access key. Falls back to TMGG_S3_SECRET_KEY.
        region
            AWS region. Falls back to TMGG_S3_REGION or 'us-east-1'.
        prefix
            Key prefix for all objects (default: 'tmgg').
        """
        self.bucket = bucket or os.environ.get("TMGG_S3_BUCKET")
        if not self.bucket:
            raise ValueError("S3 bucket not specified. Set TMGG_S3_BUCKET env var.")

        self.endpoint_url = endpoint_url or os.environ.get("TMGG_S3_ENDPOINT")
        self.access_key = access_key or os.environ.get("TMGG_S3_ACCESS_KEY")
        self.secret_key = secret_key or os.environ.get("TMGG_S3_SECRET_KEY")
        self.region = region or os.environ.get("TMGG_S3_REGION", "us-east-1")
        self.prefix = prefix

        self._client = None

    @property
    def client(self):
        """Lazy-initialize boto3 client."""
        if self._client is None:
            import boto3

            self._client = boto3.client(
                "s3",
                endpoint_url=self.endpoint_url,
                aws_access_key_id=self.access_key,
                aws_secret_access_key=self.secret_key,
                region_name=self.region,
            )
        return self._client

    def _full_key(self, key: str) -> str:
        """Add prefix to key."""
        if self.prefix:
            return f"{self.prefix}/{key}"
        return key

    def upload_file(self, local_path: Path, remote_key: str) -> str:
        """Upload a local file to S3."""
        full_key = self._full_key(remote_key)
        self.client.upload_file(str(local_path), self.bucket, full_key)
        return f"s3://{self.bucket}/{full_key}"

    def download_file(self, remote_key: str, local_path: Path) -> Path:
        """Download a file from S3."""
        full_key = self._full_key(remote_key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, full_key, str(local_path))
        return local_path

    def upload_metrics(self, metrics: dict[str, Any], run_id: str) -> str:
        """Upload metrics as JSON."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metrics, f, indent=2)
            temp_path = Path(f.name)

        try:
            remote_key = f"metrics/{run_id}.json"
            return self.upload_file(temp_path, remote_key)
        finally:
            temp_path.unlink()

    def download_metrics(self, run_id: str) -> dict[str, Any]:
        """Download metrics JSON."""
        import tempfile

        remote_key = f"metrics/{run_id}.json"
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            local_path = Path(f.name)

        try:
            self.download_file(remote_key, local_path)
            with open(local_path) as f:
                return json.load(f)
        finally:
            local_path.unlink()

    def list_checkpoints(self, prefix: str = "") -> list[str]:
        """List checkpoint files in the bucket."""
        full_prefix = self._full_key(f"checkpoints/{prefix}")
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

    def exists(self, remote_key: str) -> bool:
        """Check if a key exists."""
        from botocore.exceptions import ClientError

        full_key = self._full_key(remote_key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError:
            return False


class LocalStorage(CloudStorage):
    """Local filesystem storage for development.

    Mimics the CloudStorage interface but stores files locally.
    Useful for testing without cloud credentials.
    """

    def __init__(self, base_dir: Path | None = None):
        """Initialize local storage.

        Parameters
        ----------
        base_dir
            Base directory for storage. Defaults to ./storage.
        """
        self.base_dir = base_dir or Path("./storage")
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def upload_file(self, local_path: Path, remote_key: str) -> str:
        """Copy file to local storage directory."""
        import shutil

        dest = self.base_dir / remote_key
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return f"file://{dest.absolute()}"

    def download_file(self, remote_key: str, local_path: Path) -> Path:
        """Copy file from local storage directory."""
        import shutil

        src = self.base_dir / remote_key
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        return local_path

    def upload_metrics(self, metrics: dict[str, Any], run_id: str) -> str:
        """Save metrics as JSON to local storage."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metrics, f, indent=2)
            temp_path = Path(f.name)

        try:
            return self.upload_file(temp_path, f"metrics/{run_id}.json")
        finally:
            temp_path.unlink()

    def download_metrics(self, run_id: str) -> dict[str, Any]:
        """Load metrics JSON from local storage."""
        path = self.base_dir / f"metrics/{run_id}.json"
        with open(path) as f:
            return json.load(f)

    def list_checkpoints(self, prefix: str = "") -> list[str]:
        """List checkpoint files in local storage."""
        checkpoints_dir = self.base_dir / "checkpoints"
        if not checkpoints_dir.exists():
            return []
        return [
            str(p.relative_to(self.base_dir))
            for p in checkpoints_dir.glob(f"{prefix}**/*.ckpt")
        ]

    def exists(self, remote_key: str) -> bool:
        """Check if a file exists in local storage."""
        return (self.base_dir / remote_key).exists()
