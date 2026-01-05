"""Tigris S3 storage integration for Modal experiments.

Self-contained implementation that uses boto3 directly, avoiding
imports from tmgg.experiment_utils.cloud to prevent circular dependencies.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import Any


class TigrisStorage:
    """Tigris S3-compatible storage configured from environment variables.

    Reads configuration from Modal secrets via environment variables:
    - TMGG_TIGRIS_BUCKET: Bucket name
    - TMGG_TIGRIS_ENDPOINT: Tigris endpoint URL
    - TMGG_TIGRIS_ACCESS_KEY: Access key ID
    - TMGG_TIGRIS_SECRET_KEY: Secret access key

    These should be configured as Modal secrets and attached to functions.
    """

    DEFAULT_ENDPOINT = "https://fly.storage.tigris.dev"

    def __init__(self, prefix: str = "tmgg-experiments", path_prefix: str = ""):
        """Initialize Tigris storage from environment variables.

        Parameters
        ----------
        prefix
            Bucket-level key prefix for all objects. Default "tmgg-experiments".
        path_prefix
            Secondary path prefix for run isolation (e.g., "2025-01-05").
            Inserted between bucket prefix and object paths.

        Raises
        ------
        ValueError
            If required environment variables are not set.
        """
        self.bucket = os.environ.get("TMGG_TIGRIS_BUCKET")
        if not self.bucket:
            raise ValueError(
                "TMGG_TIGRIS_BUCKET not set. "
                "Configure Modal secrets with Tigris credentials."
            )

        self.endpoint_url = os.environ.get(
            "TMGG_TIGRIS_ENDPOINT", self.DEFAULT_ENDPOINT
        )
        self.access_key = os.environ.get("TMGG_TIGRIS_ACCESS_KEY")
        self.secret_key = os.environ.get("TMGG_TIGRIS_SECRET_KEY")

        if not self.access_key or not self.secret_key:
            raise ValueError(
                "TMGG_TIGRIS_ACCESS_KEY and TMGG_TIGRIS_SECRET_KEY must be set. "
                "Configure Modal secrets with Tigris credentials."
            )

        self.region = "auto"  # Tigris handles region automatically
        self.prefix = prefix
        self.path_prefix = path_prefix
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
        """Construct full S3 key from relative key.

        Combines bucket prefix, path prefix, and key into a single path.
        Empty components are filtered out.
        """
        parts = [self.prefix, self.path_prefix, key]
        return "/".join(p for p in parts if p)

    def upload_file(self, local_path: Path, remote_key: str) -> str:
        """Upload a local file to Tigris.

        Parameters
        ----------
        local_path
            Path to the local file.
        remote_key
            Key (path) in the storage bucket.

        Returns
        -------
        str
            The full S3 URI of the uploaded file.
        """
        full_key = self._full_key(remote_key)
        self.client.upload_file(str(local_path), self.bucket, full_key)
        return f"s3://{self.bucket}/{full_key}"

    def download_file(self, remote_key: str, local_path: Path) -> Path:
        """Download a file from Tigris.

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
        full_key = self._full_key(remote_key)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, full_key, str(local_path))
        return local_path

    def upload_metrics(self, metrics: dict[str, Any], run_id: str) -> str:
        """Upload metrics as JSON.

        Parameters
        ----------
        metrics
            Dictionary of metrics to store.
        run_id
            Unique identifier for the run.

        Returns
        -------
        str
            The S3 URI of the metrics file.
        """
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(metrics, f, indent=2)
            temp_path = Path(f.name)

        try:
            remote_key = f"metrics/{run_id}.json"
            return self.upload_file(temp_path, remote_key)
        finally:
            temp_path.unlink()

    def download_metrics(self, run_id: str) -> dict[str, Any]:
        """Download metrics JSON.

        Parameters
        ----------
        run_id
            Unique identifier for the run.

        Returns
        -------
        dict
            The stored metrics dictionary.
        """
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
        """List checkpoint files in the bucket.

        Parameters
        ----------
        prefix
            Optional prefix to filter results.

        Returns
        -------
        list[str]
            List of checkpoint keys matching the prefix.
        """
        full_prefix = self._full_key(f"checkpoints/{prefix}")
        response = self.client.list_objects_v2(Bucket=self.bucket, Prefix=full_prefix)
        return [obj["Key"] for obj in response.get("Contents", [])]

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
        from botocore.exceptions import ClientError

        full_key = self._full_key(remote_key)
        try:
            self.client.head_object(Bucket=self.bucket, Key=full_key)
            return True
        except ClientError:
            return False

    def upload_checkpoint(self, local_path: Path, run_id: str) -> str:
        """Upload a model checkpoint.

        Parameters
        ----------
        local_path
            Path to the checkpoint file.
        run_id
            Unique run identifier.

        Returns
        -------
        str
            S3 URI of the uploaded checkpoint.
        """
        remote_key = f"checkpoints/{run_id}/{local_path.name}"
        return self.upload_file(local_path, remote_key)

    def download_checkpoint(self, run_id: str, local_dir: Path) -> Path:
        """Download the latest checkpoint for a run.

        Parameters
        ----------
        run_id
            Unique run identifier.
        local_dir
            Directory to save the checkpoint.

        Returns
        -------
        Path
            Local path to the downloaded checkpoint.
        """
        checkpoints = self.list_checkpoints(f"{run_id}/")
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoints found for run {run_id}")

        # Get the latest checkpoint (by name, assuming sequential naming)
        latest = sorted(checkpoints)[-1]
        local_path = local_dir / Path(latest).name
        return self.download_file(latest.replace(f"{self.prefix}/", ""), local_path)

    def sync_results(self, local_dir: Path, run_id: str) -> list[str]:
        """Sync all result files from a local directory to storage.

        Parameters
        ----------
        local_dir
            Local directory containing result files.
        run_id
            Unique run identifier.

        Returns
        -------
        list[str]
            List of uploaded S3 URIs.
        """
        uploaded = []
        for file_path in local_dir.rglob("*"):
            if file_path.is_file():
                rel_path = file_path.relative_to(local_dir)
                remote_key = f"results/{run_id}/{rel_path}"
                uri = self.upload_file(file_path, remote_key)
                uploaded.append(uri)
        return uploaded


def get_storage_from_env(path_prefix: str = "") -> TigrisStorage | None:
    """Create TigrisStorage if environment is configured.

    Parameters
    ----------
    path_prefix
        Secondary path prefix for run isolation (e.g., "2025-01-05").

    Returns
    -------
    TigrisStorage or None
        Storage instance if configured, None otherwise.
    """
    try:
        return TigrisStorage(path_prefix=path_prefix)
    except ValueError:
        return None
