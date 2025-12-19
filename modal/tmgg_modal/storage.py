"""Tigris S3 storage integration for Modal experiments.

Provides a TigrisStorage class that extends the TMGG S3Storage
with Modal-specific configuration using environment variables
from Modal secrets.
"""

import os
from pathlib import Path

try:
    from tmgg.experiment_utils.cloud.storage import S3Storage
except ImportError as e:
    raise ImportError(
        "tmgg package is required for Modal storage. "
        "Ensure the Modal image installs tmgg correctly."
    ) from e


class TigrisStorage(S3Storage):
    """Tigris S3-compatible storage configured from environment variables.

    Reads configuration from Modal secrets via environment variables:
    - TMGG_TIGRIS_BUCKET: Bucket name
    - TMGG_TIGRIS_ENDPOINT: Tigris endpoint URL
    - TMGG_TIGRIS_ACCESS_KEY: Access key ID
    - TMGG_TIGRIS_SECRET_KEY: Secret access key

    These should be configured as Modal secrets and attached to functions.
    """

    # Default Tigris endpoint
    DEFAULT_ENDPOINT = "https://fly.storage.tigris.dev"

    def __init__(self, prefix: str = "tmgg-experiments"):
        """Initialize Tigris storage from environment variables.

        Parameters
        ----------
        prefix
            Key prefix for all objects. Default "tmgg-experiments".

        Raises
        ------
        ValueError
            If required environment variables are not set.
        """
        bucket = os.environ.get("TMGG_TIGRIS_BUCKET")
        if not bucket:
            raise ValueError(
                "TMGG_TIGRIS_BUCKET not set. "
                "Configure Modal secrets with Tigris credentials."
            )

        endpoint = os.environ.get("TMGG_TIGRIS_ENDPOINT", self.DEFAULT_ENDPOINT)
        access_key = os.environ.get("TMGG_TIGRIS_ACCESS_KEY")
        secret_key = os.environ.get("TMGG_TIGRIS_SECRET_KEY")

        if not access_key or not secret_key:
            raise ValueError(
                "TMGG_TIGRIS_ACCESS_KEY and TMGG_TIGRIS_SECRET_KEY must be set. "
                "Configure Modal secrets with Tigris credentials."
            )

        super().__init__(
            bucket=bucket,
            endpoint_url=endpoint,
            access_key=access_key,
            secret_key=secret_key,
            region="auto",  # Tigris handles region automatically
            prefix=prefix,
        )

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


def get_storage_from_env() -> TigrisStorage | None:
    """Create TigrisStorage if environment is configured.

    Returns
    -------
    TigrisStorage or None
        Storage instance if configured, None otherwise.
    """
    try:
        return TigrisStorage()
    except ValueError:
        return None
