"""Modal volume management for dataset caching.

Provides persistent volumes for caching datasets between runs,
avoiding repeated downloads and preprocessing.
"""

import modal

# Persistent volume for dataset caching
# This volume persists across function invocations
datasets_volume = modal.Volume.from_name(
    "tmgg-datasets",
    create_if_missing=True,
)

# Volume for temporary outputs (cleared between runs)
outputs_volume = modal.Volume.from_name(
    "tmgg-outputs",
    create_if_missing=True,
)

# Mount paths within containers
DATASETS_MOUNT = "/data/datasets"
OUTPUTS_MOUNT = "/data/outputs"


def get_volume_mounts() -> dict[str, modal.Volume]:
    """Get volume mount configuration for Modal functions.

    Returns
    -------
    dict
        Mapping of mount paths to volumes.
    """
    return {
        DATASETS_MOUNT: datasets_volume,
        OUTPUTS_MOUNT: outputs_volume,
    }


def ensure_dataset_cached(
    dataset_name: str,
    volume: modal.Volume | None = None,
) -> str:
    """Ensure a dataset is cached in the volume.

    Parameters
    ----------
    dataset_name
        Name of the dataset (e.g., "qm9", "enzymes").
    volume
        Volume to cache to. Defaults to datasets_volume.

    Returns
    -------
    str
        Path to the cached dataset within the volume.
    """
    from pathlib import Path

    vol = volume or datasets_volume
    cache_path = Path(DATASETS_MOUNT) / dataset_name

    # Check if already cached
    if cache_path.exists():
        return str(cache_path)

    # Download and cache the dataset
    # This depends on the dataset type
    if dataset_name in ("qm9", "enzymes", "proteins"):
        _cache_pyg_dataset(dataset_name, cache_path)
    else:
        # Synthetic datasets are generated, not cached
        pass

    # Commit changes to volume
    vol.commit()

    return str(cache_path)


def _cache_pyg_dataset(name: str, cache_path) -> None:
    """Download and cache a PyTorch Geometric dataset.

    Parameters
    ----------
    name
        Dataset name.
    cache_path
        Path to cache the dataset.
    """
    from pathlib import Path

    cache_path = Path(cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)

    try:
        from torch_geometric.datasets import QM9, TUDataset

        if name == "qm9":
            QM9(root=str(cache_path))
        elif name in ("enzymes", "proteins"):
            TUDataset(root=str(cache_path), name=name.upper())
    except ImportError:
        # PyG not available, skip caching
        pass


def clear_outputs_volume() -> None:
    """Clear the outputs volume for a fresh run."""
    outputs_volume.reload()


def list_cached_datasets() -> list[str]:
    """List all cached datasets in the volume.

    Returns
    -------
    list[str]
        Names of cached datasets.
    """
    from pathlib import Path

    datasets_path = Path(DATASETS_MOUNT)
    if not datasets_path.exists():
        return []

    return [d.name for d in datasets_path.iterdir() if d.is_dir()]
