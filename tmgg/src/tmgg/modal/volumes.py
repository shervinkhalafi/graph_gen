"""Modal volume management for dataset caching.

No ``import modal`` at module level. Volume objects are created lazily
via ``_create_volume()`` — called at decoration time inside
``_functions.py`` or at runtime inside a Modal container.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import modal

# Mount paths within containers (pure constants, no modal dependency)
DATASETS_MOUNT = "/data/datasets"
OUTPUTS_MOUNT = "/data/outputs"
EIGENSTRUCTURE_MOUNT = "/data/eigenstructure"

# Volume names in Modal
DATASETS_VOLUME_NAME = "tmgg-datasets"
OUTPUTS_VOLUME_NAME = "tmgg-outputs"
EIGENSTRUCTURE_VOLUME_NAME = "tmgg-eigenstructure"


def _create_volume(name: str) -> modal.Volume:
    """Create or retrieve a Modal volume by name.

    Parameters
    ----------
    name
        Modal volume name.

    Returns
    -------
    modal.Volume
        Volume handle (creates if missing).
    """
    import modal as _modal

    return _modal.Volume.from_name(name, create_if_missing=True)


def get_volume_mounts() -> dict[str, Any]:
    """Get volume mount configuration for Modal functions.

    Only call at decoration time (inside ``_functions.py``) or at runtime
    inside a Modal container.

    Returns
    -------
    dict
        Mapping of mount paths to volumes.
    """
    return {
        DATASETS_MOUNT: _create_volume(DATASETS_VOLUME_NAME),
        OUTPUTS_MOUNT: _create_volume(OUTPUTS_VOLUME_NAME),
        EIGENSTRUCTURE_MOUNT: _create_volume(EIGENSTRUCTURE_VOLUME_NAME),
    }


def get_eigenstructure_volume_mounts() -> dict[str, Any]:
    """Get volume mount configuration for eigenstructure study functions.

    Returns
    -------
    dict
        Mapping of mount paths to eigenstructure volume.
    """
    return {
        EIGENSTRUCTURE_MOUNT: _create_volume(EIGENSTRUCTURE_VOLUME_NAME),
    }


def get_eigenstructure_volume() -> Any:
    """Get the eigenstructure volume for ``commit()`` calls.

    Returns
    -------
    modal.Volume
        The eigenstructure volume handle.
    """
    return _create_volume(EIGENSTRUCTURE_VOLUME_NAME)


def get_datasets_volume() -> Any:
    """Get the datasets volume.

    Returns
    -------
    modal.Volume
        The datasets volume handle.
    """
    return _create_volume(DATASETS_VOLUME_NAME)


def get_outputs_volume() -> Any:
    """Get the outputs volume.

    Returns
    -------
    modal.Volume
        The outputs volume handle.
    """
    return _create_volume(OUTPUTS_VOLUME_NAME)


def ensure_dataset_cached(
    dataset_name: str,
    volume: Any | None = None,
) -> str:
    """Ensure a dataset is cached in the volume.

    Parameters
    ----------
    dataset_name
        Name of the dataset (e.g., "qm9", "enzymes").
    volume
        Volume to cache to. Defaults to datasets volume.

    Returns
    -------
    str
        Path to the cached dataset within the volume.
    """
    from pathlib import Path

    vol = volume or get_datasets_volume()
    cache_path = Path(DATASETS_MOUNT) / dataset_name

    if cache_path.exists():
        return str(cache_path)

    if dataset_name in ("qm9", "enzymes", "proteins"):
        _cache_pyg_dataset(dataset_name, cache_path)

    vol.commit()
    return str(cache_path)


def _cache_pyg_dataset(name: str, cache_path: Any) -> None:
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
        pass


def clear_outputs_volume() -> None:
    """Clear the outputs volume for a fresh run."""
    get_outputs_volume().reload()


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


def list_eigenstructure_studies() -> list[dict[str, str]]:
    """List all eigenstructure studies in the volume.

    Returns
    -------
    list[dict]
        List of dicts with 'name' and 'path' for each study directory.
    """
    from pathlib import Path

    eigen_path = Path(EIGENSTRUCTURE_MOUNT)
    if not eigen_path.exists():
        return []

    studies = []
    for d in eigen_path.iterdir():
        if d.is_dir():
            studies.append({"name": d.name, "path": str(d)})
    return studies


def get_eigenstructure_path(study_name: str) -> str:
    """Get the full path to an eigenstructure study in the volume.

    Parameters
    ----------
    study_name
        Name of the study (relative path within eigenstructure volume).

    Returns
    -------
    str
        Full path to the study directory.
    """
    from pathlib import Path

    return str(Path(EIGENSTRUCTURE_MOUNT) / study_name)
