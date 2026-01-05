"""Hydra config composition for Modal stage scripts.

Provides proper config loading that resolves all interpolations and defaults,
unlike direct OmegaConf.load() which skips Hydra's composition machinery.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig, OmegaConf

from tmgg.modal.paths import get_exp_configs_path


@contextmanager
def hydra_config_context(config_dir: Path) -> Iterator[None]:
    """Context manager for Hydra initialization.

    Ensures GlobalHydra is cleared before and after use, preventing
    "already initialized" errors when called multiple times.

    Parameters
    ----------
    config_dir
        Absolute path to the config directory.

    Yields
    ------
    None
        Context for Hydra operations.
    """
    GlobalHydra.instance().clear()
    try:
        initialize_config_dir(config_dir=str(config_dir), version_base=None)
        yield
    finally:
        GlobalHydra.instance().clear()


def compose_config(
    config_name: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Compose a Hydra config with all defaults resolved.

    Uses Hydra's composition API to properly process defaults lists and
    resolve all interpolations, which OmegaConf.load() cannot do.

    Parameters
    ----------
    config_name
        Name of the config file (without .yaml extension).
    overrides
        Optional list of Hydra overrides (e.g., ["model=models/spectral/linear_pe"]).

    Returns
    -------
    DictConfig
        Fully composed and resolved configuration.

    Raises
    ------
    RuntimeError
        If tmgg exp_configs path cannot be found.
    FileNotFoundError
        If the specified config file doesn't exist.
    """
    exp_configs = get_exp_configs_path()
    if exp_configs is None:
        raise RuntimeError(
            "Could not find tmgg exp_configs. Set TMGG_PATH or ensure "
            "modal/ and tmgg/ are siblings."
        )

    config_path = exp_configs / f"{config_name}.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config not found at {config_path}. "
            "Ensure tmgg exp_configs are properly set up."
        )

    with hydra_config_context(exp_configs):
        cfg = compose(config_name=config_name, overrides=overrides or [])
        return cfg


def compose_config_as_dict(
    config_name: str,
    overrides: list[str] | None = None,
) -> dict[str, Any]:
    """Compose config and convert to plain dict with interpolations resolved.

    Convenience wrapper that returns a dict suitable for serialization
    and passing to Modal functions.

    Parameters
    ----------
    config_name
        Name of the config file (without .yaml extension).
    overrides
        Optional list of Hydra overrides.

    Returns
    -------
    dict
        Plain dictionary with all values resolved.
    """
    cfg = compose_config(config_name, overrides)
    return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
