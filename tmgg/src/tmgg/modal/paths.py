"""Path discovery utilities for Modal experiments.

Since modal is now part of the tmgg package, path discovery is straightforward.
The module lives at src/tmgg/modal/, so paths are resolved relative to that.
"""

from __future__ import annotations

import os
from pathlib import Path


def discover_tmgg_path() -> Path:
    """Get the tmgg package root directory.

    Since modal is now inside tmgg, this is straightforward.
    The package structure is:
        tmgg/           <- returned path
          src/
            tmgg/
              modal/
                paths.py  <- this file

    Returns
    -------
    Path
        Path to tmgg package root (the directory containing src/).

    Notes
    -----
    TMGG_PATH environment variable can override for special cases
    (e.g., development with editable installs in non-standard locations).
    """
    env_path = os.environ.get("TMGG_PATH")
    if env_path:
        path = Path(env_path)
        if _is_valid_tmgg_path(path):
            return path
        raise ValueError(
            f"TMGG_PATH={env_path} does not contain valid tmgg package "
            "(missing src/tmgg directory)"
        )

    # modal/ is at src/tmgg/modal/, so go up 4 levels to get to tmgg root
    # paths.py -> modal/ -> tmgg/ -> src/ -> tmgg_root/
    return Path(__file__).parent.parent.parent.parent


def _is_valid_tmgg_path(path: Path) -> bool:
    """Check if path contains a valid tmgg package."""
    if not path.exists():
        return False

    src_tmgg = path / "src" / "tmgg"
    if not src_tmgg.exists():
        return False

    return bool(list(src_tmgg.glob("*.py")))


def get_exp_configs_path() -> Path:
    """Get path to experiment configs directory.

    Returns
    -------
    Path
        Path to exp_configs directory.
    """
    # exp_configs is a sibling to modal/ inside src/tmgg/
    return Path(__file__).parent.parent / "exp_configs"


def require_tmgg_path() -> Path:
    """Get tmgg path (always succeeds since we're inside the package).

    Returns
    -------
    Path
        Path to tmgg package root.
    """
    return discover_tmgg_path()


def require_exp_configs_path() -> Path:
    """Get exp_configs path or raise an error.

    Returns
    -------
    Path
        Path to exp_configs directory.

    Raises
    ------
    RuntimeError
        If exp_configs directory doesn't exist.
    """
    configs_path = get_exp_configs_path()
    if not configs_path.exists():
        raise RuntimeError(
            f"exp_configs not found at {configs_path}. "
            "Ensure tmgg package is properly installed."
        )
    return configs_path
