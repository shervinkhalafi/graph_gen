"""Path discovery utilities for Modal experiments.

Provides robust path resolution that doesn't rely on fragile
relative path calculations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def discover_tmgg_path() -> Optional[Path]:
    """Discover the tmgg package path.

    Resolution order:
    1. TMGG_PATH environment variable (explicit override)
    2. Sibling directory to modal/ (standard repo layout)
    3. None if not found

    Returns
    -------
    Path or None
        Path to tmgg package root, or None if not found.
    """
    # 1. Check environment variable
    env_path = os.environ.get("TMGG_PATH")
    if env_path:
        path = Path(env_path)
        if _is_valid_tmgg_path(path):
            return path
        raise ValueError(
            f"TMGG_PATH={env_path} does not contain valid tmgg package "
            "(missing src/tmgg directory)"
        )

    # 2. Try sibling directory (modal/ and tmgg/ are siblings)
    modal_root = Path(__file__).parent.parent
    sibling_path = modal_root.parent / "tmgg"
    if _is_valid_tmgg_path(sibling_path):
        return sibling_path

    # 3. Not found
    return None


def _is_valid_tmgg_path(path: Path) -> bool:
    """Check if path contains a valid tmgg package.

    A valid tmgg path has the structure:
      tmgg/
        src/
          tmgg/
            __init__.py (or other module files)
    """
    if not path.exists():
        return False

    src_tmgg = path / "src" / "tmgg"
    if not src_tmgg.exists():
        return False

    # Check for at least one Python file
    return bool(list(src_tmgg.glob("*.py")))


def get_exp_configs_path() -> Optional[Path]:
    """Get path to experiment configs directory.

    Returns
    -------
    Path or None
        Path to exp_configs, or None if tmgg not found.
    """
    tmgg_path = discover_tmgg_path()
    if tmgg_path is None:
        return None
    return tmgg_path / "src" / "tmgg" / "exp_configs"


def require_tmgg_path() -> Path:
    """Get tmgg path or raise an error.

    Use this when tmgg is required for operation.

    Returns
    -------
    Path
        Path to tmgg package root.

    Raises
    ------
    RuntimeError
        If tmgg path cannot be found.
    """
    path = discover_tmgg_path()
    if path is None:
        raise RuntimeError(
            "Could not find tmgg package. Either:\n"
            "  1. Set TMGG_PATH environment variable to tmgg package root\n"
            "  2. Ensure modal/ and tmgg/ are siblings in the same directory"
        )
    return path


def require_exp_configs_path() -> Path:
    """Get exp_configs path or raise an error.

    Returns
    -------
    Path
        Path to exp_configs directory.

    Raises
    ------
    RuntimeError
        If path cannot be found.
    """
    tmgg_path = require_tmgg_path()
    configs_path = tmgg_path / "src" / "tmgg" / "exp_configs"
    if not configs_path.exists():
        raise RuntimeError(
            f"exp_configs not found at {configs_path}. "
            "Ensure tmgg package is properly installed."
        )
    return configs_path
