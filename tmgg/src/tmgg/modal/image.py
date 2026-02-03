"""Modal image builder for TMGG experiments.

Provides functions to create Modal images with the TMGG package,
PyTorch, and all required dependencies pre-installed.
"""

from pathlib import Path

import modal

# Patterns to exclude when mounting local directories
# Uses dockerignore-like syntax
# Since we now only mount src/, these patterns are relative to src/
MODAL_IGNORE_PATTERNS = [
    # Python bytecode
    "**/__pycache__/",
    "**/*.pyc",
    # IDE
    "**/.idea/",
    "**/.vscode/",
    "**/*.swp",
    # Test artifacts (if any end up in src/)
    "**/.pytest_cache/",
    "**/.hypothesis/",
    # Build artifacts
    "**/*.egg-info/",
    # Misc
    "**/.DS_Store",
    "**/*.log",
]


def create_base_image() -> modal.Image:
    """Create base Modal image with Python and system dependencies.

    Returns
    -------
    modal.Image
        Base image with Python 3.12 and system packages.
    """
    return modal.Image.debian_slim(python_version="3.12").apt_install(
        "git",
        "build-essential",
        "libffi-dev",
    )


# PyTorch CUDA index URL for consistent CUDA-enabled installations
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"


def create_pytorch_image() -> modal.Image:
    """Create Modal image with PyTorch and CUDA 12.1.

    Returns
    -------
    modal.Image
        Image with PyTorch and CUDA support.
    """
    return create_base_image().uv_pip_install(
        "torch>=2.0.0",
        "torchvision",
        index_url=TORCH_CUDA_INDEX,
    )


def create_tmgg_image(
    tmgg_path: Path | None = None,
    include_configs: bool = True,
) -> modal.Image:
    """Create Modal image with TMGG package installed.

    Parameters
    ----------
    tmgg_path
        Path to local TMGG package for development.
        If None, installs from pip (when published).
    include_configs
        Whether to copy config files into the image.

    Returns
    -------
    modal.Image
        Image with TMGG and all dependencies.
    """
    image = create_pytorch_image()

    # Install TMGG dependencies
    # extra_index_url ensures pytorch-lightning gets CUDA-enabled torch
    # if it needs to resolve torch as a dependency
    image = image.uv_pip_install(
        "pytorch-lightning>=2.0.0",
        "hydra-core>=1.3.0",
        "omegaconf>=2.3.0",
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.5.0",
        "wandb>=0.20.1",
        "tqdm>=4.64.0",
        "networkx>=3.5",
        "tensorboard>=2.20.0",
        "loguru>=0.7.3",
        "boto3>=1.26.0",
        extra_index_url=TORCH_CUDA_INDEX,
    )

    # Install TMGG package
    if tmgg_path is not None:
        # Development mode: copy only src/ directory (not project root)
        # This avoids uploading configs/, data/, checkpoints/, results/, etc.
        # copy=True required because run_commands follows add_local_dir
        src_path = tmgg_path / "src"
        pyproject_path = tmgg_path / "pyproject.toml"

        # Mount src/ directory
        image = image.add_local_dir(
            str(src_path),
            "/app/tmgg/src",
            ignore=MODAL_IGNORE_PATTERNS,
            copy=True,
        )
        # Mount pyproject.toml and README.md for editable install
        image = image.add_local_file(
            str(pyproject_path),
            "/app/tmgg/pyproject.toml",
            copy=True,
        )
        readme_path = tmgg_path / "README.md"
        if readme_path.exists():
            image = image.add_local_file(
                str(readme_path),
                "/app/tmgg/README.md",
                copy=True,
            )
        image = image.run_commands("uv pip install --system -e /app/tmgg")
    else:
        # Production mode: install from pip
        image = image.uv_pip_install("tmgg[cloud]")

    # Copy config files if requested
    if include_configs and tmgg_path is not None:
        config_path = tmgg_path / "src" / "tmgg" / "exp_configs"
        if config_path.exists():
            image = image.add_local_dir(
                str(config_path),
                "/app/configs",
                ignore=MODAL_IGNORE_PATTERNS,
            )

    return image


def create_experiment_image(
    tmgg_path: Path | None = None,
    secrets: list[str] | None = None,
) -> modal.Image:
    """Create fully configured experiment image with secrets.

    Parameters
    ----------
    tmgg_path
        Path to local TMGG package.
    secrets
        List of Modal secret names to include.

    Returns
    -------
    modal.Image
        Complete experiment image.
    """
    image = create_tmgg_image(tmgg_path)

    # Set environment variables for experiment tracking
    image = image.env(
        {
            "WANDB_MODE": "online",
            "PYTHONUNBUFFERED": "1",
        }
    )

    return image


# Pre-built images for common configurations
# These can be referenced directly in Modal functions


# Development image (uses local TMGG)
def get_dev_image(tmgg_path: Path) -> modal.Image:
    """Get development image with local TMGG package."""
    return create_tmgg_image(tmgg_path, include_configs=True)


# Production image (uses pip-installed TMGG)
def get_prod_image() -> modal.Image:
    """Get production image with pip-installed TMGG."""
    return create_tmgg_image(tmgg_path=None, include_configs=False)
