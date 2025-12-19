"""Modal image builder for TMGG experiments.

Provides functions to create Modal images with the TMGG package,
PyTorch, and all required dependencies pre-installed.
"""

from pathlib import Path

import modal


def create_base_image() -> modal.Image:
    """Create base Modal image with Python and system dependencies.

    Returns
    -------
    modal.Image
        Base image with Python 3.11 and system packages.
    """
    return modal.Image.debian_slim(python_version="3.11").apt_install(
        "git",
        "build-essential",
        "libffi-dev",
    )


def create_pytorch_image(cuda_version: str = "12.1") -> modal.Image:
    """Create Modal image with PyTorch and CUDA.

    Parameters
    ----------
    cuda_version
        CUDA version for PyTorch. Default "12.1".

    Returns
    -------
    modal.Image
        Image with PyTorch and CUDA support.
    """
    # Determine PyTorch index URL based on CUDA version
    cuda_suffix = cuda_version.replace(".", "")
    torch_index = f"https://download.pytorch.org/whl/cu{cuda_suffix}"

    return create_base_image().pip_install(
        f"torch>=2.0.0 --index-url {torch_index}",
        "torchvision",
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
    image = image.pip_install(
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
    )

    # Install TMGG package
    if tmgg_path is not None:
        # Development mode: copy local package
        image = image.add_local_dir(str(tmgg_path), "/app/tmgg")
        image = image.run_commands("pip install -e /app/tmgg")
    else:
        # Production mode: install from pip
        image = image.pip_install("tmgg[cloud]")

    # Copy config files if requested
    if include_configs and tmgg_path is not None:
        config_path = tmgg_path / "src" / "tmgg" / "exp_configs"
        if config_path.exists():
            image = image.add_local_dir(str(config_path), "/app/configs")

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
