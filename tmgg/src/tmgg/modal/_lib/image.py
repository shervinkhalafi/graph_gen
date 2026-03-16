"""Modal image builder for TMGG experiments.

Provides a single ``create_tmgg_image`` function that builds a Modal
image with TMGG, PyTorch, and all evaluation dependencies (graph-tool,
ORCA, POT). Uses ``Image.micromamba()`` as the base so that conda-forge
packages (graph-tool) can be installed natively alongside pip packages.
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


# PyTorch CUDA index URL for consistent CUDA-enabled installations
TORCH_CUDA_INDEX = "https://download.pytorch.org/whl/cu121"


def create_tmgg_image(
    tmgg_path: Path | None = None,
) -> modal.Image:
    """Create Modal image with TMGG and all dependencies.

    Uses ``Image.micromamba()`` as the base so graph-tool (conda-forge
    only, not pip-installable) is available for SBM accuracy evaluation.
    PyTorch and all other pip dependencies are layered on top. When
    ``tmgg_path`` is provided, ORCA is compiled from bundled C++ source.

    Hydra config files are **not** included in the image — the deployed
    app receives fully resolved configs as YAML via ``modal_run_cli()``,
    which writes them to a temp directory for the CLI subprocess.

    Parameters
    ----------
    tmgg_path
        Path to local TMGG package for development. If None, installs
        from pip (when published).

    Returns
    -------
    modal.Image
        Image with TMGG, PyTorch, graph-tool, ORCA, and all deps.
    """
    # micromamba base with graph-tool from conda-forge
    image = (
        modal.Image.micromamba(python_version="3.12")
        .apt_install("git", "build-essential", "libffi-dev")
        .micromamba_install("graph-tool", channels=["conda-forge"])
    )

    # PyTorch with CUDA support
    image = image.uv_pip_install(
        "torch>=2.0.0",
        "torchvision",
        index_url=TORCH_CUDA_INDEX,
    )

    # TMGG pip dependencies
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
        src_path = tmgg_path / "src"
        pyproject_path = tmgg_path / "pyproject.toml"

        image = image.add_local_dir(
            str(src_path),
            "/app/tmgg/src",
            ignore=MODAL_IGNORE_PATTERNS,
            copy=True,
        )
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
        image = image.uv_pip_install("uv")
        image = image.run_commands(
            "uv pip install --system --compile-bytecode -e /app/tmgg"
        )

        # Compile ORCA binary from bundled C++ source
        orca_src = (
            tmgg_path / "src" / "tmgg" / "training" / "evaluation_metrics" / "orca"
        )
        if orca_src.exists():
            image = image.add_local_dir(
                str(orca_src),
                "/app/orca",
                ignore=MODAL_IGNORE_PATTERNS,
                copy=True,
            )
            image = image.run_commands(
                "cd /app/orca && g++ -O2 -std=c++11 -o orca orca.cpp"
            )
    else:
        image = image.uv_pip_install("tmgg[cloud]")

    # Environment
    image = image.env(
        {
            "WANDB_MODE": "online",
            "PYTHONUNBUFFERED": "1",
        }
    )

    return image
