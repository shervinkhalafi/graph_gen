"""Modal image builder for TMGG experiments.

Provides a single ``create_tmgg_image`` function that builds a Modal
image with TMGG, PyTorch, and all evaluation dependencies (graph-tool,
ORCA, POT). Uses ``Image.micromamba()`` as the base so binary-heavy
packages are solved in one Conda environment instead of mixing Conda
and pip native runtimes.
"""

import re
import tomllib
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


_ORCA_SOURCE_RELATIVE_DIR = Path("src") / "tmgg" / "evaluation" / "orca"
PYTORCH_VERSION = "2.5.1"
TORCHVISION_VERSION = "0.20.1"
PYTORCH_CUDA_VERSION = "12.1"
_PIP_RUNTIME_EXCLUDE = {"torch", "modal"}
_PIP_RUNTIME_EXTRA_DEPS = (
    "boto3>=1.26.0",
    "tensorboard>=2.20.0",
)


def _resolve_orca_source_dir(tmgg_path: Path) -> Path:
    """Return the in-repo ORCA source directory.

    The current ORCA source lives inside the package at
    ``src/tmgg/evaluation/orca``. We resolve it from the repo root up front so
    Modal image construction fails loudly if the source tree drifts again.
    """
    orca_src = tmgg_path / _ORCA_SOURCE_RELATIVE_DIR
    if not orca_src.exists():
        raise FileNotFoundError(f"ORCA source not found at {orca_src}")
    return orca_src


def _resolve_pyproject_path(tmgg_path: Path) -> Path:
    """Return the repo ``pyproject.toml`` used to build the image."""
    pyproject_path = tmgg_path / "pyproject.toml"
    if not pyproject_path.exists():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    return pyproject_path


def _dependency_name(requirement: str) -> str:
    """Extract the normalized package name from a PEP 508 requirement string."""
    return re.split(r"[<>= !\\[]", requirement, maxsplit=1)[0]


def _project_pip_runtime_deps(tmgg_path: Path) -> list[str]:
    """Return pip runtime deps for the experiment image from ``pyproject.toml``.

    The Modal image installs the native PyTorch/CUDA/graph-tool stack via
    micromamba, so those packages are excluded here. All remaining declared
    runtime dependencies come from the project metadata to avoid drift between
    local environments and the image.
    """
    pyproject = tomllib.loads(_resolve_pyproject_path(tmgg_path).read_text())
    project_deps = pyproject["project"]["dependencies"]

    runtime_deps = [
        dep for dep in project_deps if _dependency_name(dep) not in _PIP_RUNTIME_EXCLUDE
    ]
    runtime_dep_names = {_dependency_name(dep) for dep in runtime_deps}
    for dep in _PIP_RUNTIME_EXTRA_DEPS:
        if _dependency_name(dep) not in runtime_dep_names:
            runtime_deps.append(dep)
    return runtime_deps


def _runtime_orca_dir() -> Path:
    """Return the installed-package ORCA directory inside the image/runtime."""
    import tmgg

    return Path(tmgg.__file__).resolve().parent / "evaluation" / "orca"


def _compile_orca_in_image() -> None:
    """Compile ORCA inside the image at the package runtime location.

    This runs as a Modal image build step via ``Image.run_function()`` so the
    resulting binary is built against the container's libc/libstdc++ rather
    than copied from the host machine.
    """
    import subprocess

    orca_dir = _runtime_orca_dir()
    source = orca_dir / "orca.cpp"
    binary = orca_dir / "orca"
    if not source.exists():
        raise FileNotFoundError(f"ORCA source not found at {source}")

    binary.unlink(missing_ok=True)
    subprocess.run(
        ["g++", "-O2", "-std=c++11", "-o", str(binary), str(source)],
        check=True,
    )


def _runtime_env() -> dict[str, str]:
    """Return runtime environment variables for the Modal image."""
    return {
        "WANDB_MODE": "online",
        "PYTHONUNBUFFERED": "1",
    }


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
    # micromamba base with one solved binary environment for graph-tool + PyTorch
    image = (
        modal.Image.micromamba(python_version="3.12")
        .apt_install("git", "build-essential", "libffi-dev")
        .micromamba_install(
            "graph-tool",
            f"pytorch=={PYTORCH_VERSION}",
            f"torchvision=={TORCHVISION_VERSION}",
            f"pytorch-cuda={PYTORCH_CUDA_VERSION}",
            channels=["pytorch", "nvidia", "conda-forge"],
        )
    )

    # Install TMGG package
    if tmgg_path is not None:
        _resolve_orca_source_dir(tmgg_path)
        pip_runtime_deps = _project_pip_runtime_deps(tmgg_path)
        src_path = tmgg_path / "src"
        pyproject_path = _resolve_pyproject_path(tmgg_path)

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
        # TMGG pip dependencies.
        #
        # Keep the native stack (PyTorch/CUDA/graph-tool) in one conda solve.
        # PyG itself is installed from PyPI because its docs no longer provide
        # conda packages for torch > 2.5.0, and the base ``torch-geometric``
        # package is usable without extra compiled extensions.
        image = image.uv_pip_install(*pip_runtime_deps)
        image = image.uv_pip_install("uv")
        image = image.run_commands(
            "uv pip install --system --compile-bytecode --no-deps -e /app/tmgg"
        )
        image = image.run_function(_compile_orca_in_image)
    else:
        # Fallback for published-package installs when the source checkout is
        # unavailable at deploy time. Keep this list in sync with pyproject if
        # that workflow becomes active again.
        image = image.uv_pip_install(
            "pytorch-lightning>=2.0.0",
            "hydra-core>=1.3.0",
            "omegaconf>=2.3.0",
            "numpy>=1.21.0",
            "scipy>=1.7.0",
            "matplotlib>=3.5.0",
            "wandb>=0.20.1",
            "tqdm>=4.64.0",
            "rich>=13.0.0",
            "networkx>=3.5",
            "loguru>=0.7.3",
            "click>=8.0.0",
            "pandas>=2.0.0",
            "seaborn>=0.13.2",
            "torch-geometric>=2.5.0",
            "safetensors>=0.4.0",
            "polars>=1.0.0",
            "pot>=0.9.0",
            *_PIP_RUNTIME_EXTRA_DEPS,
        )
        image = image.uv_pip_install("uv")
        image = image.run_commands(
            "uv pip install --system --compile-bytecode --no-deps tmgg[cloud]"
        )

    # Environment
    image = image.env(_runtime_env())

    return image
