"""Modal app: deployed `tmgg-validate-gdpo-sbm` functions for the GDPO-SBM
sanity check. Three GPU tiers (A10G / A100-40GB / T4), each wrapping the
same ``validate.py`` pipeline. Outputs — metrics, report, reference-
comparison PNG, and per-graph samples — land on the ``tmgg-outputs``
volume under ``/data/outputs/validate-gdpo-sbm/<timestamp>/``.

Lifecycle
---------

    # Deploy once (re-run after editing validate.py or bumping the ckpt,
    # both of which are baked into the image at build time):
    mise run validate-gdpo-sbm-deploy

    # Spawn a call against the deployed app (no ephemeral app lifecycle;
    # the terminal can be closed immediately):
    mise run validate-gdpo-sbm-run                    # A100, paper-matching 40 samples

    # Fetch the result later via client.py, or pull from the volume.

Design notes
------------
* Standalone ``modal.App("tmgg-validate-gdpo-sbm")`` — deliberately *not*
  a new function on the production ``tmgg-spectral`` app. This is a
  one-off sanity check; isolating keeps it decoupled from training
  deployment cadence.
* Reuses :func:`tmgg.modal._lib.image.create_tmgg_image` so the container
  has the identical PyTorch/CUDA/graph-tool/ORCA stack as the training
  runs. No duplicate image definition, no drift risk.
* Reuses the ``tmgg-outputs`` volume. Reports accumulate at
  ``/data/outputs/validate-gdpo-sbm/<timestamp>/`` so multiple runs
  don't clobber each other.
* Client-side invocation lives in ``client.py`` (spawn/fetch), which uses
  ``modal.Function.from_name(...)`` so no ``modal run`` or local_entrypoint
  is involved — spawned calls persist on Modal independently of any
  client process lifetime.
"""

from __future__ import annotations

import sys
from pathlib import Path, PurePosixPath

import modal
from modal.cloud_bucket_mount import CloudBucketMount
from modal.volume import Volume

# This module is imported in two contexts:
#   - Host (deploy/client): the file lives at <repo>/analysis/.../modal_app.py
#     and we need REPO_ROOT + local checkpoint/validate.py paths to build the
#     image and resolve the tmgg package on sys.path.
#   - Container (Modal worker): the file is copied to /root/modal_app.py;
#     host paths are meaningless and parents[3] raises IndexError. The image
#     layers referencing host files have already been baked in.
_THIS_FILE: Path = Path(__file__).resolve()
_IS_HOST: bool = len(_THIS_FILE.parents) > 3


def _resolve_paths() -> tuple[Path, Path, Path]:
    """Return (repo_root, ckpt_local, validate_script) for the current env.

    On the host these point at the real repo layout; on the container
    they resolve to the baked-in copies under /app/.
    """
    if _IS_HOST:
        repo_root = _THIS_FILE.parents[3]
        ckpt = repo_root / ".local-storage/digress-checkpoints/gdpo_sbm/gdpo_sbm.ckpt"
        script = _THIS_FILE.parent / "validate.py"
        if not ckpt.exists():
            sys.exit(
                f"Checkpoint not found at {ckpt}. See .local-storage/"
                + "digress-checkpoints/README.md for how to fetch it "
                + "(`gdown` from the GDPO GDrive link in the per-folder README)."
            )
        # Repo source must be on host Python path for image introspection.
        sys.path.insert(0, str(repo_root / "src"))
        return repo_root, ckpt, script
    return Path("/"), Path("/app/ckpts/gdpo_sbm.ckpt"), Path("/app/validate.py")


REPO_ROOT, CKPT_LOCAL, VALIDATE_SCRIPT = _resolve_paths()

from tmgg.modal._lib.image import create_tmgg_image  # noqa: E402
from tmgg.modal._lib.volumes import OUTPUTS_MOUNT, _create_volume  # noqa: E402

app = modal.App("tmgg-validate-gdpo-sbm")

# Image: reuse the training image (micromamba + graph-tool + torch + ORCA)
# and bake the checkpoint and validate.py in at build time. The checkpoint
# is small (~29 MB) so this keeps the function self-contained; no separate
# volume upload flow is needed.
if _IS_HOST:
    image = (
        create_tmgg_image(tmgg_path=REPO_ROOT)
        .add_local_file(str(CKPT_LOCAL), "/app/ckpts/gdpo_sbm.ckpt", copy=True)
        .add_local_file(str(VALIDATE_SCRIPT), "/app/validate.py", copy=True)
    )
else:
    image = create_tmgg_image()

outputs_vol = _create_volume("tmgg-outputs")


def _run_validation_on_container(
    num_samples: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    """Body shared by every GPU-tier function.

    The GPU-tier wrappers below just decorate this with different
    ``gpu=`` strings so Modal's one-function-per-GPU constraint is
    honoured without duplicating the logic.
    """
    import importlib.util
    from datetime import datetime
    from pathlib import Path as _Path

    spec = importlib.util.spec_from_file_location(
        "validate_gdpo_sbm", "/app/validate.py"
    )
    assert spec is not None and spec.loader is not None
    validate_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(validate_mod)

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = _Path(f"{OUTPUTS_MOUNT}/validate-gdpo-sbm/{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = validate_mod.run(
        ckpt_path=_Path("/app/ckpts/gdpo_sbm.ckpt"),
        out_dir=out_dir,
        num_samples=num_samples,
        device_str="cuda",
        seed=seed,
        batch_size=batch_size,
    )

    # Flush results to the volume before the container exits.
    outputs_vol.commit()

    def _slurp(name: str) -> bytes | None:
        p = out_dir / name
        return p.read_bytes() if p.exists() else None

    return {
        "metrics": metrics,
        "output_dir": str(out_dir),
        "stamp": stamp,
        "files": {
            "metrics.json": _slurp("metrics.json"),
            "report.md": _slurp("report.md"),
            "metrics_vs_reference.png": _slurp("metrics_vs_reference.png"),
            "samples.jsonl": _slurp("samples.jsonl"),
        },
    }


# GPU-tier wrappers. Modal requires one @app.function per GPU spec, so we
# define three — client picks via ``modal.Function.from_name``. Each passes
# the image/timeout/volumes explicitly so pyright can see the kwarg types
# (a `**dict(...)` spread would lose them). ``_VOLUMES`` is typed to match
# the declared parameter type on ``@app.function(volumes=...)`` so dict
# invariance doesn't bite.
_TIMEOUT_SEC = 3600  # 1 h ample — A10G should finish in 2-5 min
_VOLUMES: dict[str | PurePosixPath, Volume | CloudBucketMount] = {
    OUTPUTS_MOUNT: outputs_vol
}


@app.function(gpu="A10G", image=image, timeout=_TIMEOUT_SEC, volumes=_VOLUMES)
def validate_a10g(
    num_samples: int = 40, batch_size: int = 40, seed: int = 42
) -> dict[str, object]:
    return _run_validation_on_container(num_samples, batch_size, seed)


@app.function(gpu="A100-40GB", image=image, timeout=_TIMEOUT_SEC, volumes=_VOLUMES)
def validate_a100(
    num_samples: int = 40, batch_size: int = 40, seed: int = 42
) -> dict[str, object]:
    return _run_validation_on_container(num_samples, batch_size, seed)


@app.function(gpu="T4", image=image, timeout=_TIMEOUT_SEC, volumes=_VOLUMES)
def validate_t4(
    num_samples: int = 40, batch_size: int = 8, seed: int = 42
) -> dict[str, object]:
    return _run_validation_on_container(num_samples, batch_size, seed)
