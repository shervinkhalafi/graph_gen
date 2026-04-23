"""CLI for discrete diffusion checkpoint evaluation.

Loads a trained checkpoint, samples graphs, and reports MMD metrics
against a reference distribution drawn from the checkpoint's own
datamodule (val or test split per ``--reference_set``). Separated from
``evaluate`` to avoid an import cycle with ``lightning_module``.

Only DiffusionModule checkpoints are supported. Legacy checkpoints from
earlier LightningModule implementations are incompatible.

Reference graphs come from the *training-time datamodule* rather than a
synthetic regenerator. The datamodule is reconstructed via
:func:`hydra.utils.instantiate` against the ``data:`` block of the
sibling ``config.yaml`` that ``run_experiment`` saves alongside every
checkpoint. The fail-loud contract holds: when ``config.yaml`` is missing
or its ``data:`` block lacks a ``_target_``, the call raises rather
than falling back to synthetic graphs.

EMA weights swap into the loaded model when ``use_ema in ('true',
'auto')`` and the checkpoint carries an ``EMACallback`` shadow under
``checkpoint['callbacks']``. ``--use_ema true`` without a shadow raises;
``--use_ema auto`` silently uses live weights and records
``ema_active=False`` in the result dict.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from tmgg.evaluation.mmd_metrics import compute_mmd_metrics
from tmgg.models.base import GraphModel
from tmgg.training.callbacks.ema import EMACallback
from tmgg.training.lightning_modules.diffusion_module import DiffusionModule


def _instantiate_checkpoint_model(
    checkpoint_path: Path,
    *,
    device: str,
) -> GraphModel:
    """Reconstruct the nested graph model from checkpoint metadata.

    ``DiffusionModule`` intentionally excludes the graph model from
    ``save_hyperparameters`` because the instantiated module object is not a
    stable constructor argument. The checkpoint still stores
    ``model_class`` and ``model_config`` in the hyperparameters, which is the
    canonical source for rebuilding that nested model at load time.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    hparams = checkpoint.get("hyper_parameters")
    if not isinstance(hparams, dict):
        raise TypeError(
            f"Expected checkpoint hyperparameters dict in {checkpoint_path}, "
            f"got {type(hparams).__name__}"
        )

    model_class_path = hparams.get("model_class")
    model_config = hparams.get("model_config")
    if not isinstance(model_class_path, str):
        raise TypeError(
            f"Checkpoint {checkpoint_path} is missing string model_class metadata"
        )
    if not isinstance(model_config, dict):
        raise TypeError(
            f"Checkpoint {checkpoint_path} is missing dict model_config metadata"
        )

    module_path, class_name = model_class_path.rsplit(".", 1)
    model_module = importlib.import_module(module_path)
    model_class = getattr(model_module, class_name)
    if not isinstance(model_class, type) or not issubclass(model_class, GraphModel):
        raise TypeError(
            f"Checkpoint model class {model_class_path!r} is not a GraphModel subclass"
        )

    model = model_class(**model_config)
    return model


def _load_diffusion_module(
    checkpoint_path: Path,
    *,
    device: str,
) -> DiffusionModule:
    """Load a diffusion checkpoint, rebuilding the nested model if needed."""
    try:
        return DiffusionModule.load_from_checkpoint(
            str(checkpoint_path), map_location=device
        )
    except TypeError as exc:
        if "missing 1 required keyword-only argument: 'model'" not in str(exc):
            raise

    # Newer checkpoints persist model metadata but not the instantiated model.
    # Rebuild the nested GraphModel explicitly and retry the Lightning load.
    model = _instantiate_checkpoint_model(checkpoint_path, device=device)
    return DiffusionModule.load_from_checkpoint(
        str(checkpoint_path),
        map_location=device,
        model=model,
    )


def _find_sibling_config_yaml(checkpoint_path: Path) -> Path:
    """Locate the ``config.yaml`` saved next to a Lightning run's checkpoints.

    ``run_experiment.run_single_experiment`` writes ``config.yaml`` into
    ``${paths.output_dir}`` and stores checkpoints under
    ``${paths.output_dir}/checkpoints/``. The sibling lookup walks up
    one directory from the checkpoint file.

    Raises
    ------
    FileNotFoundError
        Neither ``checkpoint_path.parent.parent / "config.yaml"`` nor
        ``checkpoint_path.parent / "config.yaml"`` exists. Both fail
        loudly per CLAUDE.md -- there is no synthetic-regeneration
        fallback.
    """
    candidates = [
        checkpoint_path.parent.parent / "config.yaml",
        checkpoint_path.parent / "config.yaml",
    ]
    for cand in candidates:
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"No config.yaml found alongside {checkpoint_path}; looked in: "
        f"{[str(c) for c in candidates]}. Reference-graph extraction "
        "requires the training-time data config to reconstruct the "
        "datamodule."
    )


def _load_datamodule_for_reference(
    checkpoint_path: Path,
    *,
    reference_set: Literal["val", "test"],
) -> Any:
    """Instantiate and ``setup`` the training-time datamodule.

    Reads the sibling ``config.yaml``, calls
    :func:`hydra.utils.instantiate` on the ``data:`` block, and runs the
    appropriate Lightning ``setup`` stage for the requested split.

    Parameters
    ----------
    checkpoint_path
        Path to the ``.ckpt`` file under evaluation. The sibling
        ``config.yaml`` is located one directory above.
    reference_set
        ``"val"`` runs ``setup("fit")`` (Lightning's standard stage for
        train+val); ``"test"`` runs ``setup("test")``.

    Returns
    -------
    Any
        The instantiated, setup datamodule. Typed ``Any`` because Hydra
        recursive instantiation returns ``Any`` and the concrete
        :class:`BaseGraphDataModule` subclass varies per experiment.

    Raises
    ------
    KeyError
        ``config.yaml`` lacks a ``data`` block.
    """
    config_path = _find_sibling_config_yaml(checkpoint_path)
    loaded = OmegaConf.load(config_path)
    if not isinstance(loaded, DictConfig):
        raise TypeError(
            f"config.yaml at {config_path} did not parse to a DictConfig; "
            f"got {type(loaded).__name__}. The training-time runner saves a "
            "mapping; a list/scalar at the top level is malformed."
        )
    data_cfg = OmegaConf.select(loaded, "data")
    if data_cfg is None:
        raise KeyError(
            f"config.yaml at {config_path} has no `data:` block; cannot "
            "reconstruct the datamodule for reference-graph extraction."
        )

    datamodule = hydra.utils.instantiate(data_cfg)
    stage = "fit" if reference_set == "val" else "test"
    datamodule.setup(stage=stage)
    return datamodule


def _collect_reference_graphs(
    datamodule: Any,
    *,
    reference_set: Literal["val", "test"],
    max_graphs: int,
) -> list[Any]:
    """Pull up to ``max_graphs`` NetworkX references from the datamodule.

    Delegates to :meth:`BaseGraphDataModule.get_reference_graphs`, which
    iterates the appropriate dataloader and converts each
    ``GraphData.binarised_adjacency()`` slice into a NetworkX graph
    while honouring ``node_mask`` for variable-size batches.
    """
    return datamodule.get_reference_graphs(reference_set, max_graphs)


def _maybe_load_ema_state(checkpoint: dict[str, Any]) -> dict[str, Any] | None:
    """Return the EMACallback state dict from the checkpoint, if present.

    Lightning persists callback state under ``checkpoint['callbacks']``
    keyed by the callback's ``state_key`` (a string containing the
    class name and constructor arg repr). The probe matches any key
    containing ``"EMA"`` to track Lightning's default state-key naming
    convention.
    """
    callbacks = checkpoint.get("callbacks")
    if not isinstance(callbacks, dict):
        return None
    for key, value in callbacks.items():
        if "EMA" in str(key) and isinstance(value, dict):
            return value
    return None


def _maybe_swap_ema_into_module(
    module: DiffusionModule,
    checkpoint: dict[str, Any],
    use_ema: Literal["auto", "true", "false"],
) -> bool:
    """Swap EMA shadow weights into ``module.model`` when configured.

    Returns ``True`` when the swap actually fired (i.e. the result
    record's ``ema_active`` flag should be ``True``).

    Raises
    ------
    RuntimeError
        ``use_ema == "true"`` requested but the checkpoint carries no
        EMA shadow.
    """
    if use_ema == "false":
        return False
    ema_state = _maybe_load_ema_state(checkpoint)
    if ema_state is None:
        if use_ema == "true":
            raise RuntimeError(
                "use_ema=true requested but the checkpoint carries no "
                "EMA shadow under checkpoint['callbacks']."
            )
        return False
    decay = float(ema_state["decay"])
    cb = EMACallback(decay=decay)
    cb.load_state_dict(ema_state)
    cb.copy_shadow_into(module.model)
    return True


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    num_samples: int = 500,
    reference_set: Literal["val", "test"] = "val",
    use_ema: Literal["auto", "true", "false"] = "auto",
    mmd_kernel: Literal["gaussian", "gaussian_tv"] = "gaussian_tv",
    mmd_sigma: float = 1.0,
    device: str = "cpu",
) -> dict[str, Any]:
    """Load a discrete diffusion checkpoint, sample graphs, and compute MMD.

    Reference graphs come from the *training-time datamodule*: the
    sibling ``config.yaml`` is loaded, the ``data:`` block is
    Hydra-instantiated, and the appropriate split's dataloader is
    walked to collect up to ``num_samples`` references. There is no
    synthetic-regeneration fallback (CLAUDE.md "single way of doing
    things").

    Parameters
    ----------
    checkpoint_path
        Path to a Lightning checkpoint (``.ckpt``).
    num_samples
        Number of graphs to generate and compare. The reference set is
        capped at the same count.
    reference_set
        ``"val"`` (default) or ``"test"``. Selects which split's
        dataloader supplies the reference graphs.
    use_ema
        ``"auto"`` swaps shadow weights when present, otherwise uses
        live weights (recording ``ema_active=False``). ``"true"``
        requires a shadow; raises if absent. ``"false"`` skips the
        swap.
    mmd_kernel
        Kernel for MMD computation.
    mmd_sigma
        Kernel bandwidth.
    device
        Torch device.

    Returns
    -------
    dict
        Evaluation results including MMD metrics and the
        ``ema_active`` boolean reflecting whether EMA weights were
        actually used.
    """
    checkpoint_path = Path(checkpoint_path)

    print(f"\n{'=' * 60}")
    print(f"Evaluating: {checkpoint_path.name}")
    print(f"{'=' * 60}")

    # Load model from checkpoint (save_hyperparameters stores constructor args)
    # Only DiffusionModule checkpoints are supported.
    print("\nLoading model...")
    module = _load_diffusion_module(checkpoint_path, device=device)
    module = module.to(device)
    module.eval()

    # Re-load the raw checkpoint dict to inspect EMA callback state.
    # ``load_from_checkpoint`` does not expose the callback section.
    raw_checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    ema_active = _maybe_swap_ema_into_module(module, raw_checkpoint, use_ema)
    if ema_active:
        print("Swapped EMA shadow weights into model for evaluation.")
    else:
        print("Evaluating live (non-EMA) weights.")

    # Pull reference graphs from the training-time datamodule. Lightning
    # `setup` reads the dataset on disk (or generates synthetic batches
    # when the datamodule does that internally), then `get_reference_graphs`
    # walks the dataloader and binarises each adjacency.
    print(
        f"\nFetching {num_samples} reference graphs from {reference_set} split "
        "via training-time datamodule..."
    )
    datamodule = _load_datamodule_for_reference(
        checkpoint_path, reference_set=reference_set
    )
    ref_graphs = _collect_reference_graphs(
        datamodule, reference_set=reference_set, max_graphs=num_samples
    )
    if not ref_graphs:
        raise RuntimeError(
            f"Reference {reference_set} split for {checkpoint_path} returned "
            "zero graphs; cannot compute MMD against an empty reference."
        )

    # Sample from model using the sampler. Generate the same count as
    # the (possibly truncated) reference list so MMD compares
    # equally-sized populations.
    num_generated = len(ref_graphs)
    print(f"Sampling {num_generated} graphs from model...")
    with torch.no_grad():
        generated_graph_data = module.generate_graphs(num_generated)

    mmd_results = compute_mmd_metrics(
        ref_graphs, generated_graph_data, kernel=mmd_kernel, sigma=mmd_sigma
    )

    results: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path.absolute()),
        "checkpoint_name": checkpoint_path.name,
        "reference_set": reference_set,
        "num_generated": num_generated,
        "num_reference": len(ref_graphs),
        "mmd_kernel": mmd_kernel,
        "mmd_sigma": mmd_sigma,
        "mmd_results": mmd_results.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "ema_active": ema_active,
    }

    print(f"\n{'=' * 60}")
    print("MMD Results")
    print(f"{'=' * 60}")
    print(f"  Degree MMD:     {mmd_results.degree_mmd:.6f}")
    print(f"  Clustering MMD: {mmd_results.clustering_mmd:.6f}")
    print(f"  Spectral MMD:   {mmd_results.spectral_mmd:.6f}")
    print(f"{'=' * 60}\n")

    return results


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for discrete diffusion checkpoint evaluation."""
    parser = argparse.ArgumentParser(
        description="Evaluate a discrete diffusion checkpoint with MMD metrics",
    )
    parser.add_argument("--checkpoint", required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--num-samples", type=int, default=500, help="Graphs to generate/compare"
    )
    parser.add_argument(
        "--reference_set",
        default="val",
        choices=["val", "test"],
        help="Which split's dataloader supplies the reference graphs.",
    )
    parser.add_argument(
        "--use_ema",
        default="auto",
        choices=["auto", "true", "false"],
        help=(
            "Swap to EMA shadow weights before sampling. auto = swap iff "
            "the checkpoint carries a shadow."
        ),
    )
    parser.add_argument(
        "--kernel",
        default="gaussian_tv",
        choices=["gaussian", "gaussian_tv"],
        help="MMD kernel (default: gaussian_tv)",
    )
    parser.add_argument("--sigma", type=float, default=1.0, help="Kernel bandwidth")
    parser.add_argument("--device", default="cpu", help="Torch device (default: cpu)")
    parser.add_argument(
        "--output", type=str, default=None, help="Write results to JSON file"
    )

    args = parser.parse_args(argv)

    results = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        reference_set=args.reference_set,
        use_ema=args.use_ema,
        mmd_kernel=args.kernel,
        mmd_sigma=args.sigma,
        device=args.device,
    )

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"Results written to {output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
