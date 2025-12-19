"""Utilities for loading checkpoints with mismatched hyperparameters."""

import inspect
import warnings
from pathlib import Path
from typing import TypeVar

import pytorch_lightning as pl
import torch

T = TypeVar("T", bound=pl.LightningModule)


def load_checkpoint_with_fallback[T: pl.LightningModule](
    module_class: type[T],
    checkpoint_path: str | Path,
    strict: bool = True,
    map_location: str | torch.device | None = None,
    **override_kwargs,
) -> T:
    """
    Load a LightningModule from checkpoint with fallback for mismatched hyperparameters.

    When checkpoint contains hyperparameters that module's __init__ no longer accepts,
    filters them out and retries loading. This handles backwards compatibility when
    module signatures change between code versions.

    Parameters
    ----------
    module_class
        The LightningModule subclass to instantiate.
    checkpoint_path
        Path to the checkpoint file.
    strict
        Whether to strictly enforce state_dict key matching.
    map_location
        Device to map tensors to during loading.
    **override_kwargs
        Additional kwargs to pass to __init__, overriding checkpoint values.

    Returns
    -------
    T
        The loaded module instance.

    Raises
    ------
    TypeError
        If loading fails for reasons other than unexpected keyword arguments.
    """
    checkpoint_path = Path(checkpoint_path)

    # Attempt normal loading
    try:
        return module_class.load_from_checkpoint(
            str(checkpoint_path),
            strict=strict,
            map_location=map_location,
            **override_kwargs,
        )
    except TypeError as e:
        if "unexpected keyword argument" not in str(e):
            raise

    # Introspect __init__ to find accepted parameters
    sig = inspect.signature(module_class.__init__)
    accepted_params = set(sig.parameters.keys()) - {"self"}

    has_var_keyword = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if has_var_keyword:
        raise  # **kwargs accepts anything, filtering won't help

    # Load checkpoint and filter hyperparameters
    checkpoint = torch.load(
        checkpoint_path, map_location=map_location, weights_only=False
    )
    stored_hparams = checkpoint.get("hyper_parameters", {})

    filtered_hparams = {}
    removed_keys = []
    for key, value in stored_hparams.items():
        if key in accepted_params:
            filtered_hparams[key] = value
        else:
            removed_keys.append(key)

    if removed_keys:
        warnings.warn(
            f"Checkpoint '{checkpoint_path}' has hyperparameters not accepted by "
            f"{module_class.__name__}.__init__. Filtered: {removed_keys}",
            UserWarning,
            stacklevel=2,
        )

    # Instantiate with filtered hparams and load state dict
    final_hparams = {**filtered_hparams, **override_kwargs}
    model = module_class(**final_hparams)
    model.load_state_dict(checkpoint["state_dict"], strict=strict)

    return model
