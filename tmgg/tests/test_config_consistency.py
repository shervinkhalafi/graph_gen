"""Config consistency tests to prevent silent misconfiguration.

Rationale
---------
Several model config YAML files hardcode scheduler parameters instead of
inheriting from the shared base config, and some don't propagate
eval_noise_levels/fixed_noise_seed. This makes experiments inconsistent
because changes to the base config don't reach configs that hardcode their
own values. These tests enforce that model configs delegate to the shared
base where appropriate.

Starting state: model configs under exp_configs/models/{gnn,hybrid,digress,baselines}
Invariants:
  - eval_noise_levels must appear in every model config (directly or via defaults)
  - scheduler_config must reference ${scheduler_config} (not define inline dicts)
    EXCEPT for digress configs that intentionally use `type: none`
"""

import glob
import os
from pathlib import Path

import yaml

_MODELS_ROOT = Path("src/tmgg/experiments/exp_configs/models")

_CONFIG_DIRS = [
    str(_MODELS_ROOT / "gnn/"),
    str(_MODELS_ROOT / "hybrid/"),
    str(_MODELS_ROOT / "digress/"),
    str(_MODELS_ROOT / "baselines/"),
]


def _resolve_with_defaults(path: str) -> dict:
    """Load a YAML config, merging one level of Hydra defaults.

    This is a simplified resolver: it reads the ``defaults`` list, loads
    each referenced YAML from the same directory, and shallow-merges them
    (child keys override base keys). Enough for flat-key consistency checks
    but does not handle Hydra's full override semantics.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    defaults = cfg.pop("defaults", None)
    if not defaults:
        return cfg

    config_dir = os.path.dirname(path)
    merged: dict = {}
    for entry in defaults:
        if isinstance(entry, str) and entry != "_self_":
            base_path = os.path.join(config_dir, entry + ".yaml")
            if os.path.exists(base_path):
                with open(base_path) as f:
                    base_cfg = yaml.safe_load(f) or {}
                merged.update(base_cfg)
    # _self_ means child overrides base (Hydra convention)
    merged.update(cfg)
    return merged


def test_all_denoising_model_configs_propagate_eval_noise():
    """All denoising model configs must pass eval_noise_levels through.

    Checks both direct keys and keys inherited via Hydra defaults.
    """
    missing = []
    for d in _CONFIG_DIRS:
        for path in glob.glob(f"{d}*.yaml"):
            cfg = _resolve_with_defaults(path)
            if cfg and "eval_noise_levels" not in cfg:
                missing.append(path)
    assert not missing, (
        "These model configs don't propagate eval_noise_levels:\n"
        + "\n".join(f"  {p}" for p in missing)
    )


def test_no_inline_scheduler_in_model_configs():
    """Model configs should reference ${scheduler_config}, not define inline.

    Checks resolved configs (including defaults) for inline scheduler dicts.
    """
    inline = []
    for d in _CONFIG_DIRS[:3]:  # gnn, hybrid, digress only
        for path in glob.glob(f"{d}*.yaml"):
            cfg = _resolve_with_defaults(path)
            if cfg and isinstance(cfg.get("scheduler_config"), dict):
                inline.append(path)
    assert not inline, (
        "These configs define scheduler inline instead of ${scheduler_config}:\n"
        + "\n".join(f"  {p}" for p in inline)
    )
