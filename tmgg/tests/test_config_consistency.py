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
  - eval_noise_levels must appear in every model config so stage overrides work
  - scheduler_config must reference ${scheduler_config} (not define inline dicts)
    EXCEPT for digress configs that intentionally use `type: none`
"""

import glob

import yaml


def test_all_denoising_model_configs_propagate_eval_noise():
    """All denoising model configs must pass eval_noise_levels through."""
    config_dirs = [
        "src/tmgg/exp_configs/models/gnn/",
        "src/tmgg/exp_configs/models/hybrid/",
        "src/tmgg/exp_configs/models/digress/",
        "src/tmgg/exp_configs/models/baselines/",
    ]
    missing = []
    for d in config_dirs:
        for path in glob.glob(f"{d}*.yaml"):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            if cfg and "eval_noise_levels" not in cfg:
                missing.append(path)
    assert not missing, (
        "These model configs don't propagate eval_noise_levels:\n" + "\n".join(missing)
    )


def test_no_inline_scheduler_in_model_configs():
    """Model configs should reference ${scheduler_config}, not define inline."""
    config_dirs = [
        "src/tmgg/exp_configs/models/gnn/",
        "src/tmgg/exp_configs/models/hybrid/",
        "src/tmgg/exp_configs/models/digress/",
    ]
    inline = []
    for d in config_dirs:
        for path in glob.glob(f"{d}*.yaml"):
            with open(path) as f:
                cfg = yaml.safe_load(f)
            if cfg and isinstance(cfg.get("scheduler_config"), dict):
                inline.append(path)
    assert not inline, (
        "These configs define scheduler inline instead of ${scheduler_config}:\n"
        + "\n".join(inline)
    )
