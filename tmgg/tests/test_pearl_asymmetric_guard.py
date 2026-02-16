"""Guard test: PEARL and asymmetric config keys must not silently pass through.

Rationale
---------
Prior to this fix, PEARL/asymmetric YAML configs set parameters like
``embedding_source``, ``pearl_num_layers``, and ``asymmetric`` that were
swallowed by ``SpectralDenoisingLightningModule.__init__``'s ``**kwargs``
without reaching the model factory. Researchers selecting these configs
unknowingly ran the default model, so experiments were mislabeled.

The offending configs were deleted. These tests ensure no new config file
sneaks PEARL/asymmetric keys back in until the pipeline is properly wired.
"""

import glob

import yaml


def test_no_pearl_configs_exist():
    """No spectral config file should set embedding_source or pearl_* keys."""
    config_dir = "src/tmgg/exp_configs/models/spectral/"
    pearl_keys = {
        "embedding_source",
        "pearl_num_layers",
        "pearl_hidden_dim",
        "pearl_input_samples",
    }
    violations = []
    for path in glob.glob(f"{config_dir}*.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if cfg and any(k in cfg for k in pearl_keys):
            found = [k for k in pearl_keys if k in cfg]
            violations.append(f"{path}: {found}")
    assert not violations, (
        "PEARL config keys found in spectral configs but pipeline is not wired. "
        "Either wire the pipeline or remove the keys:\n" + "\n".join(violations)
    )


def test_no_asymmetric_configs_exist():
    """No spectral config should set asymmetric key (not wired through factory)."""
    config_dir = "src/tmgg/exp_configs/models/spectral/"
    violations = []
    for path in glob.glob(f"{config_dir}*.yaml"):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if cfg and "asymmetric" in cfg:
            violations.append(path)
    assert not violations, (
        "asymmetric key found in spectral configs but factory doesn't pass it. "
        "Either wire through factory or remove:\n" + "\n".join(violations)
    )
