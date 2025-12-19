import sys

sys.path.append("src")

import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Clear any existing Hydra instance
GlobalHydra.instance().clear()

with initialize(config_path="src/tmgg/exp_configs/base/data", version_base=None):
    # Compose config with command line overrides similar to the script
    cfg = compose(
        config_name="legacy_match",
        overrides=["+data.noise_type=digress", "+data.noise_levels=[0.3]"],
    )

    print("Config:", cfg)
    print("\n_target_:", cfg._target_)
    print("dataset_name:", cfg.dataset_name)
    print("noise_type:", cfg.noise_type)
    print("noise_levels:", cfg.noise_levels)

    # Try to instantiate
    dm = hydra.utils.instantiate(cfg)
    print("\nInstantiated type:", type(dm))
    print("Module:", dm.__class__.__module__)
