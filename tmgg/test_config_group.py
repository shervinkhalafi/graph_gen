import sys
sys.path.append('src')

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Clear any existing Hydra instance
GlobalHydra.instance().clear()

# Test what happens with the exact overrides
with initialize(config_path="src/tmgg/exp_configs", version_base="1.3"):
    cfg = compose(
        config_name="base_config_attention",
        overrides=[
            "data=legacy_match",
            "+data.noise_type=digress",
            "+data.noise_levels=[0.3]",
        ]
    )
    
    print(f"Type of cfg.data: {type(cfg.data)}")
    if hasattr(cfg.data, '__dict__'):
        print("Data config keys:", list(cfg.data.keys()))
        if hasattr(cfg.data, '_target_'):
            print(f"Data _target_: {cfg.data._target_}")
    else:
        print(f"Value of cfg.data: {cfg.data}")
