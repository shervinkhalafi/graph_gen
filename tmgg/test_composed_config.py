import sys
sys.path.append('src')

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Clear any existing Hydra instance
GlobalHydra.instance().clear()

with initialize(config_path="src/tmgg/exp_configs", version_base=None):
    # Compose config with command line overrides similar to the script
    cfg = compose(
        config_name="base_config_attention",
        overrides=[
            "data=legacy_match",
            "+data.noise_type=digress",
            "+data.noise_levels=[0.3]",
            "model.noise_type=digress",
            "model.noise_levels=[0.3]",
            "model.num_heads=8",
            "model.num_layers=8",
        ]
    )
    
    print("Full config:")
    print(OmegaConf.to_yaml(cfg))
    
    print("\n\nData config value:", cfg.data)
