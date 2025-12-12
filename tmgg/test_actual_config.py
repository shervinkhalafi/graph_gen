import sys
sys.path.append('src')

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
import hydra

# Clear any existing Hydra instance
GlobalHydra.instance().clear()

with initialize(config_path="src/tmgg/exp_configs", version_base="1.3"):
    cfg = compose(
        config_name="base_config_attention",
        overrides=[
            "data=legacy_match",
            "+data.noise_type=digress",
            "+data.noise_levels=[0.3]",
            "model.noise_type=digress",
            "model.noise_levels=[0.3]",
        ]
    )
    
    print("Data config contents:")
    print(OmegaConf.to_yaml(cfg.data))
    
    print("\nAttempting to instantiate data module:")
    try:
        dm = hydra.utils.instantiate(cfg.data)
        print(f"Success! Type: {type(dm)}")
    except Exception as e:
        print(f"Failed: {e}")
