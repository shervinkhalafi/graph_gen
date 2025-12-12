import sys
sys.path.append('src')

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

# Clear any existing Hydra instance
GlobalHydra.instance().clear()

with initialize(config_path="src/tmgg/exp_configs", version_base="1.3"):
    # First, compose with data=legacy_match
    cfg = compose(
        config_name="base_config_attention",
        overrides=[
            "data=legacy_match",
        ]
    )
    
    print("Data config after data=legacy_match:")
    print(OmegaConf.to_yaml(cfg.data))
    
    print("\nIs data a struct?", OmegaConf.is_struct(cfg.data))
    print("Is cfg a struct?", OmegaConf.is_struct(cfg))
    
    # Now try with the noise overrides
    GlobalHydra.instance().clear()
    
with initialize(config_path="src/tmgg/exp_configs", version_base="1.3"):
    try:
        cfg2 = compose(
            config_name="base_config_attention",
            overrides=[
                "data=legacy_match",
                "data.noise_type=gaussian",
            ]
        )
        print("\nSuccessfully composed with noise_type override")
    except Exception as e:
        print(f"\nFailed to compose with noise_type override: {e}")
