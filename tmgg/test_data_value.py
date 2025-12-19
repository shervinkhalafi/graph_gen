import sys

sys.path.append("src")

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Clear any existing Hydra instance
GlobalHydra.instance().clear()

with initialize(config_path="src/tmgg/exp_configs", version_base="1.3"):
    cfg = compose(
        config_name="base_config_attention",
        overrides=[
            "data=legacy_match",
        ],
    )

    print(f"Type of cfg.data: {type(cfg.data)}")
    print(f"Value of cfg.data: {cfg.data}")
