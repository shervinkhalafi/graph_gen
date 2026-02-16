import sys

sys.path.append("src")

from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

# Clear any existing Hydra instance
GlobalHydra.instance().clear()

# Test 1: Try with full path
with initialize(config_path="src/tmgg/exp_configs", version_base="1.3"):
    try:
        cfg = compose(
            config_name="base_config_attention",
            overrides=[
                "data=base/data/legacy_match",
            ],
        )
        print("Test 1 - Full path: Success")
        print(f"Type of cfg.data: {type(cfg.data)}")
        if hasattr(cfg.data, "_target_"):
            print(f"Data _target_: {cfg.data._target_}")
    except Exception as e:
        print(f"Test 1 - Full path failed: {e}")

GlobalHydra.instance().clear()

# Test 2: Try overriding the defaults
with initialize(config_path="src/tmgg/exp_configs", version_base="1.3"):
    try:
        cfg = compose(
            config_name="base_config_attention",
            overrides=[
                "defaults.1=base/data/legacy_match@data",  # Override the second item in defaults list
            ],
        )
        print("\nTest 2 - Override defaults: Success")
        print(f"Type of cfg.data: {type(cfg.data)}")
        if hasattr(cfg.data, "_target_"):
            print(f"Data _target_: {cfg.data._target_}")
    except Exception as e:
        print(f"Test 2 - Override defaults failed: {e}")
