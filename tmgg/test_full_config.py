import sys

sys.path.append("src")

import hydra
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

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
        ],
    )

    print("Data config _target_:", cfg.data._target_)
    print("Model config _target_:", cfg.model._target_)

    # Try to instantiate data module
    dm = hydra.utils.instantiate(cfg.data)
    print("\nData module type:", type(dm))

    # Try to instantiate model
    model = hydra.utils.instantiate(cfg.model)
    print("Model type:", type(model))

    # Setup data
    dm.prepare_data()
    dm.setup("fit")

    # Get a batch
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    print("\nBatch type:", type(batch))
    print("Batch shape:", batch.shape)
