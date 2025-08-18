"""Runner for grid search experiments."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tmgg.experiment_utils.run_experiment import run_experiment

# Navigate to the centralized config location
TMGG_ROOT = Path(__file__).parent.parent  # Navigate to tmgg/src/tmgg
CONFIG_PATH = str(TMGG_ROOT / "exp_configs")


@hydra.main(  # pyright: ignore[reportAny]
    version_base="1.3", 
    config_path=CONFIG_PATH, 
    config_name="grid_search_base"
)
def main(config: DictConfig):
    """
    Main training function for grid search.

    Args:
        config: Hydra configuration

    Returns:
        Dictionary with training results
    """
    return run_experiment(config)


if __name__ == "__main__":
    main()