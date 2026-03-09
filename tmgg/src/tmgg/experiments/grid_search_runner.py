"""Runner for grid search experiments."""

from pathlib import Path

import hydra
from omegaconf import DictConfig

from tmgg.experiments._shared_utils.orchestration.run_experiment import run_experiment

# Config directory is a sibling under experiments/
CONFIG_PATH = str(Path(__file__).parent / "exp_configs")


@hydra.main(  # pyright: ignore[reportAny]
    version_base=None, config_path=CONFIG_PATH, config_name="grid_search_base"
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
