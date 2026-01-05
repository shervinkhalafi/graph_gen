"""Hydra entry point for generative graph modeling experiments.

Run experiments with:
    python -m tmgg.experiments.generative.runner

Override configuration with Hydra:
    python -m tmgg.experiments.generative.runner model.model_type=gnn data.num_nodes=100
"""

import hydra
from omegaconf import DictConfig

from tmgg.experiment_utils.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../../exp_configs",
    config_name="base_config_generative",
)
def main(cfg: DictConfig) -> None:
    """Run generative graph modeling experiment.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    run_experiment(cfg)


if __name__ == "__main__":
    main()
