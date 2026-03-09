"""Hydra entry point for baseline denoising experiments.

Run experiments with::

    tmgg-baseline
    tmgg-baseline model=baselines/mlp model.hidden_dim=512
"""

import hydra
from omegaconf import DictConfig

from tmgg.experiments._shared_utils.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_baseline",
)
def main(cfg: DictConfig) -> None:
    """Run baseline denoising experiment.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    run_experiment(cfg)


if __name__ == "__main__":
    main()
