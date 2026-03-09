"""Hydra entry point for DiGress Transformer-based denoising experiments.

Run experiments with::

    tmgg-digress
    tmgg-digress model.n_layers=6 data.num_nodes=100
"""

import hydra
from omegaconf import DictConfig

from tmgg.experiments._shared_utils.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_digress",
)
def main(cfg: DictConfig) -> None:
    """Run DiGress denoising experiment.

    Parameters
    ----------
    cfg
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    run_experiment(cfg)


if __name__ == "__main__":
    main()
