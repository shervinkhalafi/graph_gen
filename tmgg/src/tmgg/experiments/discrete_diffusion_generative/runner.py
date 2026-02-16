"""Hydra entry point for discrete diffusion graph generation experiments.

Run experiments with::

    tmgg-discrete-gen
    tmgg-discrete-gen model.diffusion_steps=100 data.num_nodes=12
"""

import hydra
from omegaconf import DictConfig

from tmgg.experiment_utils.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../../exp_configs",
    config_name="base_config_discrete_diffusion_generative",
)
def main(cfg: DictConfig) -> None:
    """Run discrete diffusion graph generation experiment.

    Parameters
    ----------
    cfg
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    run_experiment(cfg, skip_final_eval=True)


if __name__ == "__main__":
    main()
