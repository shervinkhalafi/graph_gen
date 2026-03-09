"""Hydra entry point for spectral denoising experiments.

Run experiments with:
    python -m tmgg.experiments.spectral_arch_denoising.runner

Override configuration with Hydra:
    python -m tmgg.experiments.spectral_arch_denoising.runner model.k=16 data.num_nodes=100
"""

import hydra
from omegaconf import DictConfig

from tmgg.experiments._shared_utils.orchestration.run_experiment import run_experiment


@hydra.main(
    version_base=None,
    config_path="../exp_configs",
    config_name="base_config_spectral_arch",
)
def main(cfg: DictConfig) -> None:
    """Run spectral denoising experiment.

    Parameters
    ----------
    cfg : DictConfig
        Hydra configuration containing model, data, trainer, and logger settings.
    """
    run_experiment(cfg)


if __name__ == "__main__":
    main()
