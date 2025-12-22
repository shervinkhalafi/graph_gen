"""Configuration dataclass for TmggLauncher.

Defines the configuration schema and registers it with Hydra's ConfigStore.
"""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class TmggLauncherConf:
    """Configuration for the TMGG experiment launcher.

    Attributes
    ----------
    _target_
        Fully qualified class name for Hydra instantiation.
    use_modal
        If True, dispatch jobs to Modal cloud. Default False (local).
    use_ray
        If True, dispatch jobs to Ray cluster. Default False.
    use_slurm
        If True, dispatch jobs to SLURM cluster via Ray symmetric-run.
    gpu_type
        GPU tier to request ('debug', 'standard', 'fast', 'multi').
    parallelism
        Maximum concurrent experiments for sweeps.
    timeout_seconds
        Maximum runtime per experiment before termination.
    slurm_partition
        SLURM partition to submit jobs to.
    slurm_nodes
        Number of nodes to request for SLURM jobs.
    slurm_cpus_per_task
        CPUs per node for SLURM jobs.
    slurm_gpus_per_task
        GPUs per node for SLURM jobs.
    slurm_time_limit
        Job time limit for SLURM jobs (HH:MM:SS format).
    slurm_mem_per_cpu
        Memory per CPU for SLURM jobs (e.g., "4GB").
    slurm_setup_commands
        Shell commands to run before experiment (e.g., module loads).
    """

    _target_: str = field(
        default="tmgg.hydra_plugins.tmgg_launcher.TmggLauncher",
        metadata={"omegaconf_ignore": True},
    )
    use_modal: bool = False
    use_ray: bool = False
    use_slurm: bool = False
    gpu_type: str = "debug"
    parallelism: int = 4
    timeout_seconds: int = 3600
    # SLURM-specific settings
    slurm_partition: str = "gpu"
    slurm_nodes: int = 1
    slurm_cpus_per_task: int = 4
    slurm_gpus_per_task: int = 1
    slurm_time_limit: str = "04:00:00"
    slurm_mem_per_cpu: str = "4GB"
    slurm_setup_commands: list[str] = field(default_factory=list)


def register_configs() -> None:
    """Register TmggLauncher configuration with Hydra's ConfigStore."""
    cs = ConfigStore.instance()
    cs.store(
        group="hydra/launcher",
        name="tmgg",
        node=TmggLauncherConf,
    )


# Auto-register on import
register_configs()
