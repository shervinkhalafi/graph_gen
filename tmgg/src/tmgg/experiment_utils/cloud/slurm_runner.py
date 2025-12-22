"""SLURM-based CloudRunner for HPC cluster execution.

Uses Ray's symmetric-run to start a Ray cluster on SLURM-allocated nodes,
then executes experiments via the unified task abstraction. This leverages
Ray's distributed execution while SLURM handles resource allocation and
job scheduling.

Requires Ray 2.49+ for the symmetric-run command.

Example
-------
    >>> from tmgg.experiment_utils.cloud.slurm_runner import SlurmRunner, SlurmConfig
    >>> config = SlurmConfig(partition="gpu", nodes=4, gpus_per_task=1)
    >>> runner = SlurmRunner(slurm_config=config)
    >>> result = runner.run_experiment(experiment_config)

Note
----
Requires SLURM cluster access with sbatch, squeue, sacct, and scancel commands.
Python environment with TMGG must be accessible on compute nodes.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, cast, override

from omegaconf import DictConfig, OmegaConf

from tmgg.experiment_utils.cloud.base import CloudRunner, ExperimentResult, SpawnedTask
from tmgg.experiment_utils.task import TaskInput, prepare_config_for_remote

logger = logging.getLogger(__name__)


@dataclass
class SlurmConfig:
    """SLURM job configuration.

    Attributes
    ----------
    partition
        SLURM partition to submit jobs to.
    nodes
        Number of nodes to request.
    cpus_per_task
        CPUs per node (maps to --cpus-per-task).
    gpus_per_task
        GPUs per node (maps to --gpus-per-task).
    mem_per_cpu
        Memory per CPU (e.g., "4GB").
    time_limit
        Job time limit in HH:MM:SS format.
    job_name
        Name prefix for SLURM jobs.
    ray_port
        Port for Ray head node (default 6379).
    min_worker_port
        Minimum port for Ray worker connections.
    max_worker_port
        Maximum port for Ray worker connections.
    setup_commands
        Shell commands to run before starting Ray (e.g., module loads, conda activate).
    exclusive
        Request exclusive node access (default True).
    """

    partition: str = "gpu"
    nodes: int = 1
    cpus_per_task: int = 4
    gpus_per_task: int = 1
    mem_per_cpu: str = "4GB"
    time_limit: str = "04:00:00"
    job_name: str = "tmgg-experiment"
    ray_port: int = 6379
    min_worker_port: int = 10002
    max_worker_port: int = 19999
    setup_commands: list[str] = field(default_factory=list)
    exclusive: bool = True


@dataclass
class SlurmSpawnedTask:
    """Handle for a spawned SLURM job.

    Attributes
    ----------
    run_id
        Unique identifier for the experiment.
    job_id
        SLURM job ID returned by sbatch.
    """

    run_id: str
    job_id: str


class SlurmRunner(CloudRunner):
    """SLURM-based experiment runner using Ray symmetric-run.

    Generates sbatch scripts that use Ray's symmetric-run command to
    initialize a Ray cluster on SLURM-allocated nodes, then executes
    experiments via the unified task abstraction.

    Attributes
    ----------
    slurm_config
        SLURM job configuration.
    output_dir
        Directory for sbatch scripts and output files.
    poll_interval
        Seconds between status checks when waiting for completion.
    """

    slurm_config: SlurmConfig
    output_dir: Path
    poll_interval: float

    def __init__(
        self,
        slurm_config: SlurmConfig | None = None,
        output_dir: Path | None = None,
        poll_interval: float = 30.0,
    ):
        """Initialize SLURM runner.

        Parameters
        ----------
        slurm_config
            SLURM job configuration. Uses defaults if not provided.
        output_dir
            Directory for sbatch scripts and output files.
            Defaults to ./slurm_outputs.
        poll_interval
            Seconds between status checks when waiting for job completion.
        """
        self.slurm_config = slurm_config or SlurmConfig()
        self.output_dir = output_dir or Path("slurm_outputs")
        self.poll_interval = poll_interval
        self._active_jobs: dict[str, SlurmSpawnedTask] = {}

    def _create_task_input(
        self,
        config: DictConfig,
        timeout_seconds: int,
    ) -> TaskInput:
        """Create TaskInput from Hydra config.

        Parameters
        ----------
        config
            Hydra configuration.
        timeout_seconds
            Timeout for the task.

        Returns
        -------
        TaskInput
            Serializable task input for remote execution.
        """
        config_dict = prepare_config_for_remote(config)
        run_id: str = str(config_dict["run_id"])

        return TaskInput(
            config=config_dict,
            run_id=run_id,
            gpu_tier="slurm",  # SLURM handles GPU allocation
            timeout_seconds=timeout_seconds,
        )

    def _generate_sbatch_script(
        self,
        task_input: TaskInput,
    ) -> str:
        """Generate sbatch script content.

        Parameters
        ----------
        task_input
            Task input with config and run_id.

        Returns
        -------
        str
            Complete sbatch script content.
        """
        cfg = self.slurm_config
        output_dir = self.output_dir.absolute()

        # Escape JSON for embedding in shell script
        task_json = json.dumps(asdict(task_input)).replace("'", "'\"'\"'")

        # Build setup commands section
        setup_section = "\n".join(cfg.setup_commands) if cfg.setup_commands else ""

        # Build exclusive directive if requested
        exclusive_directive = "#SBATCH --exclusive" if cfg.exclusive else ""

        return f"""#!/bin/bash
#SBATCH --job-name={cfg.job_name}-{task_input.run_id[:8]}
#SBATCH --nodes={cfg.nodes}
{exclusive_directive}
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task={cfg.cpus_per_task}
#SBATCH --gpus-per-task={cfg.gpus_per_task}
#SBATCH --mem-per-cpu={cfg.mem_per_cpu}
#SBATCH --time={cfg.time_limit}
#SBATCH --partition={cfg.partition}
#SBATCH --output={output_dir}/{task_input.run_id}.out
#SBATCH --error={output_dir}/{task_input.run_id}.err

# Environment setup
{setup_section}

# Get head node IP address
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${{nodes_array[0]}}
ip_head=$head_node:{cfg.ray_port}
export ip_head
echo "Head node: $head_node"
echo "IP Head: $ip_head"

# Start Ray cluster using symmetric-run and execute task
# Ray symmetric-run starts Ray on all nodes and runs the entrypoint ONLY on head
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \\
    ray symmetric-run \\
    --address "$ip_head" \\
    --min-nodes "$SLURM_JOB_NUM_NODES" \\
    --num-cpus="${{SLURM_CPUS_PER_TASK}}" \\
    --num-gpus="${{SLURM_GPUS_PER_TASK}}" \\
    -- \\
    python -u -c '
import json
from dataclasses import asdict
from tmgg.experiment_utils.task import TaskInput, execute_task

task_dict = json.loads('"'"'{task_json}'"'"')
task = TaskInput(**task_dict)
output = execute_task(task, get_storage=None)
print("TMGG_RESULT:" + json.dumps(asdict(output)))
'

echo "SLURM job completed"
"""

    @override
    def spawn_experiment(
        self,
        config: DictConfig,
        gpu_type: str = "debug",
        timeout_seconds: int = 3600,
    ) -> SlurmSpawnedTask:
        """Submit experiment to SLURM without waiting.

        Parameters
        ----------
        config
            Hydra configuration.
        gpu_type
            Ignored for SLURM (GPU allocation via slurm_config).
        timeout_seconds
            Timeout for the task.

        Returns
        -------
        SlurmSpawnedTask
            Handle with run_id and SLURM job_id.
        """
        task_input = self._create_task_input(config, timeout_seconds)

        # Ensure output directory exists
        _ = self.output_dir.mkdir(parents=True, exist_ok=True)

        # Generate and write sbatch script
        script_path = self.output_dir / f"{task_input.run_id}.sbatch"
        script_content = self._generate_sbatch_script(task_input)
        _ = script_path.write_text(script_content)

        # Submit via sbatch
        result = subprocess.run(
            ["sbatch", str(script_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        # Parse job ID from "Submitted batch job 123456"
        job_id = result.stdout.strip().split()[-1]

        spawned = SlurmSpawnedTask(run_id=task_input.run_id, job_id=job_id)
        self._active_jobs[task_input.run_id] = spawned

        logger.info(f"Submitted SLURM job {job_id} for experiment {task_input.run_id}")
        return spawned

    @override
    def spawn_sweep(
        self,
        configs: list[DictConfig],
        gpu_type: str = "debug",
        timeout_seconds: int = 3600,
    ) -> list[SpawnedTask]:
        """Spawn multiple experiments without waiting.

        Parameters
        ----------
        configs
            List of configurations.
        gpu_type
            Ignored for SLURM.
        timeout_seconds
            Timeout per experiment.

        Returns
        -------
        list[SpawnedTask]
            Handles for all spawned jobs.
        """
        spawned: list[SpawnedTask] = [
            self.spawn_experiment(config, gpu_type, timeout_seconds)
            for config in configs
        ]
        return spawned

    @override
    def run_experiment(
        self,
        config: DictConfig,
        gpu_type: str = "debug",
        timeout_seconds: int = 3600,
    ) -> ExperimentResult:
        """Submit experiment and wait for completion.

        Parameters
        ----------
        config
            Hydra configuration.
        gpu_type
            Ignored for SLURM.
        timeout_seconds
            Timeout for the task.

        Returns
        -------
        ExperimentResult
            Result of the experiment.
        """
        spawned = self.spawn_experiment(config, gpu_type, timeout_seconds)
        return self._wait_for_job(spawned, config)

    @override
    def run_sweep(
        self,
        configs: list[DictConfig],
        gpu_type: str = "debug",
        parallelism: int = 4,
        timeout_seconds: int = 3600,
    ) -> list[ExperimentResult]:
        """Submit multiple experiments and wait for all.

        Parameters
        ----------
        configs
            List of configurations.
        gpu_type
            Ignored for SLURM.
        parallelism
            Ignored (SLURM handles scheduling).
        timeout_seconds
            Timeout per experiment.

        Returns
        -------
        list[ExperimentResult]
            Results from all experiments.
        """
        # Submit all jobs (use spawn_experiment directly to get proper types)
        spawned_tasks: list[SlurmSpawnedTask] = [
            self.spawn_experiment(config, gpu_type, timeout_seconds)
            for config in configs
        ]

        # Wait for each job
        results: list[ExperimentResult] = []
        for spawned, config in zip(spawned_tasks, configs, strict=True):
            result = self._wait_for_job(spawned, config)
            results.append(result)

        return results

    def _wait_for_job(
        self,
        spawned: SlurmSpawnedTask,
        config: DictConfig,
    ) -> ExperimentResult:
        """Poll SLURM until job completes, then parse output.

        Parameters
        ----------
        spawned
            Spawned task handle.
        config
            Original Hydra config.

        Returns
        -------
        ExperimentResult
            Parsed result from job output.
        """
        start_time = time.time()

        while True:
            status = self.get_status(spawned.run_id)
            if status in ("completed", "failed", "unknown"):
                break
            logger.debug(f"Job {spawned.job_id} status: {status}")
            time.sleep(self.poll_interval)

        duration = time.time() - start_time
        return self._parse_result(spawned, config, duration)

    def _parse_result(
        self,
        spawned: SlurmSpawnedTask,
        config: DictConfig,
        duration: float,
    ) -> ExperimentResult:
        """Parse result from SLURM output file.

        Looks for a line starting with TMGG_RESULT: followed by JSON.

        Parameters
        ----------
        spawned
            Spawned task handle.
        config
            Original Hydra config.
        duration
            Wall-clock time for the job.

        Returns
        -------
        ExperimentResult
            Parsed experiment result.
        """
        config_dict = OmegaConf.to_container(config, resolve=True)
        if not isinstance(config_dict, dict):
            raise TypeError(
                f"Expected dict from OmegaConf.to_container, got {type(config_dict)}"
            )

        output_file = self.output_dir / f"{spawned.run_id}.out"
        error_file = self.output_dir / f"{spawned.run_id}.err"

        # Default to failed result
        result = ExperimentResult(
            run_id=spawned.run_id,
            config=cast(dict[str, Any], config_dict),
            metrics={},
            status="failed",
            duration_seconds=duration,
        )

        # Try to parse output file
        if output_file.exists():
            content = output_file.read_text()
            for line in content.splitlines():
                if line.startswith("TMGG_RESULT:"):
                    try:
                        result_json = line[len("TMGG_RESULT:") :]
                        output_dict: dict[str, object] = json.loads(result_json)
                        metrics_val = output_dict.get("metrics", {})
                        metrics = (
                            cast(dict[str, float], metrics_val)
                            if isinstance(metrics_val, dict)
                            else {}
                        )
                        checkpoint_val = output_dict.get("checkpoint_uri")
                        checkpoint_path = (
                            str(checkpoint_val) if checkpoint_val is not None else None
                        )
                        status_val = output_dict.get("status", "completed")
                        status = str(status_val) if status_val else "completed"
                        error_val = output_dict.get("error_message")
                        error_message = (
                            str(error_val) if error_val is not None else None
                        )
                        duration_val = output_dict.get("duration_seconds", duration)
                        duration_secs = (
                            float(duration_val)
                            if isinstance(duration_val, int | float)
                            else duration
                        )
                        result = ExperimentResult(
                            run_id=spawned.run_id,
                            config=cast(dict[str, Any], config_dict),
                            metrics=metrics,
                            checkpoint_path=checkpoint_path,
                            status=status,
                            error_message=error_message,
                            duration_seconds=duration_secs,
                        )
                        break
                    except json.JSONDecodeError as e:
                        result.error_message = f"Failed to parse result JSON: {e}"

        # Check error file for additional context
        if result.status == "failed" and error_file.exists():
            error_content = error_file.read_text()
            if error_content.strip():
                # Truncate if too long
                if len(error_content) > 1000:
                    error_content = error_content[-1000:]
                result.error_message = (
                    result.error_message or ""
                ) + f"\nSTDERR:\n{error_content}"

        # Clean up tracking
        if spawned.run_id in self._active_jobs:
            del self._active_jobs[spawned.run_id]

        return result

    @override
    def get_status(self, run_id: str) -> str:
        """Query SLURM job status.

        Checks squeue for running jobs, then sacct for completed jobs.

        Parameters
        ----------
        run_id
            Run identifier.

        Returns
        -------
        str
            One of 'pending', 'running', 'completed', 'failed', 'unknown'.
        """
        if run_id not in self._active_jobs:
            return "unknown"

        job_id = self._active_jobs[run_id].job_id

        # Check squeue first (for running/pending jobs)
        try:
            result = subprocess.run(
                ["squeue", "-j", job_id, "-h", "-o", "%T"],
                capture_output=True,
                text=True,
                check=False,
            )
            state = result.stdout.strip()
            if state:
                if state in ("PENDING", "CONFIGURING"):
                    return "pending"
                elif state in ("RUNNING", "COMPLETING"):
                    return "running"
                elif state in ("COMPLETED",):
                    return "completed"
                elif state in ("FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL"):
                    return "failed"
                return "running"  # Default for other states
        except FileNotFoundError:
            # squeue not available
            pass

        # Check sacct for completed jobs
        try:
            result = subprocess.run(
                ["sacct", "-j", job_id, "-n", "-X", "-o", "State"],
                capture_output=True,
                text=True,
                check=False,
            )
            state = result.stdout.strip()
            if "COMPLETED" in state:
                return "completed"
            elif state:
                return "failed"
        except FileNotFoundError:
            # sacct not available
            pass

        return "unknown"

    @override
    def cancel(self, run_id: str) -> bool:
        """Cancel a running SLURM job.

        Parameters
        ----------
        run_id
            Run identifier.

        Returns
        -------
        bool
            True if cancellation was requested.
        """
        if run_id not in self._active_jobs:
            return False

        job_id = self._active_jobs[run_id].job_id
        try:
            _ = subprocess.run(["scancel", job_id], check=False)
            logger.info(f"Cancelled SLURM job {job_id}")
            del self._active_jobs[run_id]
            return True
        except FileNotFoundError:
            # scancel not available
            return False

    @property
    @override
    def supports_spawn(self) -> bool:
        """SlurmRunner supports detached (spawn) execution."""
        return True
