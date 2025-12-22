# SLURM Execution

TMGG supports execution on SLURM-managed HPC clusters using Ray's distributed computing capabilities. This document explains the architecture, configuration, and troubleshooting.

## Architecture

```
Hydra CLI / TmggLauncher
    │
    └── SlurmRunner
            │
            ├── Generates sbatch script
            ├── Submits via sbatch
            │
            └── SLURM Node Allocation
                    │
                    └── Ray symmetric-run
                            │
                            ├── Head node: Ray head + experiment
                            └── Worker nodes: Ray workers
```

The `SlurmRunner` generates sbatch scripts that use Ray's `symmetric-run` command (available in Ray 2.49+) to start a Ray cluster across SLURM-allocated nodes. This approach combines SLURM's resource management with Ray's distributed execution.

## Prerequisites

Before using SLURM execution, you need:

1. **SLURM cluster access** with standard commands (`sbatch`, `squeue`, `sacct`, `scancel`)
2. **Ray 2.49+** installed on the cluster (the `symmetric-run` feature requires this version)
3. **Python environment** accessible on compute nodes (via modules or conda)
4. **Network connectivity** between nodes for Ray's distributed communication

Install the Ray dependency group:

```bash
uv add "tmgg[ray]"
```

## Configuration

### Hydra Launcher

The simplest way to use SLURM is via the Hydra launcher configuration:

```bash
tmgg-experiment +stage=stage1_poc hydra/launcher=tmgg_slurm
```

This uses the default SLURM configuration from `src/tmgg/exp_configs/hydra/launcher/tmgg_slurm.yaml`.

### Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `slurm_partition` | `gpu` | SLURM partition to submit to |
| `slurm_nodes` | `4` | Number of nodes to allocate |
| `slurm_cpus_per_task` | `8` | CPUs per node |
| `slurm_gpus_per_task` | `1` | GPUs per node |
| `slurm_time_limit` | `"08:00:00"` | Job time limit (HH:MM:SS) |
| `slurm_mem_per_cpu` | `"4GB"` | Memory per CPU |
| `slurm_setup_commands` | `[]` | Environment setup commands |
| `timeout_seconds` | `28800` | Python-level timeout (should match time_limit) |

### Overriding Defaults

Override parameters on the command line:

```bash
tmgg-experiment +stage=stage1_poc hydra/launcher=tmgg_slurm \
    hydra.launcher.slurm_partition=batch \
    hydra.launcher.slurm_nodes=8 \
    hydra.launcher.slurm_time_limit="12:00:00"
```

Or create a custom launcher config:

```yaml
# exp_configs/hydra/launcher/my_cluster.yaml
# @package hydra.launcher
_target_: tmgg.hydra_plugins.tmgg_launcher.TmggLauncher

use_slurm: true
slurm_partition: ml-gpu
slurm_nodes: 2
slurm_cpus_per_task: 16
slurm_gpus_per_task: 4
slurm_time_limit: "04:00:00"
slurm_mem_per_cpu: "8GB"

slurm_setup_commands:
  - "module load cuda/12.1"
  - "module load anaconda3"
  - "source activate tmgg"
```

### Environment Setup

Most clusters require module loads or conda activation before experiments run. Configure these via `slurm_setup_commands`:

```yaml
slurm_setup_commands:
  - "module load cuda/12.1"
  - "module load cudnn/8.6"
  - "source ~/.bashrc && conda activate tmgg"
```

These commands execute at the start of each sbatch script before Ray initializes.

## Programmatic Usage

For more control, use `SlurmRunner` directly:

```python
from tmgg.experiment_utils.cloud import SlurmRunner, SlurmConfig

config = SlurmConfig(
    partition="gpu",
    nodes=4,
    cpus_per_task=8,
    gpus_per_task=1,
    time_limit="04:00:00",
    mem_per_cpu="4GB",
    setup_commands=[
        "module load cuda/12.1",
        "conda activate tmgg",
    ],
)

runner = SlurmRunner(slurm_config=config)
result = runner.run_experiment(experiment_config)
```

### Spawning Without Blocking

For long-running jobs, spawn experiments without blocking:

```python
spawned = runner.spawn_experiment(config, timeout_seconds=14400)
print(f"Submitted SLURM job {spawned.job_id}")

# Check status later
status = runner.get_status(spawned.run_id)  # "pending", "running", "completed", "failed"

# Or wait for completion
result = runner.get_result(spawned, config)
```

### Sweeps

Run multiple configurations as separate SLURM jobs:

```python
configs = [config1, config2, config3]
results = runner.run_sweep(configs, parallelism=len(configs))
```

Each configuration becomes a separate sbatch submission, allowing SLURM's scheduler to optimize placement.

## Generated Script Structure

The runner generates sbatch scripts with this structure:

```bash
#!/bin/bash
#SBATCH --job-name=tmgg-experiment
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=4GB
#SBATCH --time=08:00:00
#SBATCH --partition=gpu
#SBATCH --output=slurm_outputs/<run_id>.out
#SBATCH --error=slurm_outputs/<run_id>.err

# Environment setup (from slurm_setup_commands)
module load cuda/12.1
source activate tmgg

# Get head node IP for Ray cluster
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
ip_head=$head_node:6379
export ip_head

# Start Ray cluster using symmetric-run
srun --nodes="$SLURM_JOB_NUM_NODES" --ntasks="$SLURM_JOB_NUM_NODES" \
    ray symmetric-run \
    --address "$ip_head" \
    --min-nodes "$SLURM_JOB_NUM_NODES" \
    --num-cpus="${SLURM_CPUS_PER_TASK}" \
    --num-gpus="${SLURM_GPUS_PER_TASK}" \
    -- \
    python -u -c "... execute_task() call ..."
```

## Multi-Tenant Considerations

On shared clusters, Ray's default ports may conflict with other users' jobs. Configure port ranges to avoid collisions:

```python
config = SlurmConfig(
    # ...
    ray_port=6380,              # Different users use different base ports
    min_worker_port=20002,
    max_worker_port=29999,
)
```

Coordinate port assignments with your cluster's usage policy or use a unique base port derived from your user ID.

## Monitoring Jobs

### Check Job Status

```bash
# List your pending/running jobs
squeue -u $USER

# Check specific job
squeue -j <job_id>

# Detailed job info
scontrol show job <job_id>
```

### View Output

```bash
# Follow output in real-time
tail -f slurm_outputs/<run_id>.out

# Check for errors
cat slurm_outputs/<run_id>.err
```

### Cancel Jobs

```bash
# Cancel single job
scancel <job_id>

# Cancel all your jobs
scancel -u $USER
```

### Job History

```bash
# View completed job details
sacct -j <job_id> --format=JobID,JobName,State,ExitCode,Elapsed

# View all your recent jobs
sacct -u $USER --starttime=$(date -d "7 days ago" +%Y-%m-%d)
```

## Result Parsing

The runner parses experiment results from the job output file. Look for the `TMGG_RESULT:` line:

```
... Ray initialization output ...
TMGG_RESULT:{"run_id":"abc123","metrics":{"val_loss":0.123},"status":"completed",...}
```

If parsing fails, check the output file for Python exceptions or Ray errors that prevented the result line from being written.

## Troubleshooting

### Job Stays in PENDING

Check partition availability and your allocation limits:

```bash
# View partition status
sinfo -p <partition>

# Check your allocation limits
sacctmgr show assoc user=$USER format=Account,GrpTRES,MaxTRES
```

Common causes:
- Requested resources exceed limits
- Partition is full or in maintenance
- Account has no remaining allocation

### Ray Fails to Connect

Ensure nodes can communicate on the Ray port range. Check:

```bash
# Verify nodes are allocated
scontrol show hostnames "$SLURM_JOB_NODELIST"

# Test connectivity from within a job
srun --nodes=1 ping -c 1 <other_node>
```

Firewall rules may block Ray's ports. Consult your cluster administrator.

### CUDA Not Found

Add the appropriate module load to `slurm_setup_commands`:

```yaml
slurm_setup_commands:
  - "module load cuda/12.1"
  - "module load cudnn/8.6"
```

List available modules with `module avail cuda`.

### Python Environment Not Found

Ensure the conda/virtualenv activation runs correctly:

```yaml
slurm_setup_commands:
  - "source ~/.bashrc"  # May be needed for conda init
  - "conda activate tmgg"
```

Test manually:

```bash
srun --partition=<partition> --time=00:05:00 bash -c "source ~/.bashrc && conda activate tmgg && python --version"
```

### Out of Memory

Increase `slurm_mem_per_cpu` or reduce batch sizes in your experiment config:

```yaml
slurm_mem_per_cpu: "8GB"  # or "16GB"
```

Monitor memory usage with:

```bash
sacct -j <job_id> --format=JobID,MaxRSS,MaxVMSize
```

### Job Timeout

The `slurm_time_limit` must accommodate both Ray startup and experiment execution. For large clusters, Ray initialization takes longer:

```yaml
# For 4+ node jobs, add 10-15 minutes buffer
slurm_time_limit: "04:15:00"
timeout_seconds: 14400  # 4 hours in seconds
```

### Result Not Parsed

If the job completes but the result is not found:

1. Check the output file for the `TMGG_RESULT:` line
2. Look for Python exceptions in stderr
3. Verify the experiment ran to completion (check for "Experiment completed" log)

The runner searches for `TMGG_RESULT:` followed by JSON. Ensure no logging output corrupts this line.

## Comparison with Other Backends

| Feature | LocalRunner | ModalRunner | SlurmRunner |
|---------|-------------|-------------|-------------|
| Setup | None | Modal account | Cluster access |
| GPU access | Local only | On-demand cloud | Cluster allocation |
| Scaling | Single machine | Auto-scaling | Fixed allocation |
| Cost | Free | Pay-per-use | Allocation-based |
| Queue time | None | ~seconds | Varies |
| Best for | Development | Production sweeps | HPC environments |

## References

- [Ray SLURM deployment guide](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html)
- [Ray symmetric-run documentation](https://docs.ray.io/en/latest/cluster/vms/user-guides/community/slurm.html#symmetric-run-for-ray-cluster-deployment)
- [SLURM documentation](https://slurm.schedmd.com/documentation.html)
