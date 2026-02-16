# RayRunner Test Rationale

## Purpose

The RayRunner tests verify the local distributed execution backend that uses Ray for parallel experiment runs. Since Ray is an optional dependency, tests mock the Ray module to validate orchestration logic without requiring Ray to be installed.

## Test Categories

### TestRayRunnerInit

**Invariant:** RayRunner correctly initializes the Ray cluster when needed.

- `test_init_auto_initializes_ray`: Verifies Ray cluster starts if not connected
- `test_init_skips_if_ray_connected`: Verifies Ray is not re-initialized if already running
- `test_init_with_gpu_config`: Verifies GPU/CPU configuration is stored correctly

### TestRaySpawnedTask

**Invariant:** RaySpawnedTask dataclass correctly stores task metadata.

- `test_spawned_task_creation`: Verifies run_id and object_ref are stored

### TestRayRunnerSpawn

**Invariant:** Spawn methods create non-blocking tasks with proper tracking.

- `test_spawn_experiment_returns_spawned_task`: Verifies spawn returns valid RaySpawnedTask
- `test_spawn_experiment_tracks_active_runs`: Verifies task is tracked in _active_runs
- `test_spawn_sweep_spawns_all_configs`: Verifies all configs are spawned with unique run_ids

### TestRayRunnerBlocking

**Invariant:** Blocking methods wait for results and return ExperimentResult.

- `test_run_experiment_returns_result`: Verifies blocking execution returns ExperimentResult
- `test_run_experiment_cleans_up_tracking`: Verifies completed tasks are removed from tracking
- `test_run_sweep_returns_all_results`: Verifies all results are collected

### TestRayRunnerStatus

**Invariant:** Status and control methods accurately report task state.

- `test_get_status_unknown_for_missing_run`: Unknown run returns "unknown"
- `test_get_status_running_if_not_ready`: Running task returns "running"
- `test_get_status_completed_if_ready`: Completed task returns status from result
- `test_cancel_returns_false_for_missing_run`: Missing run returns False
- `test_cancel_removes_from_tracking`: Cancelled tasks are removed from tracking
- `test_shutdown_calls_ray_shutdown`: Shutdown properly cleans up Ray

### TestRayNotInstalled

**Invariant:** Proper error when Ray is not installed.

- `test_ensure_ray_available_raises_if_not_installed`: ImportError raised with helpful message

## Mocking Strategy

All tests mock the `ray` module at import time. The mock provides:
- `is_initialized()` - controls whether Ray appears to be running
- `init()` - mock initialization
- `get()` - returns mock task results
- `wait()` - simulates task completion checking
- `cancel()` - mock task cancellation
- `remote` decorator - creates mock remote functions

This allows testing the RayRunner orchestration logic without requiring Ray installation.

## Related Components

- `tmgg.experiment_utils.task`: TaskInput/TaskOutput abstraction
- `tmgg.experiment_utils.cloud.base`: CloudRunner ABC, ExperimentResult
- `tmgg.modal.runner`: ModalRunner (parallel cloud implementation)
