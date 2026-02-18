"""Integration tests for all experiment CLI runners.

Test rationale:
    These tests verify that each CLI runner can execute a brief training run
    without errors. They catch configuration issues, import errors, and
    instantiation failures before expensive Modal deployments.

Assumptions:
    - Tests run on CPU with minimal data (8 train samples, 2 steps)
    - Each test should complete in under 120 seconds
    - Tests use subprocess isolation to avoid Hydra state conflicts

Invariants:
    - Exit code 0 (no Python exceptions)
    - Output directory created
    - No exception tracebacks in stderr

The CLI runners tested cover all experiment types:
    - Experiment runners: digress, gnn, gnn-transformer, spectral-arch, grid-search
    - Unified experiment runner: tmgg-experiment +stage=<stage_name>
"""

import subprocess
from pathlib import Path

import pytest

from tests.integration_utils import (
    assert_training_success,
    get_quick_training_overrides,
    run_cli_command,
)

# All CLI runners to test with their expected base config and DataModule category.
# The data_module category determines which data-reduction overrides are safe.
EXPERIMENT_RUNNERS = [
    ("tmgg-digress", "base_config_digress", "denoising"),
    ("tmgg-gnn", "base_config_gnn", "denoising"),
    ("tmgg-gnn-transformer", "base_config_gnn_transformer", "denoising"),
    ("tmgg-spectral-arch", "base_config_spectral_arch", "denoising"),
    ("tmgg-grid-search", "grid_search_base", "denoising"),
    ("tmgg-discrete-gen", "base_config_discrete_diffusion_generative", "generative"),
]

# Stage configs tested via unified tmgg-experiment CLI, with DataModule category.
# Most stages use SingleGraphDataModule; stage1_sanity uses multi-graph GraphDataModule.
STAGE_CONFIGS = [
    ("stage1_poc", "single_graph"),
    ("stage1_sanity", "denoising"),
    ("stage2_validation", "single_graph"),
    ("stage3_diversity", "single_graph"),
    ("stage4_benchmarks", "single_graph"),
    ("stage5_full", "single_graph"),
]


@pytest.mark.integration
@pytest.mark.slow
class TestExperimentRunners:
    """Tests for experiment-type CLI runners (attention, gnn, etc.)."""

    @pytest.mark.parametrize("runner_cmd,base_config,dm_type", EXPERIMENT_RUNNERS)
    def test_runner_executes_brief_training(
        self, runner_cmd: str, base_config: str, dm_type: str, tmp_path: Path
    ) -> None:
        """Verify runner completes 2 training steps without error.

        This test invokes the runner via subprocess with overrides that
        limit training to 2 steps on CPU with minimal data.
        """
        overrides = get_quick_training_overrides(tmp_path, data_module=dm_type)
        cmd = ["uv", "run", runner_cmd, *overrides]

        try:
            result = run_cli_command(cmd, timeout=120)
        except subprocess.TimeoutExpired as e:
            pytest.fail(
                f"Runner {runner_cmd} timed out after 120s.\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            )

        assert_training_success(result)

        # Verify output directory was created
        assert tmp_path.exists(), f"Output directory not created: {tmp_path}"


@pytest.mark.integration
@pytest.mark.slow
class TestUnifiedExperimentRunner:
    """Tests for the unified tmgg-experiment CLI with stage configs."""

    @pytest.mark.parametrize("stage_config,dm_type", STAGE_CONFIGS)
    def test_stage_executes_brief_training(
        self, stage_config: str, dm_type: str, tmp_path: Path
    ) -> None:
        """Verify tmgg-experiment with stage override completes training.

        The unified CLI uses +stage=<name> to compose stage configs on top
        of base_config_spectral_arch.
        """
        overrides = get_quick_training_overrides(tmp_path, data_module=dm_type)
        # Ensure single-experiment mode (not sweep)
        overrides.append("sweep=false")
        cmd = ["uv", "run", "tmgg-experiment", f"+stage={stage_config}", *overrides]

        try:
            result = run_cli_command(cmd, timeout=120)
        except subprocess.TimeoutExpired as e:
            pytest.fail(
                f"Stage {stage_config} timed out after 120s.\n"
                f"stdout: {e.stdout}\nstderr: {e.stderr}"
            )

        assert_training_success(result)
        assert tmp_path.exists(), f"Output directory not created: {tmp_path}"


@pytest.mark.integration
class TestRunnerImports:
    """Quick smoke tests that verify runner modules can be imported.

    These tests don't run training but verify that all imports and
    Hydra decorators are correctly configured. Faster than full runs.
    """

    def test_experiment_runner_imports(self) -> None:
        """Verify experiment runner modules import without error."""
        from tmgg.experiments import grid_search_runner
        from tmgg.experiments.digress_denoising import runner as digress
        from tmgg.experiments.gnn_denoising import runner as gnn
        from tmgg.experiments.gnn_transformer_denoising import runner as hybrid
        from tmgg.experiments.spectral_arch_denoising import runner as spectral

        # Verify each has a main function
        assert callable(digress.main)
        assert callable(gnn.main)
        assert callable(hybrid.main)
        assert callable(spectral.main)
        assert callable(grid_search_runner.main)

    def test_unified_runner_imports(self) -> None:
        """Verify unified stage runner module imports without error."""
        from tmgg.experiments.stages import runner

        # Verify the unified main function exists
        assert callable(runner.main)
        # Verify internal _run_stage helper exists
        assert callable(runner._run_stage)
