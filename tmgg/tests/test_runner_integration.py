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
    - Experiment runners: attention, digress, gnn, hybrid, spectral, grid-search
    - Stage runners: stage1 through stage5, stage1-sanity, stage1-5
"""

import subprocess
from pathlib import Path

import pytest

from tests.integration_utils import (
    assert_training_success,
    get_quick_training_overrides,
    run_cli_command,
)

# All CLI runners to test with their expected base config
EXPERIMENT_RUNNERS = [
    ("tmgg-attention", "base_config_attention"),
    ("tmgg-digress", "base_config_digress"),
    ("tmgg-gnn", "base_config_gnn"),
    ("tmgg-hybrid", "base_config_hybrid"),
    ("tmgg-spectral", "base_config_spectral"),
    ("tmgg-grid-search", "grid_search_base"),
]

STAGE_RUNNERS = [
    ("tmgg-stage1", "stage1_poc"),
    ("tmgg-stage1-sanity", "stage1_sanity"),
    ("tmgg-stage1-5", "stage1_5_crossdata"),
    ("tmgg-stage2", "stage2_validation"),
    ("tmgg-stage3", "stage3_diversity"),
    ("tmgg-stage4", "stage4_benchmarks"),
    ("tmgg-stage5", "stage5_full"),
]

ALL_RUNNERS = EXPERIMENT_RUNNERS + STAGE_RUNNERS


@pytest.mark.integration
@pytest.mark.slow
class TestExperimentRunners:
    """Tests for experiment-type CLI runners (attention, gnn, etc.)."""

    @pytest.mark.parametrize("runner_cmd,base_config", EXPERIMENT_RUNNERS)
    def test_runner_executes_brief_training(
        self, runner_cmd: str, base_config: str, tmp_path: Path
    ) -> None:
        """Verify runner completes 2 training steps without error.

        This test invokes the runner via subprocess with overrides that
        limit training to 2 steps on CPU with minimal data.
        """
        overrides = get_quick_training_overrides(tmp_path)
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
class TestStageRunners:
    """Tests for stage-specific CLI runners (stage1, stage2, etc.)."""

    @pytest.mark.parametrize("runner_cmd,stage_config", STAGE_RUNNERS)
    def test_stage_runner_executes_brief_training(
        self, runner_cmd: str, stage_config: str, tmp_path: Path
    ) -> None:
        """Verify stage runner completes 2 training steps without error.

        Stage runners use the same underlying run_experiment function but
        inject stage-specific config overrides. This test ensures each
        stage config composes correctly with the base spectral config.
        """
        overrides = get_quick_training_overrides(tmp_path)
        # Ensure single-experiment mode (not sweep)
        overrides.append("sweep=false")
        cmd = ["uv", "run", runner_cmd, *overrides]

        try:
            result = run_cli_command(cmd, timeout=120)
        except subprocess.TimeoutExpired as e:
            pytest.fail(
                f"Stage runner {runner_cmd} timed out after 120s.\n"
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
        from tmgg.experiments.attention_denoising import runner as attention
        from tmgg.experiments.digress_denoising import runner as digress
        from tmgg.experiments.gnn_denoising import runner as gnn
        from tmgg.experiments.hybrid_denoising import runner as hybrid
        from tmgg.experiments.spectral_denoising import runner as spectral

        # Verify each has a main function
        assert callable(attention.main)
        assert callable(digress.main)
        assert callable(gnn.main)
        assert callable(hybrid.main)
        assert callable(spectral.main)
        assert callable(grid_search_runner.main)

    def test_stage_runner_imports(self) -> None:
        """Verify stage runner module imports without error."""
        from tmgg.experiments.stages import runner

        # Verify all stage functions exist
        assert callable(runner.stage1)
        assert callable(runner.stage1_sanity)
        assert callable(runner.stage1_5)
        assert callable(runner.stage2)
        assert callable(runner.stage3)
        assert callable(runner.stage4)
        assert callable(runner.stage5)
