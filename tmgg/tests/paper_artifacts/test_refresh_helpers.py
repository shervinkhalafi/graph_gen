"""Unit tests for paper-artifacts/repro-ablations/scripts/refresh.py helpers.

Rationale: classify_metric_namespace and build_run_slug are pure functions
called many times during refresh. Wrong classification breaks downstream
pivots; wrong slug breaks lineage joins. Both must be deterministic and
tested in isolation since the wandb-fetching parts of refresh.py can't
be unit-tested without a live API.
"""

import sys
from pathlib import Path

# Add the script's directory to sys.path so the test can import refresh.py
SCRIPT_DIR = (
    Path(__file__).resolve().parents[2]
    / "paper-artifacts"
    / "repro-ablations"
    / "scripts"
)
sys.path.insert(0, str(SCRIPT_DIR))

import refresh  # noqa: E402


class TestClassifyMetricNamespace:
    def test_gen_val_mmd_metrics(self):
        assert refresh.classify_metric_namespace("gen-val/degree_mmd") == "mmd"
        assert refresh.classify_metric_namespace("gen-val/clustering_mmd") == "mmd"
        assert refresh.classify_metric_namespace("gen-val/orbit_mmd") == "mmd"
        assert refresh.classify_metric_namespace("gen-val/spectral_mmd") == "mmd"

    def test_train_metrics(self):
        assert refresh.classify_metric_namespace("train/loss_epoch") == "train"
        assert refresh.classify_metric_namespace("train/loss_step") == "train"

    def test_val_metrics(self):
        assert refresh.classify_metric_namespace("val/epoch_NLL") == "val"
        assert refresh.classify_metric_namespace("val/loss") == "val"

    def test_diagnostics_progress_overrides_diagnostics(self):
        # Progress paths sit under diagnostics-train/ but should classify as 'progress'
        assert (
            refresh.classify_metric_namespace(
                "diagnostics-train/progress/loss_per_t/bin_0"
            )
            == "progress"
        )
        assert (
            refresh.classify_metric_namespace("diagnostics-val/progress/bits_per_edge")
            == "progress"
        )

    def test_diagnostics_opt_health(self):
        assert (
            refresh.classify_metric_namespace(
                "diagnostics-train/opt-health/grad_norm_total"
            )
            == "diagnostics"
        )
        assert (
            refresh.classify_metric_namespace(
                "diagnostics-train/opt-health/effective_lr"
            )
            == "diagnostics"
        )

    def test_system_metrics(self):
        assert (
            refresh.classify_metric_namespace("system/gpu.0.memoryAllocated")
            == "system"
        )
        assert refresh.classify_metric_namespace("system/cpu") == "system"
        assert refresh.classify_metric_namespace("system/memory") == "system"

    def test_impl_perf(self):
        assert (
            refresh.classify_metric_namespace("impl-perf/train/step_time_s")
            == "impl_perf"
        )

    def test_lr_keys(self):
        assert refresh.classify_metric_namespace("lr-AdamW") == "lr"
        assert refresh.classify_metric_namespace("train/lr") == "lr"

    def test_other_falls_through(self):
        assert refresh.classify_metric_namespace("epoch") == "other"
        assert refresh.classify_metric_namespace("trainer/global_step") == "other"
        assert refresh.classify_metric_namespace("_runtime") == "other"


class TestBuildRunSlug:
    def test_postfix_true(self):
        assert (
            refresh.build_run_slug("sbm", "vignac", True, "cgfv3f85")
            == "sbm_vignac_postfix_cgfv3f85"
        )

    def test_postfix_false(self):
        assert (
            refresh.build_run_slug("sbm", "vignac", False, "12s2b4a7")
            == "sbm_vignac_prefix_12s2b4a7"
        )

    def test_compound_variant(self):
        assert (
            refresh.build_run_slug("enzymes", "pearl_gnnconv_norm", True, "xsmz6yql")
            == "enzymes_pearl_gnnconv_norm_postfix_xsmz6yql"
        )
