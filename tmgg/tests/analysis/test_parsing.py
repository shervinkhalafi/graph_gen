"""Tests for tmgg.analysis.parsing architecture and flag extraction.

Run names derive from YAML config filenames via ``generate_run_id()`` in
``modal/cli/generate_configs.py``.  Every model config under
``exp_configs/models/`` must map to a known architecture label, and modifier
flags (asymmetric, pearl) must be extracted independently.  These tests
verify the first-match pattern ordering is correct and that no active
config filename falls through to ``"other"``.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tmgg.analysis.parsing import parse_architecture, parse_run_name_fields


# Each entry is (run_name_substring, expected_arch).
# Run names look like: "stage2_{arch_name}_sbm_default_lr1e-3_wd1e-4_k8_s1"
@pytest.mark.parametrize(
    "run_name, expected_arch",
    [
        # --- Spectral: filter_bank variants ---
        ("stage2_filter_bank_sbm_default_lr1e-3_wd1e-4_k8_s1", "filter_bank"),
        ("stage1e_filter_bank_pearl_lr1e-3_wd1e-4_k8_s1", "filter_bank"),
        ("stage1d_filter_bank_asymmetric_lr1e-3_wd1e-4_k8_s1", "filter_bank"),
        # --- Spectral: linear_pe variants ---
        ("stage2_linear_pe_sbm_default_lr1e-3_wd1e-4_k16_s1", "linear_pe"),
        ("stage1e_linear_pe_pearl_lr1e-3_wd1e-4_k16_s1", "linear_pe"),
        ("stage1d_linear_pe_asymmetric_lr1e-3_wd1e-4_k16_s1", "linear_pe"),
        # --- Spectral: self_attention variants ---
        ("stage2_self_attention_sbm_default_lr1e-3_wd1e-4_k8_s1", "self_attention"),
        ("stage2_self_attention_mlp_sbm_default_lr1e-3_s1", "self_attention_mlp"),
        ("stage2_multilayer_self_attention_sbm_lr1e-3_s1", "multilayer_self_attention"),
        ("stage2_self_attention_pearl_lr1e-3_k8_s1", "self_attention"),
        ("stage2_self_attention_strict_shrinkage_lr1e-3_s1", "self_attention"),
        ("stage2_self_attention_relaxed_shrinkage_lr1e-3_s1", "self_attention"),
        # --- DiGress compound: gnn_* matches first ---
        ("stage1c_digress_transformer_gnn_qk_lr1e-3_wd1e-4_k8_s1", "gnn_qk"),
        ("stage1c_digress_transformer_gnn_all_lr1e-3_wd1e-4_k8_s1", "gnn_all"),
        ("stage1c_digress_transformer_gnn_v_lr1e-3_wd1e-4_k8_s1", "gnn_v"),
        # --- DiGress: transformer ---
        (
            "stage2_digress_transformer_sbm_default_lr1e-3_wd1e-4_s1",
            "digress_transformer",
        ),
        ("stage2_digress_transformer_spectral_qk_lr1e-3_s1", "digress_transformer"),
        ("stage2_digress_transformer_spectral_all_lr1e-3_s1", "digress_transformer"),
        # --- DiGress: catch-all ---
        ("stage1_digress_sbm_small_lr1e-3_s1", "digress"),
        ("stage1_digress_sbm_vanilla_lr1e-3_s1", "digress"),
        ("stage1_digress_sbm_vanilla_gnn_lr1e-3_s1", "digress"),
        # --- Baselines ---
        ("stage2_mlp_sbm_default_lr1e-3_wd1e-4_s1", "mlp"),
        ("stage2_linear_sbm_default_lr1e-3_wd1e-4_s1", "linear"),
        # --- Edge cases ---
        ("", "other"),
        ("totally_unknown_architecture_s1", "other"),
    ],
)
def test_parse_architecture(run_name: str, expected_arch: str) -> None:
    assert parse_architecture(run_name) == expected_arch


def test_parse_architecture_nan() -> None:
    assert parse_architecture(pd.NA) == "unknown"  # type: ignore[arg-type]


class TestParseRunNameFields:
    """Verify pearl_flag and asymmetric_flag extraction from run names."""

    def test_pearl_flag_detected(self) -> None:
        result = parse_run_name_fields("stage1e_filter_bank_pearl_lr1e-3_k8_s1")
        assert result["pearl_flag"] is True

    def test_pearl_flag_absent(self) -> None:
        result = parse_run_name_fields("stage2_filter_bank_lr1e-3_k8_s1")
        assert result["pearl_flag"] is False

    def test_asymmetric_flag_detected(self) -> None:
        result = parse_run_name_fields("stage1d_filter_bank_asymmetric_lr1e-3_k8_s1")
        assert result["asymmetric_flag"] is True

    def test_asymmetric_flag_absent(self) -> None:
        result = parse_run_name_fields("stage2_filter_bank_lr1e-3_k8_s1")
        assert result["asymmetric_flag"] is False

    def test_both_flags_independent(self) -> None:
        """A name with 'pearl' should not set asymmetric, and vice versa."""
        pearl_result = parse_run_name_fields("stage1e_linear_pe_pearl_lr1e-3_k8_s1")
        assert pearl_result["pearl_flag"] is True
        assert pearl_result["asymmetric_flag"] is False

        asym_result = parse_run_name_fields("stage1d_linear_pe_asymmetric_lr1e-3_k8_s1")
        assert asym_result["asymmetric_flag"] is True
        assert asym_result["pearl_flag"] is False
