"""Tests for scripts.sweep.watch_runs.

Rationale
---------
The Option-C in-flight watcher is a snapshot fetcher: it pulls partial
W&B history for running runs, fits a saturating-exponential to the
gen-val/<quality> curve when ≥ 8 points are available, and applies a
literature-informed decision flowchart to compute a *recommendation*
(keep / kill: <reason> / extend_watch). The decision is non-binding —
Claude reads the snapshot and writes the actual decision to
``watches.jsonl`` and ``progress.md``.

These tests pin the pure-logic layer:
- ``find_running_launches`` from a rounds.jsonl row mix.
- ``find_prior_watches`` for freshness + extension-count rules.
- ``saturation_fit_partial`` recovers a known terminal value to within 5%.
- ``apply_flowchart`` returns the right recommendation for synthetic
  trajectories matching each branch of the flowchart.

The W&B query layer is mocked via FakeRun (same pattern as
``test_fetch_outcomes``).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.sweep.watch_runs import (
    FlowchartInput,
    apply_flowchart,
    build_recommendation,
    find_prior_watches,
    find_running_launches,
    read_watches,
    saturation_fit_partial,
)


@dataclass
class FakeRun:
    """Minimal stand-in for ``wandb.apis.public.runs.Run`` in unit tests."""

    id: str
    state: str
    summary: dict[str, Any]
    history_records: list[dict[str, Any]] = field(default_factory=list)

    def history(self, **_kwargs: Any) -> list[dict[str, Any]]:
        return self.history_records


def test_find_running_launches_returns_running_only() -> None:
    """Running launches = launched rows without paired outcome."""
    rows = [
        {"kind": "schema", "version": 1},
        {"kind": "launched", "run_uid": "a", "step_cap": 100000},
        {"kind": "launched", "run_uid": "b", "step_cap": 100000},
        {"kind": "outcome", "run_uid": "a"},
    ]
    running = find_running_launches(rows)
    assert [r["run_uid"] for r in running] == ["b"]


def test_find_prior_watches_returns_chrono(tmp_path: Path) -> None:
    """Prior watches for a run_uid are returned in append order."""
    p = tmp_path / "watches.jsonl"
    p.write_text(
        '{"kind":"schema","version":1}\n'
        '{"kind":"watch","run_uid":"a","current_step":10000,"decision":"keep"}\n'
        '{"kind":"watch","run_uid":"b","current_step":5000,"decision":"keep"}\n'
        '{"kind":"watch","run_uid":"a","current_step":20000,"decision":"keep"}\n'
    )
    watches = read_watches(p)
    prior_a = find_prior_watches(watches, run_uid="a")
    assert [w["current_step"] for w in prior_a] == [10000, 20000]
    prior_b = find_prior_watches(watches, run_uid="b")
    assert [w["current_step"] for w in prior_b] == [5000]


def test_saturation_fit_partial_recovers_terminal() -> None:
    """Synthetic curve f(s) = 0.85 * (1 - exp(-s/30000)) + 0.05.

    Asymptote = 0.85 + 0.05 = 0.90. With 12 noiseless points spaced
    over [5000, 60000] the fitted q_infinity_hat must be within 5% of
    0.90.
    """
    rng = np.random.default_rng(0)
    a_true, tau_true, c_true = 0.85, 30000.0, 0.05
    steps = np.linspace(5000, 60000, 12)
    values = a_true * (1.0 - np.exp(-steps / tau_true)) + c_true
    values = values + rng.normal(0.0, 0.001, size=values.shape)

    fit = saturation_fit_partial(steps=steps, values=values)
    assert fit is not None
    assert fit["q_infinity_hat"] == pytest.approx(0.90, rel=0.05)
    assert fit["tau"] == pytest.approx(30000.0, rel=0.10)
    assert fit["n_points_used"] == 12


def test_saturation_fit_returns_none_for_too_few_points() -> None:
    """Below 8 points the fit is not attempted; returns None."""
    steps = np.array([1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
    values = np.array([0.1, 0.2, 0.25, 0.27, 0.28])
    fit = saturation_fit_partial(steps=steps, values=values)
    assert fit is None


def _make_input(
    *,
    nll_history: list[float] | None = None,
    quality_history: tuple[list[float], list[float]] | None = None,
    samples_degenerate_count: int = 0,
    wall_time_ratio: float = 0.5,
    grad_cosine_blocks: dict[str, float] | None = None,
    update_to_weight_blocks: dict[str, float] | None = None,
    quality_flat_count: int = 0,
    modularity_rising: bool = False,
    bits_per_edge_descending: bool = False,
    extension_count: int = 0,
    current_step: int = 30000,
    step_cap: int = 100000,
    anchor_nll: float = 1000.0,
) -> FlowchartInput:
    return FlowchartInput(
        run_uid="test",
        current_step=current_step,
        step_cap=step_cap,
        anchor_nll=anchor_nll,
        wall_time_ratio=wall_time_ratio,
        nll_history=nll_history if nll_history is not None else [3000.0, 2900.0, 2800.0],
        samples_degenerate_count=samples_degenerate_count,
        grad_cosine_blocks=grad_cosine_blocks or {},
        update_to_weight_blocks=update_to_weight_blocks or {},
        quality_history=quality_history,
        quality_flat_count=quality_flat_count,
        modularity_rising=modularity_rising,
        bits_per_edge_descending=bits_per_edge_descending,
        extension_count=extension_count,
        peer_quality_quantile=None,
        peer_count=0,
    )


def test_flowchart_kills_diverged_on_inf_nll() -> None:
    """NLL=inf triggers immediate KILL: diverged."""
    inp = _make_input(nll_history=[2800.0, 2900.0, float("inf")])
    rec = apply_flowchart(inp)
    assert rec["recommendation"] == "kill"
    assert rec["reason"] == "diverged"


def test_flowchart_kills_diverged_on_nll_jump_50pct() -> None:
    """NLL jumping >50% upward triggers KILL: diverged."""
    inp = _make_input(nll_history=[2000.0, 2050.0, 3300.0])
    rec = apply_flowchart(inp)
    assert rec["recommendation"] == "kill"
    assert rec["reason"] == "diverged"


def test_flowchart_kills_cost_cap() -> None:
    """Wall-time > 1.5x budget triggers KILL: cost-cap (when not diverged)."""
    inp = _make_input(wall_time_ratio=1.6)
    rec = apply_flowchart(inp)
    assert rec["recommendation"] == "kill"
    assert rec["reason"] == "cost_cap"


def test_flowchart_kills_degenerate_samples() -> None:
    """3+ consecutive watches with degenerate samples triggers KILL: samples."""
    inp = _make_input(samples_degenerate_count=3)
    rec = apply_flowchart(inp)
    assert rec["recommendation"] == "kill"
    assert rec["reason"] == "samples_degenerate"


def test_flowchart_kills_opt_health_confluence() -> None:
    """All blocks dead + flat quality for 3 watches triggers KILL: opt-health."""
    inp = _make_input(
        grad_cosine_blocks={
            "transformer_block_0": 0.05,
            "transformer_block_3": 0.07,
            "transformer_block_6": 0.06,
        },
        update_to_weight_blocks={
            "transformer_block_0": 5e-5,
            "transformer_block_3": 6e-5,
            "transformer_block_6": 4e-5,
        },
        quality_flat_count=3,
    )
    rec = apply_flowchart(inp)
    assert rec["recommendation"] == "kill"
    assert rec["reason"] == "opt_health_confluence"


def test_flowchart_extends_watch_on_phase_transition() -> None:
    """Modularity rising suggests phase transition; extend watch."""
    inp = _make_input(modularity_rising=True)
    rec = apply_flowchart(inp)
    assert rec["recommendation"] == "extend_watch"


def test_flowchart_caps_extension_at_one() -> None:
    """Already extended once + still phase-transitioning -> KEEP, not extend twice."""
    inp = _make_input(modularity_rising=True, extension_count=1)
    rec = apply_flowchart(inp)
    # Extension is capped; further phase-transition signal becomes a KEEP.
    assert rec["recommendation"] == "keep"


def test_flowchart_kills_saturation_when_fit_predicts_terminal_close() -> None:
    """Saturation fit predicts terminal within 3% of current value -> kill."""
    # Fake a quality history that's clearly saturating.
    steps = list(np.linspace(5000, 60000, 12).tolist())
    values = list((0.85 * (1.0 - np.exp(-np.array(steps) / 8000.0)) + 0.05).tolist())
    inp = _make_input(quality_history=(steps, values))
    rec = apply_flowchart(inp)
    # Latest value is ~0.90, q_infinity ~0.90; well within 3%.
    assert rec["recommendation"] == "kill"
    assert rec["reason"] == "saturation_heuristic"


def test_flowchart_keep_when_phase_transition_then_late() -> None:
    """No kill signals + no saturation fit -> KEEP."""
    inp = _make_input(
        nll_history=[3000.0, 2950.0, 2900.0, 2850.0],
        wall_time_ratio=0.5,
        samples_degenerate_count=0,
        grad_cosine_blocks={"b0": 0.5, "b1": 0.6},
        update_to_weight_blocks={"b0": 1e-3, "b1": 8e-4},
    )
    rec = apply_flowchart(inp)
    assert rec["recommendation"] == "keep"


def test_flowchart_kills_relative_at_late_stage() -> None:
    """At >=33% step_cap + bottom 1/eta of peers + stable 3 watches -> kill: hyperband."""
    inp = _make_input(
        current_step=50000,
        step_cap=100000,
        quality_flat_count=3,
    )
    # Override the peer-quantile fields directly on the dataclass.
    inp.peer_quality_quantile = 0.20
    inp.peer_count = 6
    rec = apply_flowchart(inp)
    assert rec["recommendation"] == "kill"
    assert rec["reason"] == "hyperband_relative"


def test_build_recommendation_serialises_to_dict_with_required_fields() -> None:
    """build_recommendation returns dict with the schema fields."""
    inp = _make_input()
    rec = build_recommendation(
        run_uid="test",
        flowchart_input=inp,
        observed_metrics={"gen-val/sbm_accuracy": 0.6},
        observed_diagnostics={"grad_cosine": {"b0": 0.4}},
        snapshot_sha256="abc",
        previous_decision_ref=None,
    )
    for key in [
        "kind",
        "ts_utc",
        "run_uid",
        "current_step",
        "step_cap",
        "previous_decision_ref",
        "snapshot_sha256",
        "observed_metrics",
        "observed_diagnostics",
        "recommendation",
        "fit_diagnostics",
    ]:
        assert key in rec, f"missing {key} in recommendation"
    assert rec["kind"] == "watch_recommendation"


def test_freshness_gate_skips_when_no_new_step() -> None:
    """If current_step <= prior watch's current_step, the freshness gate fires."""
    from scripts.sweep.watch_runs import freshness_gate_passes

    prior = [{"current_step": 30000}]
    assert freshness_gate_passes(current_step=35000, prior_watches=prior) is True
    assert freshness_gate_passes(current_step=30000, prior_watches=prior) is False
    # No prior watches -> always passes.
    assert freshness_gate_passes(current_step=10000, prior_watches=[]) is True


def test_read_watches_skips_schema_and_blanks(tmp_path: Path) -> None:
    """read_watches skips schema row + blank lines."""
    p = tmp_path / "watches.jsonl"
    p.write_text(
        '{"kind":"schema","version":1}\n'
        "\n"
        '{"kind":"watch","run_uid":"a","current_step":1000,"decision":"keep"}\n'
    )
    watches = read_watches(p)
    assert len(watches) == 1
    assert watches[0]["run_uid"] == "a"


def test_read_watches_returns_empty_when_file_missing(tmp_path: Path) -> None:
    """Missing watches.jsonl -> empty list (not raised)."""
    p = tmp_path / "does-not-exist.jsonl"
    assert read_watches(p) == []
