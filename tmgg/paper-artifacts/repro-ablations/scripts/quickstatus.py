#!/usr/bin/env -S uv run --with pandas --with pyarrow
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "pandas>=2.2",
#     "pyarrow>=18",
# ]
# ///
"""Emit markdown rows for `runlog.md`'s Quick Status table.

Reads the refreshed bundle (`data/runs_index.csv` + per-run parquets
under `data/per_run_history/`) and prints one markdown row per run, in
the same column layout as the runlog's quick-status table. Pipe the
output into a temp file and copy-paste the relevant rows into
`runlog.md` — the diff against the existing rows is what changed since
the last snapshot.

Usage (from bundle root)::

    uv run scripts/quickstatus.py [--postfix] [--dataset {sbm,enzymes}]

Flags filter the rows; default is "all 28 runs in the index".

Schema follows the table at the bottom of `runlog.md`:

    | Config | Run ID | Launched (UTC) | Status | Step (cycles) | degree MMD² | grad_norm | Stable? | Detail |

Stability heuristic mirrors the runlog's own (`grad_norm_total < 5.0`
→ ✓; otherwise ⚠ if finite, ✗ if non-finite). Health overrides from
`runs_index.source.yaml` (`invalidated_mask_bug`, `blew_up`) are
respected — those rows render as "n/a — invalidated" or "✗ blew up".
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
BUNDLE_ROOT = SCRIPT_DIR.parent
INDEX_CSV = BUNDLE_ROOT / "data" / "runs_index.csv"
HISTORY_DIR = BUNDLE_ROOT / "data" / "per_run_history"

GEN_VAL_DEGREE = "gen-val/degree_mmd"
GRAD_NORM_KEY = "diagnostics-train/opt-health/grad_norm_total"


def _stability_flag(grad_norm: float | None, health: str | None) -> str:
    if health == "invalidated_mask_bug":
        return "n/a — invalidated"
    if health == "blew_up":
        return "✗ blew up"
    if health == "elevated_grad":
        return "⚠ (high norm)"
    if grad_norm is None or (isinstance(grad_norm, float) and math.isnan(grad_norm)):
        return "—"
    if not math.isfinite(grad_norm):
        return "✗ blew up"
    if grad_norm < 5.0:
        return "✓"
    return "⚠"


def _format_step(step: int) -> str:
    if step >= 1_000_000:
        return f"{step / 1_000_000:.2f}M"
    if step >= 1_000:
        return f"{step / 1_000:.1f}k".replace(".0k", "k")
    return str(step)


def _format_mmd(value: float | None) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "—"
    return f"{value:.3f}"


def extract_latest(run_id: str) -> dict:
    """Return latest gen-val degree MMD², latest grad_norm, eval-cycle count."""
    parquet_path = HISTORY_DIR / f"{run_id}.parquet"
    if not parquet_path.exists():
        return {
            "latest_degree_mmd": None,
            "latest_degree_mmd_step": None,
            "eval_cycles": 0,
            "grad_norm_latest": None,
        }
    h = pd.read_parquet(parquet_path)
    step_col = "trainer/global_step" if "trainer/global_step" in h.columns else "_step"

    out = {
        "latest_degree_mmd": None,
        "latest_degree_mmd_step": None,
        "eval_cycles": 0,
        "grad_norm_latest": None,
    }

    if GEN_VAL_DEGREE in h.columns:
        s = h[[step_col, GEN_VAL_DEGREE]].dropna(subset=[GEN_VAL_DEGREE])
        if not s.empty:
            row = s.sort_values(step_col).iloc[-1]
            out["latest_degree_mmd"] = float(row[GEN_VAL_DEGREE])
            out["latest_degree_mmd_step"] = int(row[step_col])
            out["eval_cycles"] = int(s[step_col].nunique())

    if GRAD_NORM_KEY in h.columns:
        gn_series = h[GRAD_NORM_KEY].dropna()
        if not gn_series.empty:
            out["grad_norm_latest"] = float(gn_series.iloc[-1])

    return out


def render_row(idx_row: pd.Series) -> str:
    extras = extract_latest(idx_row["run_id"])
    config = idx_row["config_name"]
    run_id = idx_row["run_id"]
    launched = (idx_row["launched_at_utc"] or "")[:16].replace("T", " ")
    state = idx_row["final_state"]

    final_step = int(idx_row["final_step"] or 0)
    cycles = extras["eval_cycles"]
    if cycles:
        step_cell = (
            f"{_format_step(final_step)} ({cycles} cycle{'s' if cycles != 1 else ''})"
        )
    else:
        step_cell = _format_step(final_step) if final_step else "—"

    mmd_cell = _format_mmd(extras["latest_degree_mmd"])
    if (
        extras["latest_degree_mmd_step"]
        and extras["latest_degree_mmd_step"] != final_step
    ):
        mmd_cell = f"{mmd_cell} @ {_format_step(extras['latest_degree_mmd_step'])}"

    gn = extras["grad_norm_latest"]
    if gn is None or (isinstance(gn, float) and math.isnan(gn)):
        gn_cell = "—"
    elif math.isfinite(gn):
        gn_cell = f"{gn:.3f}" if abs(gn) < 100 else f"{gn:.2e}"
    else:
        gn_cell = "Inf"

    stable = _stability_flag(gn, idx_row.get("health"))

    # date subdir for the detail link mirrors launch date
    launch_date = (idx_row["launched_at_utc"] or "")[:10]
    detail_name = (
        idx_row.get("display_name")
        if idx_row.get("postfix") and idx_row.get("display_name")
        else run_id
    )
    detail_link = f"run_details/{launch_date}/{config}_{detail_name}_details.md"

    config_cell = f"`{config}`"
    if idx_row.get("postfix"):
        config_cell = f"**{config_cell}** (post-fix)"
    rid_cell = f"`{run_id}`"
    if idx_row.get("postfix") and idx_row.get("display_name"):
        rid_cell = f"`{idx_row['display_name']}` (`{run_id}`)"

    return (
        f"| {config_cell} | {rid_cell} | {launched} | {state} | {step_cell} | "
        f"{mmd_cell} | {gn_cell} | {stable} | [link]({detail_link}) |"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--postfix",
        action="store_true",
        help="Only render postfix=True rows (the active panel).",
    )
    parser.add_argument(
        "--dataset",
        choices=("sbm", "enzymes"),
        help="Filter by dataset.",
    )
    args = parser.parse_args(argv)

    if not INDEX_CSV.exists():
        print(
            f"error: {INDEX_CSV} not found — run scripts/refresh.py first",
            file=sys.stderr,
        )
        return 1

    idx: pd.DataFrame = pd.read_csv(INDEX_CSV)
    if args.postfix:
        idx = idx.loc[idx["postfix"]]
    if args.dataset:
        idx = idx.loc[idx["dataset"] == args.dataset]

    header = "| Config | Run ID | Launched (UTC) | Status | Step (cycles) | degree MMD² | grad_norm | Stable? | Detail |"
    sep = "|--------|--------|----------------|:------:|---------------|------------:|----------:|:-------:|--------|"
    print(header)
    print(sep)
    for _, row in idx.iterrows():
        print(render_row(row))
    return 0


if __name__ == "__main__":
    sys.exit(main())
