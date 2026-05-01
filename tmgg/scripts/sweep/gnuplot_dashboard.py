"""Single-view gnuplot dashboard over the cached W&B histories.

Reads ``wandb_export/sweep_cache/<safe_uid>/history.jsonl`` produced
by ``scripts.sweep.cache_history`` and ``rounds.jsonl`` for the
terminal outcomes, then drives gnuplot to render three views in the
terminal (ANSI-256 dumb terminal, no sixel required):

  1. **Per-metric line plots** — one panel per gen-val metric, each
     pod overlaid as a separate line. (Same idea as ``status_plots``
     but driven by gnuplot for consistent styling and offline data.)
  2. **Pareto scatter** — sbm_accuracy (x) vs degree_mmd (y, log) at
     each pod's terminal step, with run_uid label per point. Pareto-
     optimal points marked.
  3. **Parallel coordinates** — one polyline per pod across all
     gen-val metric axes, normalized to [0, 1] per axis. Lets you
     spot which axes a pod dominates / fails on at a glance.

Each view is a separate gnuplot run printed sequentially. Pipe to
``less -R`` if you want to scroll, or run on its own for a one-shot
status pulse.

Invocation::

    doppler run -- uv run python -m scripts.sweep.cache_history    # extend caches
    uv run python -m scripts.sweep.gnuplot_dashboard               # render

Optional flags:
    --view {curves,pareto,parallel,all}   default: all
    --pareto-x METRIC                     default: gen-val/sbm_accuracy
    --pareto-y METRIC                     default: gen-val/degree_mmd
    --include uid_substring,...           restrict to matching pods
    --term-cols N --term-rows N           override per-panel sizing
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from scripts.sweep.cache_history import history_path, manifest_path, safe_uid
from scripts.sweep.fetch_outcomes import read_rounds

DEFAULT_METRICS = [
    "gen-val/sbm_accuracy",
    "gen-val/modularity_q",
    "gen-val/degree_mmd",
    "gen-val/clustering_mmd",
    "gen-val/orbit_mmd",
    "gen-val/spectral_mmd",
    "val/epoch_NLL",
]

# Gnuplot dumb-terminal supports ansi256; we cycle distinguishable
# 256-color codes per pod.
COLOR_CYCLE = [
    "#00d7ff",
    "#ff5fff",
    "#5fff5f",
    "#ffff5f",
    "#5fafff",
    "#ff5f5f",
    "#ffffff",
]

PARETO_X_DEFAULT = "gen-val/sbm_accuracy"
PARETO_Y_DEFAULT = "gen-val/degree_mmd"

# Direction per metric: True = larger_is_better, False = smaller_is_better.
DIRECTION_LARGER_IS_BETTER = {
    "gen-val/sbm_accuracy": True,
    "gen-val/modularity_q": True,
    "gen-val/uniqueness": True,
    "gen-val/planarity_accuracy": True,
}


def short_pod(uid: str) -> str:
    parts = uid.split("/")
    if len(parts) >= 4:
        return f"{parts[-2]}/{parts[-1][:8]}"
    return uid


def load_history(cache_root: Path, uid: str) -> list[dict[str, Any]]:
    p = history_path(cache_root, uid)
    if not p.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    rows.sort(key=lambda r: r.get("trainer/global_step", 0))
    return rows


def all_pods_with_cache(cache_root: Path, rounds: list[dict[str, Any]]) -> list[str]:
    uids = sorted({r["run_uid"] for r in rounds if r.get("kind") == "launched"})
    return [u for u in uids if manifest_path(cache_root, u).exists()]


def filter_pods(uids: list[str], include: str | None) -> list[str]:
    if not include:
        return uids
    wanted = [s.strip() for s in include.split(",") if s.strip()]
    return [u for u in uids if any(w in u for w in wanted)]


def assign_colors(uids: list[str]) -> dict[str, str]:
    return {uid: COLOR_CYCLE[i % len(COLOR_CYCLE)] for i, uid in enumerate(uids)}


def write_curve_data(
    *,
    tmpdir: Path,
    metric: str,
    pods: list[str],
    histories: dict[str, list[dict[str, Any]]],
) -> dict[str, Path]:
    """Write one TSV per pod with (step, value); return uid -> path."""
    out: dict[str, Path] = {}
    for uid in pods:
        rows = histories.get(uid, [])
        path = tmpdir / f"curve__{safe_uid(uid)}__{metric.replace('/', '_')}.tsv"
        with path.open("w") as f:
            for r in rows:
                v = r.get(metric)
                s = r.get("trainer/global_step")
                if v is None or s is None:
                    continue
                try:
                    f.write(f"{int(s)}\t{float(v)}\n")
                except (TypeError, ValueError):
                    continue
        out[uid] = path
    return out


def gnuplot_run(script: str) -> None:
    """Pipe a gnuplot script to gnuplot; print stdout/stderr to user."""
    try:
        result = subprocess.run(
            ["gnuplot"],
            input=script,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError:
        print("ERROR: gnuplot not on PATH. Install via your package manager.")
        return
    if result.stdout:
        print(result.stdout, end="")
    if result.returncode != 0 and result.stderr:
        print("--- gnuplot stderr ---")
        print(result.stderr)


def render_curves_view(
    *,
    pods: list[str],
    pod_colors: dict[str, str],
    histories: dict[str, list[dict[str, Any]]],
    metrics: list[str],
    term_cols: int,
    term_rows: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="gnuplot_dash_curves_") as td:
        tmpdir = Path(td)
        for metric in metrics:
            paths = write_curve_data(
                tmpdir=tmpdir, metric=metric, pods=pods, histories=histories
            )
            plot_specs: list[str] = []
            for uid in pods:
                p = paths[uid]
                if p.stat().st_size == 0:
                    continue
                color = pod_colors[uid]
                title = short_pod(uid)
                plot_specs.append(
                    f"'{p}' using 1:2 with linespoints "
                    f"linewidth 2 pointtype 7 pointsize 0.5 "
                    f"linecolor rgb '{color}' title '{title}'"
                )
            if not plot_specs:
                continue
            script = (
                f"set term dumb ansi256 size {term_cols},{term_rows}\n"
                f"set title '{metric}'\n"
                "set xlabel 'trainer/global_step'\n"
                "set key outside top right\n"
                "set grid\n"
                "plot " + ", ".join(plot_specs) + "\n"
            )
            gnuplot_run(script)


def terminal_metrics_per_pod(
    *,
    pods: list[str],
    histories: dict[str, list[dict[str, Any]]],
    metrics: list[str],
) -> dict[str, dict[str, float]]:
    """For each pod, take the LATEST observed value per metric."""
    out: dict[str, dict[str, float]] = {}
    for uid in pods:
        latest: dict[str, float] = {}
        for r in histories.get(uid, []):
            for m in metrics:
                v = r.get(m)
                if v is not None:
                    try:
                        latest[m] = float(v)
                    except (TypeError, ValueError):
                        continue
        out[uid] = latest
    return out


def is_pareto_optimal(
    uid: str,
    *,
    terminal: dict[str, dict[str, float]],
    x_key: str,
    y_key: str,
) -> bool:
    """Pareto on (x_key larger-is-better, y_key smaller-is-better)."""
    me = terminal.get(uid, {})
    x_me = me.get(x_key)
    y_me = me.get(y_key)
    if x_me is None or y_me is None:
        return False
    for other_uid, other in terminal.items():
        if other_uid == uid:
            continue
        x_o = other.get(x_key)
        y_o = other.get(y_key)
        if x_o is None or y_o is None:
            continue
        x_better = x_o >= x_me  # larger is better
        y_better = y_o <= y_me  # smaller is better
        strict = (x_o > x_me) or (y_o < y_me)
        if x_better and y_better and strict:
            return False
    return True


def render_pareto_view(
    *,
    pods: list[str],
    pod_colors: dict[str, str],
    terminal: dict[str, dict[str, float]],
    x_key: str,
    y_key: str,
    term_cols: int,
    term_rows: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="gnuplot_dash_pareto_") as td:
        tmpdir = Path(td)
        plot_specs: list[str] = []
        any_data = False
        for uid in pods:
            t = terminal.get(uid, {})
            x = t.get(x_key)
            y = t.get(y_key)
            if x is None or y is None:
                continue
            any_data = True
            pareto = is_pareto_optimal(uid, terminal=terminal, x_key=x_key, y_key=y_key)
            label = short_pod(uid) + (" *" if pareto else "")
            data_path = tmpdir / f"pareto__{safe_uid(uid)}.tsv"
            with data_path.open("w") as f:
                f.write(f"{x}\t{y}\t{label}\n")
            color = pod_colors[uid]
            ptype = 5 if pareto else 7  # filled square for pareto, circle else
            psize = 2 if pareto else 1.2
            plot_specs.append(
                f"'{data_path}' using 1:2 with points pointtype {ptype} "
                f"pointsize {psize} linecolor rgb '{color}' title '{label}', "
                f"'{data_path}' using 1:2:3 with labels offset 1,0.5 "
                f"font ',8' notitle"
            )
        if not any_data:
            print("(no terminal data for pareto view)")
            return

        # Try log y if all values positive.
        log_y = all(
            (terminal.get(uid, {}).get(y_key) or 0) > 0
            for uid in pods
            if terminal.get(uid)
        )
        log_y_cmd = "set logscale y" if log_y else ""

        script = (
            f"set term dumb ansi256 size {term_cols},{term_rows + 4}\n"
            f"set title 'Pareto: {x_key} (larger better, x) vs {y_key} (smaller better, y) — * = non-dominated'\n"
            f"set xlabel '{x_key}'\n"
            f"set ylabel '{y_key}'\n"
            f"{log_y_cmd}\n"
            "set grid\n"
            "set key outside top right\n"
            "plot " + ", ".join(plot_specs) + "\n"
        )
        gnuplot_run(script)


def render_parallel_view(
    *,
    pods: list[str],
    pod_colors: dict[str, str],
    terminal: dict[str, dict[str, float]],
    metrics: list[str],
    term_cols: int,
    term_rows: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix="gnuplot_dash_parallel_") as td:
        tmpdir = Path(td)

        # Per-axis min/max across pods for normalization.
        axis_stats: dict[str, tuple[float, float]] = {}
        for m in metrics:
            vals = [
                terminal[u].get(m)
                for u in pods
                if terminal.get(u, {}).get(m) is not None
            ]
            vals = [v for v in vals if v is not None and not math.isnan(float(v))]
            if not vals:
                continue
            lo = float(min(vals))
            hi = float(max(vals))
            if lo == hi:
                hi = lo + 1.0  # avoid divide-by-zero
            axis_stats[m] = (lo, hi)

        active_metrics = [m for m in metrics if m in axis_stats]
        if not active_metrics:
            print("(no terminal data for parallel-coordinates view)")
            return

        # For each pod, write a TSV: x = axis_index (0..N-1), y = normalized_value.
        # "Larger is better" axes are inverted so that 1.0 is always "better" across all axes.
        plot_specs: list[str] = []
        for uid in pods:
            t = terminal.get(uid, {})
            data_path = tmpdir / f"parallel__{safe_uid(uid)}.tsv"
            with data_path.open("w") as f:
                for i, m in enumerate(active_metrics):
                    v = t.get(m)
                    if v is None:
                        continue
                    lo, hi = axis_stats[m]
                    norm = (float(v) - lo) / (hi - lo)
                    if not DIRECTION_LARGER_IS_BETTER.get(m, False):
                        norm = 1.0 - norm  # invert so "up" is always good
                    f.write(f"{i}\t{norm}\n")
            if data_path.stat().st_size == 0:
                continue
            color = pod_colors[uid]
            plot_specs.append(
                f"'{data_path}' using 1:2 with linespoints "
                f"linewidth 2 pointtype 7 pointsize 1 "
                f"linecolor rgb '{color}' title '{short_pod(uid)}'"
            )

        if not plot_specs:
            print("(no parallel-coordinates data)")
            return

        # Custom xtics: axis index -> short metric name.
        def _short(metric_name: str) -> str:
            return metric_name.replace("gen-val/", "").replace("_mmd", "")

        xtics = ", ".join(f"'{_short(m)}' {i}" for i, m in enumerate(active_metrics))

        script = (
            f"set term dumb ansi256 size {term_cols},{term_rows + 4}\n"
            "set title 'Parallel coordinates (terminal metrics, normalized; up = better)'\n"
            "set xlabel 'metric axis'\n"
            "set ylabel 'normalized [0=worst, 1=best]'\n"
            f"set xtics ({xtics})\n"
            "set yrange [-0.05:1.10]\n"
            "set grid\n"
            "set key outside top right\n"
            "plot " + ", ".join(plot_specs) + "\n"
        )
        gnuplot_run(script)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _ = p.add_argument(
        "--rounds-jsonl",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/rounds.jsonl"),
    )
    _ = p.add_argument(
        "--cache-root", type=Path, default=Path("wandb_export/sweep_cache")
    )
    _ = p.add_argument(
        "--view",
        choices=["curves", "pareto", "parallel", "all"],
        default="all",
    )
    _ = p.add_argument("--include", default=None)
    _ = p.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    _ = p.add_argument("--pareto-x", default=PARETO_X_DEFAULT)
    _ = p.add_argument("--pareto-y", default=PARETO_Y_DEFAULT)
    _ = p.add_argument("--term-cols", type=int, default=120)
    _ = p.add_argument("--term-rows", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not shutil.which("gnuplot"):
        raise SystemExit("gnuplot not found on PATH; install it first.")

    rounds = read_rounds(args.rounds_jsonl)
    pods = all_pods_with_cache(args.cache_root, rounds)
    pods = filter_pods(pods, args.include)
    if not pods:
        raise SystemExit(
            f"no cached pods under {args.cache_root}; " "run cache_history first."
        )
    pod_colors = assign_colors(pods)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    histories = {uid: load_history(args.cache_root, uid) for uid in pods}
    terminal = terminal_metrics_per_pod(pods=pods, histories=histories, metrics=metrics)

    print(f"# {len(pods)} pods cached:")
    for uid in pods:
        print(
            f"  {pod_colors[uid]:>8}  {short_pod(uid)}  history_rows={len(histories[uid])}"
        )

    if args.view in {"curves", "all"}:
        render_curves_view(
            pods=pods,
            pod_colors=pod_colors,
            histories=histories,
            metrics=metrics,
            term_cols=args.term_cols,
            term_rows=args.term_rows,
        )
    if args.view in {"pareto", "all"}:
        render_pareto_view(
            pods=pods,
            pod_colors=pod_colors,
            terminal=terminal,
            x_key=args.pareto_x,
            y_key=args.pareto_y,
            term_cols=args.term_cols,
            term_rows=args.term_rows,
        )
    if args.view in {"parallel", "all"}:
        render_parallel_view(
            pods=pods,
            pod_colors=pod_colors,
            terminal=terminal,
            metrics=metrics,
            term_cols=args.term_cols,
            term_rows=args.term_rows,
        )


if __name__ == "__main__":  # pragma: no cover
    main()
