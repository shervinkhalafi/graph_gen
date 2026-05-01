# /// script
# dependencies = [
#   "wandb",
#   "plotext",
#   "pyyaml",
# ]
# ///
"""Quick-look terminal plots of every running launch's gen-val trajectory.

Reusable status pulse: pulls W&B history for every launched-but-unresolved
pod in ``rounds.jsonl``, overlays each pod's curve per metric, prints
colored ANSI line plots to the terminal. Also draws the corresponding
``anchors.yaml`` threshold as a horizontal line per panel where applicable.

Invocation::

    doppler run -- uv run python -m scripts.sweep.status_plots

Optional flags:
    --rounds-jsonl <path>     default: docs/.../rounds.jsonl
    --anchors-yaml <path>     default: docs/.../anchors.yaml
    --entity <str>            default: graph_denoise_team
    --project <str>           default: tmgg-smallest-config-sweep
    --metrics m1,m2,...       restrict to a subset of gen-val keys
    --max-rows N              cap on history rows per pod (default 200)

The terminal panel layout is one panel per metric, ~30 cols x ~10 rows
each, packed two per row when possible. Only pods with current_step >
0 are plotted; failed/finished pods are omitted (use fetch_outcomes
for terminal numbers).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import plotext as plt  # type: ignore[import-untyped]
import yaml
from scripts.sweep.fetch_outcomes import find_pending_launches, read_rounds

DEFAULT_METRICS = [
    "gen-val/sbm_accuracy",
    "gen-val/modularity_q",
    "gen-val/degree_mmd",
    "gen-val/clustering_mmd",
    "gen-val/orbit_mmd",
    "gen-val/spectral_mmd",
    "val/epoch_NLL",
]

# Stable color cycle (plotext supports named colors); cycles per pod.
POD_COLORS = ["cyan", "magenta", "green", "yellow", "blue", "red", "white"]


def fetch_history(
    *, entity: str, project: str, run_uid: str, max_rows: int
) -> list[dict[str, Any]]:
    import wandb  # local; only needed at runtime

    api = wandb.Api()
    runs = list(api.runs(f"{entity}/{project}", filters={"display_name": run_uid}))
    if not runs:
        return []
    runs.sort(key=lambda r: getattr(r, "created_at", "") or "", reverse=True)
    run = runs[0]
    rows = list(
        run.scan_history(
            keys=["trainer/global_step", *DEFAULT_METRICS],
            page_size=max_rows,
        )
    )
    by_step: dict[int, dict[str, Any]] = {}
    for r in rows:
        if r is None:
            continue
        s_raw = r.get("trainer/global_step")
        if s_raw is None:
            continue
        try:
            s = int(s_raw)
        except (TypeError, ValueError):
            continue
        merged = by_step.setdefault(s, {})
        for k, v in r.items():
            if v is not None:
                merged[k] = v
    out = [{**v, "trainer/global_step": k} for k, v in sorted(by_step.items())]
    return out[-max_rows:]


def load_anchors(anchors_path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    if not anchors_path.exists():
        return {}
    return yaml.safe_load(anchors_path.read_text()) or {}


def threshold_for_metric(
    anchors: dict[str, dict[str, dict[str, Any]]], dataset: str, metric_key: str
) -> float | None:
    short = metric_key.removeprefix("gen-val/")
    ds_anchors = anchors.get(dataset, {})
    info = ds_anchors.get(short)
    if not info:
        return None
    target = info.get("target")
    tol = info.get("tolerance_x", 1.0)
    if target is None:
        return None
    if info.get("direction") == "smaller_is_better":
        return float(target) * float(tol)
    return float(target)


def plot_metric(
    *,
    metric_key: str,
    pod_series: dict[str, tuple[list[int], list[float]]],
    pod_colors: dict[str, str],
    threshold: float | None,
    width: int,
    height: int,
) -> None:
    plt.clf()
    plt.theme("clear")
    plt.plotsize(width, height)
    plt.title(metric_key)
    plt.xlabel("step")

    has_data = False
    for pod, (xs, ys) in pod_series.items():
        if not xs:
            continue
        plt.plot(xs, ys, color=pod_colors[pod], marker="braille", label=pod[:24])
        has_data = True

    if threshold is not None and has_data:
        all_xs = [x for xs, _ in pod_series.values() for x in xs]
        if all_xs:
            xmin, xmax = min(all_xs), max(all_xs)
            plt.plot([xmin, xmax], [threshold, threshold], color="white", marker="dot")

    if not has_data:
        plt.text("no data", x=0.5, y=0.5, alignment="center")

    plt.show()


def short_pod(uid: str) -> str:
    parts = uid.split("/")
    if len(parts) >= 4:
        return f"{parts[-2]}/{parts[-1][:8]}"
    return uid


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    _ = p.add_argument(
        "--rounds-jsonl",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/rounds.jsonl"),
    )
    _ = p.add_argument(
        "--anchors-yaml",
        type=Path,
        default=Path("docs/experiments/sweep/smallest-config-2026-04-29/anchors.yaml"),
    )
    _ = p.add_argument("--entity", default="graph_denoise_team")
    _ = p.add_argument("--project", default="tmgg-smallest-config-sweep")
    _ = p.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    _ = p.add_argument("--max-rows", type=int, default=200)
    _ = p.add_argument("--width", type=int, default=88, help="per-panel width chars")
    _ = p.add_argument("--height", type=int, default=14, help="per-panel height chars")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]

    rounds = read_rounds(args.rounds_jsonl)
    pending = find_pending_launches(rounds)
    if not pending:
        print("no running launches in rounds.jsonl", file=sys.stderr)
        return 1

    anchors = load_anchors(args.anchors_yaml)

    pod_uids = [r["run_uid"] for r in pending]
    datasets = {r["run_uid"]: r.get("dataset", "") for r in pending}
    pod_colors = {
        uid: POD_COLORS[i % len(POD_COLORS)] for i, uid in enumerate(pod_uids)
    }

    print(f"# {len(pending)} running pods")
    for uid in pod_uids:
        print(f"  {pod_colors[uid]:>8}  {short_pod(uid)}  ds={datasets[uid]}")

    histories: dict[str, list[dict[str, Any]]] = {}
    for uid in pod_uids:
        try:
            histories[uid] = fetch_history(
                entity=args.entity,
                project=args.project,
                run_uid=uid,
                max_rows=args.max_rows,
            )
        except Exception as exc:  # noqa: BLE001 — diagnostic CLI
            print(f"# SKIP {uid}: {type(exc).__name__}: {exc}")
            histories[uid] = []

    for metric in metrics:
        pod_series: dict[str, tuple[list[int], list[float]]] = {}
        for uid in pod_uids:
            hist = histories.get(uid, [])
            xs: list[int] = []
            ys: list[float] = []
            for r in hist:
                v = r.get(metric)
                s = r.get("trainer/global_step")
                if v is None or s is None:
                    continue
                try:
                    xs.append(int(s))
                    ys.append(float(v))
                except (TypeError, ValueError):
                    continue
            pod_series[short_pod(uid)] = (xs, ys)

        # Use the first running pod's dataset for anchor lookup; the
        # sweep is dataset-uniform per round so this is safe.
        dataset = next(iter(datasets.values()))
        threshold = threshold_for_metric(anchors, dataset, metric)
        plot_metric(
            metric_key=metric,
            pod_series=pod_series,
            pod_colors={short_pod(uid): pod_colors[uid] for uid in pod_uids},
            threshold=threshold,
            width=args.width,
            height=args.height,
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
