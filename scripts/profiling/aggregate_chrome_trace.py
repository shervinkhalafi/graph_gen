"""Aggregate a chrome-format torch.profiler trace into a key_averages-style table.

Stream-parses the (potentially many-GB) JSON to avoid loading the whole
trace into memory. Replicates the shape of
``prof.key_averages().table(sort_by="cuda_time_total", row_limit=N)``
closely enough to be actionable: groups by ``name``, separates CPU vs
CUDA events by chrome-trace ``cat``/``ph`` heuristics, sums durations
and counts.

This exists because torch's own ``key_averages().table()`` runs
in-container at the end of a profiled run and can blow past Modal's
900 s heartbeat for large eval traces. The chrome trace itself
exports cheaply, so we export it from Modal and aggregate on the host.

Usage:
    uv run scripts/profiling/aggregate_chrome_trace.py <trace.json> [--row-limit 30]
"""

# /// script
# dependencies = [
#     "ijson>=3.2",
# ]
# ///

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import ijson


def aggregate(trace_path: Path) -> dict[str, dict[str, dict[str, float]]]:
    """Stream the trace and bucket by (op-class, name).

    Returns
    -------
    dict
        ``{class: {name: {"count": int, "total_dur_us": float}}}`` where
        ``class`` is one of ``"cuda"`` or ``"cpu"``.
    """
    buckets: dict[str, dict[str, dict[str, float]]] = {
        "cuda": defaultdict(lambda: {"count": 0, "total_dur_us": 0.0}),
        "cpu": defaultdict(lambda: {"count": 0, "total_dur_us": 0.0}),
    }
    n_seen = 0
    truncated = False
    with trace_path.open("rb") as fh:
        try:
            for ev in ijson.items(fh, "traceEvents.item"):
                n_seen += 1
                if n_seen % 1_000_000 == 0:
                    print(f"  parsed {n_seen:,} events", file=sys.stderr, flush=True)
                ph = ev.get("ph")
                if ph != "X":
                    continue
                cat = ev.get("cat") or ""
                name = ev.get("name") or "<anon>"
                dur = ev.get("dur")
                if dur is None:
                    continue
                if "kernel" in cat or "gpu_memcpy" in cat or cat == "kernel":
                    cls = "cuda"
                elif cat in ("cuda_runtime", "cuda_driver"):
                    # API call on CPU side — count under cpu, since it's host time
                    cls = "cpu"
                else:
                    cls = "cpu"
                row = buckets[cls][name]
                row["count"] += 1
                row["total_dur_us"] += float(dur)
        except ijson.common.IncompleteJSONError as e:
            # torch.profiler's chrome-trace exporter occasionally writes a
            # malformed event late in the stream (often around stack/correlation
            # metadata). The diffusion sample loop is highly repetitive
            # (num_samples * T * uniform-shape forwards), so the partial
            # aggregation is still representative for ranking hotspots.
            truncated = True
            print(
                f"  WARN: lexical error at event ~{n_seen:,}: {e}",
                file=sys.stderr,
                flush=True,
            )
    suffix = " (TRUNCATED — see warning above)" if truncated else ""
    print(f"  total events parsed: {n_seen:,}{suffix}", file=sys.stderr, flush=True)
    return buckets


def render_table(
    buckets: dict[str, dict[str, dict[str, float]]],
    row_limit: int,
    sort_by: str,
) -> str:
    """Render a key_averages-style table sorted by total CUDA or CPU time."""
    lines: list[str] = []
    bar = "-" * 110
    lines.append(bar)
    lines.append(
        f"{'Name':70s}  {'CUDA total':>14s}  {'CUDA %':>7s}  "
        f"{'Calls':>10s}  {'Avg (us)':>10s}"
    )
    lines.append(bar)

    def _fmt_total_us(us: float) -> str:
        if us >= 1_000_000:
            return f"{us / 1_000_000:.3f}s"
        if us >= 1_000:
            return f"{us / 1_000:.3f}ms"
        return f"{us:.3f}us"

    cuda_total = sum(r["total_dur_us"] for r in buckets["cuda"].values())
    cpu_total = sum(r["total_dur_us"] for r in buckets["cpu"].values())

    target = buckets["cuda"] if sort_by == "cuda" else buckets["cpu"]
    target_total = cuda_total if sort_by == "cuda" else cpu_total
    rows = sorted(target.items(), key=lambda kv: kv[1]["total_dur_us"], reverse=True)[
        :row_limit
    ]
    for name, row in rows:
        share = row["total_dur_us"] / target_total * 100 if target_total else 0
        avg = row["total_dur_us"] / row["count"] if row["count"] else 0
        name_show = name if len(name) <= 70 else name[:67] + "..."
        lines.append(
            f"{name_show:70s}  {_fmt_total_us(row['total_dur_us']):>14s}  "
            f"{share:6.2f}%  {row['count']:>10,}  {avg:>10.2f}"
        )
    lines.append(bar)
    lines.append(
        f"Self CUDA time total: {_fmt_total_us(cuda_total)}    "
        f"Self CPU time total: {_fmt_total_us(cpu_total)}"
    )
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    _ = p.add_argument("trace", type=Path)
    _ = p.add_argument("--row-limit", type=int, default=30)
    _ = p.add_argument(
        "--sort-by",
        choices=["cuda", "cpu"],
        default="cuda",
        help="Which dimension to rank rows by.",
    )
    _ = p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="If set, write the table to this file in addition to stdout.",
    )
    args = p.parse_args()

    print(
        f"# Streaming {args.trace} ({args.trace.stat().st_size / 1e9:.1f} GB)",
        file=sys.stderr,
    )
    buckets = aggregate(args.trace)

    cuda_table = render_table(buckets, args.row_limit, sort_by="cuda")
    cpu_table = render_table(buckets, args.row_limit, sort_by="cpu")
    body = (
        "Sorted by CUDA total\n"
        f"{cuda_table}\n\n"
        "Sorted by CPU total\n"
        f"{cpu_table}\n"
    )
    print(body)
    if args.out is not None:
        args.out.write_text(body)
        print(f"# wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
