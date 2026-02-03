# /// script
# dependencies = [
#     "pandas>=2.0",
#     "pyarrow",
#     "rich",
# ]
# ///
"""Convert MCP query results to parquet format.

This script processes the JSON output files from W&B MCP queries
and converts them to the unified parquet format used by wandb-tools.

Usage:
    uv run scripts/export_via_mcp.py <json_file> -o output.parquet
"""

import argparse
import json
from pathlib import Path

import pandas as pd
from rich.console import Console

console = Console()


def flatten_config(config_str: str) -> dict:
    """Parse and flatten W&B config JSON string."""
    try:
        config = json.loads(config_str)
    except (json.JSONDecodeError, TypeError):
        return {}

    flat = {}
    for key, value in config.items():
        if key.startswith("_"):  # Skip internal W&B keys
            continue
        if isinstance(value, dict) and "value" in value:
            val = value["value"]
            if isinstance(val, dict):
                # Flatten nested dict
                for k2, v2 in val.items():
                    if not k2.startswith("_"):
                        flat[f"config_{key}_{k2}"] = (
                            str(v2) if isinstance(v2, dict | list) else v2
                        )
            else:
                flat[f"config_{key}"] = (
                    str(val) if isinstance(val, dict | list) else val
                )
        else:
            flat[f"config_{key}"] = (
                str(value) if isinstance(value, dict | list) else value
            )
    return flat


def flatten_metrics(summary_str: str) -> dict:
    """Parse and flatten W&B summary metrics JSON string."""
    try:
        summary = json.loads(summary_str)
    except (json.JSONDecodeError, TypeError):
        return {}

    metrics = {}
    for key, value in summary.items():
        if key.startswith("_"):
            continue
        if isinstance(value, int | float):
            clean_key = key.replace("/", "_").replace(".", "_").replace("-", "_")
            metrics[f"metric_{clean_key}"] = value
    return metrics


def process_mcp_json(json_path: Path) -> pd.DataFrame:
    """Process MCP query result JSON file into DataFrame."""
    with open(json_path) as f:
        data = json.load(f)

    # Navigate to runs
    runs_data = (
        data.get("result", {}).get("project", {}).get("runs", {}).get("edges", [])
    )

    records = []
    for edge in runs_data:
        node = edge.get("node", {})

        record = {
            "id": node.get("id"),
            "name": node.get("name"),
            "display_name": node.get("displayName"),
            "state": node.get("state"),
            "created_at": node.get("createdAt"),
            "_config_json": node.get("config", "{}"),
            "_summary_json": node.get("summaryMetrics", "{}"),
        }

        # Flatten config and metrics
        record.update(flatten_config(node.get("config", "{}")))
        record.update(flatten_metrics(node.get("summaryMetrics", "{}")))

        records.append(record)

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="Convert MCP JSON to parquet")
    parser.add_argument("inputs", nargs="+", type=Path, help="Input JSON files")
    parser.add_argument(
        "--output", "-o", type=Path, required=True, help="Output parquet file"
    )
    args = parser.parse_args()

    all_dfs = []
    for json_path in args.inputs:
        console.print(f"[dim]Processing {json_path}...[/dim]")
        df = process_mcp_json(json_path)
        all_dfs.append(df)
        console.print(f"  -> {len(df)} runs")

    if not all_dfs:
        console.print("[red]No data found[/red]")
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Convert mixed-type columns to string
    for col in combined.columns:
        if combined[col].apply(type).nunique() > 1:
            combined[col] = combined[col].astype(str)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(args.output, index=False)
    console.print(f"\n[green]Saved {len(combined)} runs to {args.output}[/green]")


if __name__ == "__main__":
    main()
