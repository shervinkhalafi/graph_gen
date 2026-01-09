# /// script
# dependencies = [
#     "wandb>=0.15",
#     "rich",
# ]
# ///
"""
Fetch all experiment runs from W&B and save to JSON for analysis.

Usage:
    uv run scripts/fetch_wandb_runs.py
    uv run scripts/fetch_wandb_runs.py --entity graph_denoise_team
    uv run scripts/fetch_wandb_runs.py --output results/wandb_export.json
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import wandb
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

console = Console()


def extract_metrics(summary: dict) -> dict:
    """Extract relevant metrics from run summary, handling various naming conventions."""
    metrics = {}

    # Test metrics (try multiple naming conventions)
    for key in ["test/mse", "test_mse", "test/loss"]:
        if key in summary:
            metrics["test_mse"] = summary[key]
            break

    for key in ["test/subspace_distance", "test_subspace", "test/subspace"]:
        if key in summary:
            metrics["test_subspace"] = summary[key]
            break

    for key in ["test/eigenvalue_error", "test_eigenval", "test/eigenval"]:
        if key in summary:
            metrics["test_eigenval"] = summary[key]
            break

    # Final metrics at different epsilon values
    eps_values = ["0.01", "0.03", "0.1", "0.2", "0.3"]
    for eps in eps_values:
        for key in [f"final/mse_eps_{eps}", f"final_mse_{eps.replace('.', '')}"]:
            if key in summary:
                metrics[f"final_mse_{eps.replace('.', '')}"] = summary[key]
                break
        for key in [
            f"final/subspace_eps_{eps}",
            f"final_subspace_{eps.replace('.', '')}",
        ]:
            if key in summary:
                metrics[f"final_subspace_{eps.replace('.', '')}"] = summary[key]
                break

    # Training metrics
    for key in ["train/loss", "train_loss"]:
        if key in summary:
            metrics["train_loss"] = summary[key]
            break

    for key in ["val/loss", "val_loss"]:
        if key in summary:
            metrics["val_loss"] = summary[key]
            break

    # Step/epoch info
    for key in ["_step", "trainer/global_step", "epoch"]:
        if key in summary:
            metrics["steps"] = summary[key]
            break

    return metrics


def fetch_project_runs(
    api: wandb.Api, entity: str, project: str, since_date: datetime | None = None
) -> list[dict]:
    """Fetch all runs from a single project, optionally filtered by date."""
    runs_data = []
    try:
        filters = {}
        if since_date:
            filters = {"created_at": {"$gte": since_date.isoformat()}}
        runs = api.runs(f"{entity}/{project}", filters=filters if filters else None)
        for run in runs:
            summary = dict(run.summary) if run.summary else {}
            config = dict(run.config) if run.config else {}

            run_data = {
                "id": run.id,
                "name": run.name,
                "display_name": run.display_name,
                "state": run.state,
                "entity": entity,
                "project": project,
                "created_at": run.created_at,
                "tags": list(run.tags) if run.tags else [],
                "config": {
                    "lr": config.get("lr") or config.get("optimizer", {}).get("lr"),
                    "weight_decay": config.get("weight_decay")
                    or config.get("optimizer", {}).get("weight_decay"),
                    "batch_size": config.get("batch_size")
                    or config.get("data", {}).get("batch_size"),
                    "model": config.get("model", {}).get("_target_", "").split(".")[-1]
                    if isinstance(config.get("model"), dict)
                    else None,
                },
                **extract_metrics(summary),
            }
            runs_data.append(run_data)
    except Exception as e:
        console.print(f"[red]Error fetching {entity}/{project}: {e}[/red]")

    return runs_data


def fetch_all_runs(
    entities: list[str],
    project_filter: str | None = None,
    since_date: datetime | None = None,
) -> dict[str, list[dict]]:
    """Fetch runs from all projects across specified entities."""
    api = wandb.Api()
    all_data = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        for entity in entities:
            task = progress.add_task(f"Fetching projects for {entity}...", total=None)
            try:
                projects = list(api.projects(entity))
                progress.update(
                    task, description=f"Found {len(projects)} projects in {entity}"
                )

                for proj in projects:
                    project_name = proj.name
                    if project_filter and project_filter not in project_name:
                        continue

                    progress.update(
                        task, description=f"Fetching {entity}/{project_name}..."
                    )
                    runs = fetch_project_runs(api, entity, project_name, since_date)
                    if runs:
                        key = f"{entity}/{project_name}"
                        all_data[key] = runs
                        progress.update(
                            task,
                            description=f"  → {len(runs)} runs from {project_name}",
                        )

            except Exception as e:
                console.print(f"[red]Error listing projects for {entity}: {e}[/red]")

    return all_data


def print_summary(data: dict[str, list[dict]]) -> None:
    """Print a summary table of fetched data."""
    table = Table(title="W&B Export Summary")
    table.add_column("Project", style="cyan")
    table.add_column("Runs", justify="right")
    table.add_column("Finished", justify="right")
    table.add_column("Best test_mse", justify="right")

    total_runs = 0
    for project, runs in sorted(data.items()):
        n_runs = len(runs)
        n_finished = sum(1 for r in runs if r["state"] == "finished")
        test_mses = [r.get("test_mse") for r in runs if r.get("test_mse") is not None]
        best_mse = f"{min(test_mses):.4f}" if test_mses else "-"

        table.add_row(project, str(n_runs), str(n_finished), best_mse)
        total_runs += n_runs

    console.print(table)
    console.print(
        f"\n[bold]Total: {total_runs} runs across {len(data)} projects[/bold]"
    )


def main():
    parser = argparse.ArgumentParser(description="Fetch W&B experiment data")
    parser.add_argument(
        "--entity",
        "-e",
        action="append",
        default=[],
        help="W&B entity (can specify multiple). Default: igorkraw, graph_denoise_team",
    )
    parser.add_argument(
        "--project-filter",
        "-p",
        default=None,
        help="Only fetch projects containing this string",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="wandb_runs_export.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--flat",
        action="store_true",
        help="Output as flat list instead of grouped by project",
    )
    parser.add_argument(
        "--since",
        "-s",
        default=None,
        help="Only fetch runs created since this date (YYYY-MM-DD) or relative like '2d' for 2 days",
    )

    args = parser.parse_args()

    entities = args.entity if args.entity else ["igorkraw", "graph_denoise_team"]

    # Parse since date
    since_date = None
    if args.since:
        if args.since.endswith("d"):
            days = int(args.since[:-1])
            since_date = datetime.now() - __import__("datetime").timedelta(days=days)
        else:
            since_date = datetime.fromisoformat(args.since)
        console.print(f"[dim]Filtering runs since {since_date.isoformat()}[/dim]\n")

    console.print(f"[bold]Fetching W&B data from entities: {entities}[/bold]\n")

    data = fetch_all_runs(entities, args.project_filter, since_date)

    if not data:
        console.print(
            "[yellow]No data fetched. Check entity names and permissions.[/yellow]"
        )
        return

    print_summary(data)

    # Prepare output
    if args.flat:
        output_data = []
        for runs in data.values():
            output_data.extend(runs)
    else:
        output_data = data

    # Add metadata
    export = {
        "metadata": {
            "exported_at": datetime.now().isoformat(),
            "entities": entities,
            "project_filter": args.project_filter,
            "total_runs": sum(len(runs) for runs in data.values()),
            "total_projects": len(data),
        },
        "data": output_data,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(export, f, indent=2, default=str)

    console.print(f"\n[green]Saved to {output_path}[/green]")


if __name__ == "__main__":
    main()
