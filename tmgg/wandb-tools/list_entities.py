# /// script
# dependencies = [
#     "wandb>=0.15",
#     "rich",
#     "python-dotenv",
# ]
# ///
"""List accessible W&B teams/entities and their projects.

Provides a quick overview of all entities the user has access to,
along with project counts and run statistics.

Usage:
    uv run wandb-tools/list_entities.py
    uv run wandb-tools/list_entities.py --entity graph_denoise_team
    uv run wandb-tools/list_entities.py --json > entities.json
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass

import wandb
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

console = Console()


def get_wandb_api(api_key: str | None = None) -> wandb.Api:
    """Get W&B API with credentials from .env if available."""
    load_dotenv()
    key = (
        api_key
        or os.environ.get("WANDB_API_KEY")
        or os.environ.get("GRAPH_DENOISE_TEAM_SERVICE")
    )
    if key:
        return wandb.Api(api_key=key)
    return wandb.Api()


@dataclass
class ProjectInfo:
    """Information about a single W&B project."""

    name: str
    entity: str
    run_count: int
    finished_count: int
    created_at: str | None = None


@dataclass
class EntityInfo:
    """Information about a W&B entity (user or team)."""

    name: str
    projects: list[ProjectInfo]

    @property
    def total_runs(self) -> int:
        return sum(p.run_count for p in self.projects)

    @property
    def total_finished(self) -> int:
        return sum(p.finished_count for p in self.projects)


def get_project_info(api: wandb.Api, entity: str, project_name: str) -> ProjectInfo:
    """Fetch info for a single project including run counts."""
    try:
        runs = api.runs(f"{entity}/{project_name}", per_page=1000)
        run_list = list(runs)
        run_count = len(run_list)
        finished_count = sum(1 for r in run_list if r.state == "finished")
    except Exception as e:
        console.print(
            f"[yellow]Warning: Could not fetch runs for {entity}/{project_name}: {e}[/yellow]",
            file=sys.stderr,
        )
        run_count = 0
        finished_count = 0

    return ProjectInfo(
        name=project_name,
        entity=entity,
        run_count=run_count,
        finished_count=finished_count,
    )


def list_entity_projects(api: wandb.Api, entity: str) -> EntityInfo:
    """List all projects for an entity with run counts."""
    projects = []
    try:
        api_projects = list(api.projects(entity))
        for proj in api_projects:
            info = get_project_info(api, entity, proj.name)
            projects.append(info)
    except wandb.errors.CommError as e:
        console.print(
            f"[red]Error accessing entity '{entity}': {e}[/red]", file=sys.stderr
        )
    except Exception as e:
        console.print(
            f"[red]Unexpected error for entity '{entity}': {e}[/red]", file=sys.stderr
        )

    return EntityInfo(name=entity, projects=projects)


def discover_entities(api: wandb.Api) -> list[str]:
    """Discover entities the current user has access to.

    W&B API doesn't provide a direct way to list all accessible entities,
    so we start with the authenticated user and any known team entities.
    """
    entities = []

    # Get current user
    try:
        user = api.viewer
        if user:
            entities.append(user.entity)
    except Exception:
        pass

    # Known team entities for this project
    known_teams = ["graph_denoise_team", "igorkraw"]
    for team in known_teams:
        if team not in entities:
            try:
                # Test if we can access this entity
                list(api.projects(team))
                entities.append(team)
            except Exception:
                pass

    return entities


def print_entities_table(entities: list[EntityInfo]) -> None:
    """Print entity/project information as a rich table."""
    for entity_info in entities:
        table = Table(
            title=f"[bold]{entity_info.name}[/bold] ({len(entity_info.projects)} projects)"
        )
        table.add_column("Project", style="cyan")
        table.add_column("Runs", justify="right")
        table.add_column("Finished", justify="right")
        table.add_column("Completion %", justify="right")

        # Sort by run count descending
        sorted_projects = sorted(
            entity_info.projects, key=lambda p: p.run_count, reverse=True
        )

        for proj in sorted_projects:
            completion = (
                f"{100 * proj.finished_count / proj.run_count:.0f}%"
                if proj.run_count > 0
                else "-"
            )
            table.add_row(
                proj.name,
                str(proj.run_count),
                str(proj.finished_count),
                completion,
            )

        console.print(table)
        console.print(
            f"[dim]Total: {entity_info.total_runs} runs, "
            f"{entity_info.total_finished} finished[/dim]\n"
        )


def entities_to_dict(entities: list[EntityInfo]) -> dict:
    """Convert entity info to JSON-serializable dict."""
    return {
        "entities": [
            {
                "name": e.name,
                "total_projects": len(e.projects),
                "total_runs": e.total_runs,
                "total_finished": e.total_finished,
                "projects": [
                    {
                        "name": p.name,
                        "run_count": p.run_count,
                        "finished_count": p.finished_count,
                    }
                    for p in e.projects
                ],
            }
            for e in entities
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="List W&B entities and their projects")
    parser.add_argument(
        "--entity",
        "-e",
        action="append",
        default=[],
        help="Specific entity to query (can specify multiple)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of table",
    )
    parser.add_argument(
        "--discover",
        action="store_true",
        help="Attempt to discover all accessible entities",
    )
    args = parser.parse_args()

    api = get_wandb_api()

    # Determine which entities to query
    if args.entity:
        entity_names = args.entity
    elif args.discover:
        entity_names = discover_entities(api)
        if not args.json:
            console.print(f"[dim]Discovered entities: {entity_names}[/dim]\n")
    else:
        # Default entities
        entity_names = ["igorkraw", "graph_denoise_team"]

    # Fetch entity info
    entities = []
    for entity_name in entity_names:
        if not args.json:
            console.print(f"[dim]Fetching projects for {entity_name}...[/dim]")
        info = list_entity_projects(api, entity_name)
        entities.append(info)

    # Output
    if args.json:
        print(json.dumps(entities_to_dict(entities), indent=2))
    else:
        console.print()
        print_entities_table(entities)


if __name__ == "__main__":
    main()
