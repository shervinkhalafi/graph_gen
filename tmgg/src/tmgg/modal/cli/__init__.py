"""CLI tools for Modal cloud experiment orchestration.

Entry point: ``tmgg-modal``
"""

from __future__ import annotations

import click


@click.group()
def main() -> None:
    """Manage TMGG experiments on Modal cloud GPUs.

    \b
    Subcommands:
      run        Compose config locally and dispatch a single experiment
      evaluate   Dispatch MMD evaluation of checkpoints to Modal GPUs
      aggregate  Pull evaluation results from the Modal volume into Parquet
      datasets   Prepare or validate datasets on the shared volume
    """


from tmgg.modal.cli.aggregate import aggregate  # noqa: E402
from tmgg.modal.cli.datasets import datasets  # noqa: E402
from tmgg.modal.cli.evaluate import evaluate  # noqa: E402
from tmgg.modal.cli.run import run  # noqa: E402

main.add_command(run)
main.add_command(evaluate)
main.add_command(aggregate)
main.add_command(datasets)

if __name__ == "__main__":
    main()
