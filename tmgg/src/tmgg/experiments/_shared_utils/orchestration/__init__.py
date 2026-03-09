"""Experiment execution pipeline.

Wires Hydra configuration to data modules, models, Lightning trainers,
callbacks, sanity checks, and checkpoint resumption.
"""

from tmgg.experiments._shared_utils.orchestration.run_experiment import (
    generate_run_id,
    run_experiment,
)

__all__ = [
    "generate_run_id",
    "run_experiment",
]
