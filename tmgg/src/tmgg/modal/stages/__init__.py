"""Stage-specific configuration generators.

.. deprecated::
    The ``run_stage1`` and ``run_stage2`` Modal functions have been removed.
    Use YAML ``stage_definitions/`` + ``generate_configs`` + ``launch_sweep``
    pipeline instead.

    ``generate_stage1_configs`` and ``generate_stage2_configs`` are retained
    for downstream use.
"""

from tmgg.modal.stages.stage1 import generate_stage1_configs
from tmgg.modal.stages.stage2 import generate_stage2_configs

__all__ = ["generate_stage1_configs", "generate_stage2_configs"]
