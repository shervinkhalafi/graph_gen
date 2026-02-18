"""CLI helper for running a single Modal experiment.

The ``@app.local_entrypoint()`` now lives in ``_functions.py``. This
module provides a thin helper used by CLI commands that submit
experiments via ``modal.Function.from_name()``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tmgg.modal.app import MODAL_APP_NAME


def _select_modal_function(gpu_tier: str) -> Any:
    """Select the appropriate Modal function for the GPU tier.

    Tier mapping:
    - debug -> T4 (modal_execute_task_debug)
    - standard -> A10G (modal_execute_task)
    - fast, multi, h100 -> A100 (modal_execute_task_fast)
    """
    import modal

    if gpu_tier == "debug":
        func_name = "modal_execute_task_debug"
    elif gpu_tier in ("fast", "multi", "h100"):
        func_name = "modal_execute_task_fast"
    else:
        func_name = "modal_execute_task"
    return modal.Function.from_name(MODAL_APP_NAME, func_name)


def load_config(config_path: Path) -> dict[str, Any]:
    """Load experiment config from JSON file.

    Parameters
    ----------
    config_path
        Path to config JSON file.

    Returns
    -------
    dict
        Config dictionary containing run_id and nested config.
    """
    with open(config_path) as f:
        return json.load(f)
