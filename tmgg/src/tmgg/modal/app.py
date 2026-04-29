"""Modal app constants and GPU configurations.

Pure constants — no ``import modal`` at module level. The actual
``modal.App`` instance lives in ``_functions.py``, the sole deployment
entry-point.
"""

# App name shared between deployment (_functions.py) and runtime
# lookups (runner.py uses modal.Function.from_name with this name).
MODAL_APP_NAME = "tmgg-spectral"

# GPU tier configurations
# Maps logical names to Modal GPU specifications (string format)
GPU_CONFIGS = {
    "debug": "T4",
    "standard": "A10G",
    "fast": "A100-40GB",
    "multi": "A100-40GB:2",
    "h100": "H100",
}

# Default timeouts per GPU tier (in seconds)
# Set to 24 hours to avoid premature termination during long experiments
DEFAULT_TIMEOUTS = {
    "debug": 86400,
    "standard": 86400,
    "fast": 86400,
    "multi": 86400,
    "h100": 86400,
}

# How long containers stay warm after completing a task (in seconds)
# Higher values improve container reuse during sweeps but cost more
DEFAULT_SCALEDOWN_WINDOW = 60


def resolve_modal_function_name(base_name: str, gpu_tier: str) -> str:
    """Build the Modal function name for a given GPU tier.

    Maps logical GPU tier to the appropriate function name suffix:
    ``debug`` -> ``_debug``, ``fast|multi|h100`` -> ``_fast``,
    everything else (e.g. ``standard``) -> no suffix.

    Parameters
    ----------
    base_name : str
        Base function name without tier suffix (e.g. ``"modal_run_cli"``).
    gpu_tier : str
        Logical GPU tier from ``GPU_CONFIGS``.

    Returns
    -------
    str
        Full function name with appropriate suffix.
    """
    if gpu_tier not in GPU_CONFIGS:
        raise ValueError(
            f"Unknown GPU tier {gpu_tier!r}. Valid tiers: {sorted(GPU_CONFIGS)}"
        )
    if gpu_tier == "debug":
        return f"{base_name}_debug"
    if gpu_tier in ("fast", "multi", "h100"):
        return f"{base_name}_fast"
    return base_name


# Per-tier CPU reservations (vCPU floor; bursts up to host limit).
# Memory is intentionally left at Modal's default — RAM on Modal is
# billed for what is reserved, so let workers grow to host capacity
# instead of paying for unused headroom. CPU reservations matter more
# for guaranteeing dataloader / RDKit-preprocessing throughput.
CPU_PROFILES = {
    "debug": 2.0,
    "standard": 4.0,
    "fast": 8.0,
    "multi": 12.0,
    "h100": 8.0,
}
