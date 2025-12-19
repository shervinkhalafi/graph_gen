"""Modal App definition and GPU configurations.

Provides the central Modal App instance and GPU tier mappings
used across all stage runners.
"""

import modal

# Modal App for TMGG spectral denoising experiments
app = modal.App("tmgg-spectral")

# GPU tier configurations
# Maps logical names to Modal GPU specifications
GPU_CONFIGS = {
    # Development/testing tier (cheapest)
    "debug": modal.gpu.T4(),
    # Standard tier for most experiments
    "standard": modal.gpu.A10G(),
    # Fast tier for larger models or time-sensitive runs
    "fast": modal.gpu.A100(size="40GB"),
    # Multi-GPU for DiGress or very large experiments
    "multi": modal.gpu.A100(size="40GB", count=2),
    # H100 tier for maximum performance (if available)
    "h100": modal.gpu.H100(),
}

# Default timeouts per GPU tier (in seconds)
DEFAULT_TIMEOUTS = {
    "debug": 600,  # 10 minutes
    "standard": 1800,  # 30 minutes
    "fast": 3600,  # 1 hour
    "multi": 7200,  # 2 hours
    "h100": 3600,  # 1 hour
}

# Memory configurations (for non-GPU containers)
MEMORY_CONFIGS = {
    "small": 2048,  # 2 GB
    "medium": 8192,  # 8 GB
    "large": 32768,  # 32 GB
}
