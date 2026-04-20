"""Illustrate the SBM-diversity knob: 4 diversity levels × 4 example graphs.

Renders a 4×4 grid of adjacency-matrix spy plots. Rows correspond to
``diversity ∈ {0.0, 0.33, 0.67, 1.0}``; columns are four independent samples
at each diversity level, drawn under the Phase 4 parametrisation
(``num_nodes=50``, ``num_blocks=4``, ``p_intra ∈ (0.3, 0.9)``,
``p_inter ∈ (0.01, 0.2)``).

Runs in the project env — do NOT add a PEP 723 inline header. Execute with
``uv run scripts/plot_phase4_sbm_illustration.py``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tmgg.data.datasets.sbm import generate_sbm_batch

OUT_DIR = Path("docs/reports/2026-04-19-phase4-eigenvalue-study/figures")
FILENAME_STEM = "F0_sbm_diversity_examples"

DIVERSITIES = (0.0, 0.33, 0.67, 1.0)
N_EXAMPLES = 4
NUM_NODES = 50
NUM_BLOCKS = 4
P_INTRA_RANGE = (0.3, 0.9)
P_INTER_RANGE = (0.01, 0.2)
SEED = 42


def _midpoint(lo: float, hi: float) -> float:
    return (lo + hi) / 2.0


def sample_batch(diversity: float, seed: int) -> np.ndarray:
    if diversity == 0.0:
        return generate_sbm_batch(
            N_EXAMPLES,
            NUM_NODES,
            num_blocks=NUM_BLOCKS,
            p_intra=_midpoint(*P_INTRA_RANGE),
            p_inter=_midpoint(*P_INTER_RANGE),
            seed=seed,
        )
    return generate_sbm_batch(
        N_EXAMPLES,
        NUM_NODES,
        num_blocks=NUM_BLOCKS,
        p_intra=P_INTRA_RANGE,
        p_inter=P_INTER_RANGE,
        diversity=diversity,
        seed=seed,
    )


def main() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "axes.titlesize": 8,
            "axes.labelsize": 8,
        }
    )

    fig, axes = plt.subplots(
        len(DIVERSITIES),
        N_EXAMPLES,
        figsize=(6.75, 6.75),
        constrained_layout=True,
    )

    for row, diversity in enumerate(DIVERSITIES):
        batch = sample_batch(diversity, seed=SEED)
        for col in range(N_EXAMPLES):
            ax = axes[row, col]
            ax.imshow(batch[col], cmap="Greys", interpolation="nearest", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            edge_count = int(batch[col].sum() // 2)
            if col == 0:
                ax.set_ylabel(f"d={diversity:g}", fontweight="bold")
            ax.set_title(f"|E|={edge_count}", fontsize=7)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"{FILENAME_STEM}.{ext}", bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(
        f"wrote F0 (rows={len(DIVERSITIES)}, cols={N_EXAMPLES}) "
        f"→ figures/{FILENAME_STEM}.{{pdf,png}}"
    )


if __name__ == "__main__":
    main()
