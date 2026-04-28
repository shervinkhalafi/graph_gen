"""Side-by-side comparison of generated SBM graphs vs the train distribution.

Renders a single PNG with two stacked panels:
* top: 40 graphs sampled by the GDPO checkpoint (from ``samples.jsonl``)
* bottom: 40 graphs drawn from the SPECTRE SBM **train** split

Run from the project root with ``uv run python
analysis/digress-loss-check/validate-gdpo-sbm/visualize_samples.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch

from tmgg.data.data_modules.spectre_sbm import SpectreSBMDataModule

HERE = Path(__file__).resolve().parent
RUN_STAMP = "modal-20260427T071513"  # post-MMD-fix run
SAMPLES_PATH = HERE / "outputs" / RUN_STAMP / "samples.jsonl"
OUTPUT_PNG = HERE / "outputs" / RUN_STAMP / "graphs_vs_train.png"
OUTPUT_PDF = HERE / "outputs" / RUN_STAMP / "graphs_vs_train.pdf"
N_GRAPHS = 40
GRID_ROWS = 5
GRID_COLS = 8
SEED = 42


def load_generated_graphs(path: Path) -> list[nx.Graph]:
    """Load JSONL samples into NetworkX graphs preserving isolated nodes."""
    graphs: list[nx.Graph] = []
    with path.open() as fh:
        for line in fh:
            record = json.loads(line)
            g = nx.Graph()
            g.add_nodes_from(range(int(record["num_nodes"])))
            g.add_edges_from((int(u), int(v)) for u, v in record["edges"])
            graphs.append(g)
    return graphs


def load_train_graphs(num_graphs: int) -> list[nx.Graph]:
    """Pull *num_graphs* networkx graphs from the SBM train split.

    Mirrors :meth:`BaseGraphDataModule.get_reference_graphs` (which only
    supports val/test) for the train loader.
    """
    datamodule = SpectreSBMDataModule(batch_size=12, num_workers=0)
    datamodule.setup()

    loader = datamodule.train_dataloader()
    graphs: list[nx.Graph] = []
    for batch in loader:
        adj = batch.binarised_adjacency()  # (B, N, N)
        bs = adj.shape[0]
        for i in range(bs):
            if len(graphs) >= num_graphs:
                return graphs
            n = int(batch.node_mask[i].sum().item())
            A_np = adj[i, :n, :n].cpu().numpy()
            graphs.append(nx.from_numpy_array(A_np))
    if len(graphs) < num_graphs:
        raise RuntimeError(
            f"train split only yielded {len(graphs)} graphs, asked for {num_graphs}"
        )
    return graphs


def draw_panel(subfig, graphs: list[nx.Graph], title: str) -> None:
    """Render *graphs* into a 5x8 grid inside *subfig* with a panel title."""
    subfig.suptitle(title, fontsize=14, fontweight="bold")
    axes = subfig.subplots(GRID_ROWS, GRID_COLS)
    for idx, ax in enumerate(axes.flat):
        ax.set_axis_off()
        if idx >= len(graphs):
            continue
        g = graphs[idx]
        pos = nx.spring_layout(g, seed=0)
        nx.draw_networkx(
            g,
            pos=pos,
            ax=ax,
            node_size=8,
            width=0.3,
            with_labels=False,
            edge_color="#444",
            node_color="#1f77b4",
        )
        ax.set_title(f"n={g.number_of_nodes()}", fontsize=7)


def main() -> None:
    if not SAMPLES_PATH.exists():
        raise FileNotFoundError(f"samples.jsonl missing at {SAMPLES_PATH}")

    print(f"[1/3] Loading {N_GRAPHS} generated graphs from {SAMPLES_PATH} ...")
    gen_graphs = load_generated_graphs(SAMPLES_PATH)
    if len(gen_graphs) != N_GRAPHS:
        raise RuntimeError(
            f"expected {N_GRAPHS} generated graphs, got {len(gen_graphs)}"
        )

    print(f"[2/3] Loading {N_GRAPHS} train SBM graphs (seed={SEED}) ...")
    torch.manual_seed(SEED)
    train_graphs = load_train_graphs(N_GRAPHS)

    print(f"[3/3] Rendering panels and writing PNG+PDF to {OUTPUT_PNG.parent} ...")
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(GRID_COLS * 1.6, GRID_ROWS * 1.6 * 2 + 1.0))
    top_sub, bottom_sub = fig.subfigures(2, 1, hspace=0.05)
    draw_panel(top_sub, gen_graphs, f"Generated (n={N_GRAPHS})")
    draw_panel(bottom_sub, train_graphs, f"Train SBM (n={N_GRAPHS})")
    fig.savefig(OUTPUT_PNG, dpi=140, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print(f"      Wrote {OUTPUT_PNG} ({OUTPUT_PNG.stat().st_size / 1024:.1f} KiB)")
    print(f"      Wrote {OUTPUT_PDF} ({OUTPUT_PDF.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
