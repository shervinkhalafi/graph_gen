"""Plotting utilities for graph denoising experiments."""

from collections.abc import Callable
from typing import Any, Literal

import matplotlib.axes
import matplotlib.figure
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from tmgg.data.datasets.graph_types import GraphData


def _to_2d_numpy(x: np.ndarray | torch.Tensor) -> np.ndarray:
    """Convert input to a 2D numpy array, stripping batch dimensions."""
    arr: np.ndarray = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    while arr.ndim > 2:
        arr = arr[0]
    return arr


def plot_graph_denoising_comparison(
    A_clean: np.ndarray | torch.Tensor,
    noise_fn: Callable[
        [np.ndarray | torch.Tensor, float],
        np.ndarray | torch.Tensor | tuple[np.ndarray | torch.Tensor, Any, Any],
    ],
    denoise_fn: Callable[
        [np.ndarray | torch.Tensor], np.ndarray | torch.Tensor | tuple[Any, ...]
    ],
    noise_level: float,
    noise_type: str = "Unknown",
    title_prefix: str = "",
    cmap: str = "viridis",
    figsize: tuple[int, int] = (30, 5),
    save_path: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> matplotlib.figure.Figure:
    """
    General-purpose visualization for graph denoising experiments.

    This function creates a 6-panel visualization showing:
    1. Clean graph (binary ground truth)
    2. Noisy graph (after noise applied)
    3. Predicted graph (thresholded at 0.5)
    4. Prediction error (binary difference)
    5. Logits (raw model output)
    6. Logit error (logits - clean)

    Args:
        A_clean: Clean adjacency matrix (numpy array or torch tensor)
        noise_fn: Function that adds noise. Should accept (matrix, noise_level) and
                 return noisy_matrix (or tuple with noisy_matrix first)
        denoise_fn: Function that denoises. Should accept noisy matrix and return
                   either predictions alone or (predictions, logits) tuple
        noise_level: Noise level parameter (e.g., epsilon)
        noise_type: Name of the noise type for labeling
        title_prefix: Optional prefix for the plot title
        cmap: Colormap to use for visualization
        figsize: Figure size as (width, height)
        save_path: Optional path to save the plot
        vmin: Minimum value for color scale (if None, auto-determined)
        vmax: Maximum value for color scale (if None, auto-determined)

    Returns:
        Matplotlib figure object
    """
    A_clean_np = _to_2d_numpy(A_clean)

    # Add noise
    A_noisy_raw = noise_fn(A_clean, noise_level)
    A_noisy_np = _to_2d_numpy(
        A_noisy_raw[0] if isinstance(A_noisy_raw, tuple) else A_noisy_raw
    )

    # Denoise — may return (predictions, logits) or just predictions
    denoise_result: np.ndarray | torch.Tensor | tuple[Any, ...] = denoise_fn(
        A_noisy_raw[0] if isinstance(A_noisy_raw, tuple) else A_noisy_raw
    )
    A_logits_np: np.ndarray | None
    if isinstance(denoise_result, tuple):
        A_probs_np = _to_2d_numpy(denoise_result[0])
        A_logits_np = _to_2d_numpy(denoise_result[1])
    else:
        A_probs_np = _to_2d_numpy(denoise_result)
        A_logits_np = None

    # Threshold predictions at 0.5
    A_pred_np = (A_probs_np > 0.5).astype(float)

    # Calculate prediction error (binary difference)
    A_pred_error_np = A_pred_np - A_clean_np

    # Determine number of columns based on whether we have logits
    n_cols = 6 if A_logits_np is not None else 4
    fig, axes_raw = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    axes: np.ndarray = (
        np.array(axes_raw) if not isinstance(axes_raw, np.ndarray) else axes_raw
    )

    # Determine color scale for binary matrices
    if vmin is None:
        vmin = 0.0
    if vmax is None:
        vmax = 1.0

    # Col 0: Clean graph
    im0 = axes[0].imshow(A_clean_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    _ = axes[0].set_title(
        f"{title_prefix}Clean" if title_prefix else "Clean", fontsize=12
    )
    _ = axes[0].axis("off")
    _ = plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Col 1: Noisy graph
    im1 = axes[1].imshow(A_noisy_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    _ = axes[1].set_title(f"Noisy ({noise_type}, ε={noise_level})", fontsize=12)
    _ = axes[1].axis("off")
    _ = plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    # Col 2: Predicted graph (thresholded)
    im2 = axes[2].imshow(A_pred_np, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    _ = axes[2].set_title("Predicted (>0.5)", fontsize=12)
    _ = axes[2].axis("off")
    _ = plt.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    # Col 3: Prediction error
    im3 = axes[3].imshow(A_pred_error_np, cmap="bwr", vmin=-1, vmax=1, aspect="equal")
    _ = axes[3].set_title("Pred Error", fontsize=12)
    _ = axes[3].axis("off")
    _ = plt.colorbar(im3, ax=axes[3], fraction=0.046, pad=0.04)

    # Col 4 & 5: Logits and logit error (if available)
    if A_logits_np is not None:
        # Logits
        logit_vmax: float = max(
            abs(float(A_logits_np.min())), abs(float(A_logits_np.max()))
        )
        im4 = axes[4].imshow(
            A_logits_np, cmap="bwr", vmin=-logit_vmax, vmax=logit_vmax, aspect="equal"
        )
        _ = axes[4].set_title("Logits", fontsize=12)
        _ = axes[4].axis("off")
        _ = plt.colorbar(im4, ax=axes[4], fraction=0.046, pad=0.04)

        # Logit error (logits - clean, where clean is binary 0/1)
        # This shows how far the logits are from the "ideal" values
        A_logit_error_np: np.ndarray = A_logits_np - A_clean_np
        error_vmax: float = max(
            abs(float(A_logit_error_np.min())), abs(float(A_logit_error_np.max()))
        )
        im5 = axes[5].imshow(
            A_logit_error_np,
            cmap="bwr",
            vmin=-error_vmax,
            vmax=error_vmax,
            aspect="equal",
        )
        _ = axes[5].set_title("Logit Error", fontsize=12)
        _ = axes[5].axis("off")
        _ = plt.colorbar(im5, ax=axes[5], fraction=0.046, pad=0.04)

    _ = plt.tight_layout()

    if save_path:
        _ = plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_multi_noise_visualization(
    A_original: torch.Tensor,
    model: torch.nn.Module,
    noise_function: Callable[[torch.Tensor, float], torch.Tensor],
    noise_levels: list[float],
    device: str = "cpu",
    threshold: float = 0.5,
) -> matplotlib.figure.Figure:
    """
    Create visualization showing denoising results across multiple noise levels.

    This function creates a 5-row visualization for each noise level:
    1. Clean Graph
    2. Noisy Graph
    3. Denoised Graph (probabilities)
    4. Predicted Graph (thresholded)
    5. Delta (Denoised - Clean)

    Args:
        A_original: Original clean adjacency matrix
        model: Trained denoising model
        noise_function: Function to add noise
        noise_levels: List of noise levels to visualize
        device: Device for computation
        threshold: Threshold for binary prediction (default 0.5)

    Returns:
        Matplotlib figure
    """
    n_rows = 5
    fig_size = 4
    fig, axes_raw = plt.subplots(
        n_rows,
        len(noise_levels),
        figsize=(fig_size * len(noise_levels), fig_size * n_rows),
    )
    axes: np.ndarray = np.array(axes_raw).reshape(n_rows, -1)

    _ = model.eval()
    with torch.no_grad():
        for i, eps in enumerate(noise_levels):
            # Add noise
            A_noisy: torch.Tensor = noise_function(A_original, eps)
            A_noisy = A_noisy.to(device)

            # Get reconstruction
            A_noisy_input: torch.Tensor = (
                A_noisy.unsqueeze(0) if A_noisy.ndim == 2 else A_noisy
            )

            # Get reconstruction via GraphData interface
            result: GraphData = model(GraphData.from_adjacency(A_noisy_input))
            A_reconstructed: torch.Tensor = result.to_adjacency()

            A_original_np: np.ndarray = A_original.squeeze().cpu().numpy()
            A_noisy_np: np.ndarray = A_noisy.squeeze().cpu().numpy()
            A_reconstructed_np: np.ndarray = A_reconstructed.squeeze().cpu().numpy()
            A_predicted_np: np.ndarray = (A_reconstructed_np > threshold).astype(float)

            # Determine shared vmin/vmax for first 3 rows (continuous values)
            vmin: float = min(
                float(A_original_np.min()),
                float(A_noisy_np.min()),
                float(A_reconstructed_np.min()),
            )
            vmax: float = max(
                float(A_original_np.max()),
                float(A_noisy_np.max()),
                float(A_reconstructed_np.max()),
            )
            plot_params: dict[str, Any] = {
                "cmap": "viridis",
                "vmin": vmin,
                "vmax": vmax,
            }
            binary_params: dict[str, Any] = {"cmap": "viridis", "vmin": 0, "vmax": 1}

            # Row 0: Clean
            _ = axes[0, i].imshow(A_original_np, **binary_params)
            _ = axes[0, i].set_title(f"ε={eps:.2f}")
            if i == 0:
                _ = axes[0, i].set_ylabel("Clean", fontsize=12)

            # Row 1: Noisy
            _ = axes[1, i].imshow(A_noisy_np, **plot_params)
            if i == 0:
                _ = axes[1, i].set_ylabel("Noisy", fontsize=12)

            # Row 2: Denoised (probabilities)
            _ = axes[2, i].imshow(A_reconstructed_np, **plot_params)
            if i == 0:
                _ = axes[2, i].set_ylabel("Denoised", fontsize=12)

            # Row 3: Predicted (thresholded)
            _ = axes[3, i].imshow(A_predicted_np, **binary_params)
            if i == 0:
                _ = axes[3, i].set_ylabel("Predicted", fontsize=12)

            # Row 4: Delta
            delta: np.ndarray = A_reconstructed_np - A_original_np
            delta_vmax: float = max(float(np.abs(delta).max()), 1e-6)
            _ = axes[4, i].imshow(delta, cmap="bwr", vmin=-delta_vmax, vmax=delta_vmax)
            if i == 0:
                _ = axes[4, i].set_ylabel("Delta", fontsize=12)

            for row in range(n_rows):
                _ = axes[row, i].axis("off")

    _ = plt.tight_layout(pad=0.1, h_pad=0.5)
    return fig


# =============================================================================
# Node-Link Graph Visualization Functions
# =============================================================================


def _adjacency_to_graph(
    A: np.ndarray | torch.Tensor, threshold: float = 0.5
) -> "nx.Graph[int]":
    """
    Convert adjacency matrix to NetworkX Graph.

    Parameters
    ----------
    A
        Adjacency matrix, shape (n, n). Can be binary or probabilistic (values in [0,1]).
    threshold
        Values above this threshold are treated as edges. Default 0.5.

    Returns
    -------
    nx.Graph
        Undirected graph with nodes 0..n-1.
    """
    A_np = _to_2d_numpy(A)

    # Threshold to binary
    A_binary = (A_np > threshold).astype(float)

    # Create graph from adjacency (use upper triangle for undirected)
    G = nx.Graph()
    n = A_binary.shape[0]
    G.add_nodes_from(range(n))

    for i in range(n):
        for j in range(i + 1, n):
            if A_binary[i, j] > 0:
                G.add_edge(i, j)

    return G


def _compute_layout(
    G: "nx.Graph[int]",
    layout: Literal["spring", "spectral", "kamada_kawai"] = "spring",
    seed: int = 42,
) -> dict[int, tuple[float, float]]:
    """
    Compute node positions using the specified layout algorithm.

    Parameters
    ----------
    G
        NetworkX graph.
    layout
        Layout algorithm: "spring" (force-directed), "spectral" (eigenvector-based),
        or "kamada_kawai" (path-length optimization).
    seed
        Random seed for reproducible spring layout.

    Returns
    -------
    dict
        Node positions as {node_id: (x, y)}.
    """
    if layout == "spring":
        return nx.spring_layout(G, seed=seed, k=1.5 / (len(G) ** 0.5 + 1))
    elif layout == "spectral":
        # Spectral layout may fail for disconnected graphs
        if nx.is_connected(G) and len(G) > 1:
            try:
                return nx.spectral_layout(G)
            except Exception:
                return nx.spring_layout(G, seed=seed)
        else:
            return nx.spring_layout(G, seed=seed)
    elif layout == "kamada_kawai":
        # Kamada-Kawai requires connected graph
        if nx.is_connected(G) and len(G) > 1:
            try:
                return nx.kamada_kawai_layout(G)
            except Exception:
                return nx.spring_layout(G, seed=seed)
        else:
            return nx.spring_layout(G, seed=seed)
    else:
        raise ValueError(
            f"Unknown layout: {layout}. Use 'spring', 'spectral', or 'kamada_kawai'."
        )


def _draw_graph_with_diff(
    ax: matplotlib.axes.Axes,
    G: "nx.Graph[int]",
    G_reference: "nx.Graph[int] | None",
    pos: dict[int, tuple[float, float]],
    title: str,
    show_diff: bool = True,
    node_size: int = 100,
    node_color: str = "lightblue",
) -> None:
    """
    Draw a graph on the given axes, optionally highlighting edge differences.

    When show_diff=True and G_reference is provided, edges are colored:
    - Black: edges present in both graphs
    - Green: edges added (present in G but not in G_reference)
    - Red: edges removed (present in G_reference but not in G)

    Parameters
    ----------
    ax
        Matplotlib axes to draw on.
    G
        Graph to draw.
    G_reference
        Reference graph for computing edge differences. If None, all edges are black.
    pos
        Node positions from layout computation.
    title
        Axes title.
    show_diff
        If True and G_reference is provided, color edges by difference type.
    node_size
        Size of nodes in the plot.
    node_color
        Color of nodes in the plot.
    """
    # Draw nodes
    _ = nx.draw_networkx_nodes(
        G, pos, ax=ax, node_size=node_size, node_color=node_color, edgecolors="black"
    )

    if show_diff and G_reference is not None:
        # Compute edge sets
        edges_g = set(G.edges())
        edges_ref = set(G_reference.edges())

        # For undirected graphs, normalize edge representation
        edges_g_normalized = {tuple(sorted(e)) for e in edges_g}
        edges_ref_normalized = {tuple(sorted(e)) for e in edges_ref}

        # Categorize edges
        shared_edges = edges_g_normalized & edges_ref_normalized
        added_edges = edges_g_normalized - edges_ref_normalized
        removed_edges = edges_ref_normalized - edges_g_normalized

        # Draw shared edges (black)
        if shared_edges:
            _ = nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                edgelist=list(shared_edges),
                edge_color="black",
                width=1.0,
            )

        # Draw added edges (green)
        if added_edges:
            _ = nx.draw_networkx_edges(
                G, pos, ax=ax, edgelist=list(added_edges), edge_color="green", width=2.0
            )

        # Draw removed edges (red, dashed) - draw on reference graph
        if removed_edges:
            _ = nx.draw_networkx_edges(
                G_reference,
                pos,
                ax=ax,
                edgelist=list(removed_edges),
                edge_color="red",
                width=2.0,
                style="dashed",
            )
    else:
        # Draw all edges in black
        _ = nx.draw_networkx_edges(G, pos, ax=ax, edge_color="black", width=1.0)

    _ = ax.set_title(title, fontsize=12)
    _ = ax.axis("off")


def plot_graph_denoising_combined(
    A_clean: np.ndarray | torch.Tensor,
    A_noisy: np.ndarray | torch.Tensor,
    A_denoised: np.ndarray | torch.Tensor,
    noise_type: str = "Unknown",
    noise_level: float = 0.0,
    layout: Literal["spring", "spectral", "kamada_kawai"] = "spring",
    show_edge_diff: bool = True,
    max_nodes_for_layout: int = 50,
    figsize: tuple[int, int] | None = None,
    seed: int = 42,
    title_prefix: str = "",
    threshold: float = 0.5,
    save_path: str | None = None,
) -> matplotlib.figure.Figure:
    """
    Combined visualization showing both adjacency heatmaps and node-link diagrams.

    Creates a 2-row figure:
    - Row 1: Adjacency matrix heatmaps (always shown)
    - Row 2: Node-link network diagrams (only for graphs <= max_nodes_for_layout)

    This provides both the precise edge-level detail of heatmaps and the
    structural intuition of network layouts.

    Parameters
    ----------
    A_clean
        Clean adjacency matrix.
    A_noisy
        Noisy adjacency matrix.
    A_denoised
        Denoised adjacency matrix (can be probabilities).
    noise_type
        Label for noise type in titles.
    noise_level
        Noise level for labeling.
    layout
        Layout algorithm for network diagrams.
    show_edge_diff
        If True, highlight edge differences with colors in network diagrams.
    max_nodes_for_layout
        Skip network row for graphs exceeding this size.
    figsize
        Figure dimensions. If None, auto-computed based on content.
    seed
        Random seed for reproducible layouts.
    title_prefix
        Prefix for plot titles.
    threshold
        Threshold for binary edge prediction.
    save_path
        Optional save path.

    Returns
    -------
    matplotlib.figure.Figure
        Combined visualization figure.
    """
    A_clean_np = _to_2d_numpy(A_clean)
    A_noisy_np = _to_2d_numpy(A_noisy)
    A_denoised_np = _to_2d_numpy(A_denoised)

    n_nodes = A_clean_np.shape[0]
    include_network = n_nodes <= max_nodes_for_layout

    # Determine figure layout
    n_rows = 2 if include_network else 1
    n_cols = 4  # Clean, Noisy, Denoised, Delta

    if figsize is None:
        figsize = (5 * n_cols, 5 * n_rows)

    fig, axes_raw = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array([axes_raw]) if n_rows == 1 else np.array(axes_raw)

    # Row 0: Heatmaps
    # Determine color scale
    vmin = 0.0
    vmax = 1.0

    # Clean heatmap
    im0 = axes[0, 0].imshow(
        A_clean_np, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal"
    )
    _ = axes[0, 0].set_title(f"{title_prefix}Clean", fontsize=12)
    _ = axes[0, 0].axis("off")
    _ = plt.colorbar(im0, ax=axes[0, 0], fraction=0.046, pad=0.04)

    # Noisy heatmap
    im1 = axes[0, 1].imshow(
        A_noisy_np, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal"
    )
    _ = axes[0, 1].set_title(f"Noisy ({noise_type}, ε={noise_level:.2f})", fontsize=12)
    _ = axes[0, 1].axis("off")
    _ = plt.colorbar(im1, ax=axes[0, 1], fraction=0.046, pad=0.04)

    # Denoised heatmap
    im2 = axes[0, 2].imshow(
        A_denoised_np, cmap="viridis", vmin=vmin, vmax=vmax, aspect="equal"
    )
    _ = axes[0, 2].set_title("Denoised", fontsize=12)
    _ = axes[0, 2].axis("off")
    _ = plt.colorbar(im2, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Delta heatmap
    A_delta = A_denoised_np - A_clean_np
    delta_vmax = max(float(np.abs(A_delta).max()), 1e-6)
    im3 = axes[0, 3].imshow(
        A_delta, cmap="bwr", vmin=-delta_vmax, vmax=delta_vmax, aspect="equal"
    )
    _ = axes[0, 3].set_title("Delta (Denoised - Clean)", fontsize=12)
    _ = axes[0, 3].axis("off")
    _ = plt.colorbar(im3, ax=axes[0, 3], fraction=0.046, pad=0.04)

    # Row 1: Node-link diagrams (if graph is small enough)
    if include_network:
        G_clean = _adjacency_to_graph(A_clean_np, threshold=threshold)
        G_noisy = _adjacency_to_graph(A_noisy_np, threshold=threshold)
        G_denoised = _adjacency_to_graph(A_denoised_np, threshold=threshold)

        pos = _compute_layout(G_clean, layout=layout, seed=seed)

        # Clean network
        _draw_graph_with_diff(
            axes[1, 0], G_clean, None, pos, "Clean (network)", show_diff=False
        )

        # Noisy network
        _draw_graph_with_diff(
            axes[1, 1],
            G_noisy,
            G_clean if show_edge_diff else None,
            pos,
            "Noisy (network)",
            show_diff=show_edge_diff,
        )

        # Denoised network
        _draw_graph_with_diff(
            axes[1, 2],
            G_denoised,
            G_clean if show_edge_diff else None,
            pos,
            "Denoised (network)",
            show_diff=show_edge_diff,
        )

        # Edge diff summary panel
        axes[1, 3].axis("off")

        # Compute edge statistics
        edges_clean = {tuple(sorted(e)) for e in G_clean.edges()}
        edges_noisy = {tuple(sorted(e)) for e in G_noisy.edges()}
        edges_denoised = {tuple(sorted(e)) for e in G_denoised.edges()}

        noisy_added = len(edges_noisy - edges_clean)
        noisy_removed = len(edges_clean - edges_noisy)
        denoised_added = len(edges_denoised - edges_clean)
        denoised_removed = len(edges_clean - edges_denoised)
        denoised_correct = len(edges_denoised & edges_clean)

        summary_text = (
            f"Edge Statistics:\n\n"
            f"Clean: {len(edges_clean)} edges\n\n"
            f"Noisy:\n"
            f"  +{noisy_added} added, -{noisy_removed} removed\n\n"
            f"Denoised:\n"
            f"  +{denoised_added} added, -{denoised_removed} removed\n"
            f"  {denoised_correct}/{len(edges_clean)} correct"
        )
        _ = axes[1, 3].text(
            0.1,
            0.5,
            summary_text,
            fontsize=11,
            verticalalignment="center",
            family="monospace",
            transform=axes[1, 3].transAxes,
        )
        _ = axes[1, 3].set_title("Edge Diff Summary", fontsize=12)

    _ = plt.tight_layout()

    if save_path:
        _ = plt.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def create_network_denoising_figure(
    A_clean: np.ndarray | torch.Tensor,
    noise_fn: Callable[
        [np.ndarray | torch.Tensor, float],
        np.ndarray | torch.Tensor | tuple[np.ndarray | torch.Tensor, Any, Any],
    ],
    denoise_fn: Callable[
        [np.ndarray | torch.Tensor], np.ndarray | torch.Tensor | tuple[Any, ...]
    ],
    noise_level: float,
    noise_type: str = "Unknown",
    title_prefix: str = "",
    layout: Literal["spring", "spectral", "kamada_kawai"] = "spring",
    show_edge_diff: bool = True,
    max_nodes_for_layout: int = 50,
    seed: int = 42,
) -> matplotlib.figure.Figure | None:
    """
    Create a network denoising visualization with noise/denoise functions.

    This is a convenience wrapper that applies noise and denoising, then creates
    the combined heatmap + network visualization.

    Parameters
    ----------
    A_clean
        Clean adjacency matrix.
    noise_fn
        Function that adds noise: (matrix, noise_level) -> noisy_matrix.
    denoise_fn
        Function that denoises: (noisy_matrix) -> predictions or (predictions, logits).
    noise_level
        Noise parameter epsilon.
    noise_type
        Label for noise type in titles.
    title_prefix
        Prefix for plot titles.
    layout
        Layout algorithm for network diagrams.
    show_edge_diff
        Whether to color edges by difference from clean graph (green=added, red=removed).
    max_nodes_for_layout
        Skip network visualization for graphs exceeding this size.
    seed
        Random seed for reproducible layouts.

    Returns
    -------
    matplotlib.figure.Figure or None
        Combined visualization figure, or None if graph is too large and only
        heatmaps would be shown (in which case, use the standard heatmap functions).
    """
    # Add noise
    A_noisy_result = noise_fn(A_clean, noise_level)
    A_noisy: np.ndarray | torch.Tensor
    A_noisy = A_noisy_result[0] if isinstance(A_noisy_result, tuple) else A_noisy_result

    # Denoise
    denoise_result = denoise_fn(A_noisy)
    if isinstance(denoise_result, tuple):
        A_denoised = denoise_result[0]
    else:
        A_denoised = denoise_result

    return plot_graph_denoising_combined(
        A_clean=A_clean,
        A_noisy=A_noisy,
        A_denoised=A_denoised,
        noise_type=noise_type,
        noise_level=noise_level,
        layout=layout,
        show_edge_diff=show_edge_diff,
        max_nodes_for_layout=max_nodes_for_layout,
        seed=seed,
        title_prefix=title_prefix,
    )
