"""Shared figure utilities for TMGG analysis reports.

Provides thin, composable wrappers around matplotlib/seaborn that enforce
a consistent publication-quality style.  Every plotting function accepts
an optional ``ax`` parameter; when omitted a new figure and axes pair is
created automatically, so the functions work equally well standalone or
inside multi-panel layouts.

Usage::

    from tmgg.analysis.figures import setup_style, box_plot, save_figure
    import matplotlib.pyplot as plt

    setup_style()
    fig, ax = plt.subplots()
    box_plot(df, x="arch", y="test_mse", ax=ax)
    save_figure(fig, "arch_mse.png")
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# ---------------------------------------------------------------------------
# Style setup
# ---------------------------------------------------------------------------


def setup_style() -> None:
    """Configure matplotlib and seaborn for clean, publication-quality figures.

    Sets a white-grid seaborn theme with legible fonts, tightened tick
    labels, and a non-interactive backend so figures can be saved without
    a display server.
    """
    matplotlib.use("Agg")
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font_scale=1.1,
        rc={
            "figure.figsize": (8, 5),
            "figure.dpi": 150,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "legend.frameon": True,
            "legend.edgecolor": "0.8",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.1,
            "font.family": "sans-serif",
        },
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _ensure_ax(ax: Axes | None) -> tuple[Figure, Axes]:
    """Return a (figure, axes) pair, creating them if *ax* is None.

    Parameters
    ----------
    ax : Axes or None
        Existing axes to reuse, or None to create a fresh figure.

    Returns
    -------
    tuple[Figure, Axes]
        The figure that owns the axes and the axes themselves.
    """
    if ax is None:
        fig_or_subfig, ax = plt.subplots()
        assert isinstance(fig_or_subfig, Figure)
        fig = fig_or_subfig
    else:
        parent = ax.get_figure()
        if parent is None:
            raise ValueError(
                "Axes object has no parent Figure. "
                "Pass ax=None to create a new figure, or use "
                "plt.subplots() to create axes with a parent."
            )
        if not isinstance(parent, Figure):
            raise TypeError(
                f"Expected Figure, got {type(parent).__name__}. "
                "SubFigure axes are not supported."
            )
        fig = parent
    return fig, ax


def _set_title(ax: Axes, title: str | None) -> None:
    """Set the axes title if *title* is not None."""
    if title is not None:
        ax.set_title(title)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def box_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw a box plot with optional grouping.

    Parameters
    ----------
    data : pd.DataFrame
        Tidy dataframe.
    x : str
        Column mapped to the categorical axis.
    y : str
        Column mapped to the value axis.
    hue : str or None
        Optional column for colour-coded grouping.
    title : str or None
        Axes title.
    ax : Axes or None
        Axes to draw on.  Created if None.
    **kwargs
        Forwarded to ``seaborn.boxplot``.

    Returns
    -------
    Axes
        The axes with the plot drawn.
    """
    _, ax = _ensure_ax(ax)
    sns.boxplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    _set_title(ax, title)
    ax.tick_params(axis="x", rotation=45)
    return ax


def violin_plot(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str | None = None,
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw a violin plot with optional grouping.

    Parameters
    ----------
    data : pd.DataFrame
        Tidy dataframe.
    x : str
        Column mapped to the categorical axis.
    y : str
        Column mapped to the value axis.
    hue : str or None
        Optional column for colour-coded grouping.
    title : str or None
        Axes title.
    ax : Axes or None
        Axes to draw on.  Created if None.
    **kwargs
        Forwarded to ``seaborn.violinplot``.

    Returns
    -------
    Axes
        The axes with the plot drawn.
    """
    _, ax = _ensure_ax(ax)
    sns.violinplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    _set_title(ax, title)
    ax.tick_params(axis="x", rotation=45)
    return ax


def heatmap(
    data: pd.DataFrame | np.ndarray,
    annot: bool = True,
    cmap: str = "RdYlBu_r",
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw a heatmap.

    Parameters
    ----------
    data : pd.DataFrame or np.ndarray
        2-D data to visualise.
    annot : bool
        Annotate cells with numeric values.
    cmap : str
        Matplotlib colourmap name.
    title : str or None
        Axes title.
    ax : Axes or None
        Axes to draw on.  Created if None.
    **kwargs
        Forwarded to ``seaborn.heatmap``.

    Returns
    -------
    Axes
        The axes with the plot drawn.
    """
    _, ax = _ensure_ax(ax)
    sns.heatmap(data, annot=annot, cmap=cmap, ax=ax, **kwargs)
    _set_title(ax, title)
    return ax


def grouped_bar_chart(
    data: pd.DataFrame,
    x: str,
    y: str,
    hue: str,
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw a grouped bar chart.

    Parameters
    ----------
    data : pd.DataFrame
        Tidy dataframe.
    x : str
        Column mapped to the categorical axis.
    y : str
        Column mapped to the value axis.
    hue : str
        Column for colour-coded grouping.
    title : str or None
        Axes title.
    ax : Axes or None
        Axes to draw on.  Created if None.
    **kwargs
        Forwarded to ``seaborn.barplot``.

    Returns
    -------
    Axes
        The axes with the plot drawn.
    """
    _, ax = _ensure_ax(ax)
    sns.barplot(data=data, x=x, y=y, hue=hue, ax=ax, **kwargs)
    _set_title(ax, title)
    ax.tick_params(axis="x", rotation=45)
    return ax


def scatter_with_annotations(
    data: pd.DataFrame,
    x: str,
    y: str,
    labels: str | Sequence[str],
    title: str | None = None,
    ax: Axes | None = None,
    **kwargs: Any,
) -> Axes:
    """Draw a scatter plot with text annotations on each point.

    Parameters
    ----------
    data : pd.DataFrame
        Tidy dataframe.
    x : str
        Column mapped to the horizontal axis.
    y : str
        Column mapped to the vertical axis.
    labels : str or sequence of str
        Column name whose values label each point, or an explicit
        sequence of label strings.
    title : str or None
        Axes title.
    ax : Axes or None
        Axes to draw on.  Created if None.
    **kwargs
        Forwarded to ``seaborn.scatterplot``.

    Returns
    -------
    Axes
        The axes with the plot drawn.
    """
    _, ax = _ensure_ax(ax)
    sns.scatterplot(data=data, x=x, y=y, ax=ax, **kwargs)

    label_values: Sequence[str]
    if isinstance(labels, str):
        label_values = data[labels].astype(str).tolist()
    else:
        label_values = labels

    for i, label in enumerate(label_values):
        ax.annotate(
            label,
            (data[x].iloc[i], data[y].iloc[i]),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
            alpha=0.8,
        )

    _set_title(ax, title)
    return ax


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def save_figure(fig: Figure, path: str | Path, dpi: int = 150) -> Path:
    """Save a figure and close it to free memory.

    Parameters
    ----------
    fig : Figure
        Matplotlib figure to save.
    path : str or Path
        Destination file path.  Parent directories are created
        automatically.
    dpi : int
        Resolution in dots per inch.

    Returns
    -------
    Path
        The resolved path where the figure was saved.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return path.resolve()
