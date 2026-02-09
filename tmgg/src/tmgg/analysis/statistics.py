"""Statistical utilities for experiment analysis.

Provides group-level summary statistics, pairwise significance testing
with effect-size interpretation, and Random-Forest-based hyperparameter
importance analysis.

Extracted from ``scripts/semantic_analysis.py`` (``GroupStats``,
``ComparisonResult``, ``compute_cohens_d``, ``interpret_effect_size``,
``analyze_grouping``) and ``scripts/analyze_experiments.py``
(``compute_importance``).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import LabelEncoder


@dataclass
class GroupStats:
    """Descriptive statistics for a single group within a semantic grouping.

    Attributes
    ----------
    name : str
        Group label (e.g. ``"digress"``, ``"spectral"``).
    n : int
        Number of observations in the group.
    best : float
        Best (max or min, depending on metric direction) observed value.
    mean : float
        Arithmetic mean of the metric.
    std : float
        Sample standard deviation (0.0 when ``n < 2``).
    min_val : float
        Minimum observed value.
    max_val : float
        Maximum observed value.
    """

    name: str
    n: int
    best: float
    mean: float
    std: float
    min_val: float
    max_val: float


@dataclass
class ComparisonResult:
    """Result of a pairwise statistical comparison between two groups.

    Attributes
    ----------
    group1 : str
        Name of the first (better-performing) group.
    group2 : str
        Name of the second group.
    diff : float
        Difference in group means (group1 - group2).
    t_stat : float
        Welch's t-statistic from ``scipy.stats.ttest_ind``.
    p_value : float
        Two-sided p-value.
    cohens_d : float
        Cohen's d effect size.
    significant : bool
        Whether ``p_value < 0.05``.
    effect_size : str
        Human-readable effect size: ``"negligible"``, ``"small"``,
        ``"medium"``, or ``"large"``.
    """

    group1: str
    group2: str
    diff: float
    t_stat: float
    p_value: float
    cohens_d: float
    significant: bool
    effect_size: str  # negligible, small, medium, large


def compute_cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    """Compute Cohen's d effect size between two groups.

    Uses pooled standard deviation.  Returns 0.0 when either group has
    fewer than 2 observations or when the pooled standard deviation is
    zero.

    Parameters
    ----------
    g1 : np.ndarray
        Observations from group 1.
    g2 : np.ndarray
        Observations from group 2.

    Returns
    -------
    float
        Cohen's d (positive when ``mean(g1) > mean(g2)``).
    """
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled_std)


def interpret_effect_size(d: float) -> str:
    """Map absolute Cohen's d to a conventional label.

    Parameters
    ----------
    d : float
        Cohen's d value (sign is ignored).

    Returns
    -------
    str
        One of ``"negligible"`` (|d| < 0.2), ``"small"`` (< 0.5),
        ``"medium"`` (< 0.8), or ``"large"``.
    """
    d = abs(d)
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def analyze_grouping(
    df: pd.DataFrame,
    group_col: str,
    metric: str = "test_acc",
    higher_is_better: bool = True,
    filter_func: Callable[[pd.DataFrame], pd.Series] | None = None,
) -> tuple[list[GroupStats], list[ComparisonResult]]:
    """Compute per-group statistics and consecutive pairwise comparisons.

    Groups are sorted by best value (descending if ``higher_is_better``,
    ascending otherwise).  Pairwise comparisons are between each group
    and the next-best group in sorted order.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment results dataframe.
    group_col : str
        Column name to group by.
    metric : str
        Column name of the metric to analyse.
    higher_is_better : bool
        If True, ``best`` is the maximum; otherwise the minimum.
    filter_func : callable or None
        Optional boolean mask function applied to ``df`` before
        analysis.  Receives the dataframe and should return a boolean
        Series.

    Returns
    -------
    tuple[list[GroupStats], list[ComparisonResult]]
        Sorted group statistics and pairwise comparison results.
        Both lists are empty when no valid data remains after filtering.
    """
    if filter_func:
        df = df[filter_func(df)].copy()  # pyright: ignore[reportAssignmentType]

    df = df[df[metric].notna()].copy()  # pyright: ignore[reportAssignmentType]

    if len(df) == 0:
        return [], []

    # Compute stats per group
    group_stats = []
    groups = df[group_col].dropna().unique()

    for group in sorted(groups, key=str):
        data = df[df[group_col] == group][metric]
        if len(data) == 0:
            continue

        group_stats.append(
            GroupStats(
                name=str(group),
                n=len(data),
                best=float(data.max() if higher_is_better else data.min()),
                mean=float(data.mean()),
                std=float(data.std()) if len(data) > 1 else 0.0,
                min_val=float(data.min()),
                max_val=float(data.max()),
            )
        )

    # Sort by best performance
    group_stats.sort(key=lambda x: x.best, reverse=higher_is_better)

    # Pairwise comparisons (each group vs next best)
    comparisons = []
    for i, gs in enumerate(group_stats):
        if i + 1 >= len(group_stats):
            break

        gs2 = group_stats[i + 1]
        g1_data = np.asarray(df[df[group_col] == gs.name][metric])
        g2_data = np.asarray(df[df[group_col] == gs2.name][metric])

        if len(g1_data) < 2 or len(g2_data) < 2:
            continue

        ttest_result = stats.ttest_ind(g1_data, g2_data)
        t_stat = float(ttest_result.statistic)  # pyright: ignore[reportAttributeAccessIssue]
        p_val = float(ttest_result.pvalue)  # pyright: ignore[reportAttributeAccessIssue]

        d = compute_cohens_d(g1_data, g2_data)

        comparisons.append(
            ComparisonResult(
                group1=gs.name,
                group2=gs2.name,
                diff=gs.mean - gs2.mean,
                t_stat=t_stat,
                p_value=p_val,
                cohens_d=d,
                significant=p_val < 0.05,
                effect_size=interpret_effect_size(d),
            )
        )

    return group_stats, comparisons


def compute_importance(df: pd.DataFrame, target: str = "test_mse") -> pd.DataFrame:
    """Compute hyperparameter importance using a Random Forest.

    Trains a ``RandomForestRegressor`` on encoded feature columns and
    reports both impurity-based and permutation-based importance.

    Parameters
    ----------
    df : pd.DataFrame
        Experiment results dataframe.  Expected to contain columns
        ``stage``, ``arch``, ``model_type``, ``k``, ``lr``, ``wd``,
        ``asymmetric``, and the ``target`` metric column.
    target : str
        Name of the target metric column (e.g. ``"test_mse"``).

    Returns
    -------
    pd.DataFrame
        Dataframe with columns ``feature``, ``importance_impurity``,
        ``importance_perm``, ``importance_perm_std``, sorted by
        permutation importance descending.
    """
    # Prepare features
    feature_cols = ["stage", "arch", "model_type", "k", "lr", "wd", "asymmetric"]
    df_valid: pd.DataFrame = df[df[target].notna() & (df[target] > 0)].copy()  # pyright: ignore[reportAssignmentType]

    # Encode categorical features
    encoders: dict[str, LabelEncoder] = {}
    x_data: dict[str, np.ndarray] = {}

    for col in feature_cols:
        if col in ["k", "lr", "wd", "asymmetric"]:
            x_data[col] = np.asarray(df_valid[col].fillna(-1))
        else:
            le = LabelEncoder()
            x_data[col] = le.fit_transform(  # type: ignore[assignment]  # pyright: ignore[reportArgumentType]
                df_valid[col].fillna("unknown").astype(str)
            )
            encoders[col] = le

    features = pd.DataFrame(x_data)
    target_values = np.asarray(df_valid[target])

    # Remove rows with NaN
    mask = ~np.isnan(features.values).any(axis=1) & ~np.isnan(target_values)
    features = features[mask]
    target_values = target_values[mask]

    # Train Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)  # type: ignore[no-untyped-call]
    rf.fit(features, target_values)  # type: ignore[no-untyped-call]

    # Feature importance (impurity-based)
    importance_impurity = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_impurity": rf.feature_importances_,  # type: ignore[union-attr]
        }
    ).sort_values("importance_impurity", ascending=False)

    # Permutation importance (more reliable)
    perm_result = permutation_importance(  # type: ignore[no-untyped-call]
        rf, features, target_values, n_repeats=10, random_state=42, n_jobs=-1
    )

    importance_perm = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_perm": perm_result.importances_mean,  # type: ignore[union-attr]  # pyright: ignore[reportAttributeAccessIssue]
            "importance_perm_std": perm_result.importances_std,  # type: ignore[union-attr]  # pyright: ignore[reportAttributeAccessIssue]
        }
    ).sort_values("importance_perm", ascending=False)

    # Merge results
    importance = importance_impurity.merge(importance_perm, on="feature")
    importance = importance.sort_values("importance_perm", ascending=False)

    return importance
