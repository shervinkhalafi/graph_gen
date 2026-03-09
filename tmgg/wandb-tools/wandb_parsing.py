"""Run-name and config-column parsing for W&B experiment data.

Single source of truth for extracting structured fields from W&B run names
and flattened config columns.  Every W&B analysis script in this directory
should import from here rather than re-implementing these patterns.
"""

from __future__ import annotations

import re

import pandas as pd


def parse_stage(name: str) -> str:
    """Extract stage identifier from a run name.

    Parameters
    ----------
    name : str
        W&B run display name.

    Returns
    -------
    str
        Stage identifier (e.g. ``"stage2c"``, ``"stage1a"``) or
        ``"unknown"``/``"other"`` when the name is missing or
        unrecognised.
    """
    if pd.isna(name):
        return "unknown"
    name = str(name)
    # Check for specific stage patterns (more specific first)
    stage_patterns = [
        (r"stage2c", "stage2c"),
        (r"stage2b", "stage2b"),
        (r"stage2a", "stage2a"),
        (r"stage1f", "stage1f"),
        (r"stage1e", "stage1e"),
        (r"stage1d", "stage1d"),
        (r"stage1c", "stage1c"),
        (r"stage1b", "stage1b"),
        (r"stage1a", "stage1a"),
        (r"stage2", "stage2"),
        (r"stage1", "stage1"),
        (r"stage3", "stage3"),
    ]
    for pattern, stage in stage_patterns:
        if re.search(pattern, name.lower()):
            return stage
    return "other"


def parse_architecture(name: str) -> str:
    """Extract architecture type from a run name.

    Parameters
    ----------
    name : str
        W&B run display name.

    Returns
    -------
    str
        Architecture label (e.g. ``"gnn_all"``, ``"filter_bank"``,
        ``"linear_pe"``, ``"self_attention"``, ``"mlp"``,
        ``"digress_default"``) or ``"unknown"``/``"other"``.
    """
    if pd.isna(name):
        return "unknown"
    name = str(name).lower()
    arch_patterns = [
        # DiGress GNN variants (most specific first)
        (r"gnn_all", "gnn_all"),
        (r"gnn_qk", "gnn_qk"),
        (r"gnn_v", "gnn_v"),
        # Spectral variants (before spectral/asymmetric catch-alls)
        (r"filter_bank", "filter_bank"),
        (r"linear_pe", "linear_pe"),
        (r"self_attention_mlp", "self_attention_mlp"),
        (r"multilayer_self_attention", "multilayer_self_attention"),
        (r"self_attention", "self_attention"),
        # DiGress named variants
        (r"digress_transformer", "digress_transformer"),
        (r"digress_default", "digress_default"),
        # Cross-cutting modifier (catch-all for remaining asymmetric runs)
        (r"asymmetric", "asymmetric"),
        # Spectral catch-alls
        (r"spectral_linear", "spectral_linear"),
        (r"spectral", "spectral"),
        # Baselines
        (r"mlp", "mlp"),
        (r"linear", "linear"),
        # DiGress catch-all
        (r"digress", "digress"),
    ]
    for pattern, arch in arch_patterns:
        if re.search(pattern, name):
            return arch
    return "other"


def parse_model_type(name: str, config_target: str | None = None) -> str:
    """Extract model type from run name or config target string.

    Tries the ``config_target`` first (the Hydra ``_target_`` column) and
    falls back to substring matching on the run name.

    Parameters
    ----------
    name : str
        W&B run display name.
    config_target : str or None
        Value of the ``config_model__target_`` (or similar) column.

    Returns
    -------
    str
        One of ``"spectral"``, ``"digress"``, ``"gnn"``,
        ``"attention"``, ``"hybrid"``, or ``"unknown"``.
    """
    # Try config target first
    if config_target and not pd.isna(config_target):
        target_str = str(config_target).lower()
        if "spectral" in target_str:
            return "spectral"
        if "digress" in target_str:
            return "digress"
        if "gnn" in target_str:
            return "gnn"
        if "attention" in target_str:
            return "attention"
        if "hybrid" in target_str:
            return "hybrid"

    # Fall back to name parsing
    if pd.isna(name):
        return "unknown"
    name = str(name).lower()
    if "spectral" in name:
        return "spectral"
    if "digress" in name:
        return "digress"
    if "gnn" in name:
        return "gnn"
    return "unknown"


def parse_run_name_fields(name: str) -> dict[str, str | int | float | bool]:
    """Extract hyperparameters encoded in a run name.

    Common patterns recognised:

    - ``_k{value}``: k value for spectral models
    - ``_lr{value}``: learning rate
    - ``_wd{value}``: weight decay
    - ``_s{value}``: seed
    - ``_eps{value}``: epsilon value
    - presence of ``asymmetric`` anywhere in the name

    Parameters
    ----------
    name : str
        W&B run display name.

    Returns
    -------
    dict
        Mapping of parsed field names to values.  Keys that may appear:
        ``"k"`` (int), ``"lr_parsed"`` (str), ``"wd_parsed"`` (str),
        ``"seed_parsed"`` (int), ``"eps_parsed"`` (float),
        ``"asymmetric_flag"`` (bool), ``"pearl_flag"`` (bool).
    """
    parsed: dict[str, str | int | float | bool] = {}
    if pd.isna(name):
        return parsed
    name = str(name)

    # k value
    k_match = re.search(r"_k(\d+)", name)
    if k_match:
        parsed["k"] = int(k_match.group(1))

    # Learning rate
    lr_match = re.search(r"_lr([\d.e-]+)", name)
    if lr_match:
        parsed["lr_parsed"] = lr_match.group(1)

    # Weight decay
    wd_match = re.search(r"_wd([\d.e-]+)", name)
    if wd_match:
        parsed["wd_parsed"] = wd_match.group(1)

    # Seed
    seed_match = re.search(r"_s(\d+)(?:_|$)", name)
    if seed_match:
        parsed["seed_parsed"] = int(seed_match.group(1))

    # Epsilon
    eps_match = re.search(r"_eps([\d.]+)", name)
    if eps_match:
        parsed["eps_parsed"] = float(eps_match.group(1))

    # Asymmetric flag
    parsed["asymmetric_flag"] = "asymmetric" in name.lower()

    # Pearl embedding flag
    parsed["pearl_flag"] = "pearl" in name.lower()

    return parsed


def parse_protocol(row: pd.Series) -> str:
    """Determine whether a run used single-graph or distribution protocol.

    Single-graph means train/val/test splits use the *same* graph with
    varying noise levels.  Distribution means each split draws
    *different* graphs from a generative distribution.

    Parameters
    ----------
    row : pd.Series
        A single row of a W&B runs dataframe (flattened config columns).

    Returns
    -------
    str
        ``"single_graph"`` or ``"distribution"``.
    """
    # Check config_data_same_graph_all_splits
    same_graph = row.get("config_data_same_graph_all_splits")
    if bool(same_graph) is True or same_graph == "True":
        return "single_graph"

    # Check data module target
    target = row.get("config_data__target_", "")
    if bool(pd.notna(target)) and "SingleGraph" in str(target):
        return "single_graph"

    return "distribution"


def parse_dataset(row: pd.Series) -> str:
    """Extract dataset name from run data.

    Checks config columns first (more reliable), then falls back to
    substring matching on the run display name.

    Parameters
    ----------
    row : pd.Series
        A single row of a W&B runs dataframe.

    Returns
    -------
    str
        Dataset name string, e.g. ``"sbm"``, ``"sbm_small"``,
        ``"pyg_enzymes"``, ``"ego"``, ``"community"``, ``"grid"``,
        or ``"unknown"``.
    """
    # Priority: config columns over name parsing
    for col in [
        "config_data_dataset_name",
        "config_dataset_name",
        "config_data_graph_type",
        "config_graph_type",
    ]:
        val = row.get(col)
        if bool(pd.notna(val)) and str(val) != "nan":
            return str(val)

    # Fallback to name parsing for legacy runs
    name = str(row.get("name", "")).lower()
    if "pyg_enzymes" in name:
        return "pyg_enzymes"
    if "pyg_proteins" in name:
        return "pyg_proteins"
    if "ego" in name:
        return "ego"
    if "community" in name:
        return "community"
    if "grid" in name:
        return "grid"
    if "sbm_small" in name:
        return "sbm_small"
    if "sbm" in name:
        return "sbm"
    return "unknown"


def enrich_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add parsed/derived columns to a W&B runs dataframe.

    Adds: ``protocol``, ``stage``, ``arch``, ``model_type``, plus any
    hyperparameter fields encoded in the run name (``k``,
    ``lr_parsed``, ``wd_parsed``, ``seed_parsed``, ``eps_parsed``,
    ``asymmetric_flag``).  Also coerces known numeric config columns.

    Preserves raw JSON columns (``_config_json``, ``_summary_json``)
    if present.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe of W&B runs with at least a ``"name"`` column.

    Returns
    -------
    pd.DataFrame
        Copy of the input with additional parsed columns.
    """
    df = df.copy()

    # Parse protocol (single-graph vs distribution)
    df["protocol"] = df.apply(parse_protocol, axis=1)

    # Parse stage from name
    df["stage"] = df["name"].apply(parse_stage)

    # Parse architecture
    df["arch"] = df["name"].apply(parse_architecture)

    # Parse model type (using config if available)
    config_target_col = None
    for col in ["config_model__target_", "config_model_target"]:
        if col in df.columns:
            config_target_col = col
            break

    if config_target_col:
        df["model_type"] = df.apply(
            lambda row: parse_model_type(row["name"], row.get(config_target_col)),
            axis=1,
        )
    else:
        df["model_type"] = df["name"].apply(lambda n: parse_model_type(n, None))

    # Parse dataset
    df["dataset"] = df.apply(parse_dataset, axis=1)

    # Parse name-encoded fields
    parsed_fields = df["name"].apply(parse_run_name_fields).apply(pd.Series)
    for col in parsed_fields.columns:
        if col not in df.columns:
            df[col] = parsed_fields[col]

    # Extract numeric values from config columns where possible
    numeric_candidates = [
        "config_learning_rate",
        "config_weight_decay",
        "config_model_k",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
