"""Evaluation metrics for graph denoising and generation quality.

Combines per-graph spectral metrics (eigenvalue error, subspace distance,
accuracy) with distributional metrics (MMD on degree/clustering/spectral
distributions, ORCA orbit counting) and structural checks (SBM accuracy,
planarity, uniqueness, novelty).
"""

from tmgg.experiments._shared_utils.evaluation_metrics.graph_evaluator import (
    EvaluationResults,
    GraphEvaluator,
)
from tmgg.experiments._shared_utils.evaluation_metrics.reference_graphs import (
    generate_reference_graphs,
)

__all__ = [
    "EvaluationResults",
    "GraphEvaluator",
    "generate_reference_graphs",
]
