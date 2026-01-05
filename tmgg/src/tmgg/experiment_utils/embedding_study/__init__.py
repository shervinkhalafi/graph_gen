"""Embedding dimension study utilities.

This module provides tools for collecting embedding dimension statistics
across datasets and summarizing the results.
"""

from tmgg.experiment_utils.embedding_study.analysis import analyze_results
from tmgg.experiment_utils.embedding_study.collector import EmbeddingCollector

__all__ = ["EmbeddingCollector", "analyze_results"]
