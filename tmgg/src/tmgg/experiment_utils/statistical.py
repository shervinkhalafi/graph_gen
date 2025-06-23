"""Statistical analysis utilities for graph denoising experiments."""

import numpy as np
import torch
from scipy import stats
from typing import Dict, List, Tuple, Optional, Union
import warnings


def compute_confidence_interval(data: Union[List[float], np.ndarray], 
                               confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute confidence interval for a dataset.
    
    Args:
        data: Data points
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)  # Standard error of the mean
    
    # Use t-distribution for small samples, normal for large
    if n < 30:
        t_val = stats.t.ppf((1 + confidence) / 2, n - 1)
        margin_error = t_val * sem
    else:
        z_val = stats.norm.ppf((1 + confidence) / 2)
        margin_error = z_val * sem
    
    return mean, mean - margin_error, mean + margin_error


def paired_t_test(results_a: List[float], 
                  results_b: List[float],
                  alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Perform paired t-test between two sets of results.
    
    Args:
        results_a: Results from method A
        results_b: Results from method B
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
        
    Returns:
        Dictionary with test statistics
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)
    
    if len(results_a) != len(results_b):
        raise ValueError("Results arrays must have the same length")
    
    statistic, p_value = stats.ttest_rel(results_a, results_b, alternative=alternative)
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'mean_diff': float(np.mean(results_a - results_b)),
        'std_diff': float(np.std(results_a - results_b)),
    }


def wilcoxon_signed_rank_test(results_a: List[float], 
                             results_b: List[float],
                             alternative: str = 'two-sided') -> Dict[str, float]:
    """
    Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
    
    Args:
        results_a: Results from method A
        results_b: Results from method B
        alternative: Alternative hypothesis ('two-sided', 'less', 'greater')
        
    Returns:
        Dictionary with test statistics
    """
    results_a = np.array(results_a)
    results_b = np.array(results_b)
    
    if len(results_a) != len(results_b):
        raise ValueError("Results arrays must have the same length")
    
    statistic, p_value = stats.wilcoxon(results_a, results_b, alternative=alternative)
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'median_diff': float(np.median(results_a - results_b)),
    }


def friedman_test(*groups: List[float]) -> Dict[str, float]:
    """
    Perform Friedman test for comparing multiple related groups.
    
    Args:
        groups: Multiple groups of related measurements
        
    Returns:
        Dictionary with test statistics
    """
    statistic, p_value = stats.friedmanchisquare(*groups)
    
    return {
        'statistic': float(statistic),
        'p_value': float(p_value),
        'significant': p_value < 0.05,
        'num_groups': len(groups),
    }


def compute_effect_size_cohens_d(group_a: List[float], 
                               group_b: List[float]) -> float:
    """
    Compute Cohen's d effect size between two groups.
    
    Args:
        group_a: First group of measurements
        group_b: Second group of measurements
        
    Returns:
        Cohen's d effect size
    """
    group_a = np.array(group_a)
    group_b = np.array(group_b)
    
    mean_a = np.mean(group_a)
    mean_b = np.mean(group_b)
    
    # Pooled standard deviation
    n_a = len(group_a)
    n_b = len(group_b)
    pooled_std = np.sqrt(((n_a - 1) * np.var(group_a, ddof=1) + 
                         (n_b - 1) * np.var(group_b, ddof=1)) / (n_a + n_b - 2))
    
    cohens_d = (mean_a - mean_b) / pooled_std
    return float(cohens_d)


def bootstrap_confidence_interval(data: Union[List[float], np.ndarray],
                                statistic_func=np.mean,
                                n_bootstrap: int = 1000,
                                confidence: float = 0.95) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        data: Original data
        statistic_func: Function to compute statistic (default: mean)
        n_bootstrap: Number of bootstrap samples
        confidence: Confidence level
        
    Returns:
        Tuple of (original_statistic, lower_bound, upper_bound)
    """
    data = np.array(data)
    n = len(data)
    
    # Original statistic
    original_stat = statistic_func(data)
    
    # Bootstrap samples
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n, replace=True)
        bootstrap_stats.append(statistic_func(bootstrap_sample))
    
    bootstrap_stats = np.array(bootstrap_stats)
    
    # Compute confidence interval
    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return float(original_stat), float(lower_bound), float(upper_bound)


def multiple_comparisons_correction(p_values: List[float], 
                                  method: str = 'bonferroni') -> List[float]:
    """
    Apply multiple comparisons correction to p-values.
    
    Args:
        p_values: List of p-values
        method: Correction method ('bonferroni', 'holm', 'fdr_bh')
        
    Returns:
        List of corrected p-values
    """
    p_values = np.array(p_values)
    
    if method == 'bonferroni':
        corrected = p_values * len(p_values)
        corrected = np.minimum(corrected, 1.0)
    elif method == 'holm':
        # Holm-Bonferroni method
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        n = len(p_values)
        
        corrected_sorted = np.zeros_like(sorted_p)
        for i in range(n):
            corrected_sorted[i] = min(1.0, sorted_p[i] * (n - i))
        
        # Ensure monotonicity
        for i in range(1, n):
            corrected_sorted[i] = max(corrected_sorted[i], corrected_sorted[i-1])
        
        # Restore original order
        corrected = np.zeros_like(p_values)
        corrected[sorted_indices] = corrected_sorted
    elif method == 'fdr_bh':
        # Benjamini-Hochberg FDR correction
        sorted_indices = np.argsort(p_values)
        sorted_p = p_values[sorted_indices]
        n = len(p_values)
        
        corrected_sorted = sorted_p * n / (np.arange(1, n + 1))
        corrected_sorted = np.minimum(corrected_sorted, 1.0)
        
        # Ensure monotonicity (reverse)
        for i in range(n - 2, -1, -1):
            corrected_sorted[i] = min(corrected_sorted[i], corrected_sorted[i + 1])
        
        # Restore original order
        corrected = np.zeros_like(p_values)
        corrected[sorted_indices] = corrected_sorted
    else:
        raise ValueError(f"Unknown correction method: {method}")
    
    return corrected.tolist()


def compute_statistical_summary(results: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """
    Compute comprehensive statistical summary for experimental results.
    
    Args:
        results: Dictionary mapping method names to lists of results
        
    Returns:
        Dictionary with statistical summaries for each method
    """
    summary = {}
    
    for method_name, values in results.items():
        values = np.array(values)
        
        summary[method_name] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'iqr': float(np.percentile(values, 75) - np.percentile(values, 25)),
            'n_samples': len(values),
        }
        
        # Add confidence interval
        mean, ci_lower, ci_upper = compute_confidence_interval(values)
        summary[method_name]['ci_lower'] = ci_lower
        summary[method_name]['ci_upper'] = ci_upper
        summary[method_name]['ci_width'] = ci_upper - ci_lower
    
    return summary


def compare_methods_pairwise(results: Dict[str, List[float]], 
                           test_type: str = 'paired_t') -> Dict[str, Dict[str, float]]:
    """
    Perform pairwise statistical comparisons between methods.
    
    Args:
        results: Dictionary mapping method names to lists of results
        test_type: Type of test ('paired_t', 'wilcoxon')
        
    Returns:
        Dictionary with pairwise comparison results
    """
    method_names = list(results.keys())
    comparisons = {}
    
    for i, method_a in enumerate(method_names):
        for j, method_b in enumerate(method_names[i+1:], i+1):
            comparison_key = f"{method_a}_vs_{method_b}"
            
            if test_type == 'paired_t':
                test_result = paired_t_test(results[method_a], results[method_b])
            elif test_type == 'wilcoxon':
                test_result = wilcoxon_signed_rank_test(results[method_a], results[method_b])
            else:
                raise ValueError(f"Unknown test type: {test_type}")
            
            # Add effect size
            test_result['cohens_d'] = compute_effect_size_cohens_d(
                results[method_a], results[method_b]
            )
            
            comparisons[comparison_key] = test_result
    
    return comparisons