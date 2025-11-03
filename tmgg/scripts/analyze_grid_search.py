#!/usr/bin/env python3
"""
Analyze grid search results from wandb.

This script fetches and analyzes the grid search results logged to wandb,
creating summary tables and plots for comparison.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
from pathlib import Path
import numpy as np


def fetch_wandb_runs(project_name="tmgg-grid-search-4k", entity=None):
    """
    Fetch all runs from the wandb project.
    
    Args:
        project_name: Name of the wandb project
        entity: wandb entity/team name (optional)
    
    Returns:
        DataFrame with run data
    """
    api = wandb.Api()
    
    if entity:
        runs = api.runs(f"{entity}/{project_name}")
    else:
        runs = api.runs(project_name)
    
    summary_list = []
    for run in runs:
        # Get summary metrics
        summary = {
            "name": run.name,
            "model": run.config.get("model_name", "unknown"),
            "noise_type": run.config.get("data", {}).get("noise_type", "unknown"),
            "noise_level": run.config.get("noise_level", 0),
            "learning_rate": run.config.get("learning_rate", 0),
            "val_loss": run.summary.get("val/loss", np.nan),
            "train_loss": run.summary.get("train_loss", np.nan),
            "best_val_loss": run.summary.get("best_val_loss", np.nan),
            "total_parameters": run.summary.get("total_parameters", 0),
            "epochs": run.summary.get("epoch", 0),
            "state": run.state,
            "tags": run.tags,
        }
        
        # Add per-noise-level metrics if available
        for eps in [0.05, 0.1, 0.2, 0.3, 0.4]:
            summary[f"val_{eps}_loss"] = run.summary.get(f"val_{eps}/loss", np.nan)
            summary[f"val_{eps}_accuracy"] = run.summary.get(f"val_{eps}/accuracy", np.nan)
        
        summary_list.append(summary)
    
    return pd.DataFrame(summary_list)


def analyze_results(df):
    """
    Analyze the grid search results.
    
    Args:
        df: DataFrame with run data
    
    Returns:
        Dictionary with analysis results
    """
    results = {}
    
    # Best configuration for each model-noise combination
    print("\n" + "="*70)
    print("BEST CONFIGURATIONS")
    print("="*70)
    
    for model in df['model'].unique():
        for noise_type in df['noise_type'].unique():
            subset = df[(df['model'] == model) & (df['noise_type'] == noise_type)]
            if not subset.empty:
                best_idx = subset['val_loss'].idxmin()
                if not pd.isna(best_idx):
                    best = subset.loc[best_idx]
                    print(f"\n{model.upper()} - {noise_type}:")
                    print(f"  Best LR: {best['learning_rate']}")
                    print(f"  Best noise level trained: {best['noise_level']}")
                    print(f"  Val Loss: {best['val_loss']:.6f}")
                    print(f"  Parameters: {best['total_parameters']:,}")
    
    # Average performance by architecture
    print("\n" + "="*70)
    print("AVERAGE PERFORMANCE BY ARCHITECTURE")
    print("="*70)
    
    for model in df['model'].unique():
        subset = df[df['model'] == model]
        print(f"\n{model.upper()}:")
        print(f"  Mean Val Loss: {subset['val_loss'].mean():.6f}")
        print(f"  Std Val Loss: {subset['val_loss'].std():.6f}")
        print(f"  Best Val Loss: {subset['val_loss'].min():.6f}")
    
    # Performance by noise type
    print("\n" + "="*70)
    print("PERFORMANCE BY NOISE TYPE")
    print("="*70)
    
    for noise_type in df['noise_type'].unique():
        subset = df[df['noise_type'] == noise_type]
        print(f"\n{noise_type.upper()}:")
        for model in df['model'].unique():
            model_subset = subset[subset['model'] == model]
            if not model_subset.empty:
                print(f"  {model}: {model_subset['val_loss'].mean():.6f} ± {model_subset['val_loss'].std():.6f}")
    
    return results


def create_plots(df, output_dir="outputs/grid_search/analysis"):
    """
    Create visualization plots for the grid search results.
    
    Args:
        df: DataFrame with run data
        output_dir: Directory to save plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Set style
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # 1. Heatmap of performance across noise levels and learning rates
    for model in df['model'].unique():
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'{model.upper()} - Performance Heatmap', fontsize=16)
        
        for idx, noise_type in enumerate(df['noise_type'].unique()):
            subset = df[(df['model'] == model) & (df['noise_type'] == noise_type)]
            if not subset.empty:
                pivot = subset.pivot_table(
                    values='val_loss',
                    index='noise_level',
                    columns='learning_rate',
                    aggfunc='mean'
                )
                
                sns.heatmap(pivot, annot=True, fmt='.4f', cmap='RdYlGn_r', ax=axes[idx])
                axes[idx].set_title(f'{noise_type}')
                axes[idx].set_xlabel('Learning Rate')
                axes[idx].set_ylabel('Noise Level')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{model}_heatmap.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # 2. Box plot comparing architectures
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Architecture Comparison Across Noise Types', fontsize=16)
    
    for idx, noise_type in enumerate(df['noise_type'].unique()):
        subset = df[df['noise_type'] == noise_type]
        sns.boxplot(data=subset, x='model', y='val_loss', ax=axes[idx])
        axes[idx].set_title(f'{noise_type}')
        axes[idx].set_xlabel('Model')
        axes[idx].set_ylabel('Validation Loss')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/architecture_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 3. Learning rate sensitivity
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle('Learning Rate Sensitivity Analysis', fontsize=16)
    
    for i, model in enumerate(df['model'].unique()):
        for j, noise_type in enumerate(df['noise_type'].unique()):
            subset = df[(df['model'] == model) & (df['noise_type'] == noise_type)]
            if not subset.empty:
                grouped = subset.groupby('learning_rate')['val_loss'].agg(['mean', 'std'])
                axes[i, j].errorbar(grouped.index, grouped['mean'], yerr=grouped['std'], 
                                  marker='o', capsize=5)
                axes[i, j].set_xscale('log')
                axes[i, j].set_title(f'{model} - {noise_type}')
                axes[i, j].set_xlabel('Learning Rate')
                axes[i, j].set_ylabel('Val Loss')
                axes[i, j].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/learning_rate_sensitivity.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nPlots saved to {output_dir}/")


def save_summary_table(df, output_dir="outputs/grid_search/analysis"):
    """
    Save summary tables as CSV files.
    
    Args:
        df: DataFrame with run data
        output_dir: Directory to save tables
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Full results
    df.to_csv(f"{output_dir}/full_results.csv", index=False)
    
    # Best configurations
    best_configs = []
    for model in df['model'].unique():
        for noise_type in df['noise_type'].unique():
            subset = df[(df['model'] == model) & (df['noise_type'] == noise_type)]
            if not subset.empty and not subset['val_loss'].isna().all():
                best_idx = subset['val_loss'].idxmin()
                best_configs.append(subset.loc[best_idx])
    
    best_df = pd.DataFrame(best_configs)
    best_df.to_csv(f"{output_dir}/best_configurations.csv", index=False)
    
    # Summary statistics
    summary_stats = df.groupby(['model', 'noise_type'])['val_loss'].agg(['mean', 'std', 'min', 'max'])
    summary_stats.to_csv(f"{output_dir}/summary_statistics.csv")
    
    print(f"Tables saved to {output_dir}/")


def main():
    """Main function to run the analysis."""
    print("Fetching results from wandb...")
    
    # Fetch data from wandb
    df = fetch_wandb_runs()
    
    if df.empty:
        print("No runs found in wandb project!")
        return
    
    print(f"Found {len(df)} runs")
    
    # Filter only completed runs
    df_completed = df[df['state'] == 'finished']
    print(f"Analyzing {len(df_completed)} completed runs")
    
    # Analyze results
    analyze_results(df_completed)
    
    # Create plots
    create_plots(df_completed)
    
    # Save tables
    save_summary_table(df_completed)
    
    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)


if __name__ == "__main__":
    main()