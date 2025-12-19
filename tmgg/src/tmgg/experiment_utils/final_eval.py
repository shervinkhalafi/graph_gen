import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import Logger

from tmgg.experiment_utils.data.data_module import GraphDataModule
from tmgg.experiment_utils.data.eigendecomposition import compute_eigendecomposition
from tmgg.experiment_utils.data.noise import add_digress_noise
from tmgg.experiment_utils.metrics import compute_reconstruction_metrics


def evaluate_across_noise_levels(
    model: pl.LightningModule,
    data_module: GraphDataModule,
    noise_levels: list[float],
) -> dict[str, list[float]]:
    """
    Evaluate model across different noise levels.

    Args:
        model: Trained model
        data_module: Data module
        noise_levels: List of noise levels to evaluate

    Returns:
        Dictionary with evaluation results
    """

    model.eval()  # pyright: ignore[reportUnusedCallResult]
    results = {"mse": [], "eigenvalue_error": [], "subspace_distance": []}

    # Get a sample for evaluation
    sample_A = data_module.get_sample_adjacency_matrix("test")

    with torch.no_grad():
        for eps in noise_levels:
            # Add noise
            A_noisy = add_digress_noise(sample_A, eps)
            _, V_noisy = compute_eigendecomposition(A_noisy)

            # Predict
            V_noisy_input = V_noisy.unsqueeze(0) if A_noisy.ndim == 2 else V_noisy

            # Move to model's device
            V_noisy_input = V_noisy_input.to(model.device)
            A_pred = model(V_noisy_input)

            # Compute metrics (move prediction back to CPU)
            metrics = compute_reconstruction_metrics(sample_A, A_pred.squeeze(0).cpu())

            results["mse"].append(metrics["mse"])
            results["eigenvalue_error"].append(metrics["eigenvalue_error"])
            results["subspace_distance"].append(metrics["subspace_distance"])

    return results


def final_eval(
    model: pl.LightningModule,
    data_module: GraphDataModule,
    logger: Logger | list[Logger],
    trainer: pl.Trainer,
    best_model_path: str,
    eval_noise_levels: list[float] | None = None,
):
    # Load best model with fallback for old checkpoint hyperparameter structures
    from tmgg.experiment_utils.checkpoint_utils import load_checkpoint_with_fallback

    best_model = load_checkpoint_with_fallback(model.__class__, best_model_path)

    # Perform final evaluation across noise levels
    noise_levels = (
        eval_noise_levels if eval_noise_levels is not None else data_module.noise_levels
    )
    final_results = evaluate_across_noise_levels(best_model, data_module, noise_levels)

    # Log final results
    if logger:
        from tmgg.experiment_utils.logging import log_metrics

        # Build all metrics first for efficient logging
        final_metrics = {}
        for metric_name, values in final_results.items():
            for i, eps in enumerate(noise_levels):
                final_metrics[f"final_{metric_name}_eps_{eps}"] = values[i]

        # Log all metrics at once
        log_metrics(logger, final_metrics, step=trainer.global_step)
