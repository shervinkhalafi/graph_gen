import pytorch_lightning as pl
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
            if A_noisy.ndim == 2:
                A_noisy_input = A_noisy.unsqueeze(0)
                V_noisy_input = V_noisy.unsqueeze(0)
            else:
                A_noisy_input = A_noisy
                V_noisy_input = V_noisy

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
    logger: Logger,
    trainer: pl.Trainer,
    best_model_path: str,
):
    # hack to get class method
    best_model = model.__class__.load_from_checkpoint(best_model_path)

    # Perform final evaluation across noise levels
    noise_levels = data_module.noise_levels
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
