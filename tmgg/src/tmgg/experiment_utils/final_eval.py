import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import Logger

from tmgg.experiment_utils.data.data_module import GraphDataModule
from tmgg.experiment_utils.data.eigendecomposition import compute_eigendecomposition
from tmgg.experiment_utils.data.noise_generators import NoiseGenerator
from tmgg.experiment_utils.metrics import compute_reconstruction_metrics


def evaluate_across_noise_levels(
    model: pl.LightningModule,
    data_module: GraphDataModule,
    noise_levels: list[float],
    noise_generator: NoiseGenerator,
    num_eval_samples: int = 10,
) -> dict[str, list[float]]:
    """Evaluate model across different noise levels, averaging over multiple samples.

    For each noise level, draws ``num_eval_samples`` independent test graphs
    from the data module and averages the reconstruction metrics. This reduces
    variance compared to evaluating on a single random test graph.

    Parameters
    ----------
    model
        Trained model.
    data_module
        Data module providing test samples.
    noise_levels
        Noise intensities to evaluate at.
    noise_generator
        Noise generator matching the model's training noise type.
    num_eval_samples
        Number of test graphs to average over per noise level. Higher values
        reduce variance at the cost of evaluation time.

    Returns
    -------
    dict[str, list[float]]
        Per-noise-level MSE, eigenvalue error, and subspace distance,
        each averaged over ``num_eval_samples`` test graphs.
    """

    model.eval()  # pyright: ignore[reportUnusedCallResult]
    results: dict[str, list[float]] = {
        "mse": [],
        "eigenvalue_error": [],
        "subspace_distance": [],
    }

    with torch.no_grad():
        for eps in noise_levels:
            eps_metrics = {
                "mse": 0.0,
                "eigenvalue_error": 0.0,
                "subspace_distance": 0.0,
            }

            for _ in range(num_eval_samples):
                sample_A = data_module.get_sample_adjacency_matrix("test")

                # Add noise using the model's noise type
                A_noisy = noise_generator.add_noise(sample_A, eps)
                _, V_noisy = compute_eigendecomposition(A_noisy)

                # Predict
                V_noisy_input = V_noisy.unsqueeze(0) if A_noisy.ndim == 2 else V_noisy

                # Move to model's device
                V_noisy_input = V_noisy_input.to(model.device)
                A_pred = model(V_noisy_input)

                # Compute metrics (move prediction back to CPU)
                metrics = compute_reconstruction_metrics(
                    sample_A, A_pred.squeeze(0).cpu()
                )

                for k in eps_metrics:
                    eps_metrics[k] += metrics[k] / num_eval_samples

            for k in eps_metrics:
                results[k].append(eps_metrics[k])

    return results


def final_eval(
    model: pl.LightningModule,
    data_module: GraphDataModule,
    logger: Logger | list[Logger],
    trainer: pl.Trainer,
    best_model_path: str,
    noise_generator: NoiseGenerator,
    eval_noise_levels: list[float] | None = None,
):
    """Run final evaluation of the best checkpoint across noise levels.

    Parameters
    ----------
    model
        Original model instance (used to determine class for checkpoint loading).
    data_module
        Data module with test data already set up.
    logger
        Logger(s) for recording final metrics.
    trainer
        Trainer instance (for global_step reference).
    best_model_path
        Path to the best checkpoint file.
    noise_generator
        Noise generator matching the model's training noise type.
    eval_noise_levels
        Noise levels to evaluate at. Falls back to ``data_module.noise_levels``.
    """
    # Load best model with fallback for old checkpoint hyperparameter structures
    from tmgg.experiment_utils.checkpoint_utils import load_checkpoint_with_fallback

    best_model = load_checkpoint_with_fallback(model.__class__, best_model_path)

    # Perform final evaluation across noise levels
    noise_levels = (
        eval_noise_levels if eval_noise_levels is not None else data_module.noise_levels  # pyright: ignore[reportAttributeAccessIssue]
    )
    final_results = evaluate_across_noise_levels(
        best_model, data_module, noise_levels, noise_generator
    )

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
