"""Evaluation metrics for graph denoising experiments."""

import numpy as np
import torch
from scipy.sparse.linalg import ArpackError, ArpackNoConvergence, eigsh


def _safe_eigsh(A: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Compute top-k eigenvalues/vectors with fallback on ARPACK failure.

    ARPACK can fail for ill-conditioned, near-degenerate, sparse, or zero
    matrices. This wrapper catches failures and falls back to dense
    eigendecomposition.

    Parameters
    ----------
    A : np.ndarray
        Symmetric matrix.
    k : int
        Number of eigenvalues/vectors to compute.

    Returns
    -------
    tuple
        (eigenvalues, eigenvectors) arrays. eigenvectors may be None on fallback.
    """
    try:
        eigenvalues, eigenvectors = eigsh(A, k=k, which="LM", maxiter=10000)
        return eigenvalues, eigenvectors
    except (ArpackNoConvergence, ArpackError):
        # Fallback to dense eigendecomposition (handles zero matrices,
        # convergence failures, and other ARPACK errors)
        eigenvalues = np.linalg.eigvalsh(A)
        # Sort by magnitude descending
        sorted_idx = np.argsort(np.abs(eigenvalues))[::-1][:k]
        return eigenvalues[sorted_idx], None


def compute_eigenvalue_error(
    A_true: torch.Tensor | np.ndarray,
    A_pred: torch.Tensor | np.ndarray,
    k: int = 4,
) -> float:
    """Compute normalized eigenvalue error between adjacency matrices.

    Parameters
    ----------
    A_true
        True adjacency matrix.
    A_pred
        Predicted adjacency matrix.
    k
        Number of top eigenvalues to compare.

    Returns
    -------
    float
        Normalized eigenvalue error: ||λ_pred - λ_true|| / ||λ_true||.
    """
    # Convert to numpy arrays
    A_true_np: np.ndarray = (
        A_true.detach().cpu().numpy() if isinstance(A_true, torch.Tensor) else A_true
    )
    A_pred_np: np.ndarray = (
        A_pred.detach().cpu().numpy() if isinstance(A_pred, torch.Tensor) else A_pred
    )

    if A_true_np.ndim == 3:
        A_true_np = A_true_np[0]
    if A_pred_np.ndim == 3:
        A_pred_np = A_pred_np[0]

    l_true, _ = _safe_eigsh(A_true_np, k=k)
    l_pred, _ = _safe_eigsh(A_pred_np, k=k)

    # Guard against division by near-zero norm (e.g., graphs with mostly isolated nodes)
    norm_true = float(np.linalg.norm(l_true))
    eigval_error = float(np.linalg.norm(l_pred - l_true)) / max(norm_true, 1e-10)
    return float(eigval_error)


def compute_subspace_distance(
    A_true: torch.Tensor | np.ndarray,
    A_pred: torch.Tensor | np.ndarray,
    k: int = 4,
) -> float:
    """Compute subspace distance via projection matrix difference.

    Parameters
    ----------
    A_true
        True adjacency matrix.
    A_pred
        Predicted adjacency matrix.
    k
        Number of top eigenvectors to compare.

    Returns
    -------
    float
        Frobenius norm of ||V_true V_true^T - V_pred V_pred^T||.
    """
    if isinstance(A_true, torch.Tensor):
        A_true = A_true.detach().cpu().numpy()
    if isinstance(A_pred, torch.Tensor):
        A_pred = A_pred.detach().cpu().numpy()

    if A_true.ndim == 3:
        A_true = A_true[0]
    if A_pred.ndim == 3:
        A_pred = A_pred[0]

    # Need eigenvectors, so use full dense fallback if ARPACK fails
    try:
        _, V_true = eigsh(A_true, k=k, which="LM", maxiter=10000)
    except (ArpackNoConvergence, ArpackError):
        eigenvalues, V_all = np.linalg.eigh(A_true)
        sorted_idx = np.argsort(np.abs(eigenvalues))[::-1][:k]
        V_true = V_all[:, sorted_idx]

    try:
        _, V_pred = eigsh(A_pred, k=k, which="LM", maxiter=10000)
    except (ArpackNoConvergence, ArpackError):
        eigenvalues, V_all = np.linalg.eigh(A_pred)
        sorted_idx = np.argsort(np.abs(eigenvalues))[::-1][:k]
        V_pred = V_all[:, sorted_idx]

    Proj_true = V_true @ V_true.T
    Proj_pred = V_pred @ V_pred.T

    subspace_distance = np.linalg.norm(Proj_true - Proj_pred, "fro")
    return float(subspace_distance)


def compute_accuracy(
    A_true: torch.Tensor | np.ndarray,
    A_pred: torch.Tensor | np.ndarray,
    threshold: float = 0.5,
) -> float:
    """Compute edge prediction accuracy.

    Parameters
    ----------
    A_true
        Ground truth binary adjacency matrix.
    A_pred
        Predicted probabilities in [0,1].
    threshold
        Classification threshold for predictions.

    Returns
    -------
    float
        Fraction of correctly predicted edges.
    """
    if isinstance(A_true, np.ndarray):
        A_true = torch.tensor(A_true, dtype=torch.float32)
    if isinstance(A_pred, np.ndarray):
        A_pred = torch.tensor(A_pred, dtype=torch.float32)

    if A_true.ndim == 3:
        A_true = A_true.squeeze(0)
    if A_pred.ndim == 3:
        A_pred = A_pred.squeeze(0)

    predictions = (A_pred > threshold).float()
    return (predictions == A_true).float().mean().item()


def compute_reconstruction_metrics(
    A_true: torch.Tensor | np.ndarray,
    A_pred: torch.Tensor | np.ndarray,
) -> dict[str, float]:
    """Compute reconstruction metrics for graph denoising.

    Expects predictions (post-sigmoid, in [0,1] range).

    Parameters
    ----------
    A_true
        True adjacency matrix (binary).
    A_pred
        Predicted adjacency matrix (probabilities in [0,1]).

    Returns
    -------
    dict
        Metrics: mse, frobenius_error, eigenvalue_error, subspace_distance, accuracy.
    """
    if isinstance(A_true, np.ndarray):
        A_true = torch.tensor(A_true, dtype=torch.float32)
    if isinstance(A_pred, np.ndarray):
        A_pred = torch.tensor(A_pred, dtype=torch.float32)

    if A_true.ndim == 3:
        A_true = A_true.squeeze(0)
    if A_pred.ndim == 3:
        A_pred = A_pred.squeeze(0)

    mse = torch.mean((A_true - A_pred) ** 2).item()
    frobenius_error = torch.norm(A_true - A_pred, p="fro").item()
    eigenvalue_error = compute_eigenvalue_error(A_true, A_pred)
    subspace_distance = compute_subspace_distance(A_true, A_pred)
    accuracy = compute_accuracy(A_true, A_pred)

    return {
        "mse": mse,
        "frobenius_error": frobenius_error,
        "eigenvalue_error": eigenvalue_error,
        "subspace_distance": subspace_distance,
        "accuracy": accuracy,
    }


def compute_batch_metrics(
    A_true_batch: torch.Tensor,
    A_pred_batch: torch.Tensor,
) -> dict[str, float]:
    """Compute metrics averaged over a batch.

    Parameters
    ----------
    A_true_batch
        Batch of true adjacency matrices.
    A_pred_batch
        Batch of predicted adjacency matrices (probabilities).

    Returns
    -------
    dict
        Averaged metrics over the batch, including accuracy.
    """
    batch_size = A_true_batch.shape[0]

    metrics_sum = {
        "mse": 0.0,
        "frobenius_error": 0.0,
        "eigenvalue_error": 0.0,
        "subspace_distance": 0.0,
        "accuracy": 0.0,
    }

    for i in range(batch_size):
        sample_metrics = compute_reconstruction_metrics(
            A_true_batch[i], A_pred_batch[i]
        )
        for key in metrics_sum:
            metrics_sum[key] += sample_metrics[key]

    metrics_avg = {key: value / batch_size for key, value in metrics_sum.items()}
    return metrics_avg


def evaluate_noise_robustness(
    model: torch.nn.Module,
    A_clean: torch.Tensor,
    noise_levels: list,
    noise_function,
    device: str = "cpu",
) -> dict[str, list]:
    """Evaluate model robustness across noise levels.

    Parameters
    ----------
    model
        Trained denoising model with predict() method.
    A_clean
        Clean adjacency matrix.
    noise_levels
        List of noise levels to test.
    noise_function
        Function to add noise: (A, eps) -> A_noisy.
    device
        Device for computation.

    Returns
    -------
    dict
        Metrics by noise level.
    """
    model.eval()

    metrics_by_noise = {
        "noise_levels": noise_levels,
        "mse": [],
        "eigenvalue_error": [],
        "subspace_distance": [],
        "accuracy": [],
    }

    with torch.no_grad():
        for eps in noise_levels:
            A_noisy, _, _ = noise_function(A_clean, eps)
            A_noisy = A_noisy.to(device)
            A_clean_tensor = A_clean.to(device)

            if A_noisy.ndim == 2:
                A_noisy = A_noisy.unsqueeze(0)
            if A_clean_tensor.ndim == 2:
                A_clean_tensor = A_clean_tensor.unsqueeze(0)

            logits = model(A_noisy)
            # Call predict method (exists on denoising models but not typed on Module)
            predict_fn = getattr(model, "predict")  # noqa: B009
            A_pred = predict_fn(logits)

            sample_metrics = compute_reconstruction_metrics(A_clean_tensor, A_pred)

            metrics_by_noise["mse"].append(sample_metrics["mse"])
            metrics_by_noise["eigenvalue_error"].append(
                sample_metrics["eigenvalue_error"]
            )
            metrics_by_noise["subspace_distance"].append(
                sample_metrics["subspace_distance"]
            )
            metrics_by_noise["accuracy"].append(sample_metrics["accuracy"])

    return metrics_by_noise
