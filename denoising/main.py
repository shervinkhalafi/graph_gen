import os
import sys
import argparse
from pathlib import Path
from typing import Any
from collections.abc import Sequence

# Add denoising directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from scipy.sparse.linalg import eigsh
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.sbm import (  # pyright: ignore[reportImplicitRelativeImport]
    AdjacencyMatrixDataset,
    add_digress_noise,
    add_gaussian_noise,
    add_rotation_noise,
    generate_sbm_adjacency,
    random_skew_symmetric_matrix,
)
from src.models.attention import MultiLayerAttention  # pyright: ignore[reportImplicitRelativeImport]
from src.models.gnn import GNN, NodeVarGNN  # pyright: ignore[reportImplicitRelativeImport]


def parse_args(input_args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    # NEW#
    _ = parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="random seed for reproducibility",
    )

    _ = parser.add_argument(
        "--noise_type",
        type=str,
        default="Gaussian",
        help="noise type, can be 'Gaussian', 'Rotation', or 'Digress'",
    )

    _ = parser.add_argument(
        "--model_type",
        type=str,
        default="MultiLayerAttention",
        help="model type, can be 'MultiLayerAttention', 'GNN, or 'NodeVarGNN'",
    )

    _ = parser.add_argument(
        "--eps",
        type=float,
        default=0.5,
        help="noise level",
    )

    _ = parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="number of attention heads",
    )

    _ = parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="number of transformer layers",
    )

    _ = parser.add_argument(
        "--block_sizes",
        type=str,
        default="[10, 5, 3, 2]",
        help="block sizes",
    )

    _ = parser.add_argument(
        "--num_epochs",
        type=int,
        default=400,
        help="number of epochs",
    )

    _ = parser.add_argument(
        "--num_samples_per_epoch",
        type=int,
        default=128,
        help="number of samples per epoch",
    )

    _ = parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="batch size",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args


def main(args: argparse.Namespace) -> None:
    # Define different block sizes
    block_sizes: list[int] = eval(args.block_sizes)

    # Set probabilities for intra-block and inter-block connections
    p_intra = 1.0  # High probability for connections within blocks
    q_inter = 0.0  # Low probability for connections between blocks

    # Generate the adjacency matrix
    A: NDArray[np.float64] = generate_sbm_adjacency(block_sizes, p_intra, q_inter)

    # Create dataset and dataloader
    dataset = AdjacencyMatrixDataset(
        A, num_samples_per_epoch=args.num_samples_per_epoch
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    noise_type = args.noise_type
    eps = args.eps
    num_heads = args.num_heads
    num_layers = args.num_layers
    model_type = args.model_type

    k = 20  # the top k eigenvectors are used as denoising.
    skew: NDArray[np.float64] = random_skew_symmetric_matrix(k)
    model: nn.Module
    if model_type == "MultiLayerAttention":
        model = MultiLayerAttention(
            k, num_heads=num_heads, d_k=k, d_v=k, num_layers=num_layers, bias=True
        )
    elif model_type == "GNN":
        t = 4  # powers of the graph filter
        model = GNN(num_layers=num_layers, num_terms=t, feature_dim=k)
    elif model_type == "NodeVarGNN":
        t = 4  # powers of the graph filter
        model = NodeVarGNN(num_layers=num_layers, num_terms=t, feature_dim=k)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    criterion = nn.MSELoss()  # Mean Squared Error as the loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    n_epochs = args.num_epochs

    k_val = 4
    eigenvalues: NDArray[np.float64]
    V: NDArray[np.float64]
    eigenvalues, V = eigsh(A, k=k_val, which="LM", maxiter=10000)

    A_noisy_fixed: torch.Tensor
    V_noisy_fixed: torch.Tensor
    l_noisy_fixed: torch.Tensor
    if noise_type == "Rotation":
        A_noisy_fixed, V_noisy_fixed, l_noisy_fixed = add_rotation_noise(
            torch.tensor(A, dtype=torch.float32).unsqueeze(0), eps, skew
        )
        A_noisy_fixed = A_noisy_fixed.squeeze(0)
        V_noisy_fixed = V_noisy_fixed.squeeze(0)
        l_noisy_fixed = l_noisy_fixed.squeeze(0)

    elif noise_type == "Gaussian":
        A_noisy_fixed, V_noisy_fixed, l_noisy_fixed = add_gaussian_noise(A, eps)

    elif noise_type == "Digress":
        A_noisy_fixed, V_noisy_fixed, l_noisy_fixed = add_digress_noise(A, eps)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    loss_hist: NDArray[np.float64] = np.zeros(n_epochs)
    eigval_error_hist: NDArray[np.float64] = np.zeros(n_epochs)
    subspace_distance_hist: NDArray[np.float64] = np.zeros(n_epochs)
    for epoch in range(n_epochs):
        _ = model.train()  # Set model to training mode
        running_loss = 0.0
        for i, batch in enumerate(dataloader):
            # inputs = batch.unsqueeze(dim = 1)
            inputs = batch

            # add noise by rotating the eigenvectors
            noisy_inputs: torch.Tensor
            Vs: torch.Tensor
            l_noisy: torch.Tensor
            if noise_type == "Rotation":
                noisy_inputs, Vs, l_noisy = add_rotation_noise(inputs, eps, skew)

            elif noise_type == "Gaussian":
                noisy_inputs, Vs, l_noisy = add_gaussian_noise(inputs, eps)

            elif noise_type == "Digress":
                noisy_inputs, Vs, l_noisy = add_digress_noise(inputs, eps)
            else:
                raise ValueError(f"Unknown noise_type: {noise_type}")

            optimizer.zero_grad()  # Zero the gradients

            # Forward pass
            # the last layer's attention scores are used as the denoised adjacency matrix
            outputs: torch.Tensor
            if model_type == "MultiLayerAttention":
                outputs = model(Vs)[1][-1]
            elif model_type == "GNN":
                outputs = model(noisy_inputs)
            elif model_type == "NodeVarGNN":
                outputs = model(noisy_inputs)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")
            loss = criterion(outputs, inputs)

            # Backward pass
            loss.backward()

            # Update model parameters
            optimizer.step()

            running_loss += loss.item()

        loss_hist[epoch] = running_loss / len(dataloader)

        _ = model.eval()

        with torch.no_grad():
            A_denoised: torch.Tensor
            if model_type == "MultiLayerAttention":
                A_denoised = model(
                    torch.unsqueeze(torch.FloatTensor(V_noisy_fixed), 0)
                )[1][-1]
            elif model_type == "GNN":
                A_denoised = model(torch.unsqueeze(torch.FloatTensor(A_noisy_fixed), 0))
            elif model_type == "NodeVarGNN":
                A_denoised = model(torch.unsqueeze(torch.FloatTensor(A_noisy_fixed), 0))
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            A_denoised_np: NDArray[Any] = np.array(A_denoised.detach())

        l_denoised: NDArray[np.float64]
        V_denoised: NDArray[np.float64]
        l_denoised, V_denoised = eigsh(
            A_denoised_np[0], k=k_val, which="LM", maxiter=10000
        )

        eigval_error: np.floating[Any] = np.linalg.norm(
            l_denoised - eigenvalues
        ) / np.linalg.norm(eigenvalues)
        Proj_true: NDArray[np.float64] = V @ V.T
        Proj_denoised: NDArray[np.float64] = V_denoised @ V_denoised.T
        subspace_distance: np.floating[Any] = np.linalg.norm(
            Proj_true - Proj_denoised, "fro"
        )

        eigval_error_hist[epoch] = eigval_error
        subspace_distance_hist[epoch] = subspace_distance

        _ = model.train()

    _ = model.eval()  # Set model to evaluation mode

    # print(loss_hist)
    # plt.plot(loss_hist)
    eigenvalues, V = eigsh(A, k=k, which="LM", maxiter=10000)  # pyright: ignore[reportConstantRedefinition]

    A_noisy: torch.Tensor
    V_noisy: torch.Tensor
    if noise_type == "Rotation":
        A_noisy, V_noisy, l_noisy = add_rotation_noise(
            torch.tensor(A, dtype=torch.float32).unsqueeze(0), eps, skew
        )
        A_noisy = A_noisy.squeeze(0)
        V_noisy = V_noisy.squeeze(0)
        l_noisy = l_noisy.squeeze(0)

    elif noise_type == "Gaussian":
        A_noisy, V_noisy, l_noisy = add_gaussian_noise(A, eps)

    elif noise_type == "Digress":
        A_noisy, V_noisy, l_noisy = add_digress_noise(A, eps)
    else:
        raise ValueError(f"Unknown noise_type: {noise_type}")

    if model_type == "MultiLayerAttention":
        A_denoised = model(torch.unsqueeze(torch.FloatTensor(V_noisy), 0))[1][-1]
    elif model_type == "GNN":
        A_denoised = model(torch.unsqueeze(torch.FloatTensor(A_noisy), 0))
    elif model_type == "NodeVarGNN":
        A_denoised = model(torch.unsqueeze(torch.FloatTensor(A_noisy), 0))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    A_denoised_np = np.array(A_denoised.detach())

    l_denoised, V_denoised = eigsh(A_denoised_np[0], k=k, which="LM", maxiter=10000)

    V_noisy_np: NDArray[Any] = (
        V_noisy.numpy() if isinstance(V_noisy, torch.Tensor) else V_noisy
    )
    A_denoised_eigvalsonly: NDArray[np.float64] = (
        V_noisy_np @ np.diag(l_denoised) @ V_noisy_np.T
    )

    fig, axs = plt.subplots(3, 5, figsize=(12, 5))

    axs[0, 0].imshow(A)
    axs[0, 0].set_title("Original")

    axs[1, 0].imshow(V)
    axs[1, 0].set_title("Original Eigenvectors")

    axs[2, 0].imshow(np.diag(eigenvalues))
    axs[2, 0].set_title("Original Eigenvalues")

    axs[0, 1].imshow(A_noisy)
    axs[0, 1].set_title("Noisy")

    axs[1, 1].imshow(V_noisy)
    axs[1, 1].set_title("Noisy Eigenvectors")

    axs[2, 1].imshow(np.diag(l_noisy))
    axs[2, 1].set_title("Noisy Eigenvalues")

    axs[0, 2].imshow(A_denoised_np[0])
    axs[0, 2].set_title("Denoised")

    axs[1, 2].imshow(V_denoised)
    axs[1, 2].set_title("Denoised Eigenvectors")

    axs[2, 2].imshow(np.diag(l_denoised))
    axs[2, 2].set_title("Denoised Eigenvalues")

    axs[0, 3].imshow(A_denoised_eigvalsonly)
    axs[0, 3].set_title("Denoised using only denoised eigvals")

    axs[1, 3].imshow(V_noisy)
    axs[1, 3].set_title("Noisy Eigenvectors")

    axs[2, 3].imshow(np.diag(l_denoised))
    axs[2, 3].set_title("Denoised Eigenvalues")

    axs[0, 4].plot(np.log(loss_hist))
    axs[0, 4].set_title("Training loss (log)")

    axs[1, 4].plot(np.log(eigval_error_hist))
    axs[1, 4].set_title("Eigenvalue error (log)")

    axs[2, 4].plot(np.log(subspace_distance_hist))
    axs[2, 4].set_title("Subspace distance (log)")

    _ = axs.flatten()

    for i in range(2):
        for j in range(4):
            axs[i, j].axis("off")

    if model_type == "MultiLayerAttention":
        title = (
            noise_type
            + " Noise, eps = "
            + str(eps)
            + ", Transformer Denoiser, num_layers = "
            + str(num_layers)
            + ", num_heads = "
            + str(num_heads)
        )
    elif model_type == "GNN":
        title = (
            noise_type
            + " Noise, eps = "
            + str(eps)
            + ", GNN, num_layers = "
            + str(num_layers)
        )
    elif model_type == "NodeVarGNN":
        title = (
            noise_type
            + " Noise, eps = "
            + str(eps)
            + ", NodeVarGNN, num_layers = "
            + str(num_layers)
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    _ = fig.suptitle(title, fontsize=16)

    fig.tight_layout()
    # Create results directory if it doesn't exist

    os.makedirs("results", exist_ok=True)
    plt.savefig("results/" + title + ".png")
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    main(args)
