import numpy as np
import torch
import torch.nn as nn


class EigenDecompositionError(Exception):
    """Custom exception for eigendecomposition failures with debugging context."""

    def __init__(
        self, matrix_idx: int, matrix: torch.Tensor, original_error: Exception
    ):
        self.matrix_idx = matrix_idx
        self.matrix = matrix
        self.original_error = original_error

        # Compute debugging metrics
        self.debugging_context = self._compute_debugging_metrics(matrix)

        super().__init__(self._format_message())

    def _compute_debugging_metrics(self, A: torch.Tensor) -> dict:  # pyright: ignore[reportMissingTypeArgument]
        """Compute key metrics for debugging ill-conditioned matrices."""
        with torch.no_grad():
            # Convert to numpy for condition number calculation
            A_np = A.cpu().numpy()

            # 1. Condition number (ratio of largest to smallest singular value)
            try:
                singular_values = np.linalg.svd(A_np, compute_uv=False)
                condition_number = singular_values[0] / (singular_values[-1] + 1e-10)
            except:
                condition_number = float("inf")

            # 2. Frobenius norm (matrix magnitude)
            frobenius_norm = torch.norm(A, p="fro").item()

            # 3. Trace (sum of diagonal elements)
            trace = torch.trace(A).item()

            # 4. Check for NaN/Inf values
            has_nan = torch.isnan(A).any().item()
            has_inf = torch.isinf(A).any().item()

            # 5. Diagonal dominance metric (how dominant the diagonal is)
            diagonal = torch.diag(A)
            off_diagonal_sum = torch.sum(torch.abs(A), dim=1) - torch.abs(diagonal)
            min_diagonal_dominance = torch.min(
                torch.abs(diagonal) - off_diagonal_sum
            ).item()

            return {
                "condition_number": condition_number,
                "frobenius_norm": frobenius_norm,
                "trace": trace,
                "has_nan": has_nan,
                "has_inf": has_inf,
                "min_diagonal_dominance": min_diagonal_dominance,
                "matrix_shape": list(A.shape),
            }

    def _format_message(self) -> str:
        """Format error message with debugging context."""
        ctx = self.debugging_context
        return (
            f"Eigendecomposition failed for matrix {self.matrix_idx}. "
            f"Debugging context: "
            f"condition_number={ctx['condition_number']:.2e}, "
            f"frobenius_norm={ctx['frobenius_norm']:.2e}, "
            f"trace={ctx['trace']:.2e}, "
            f"has_nan={ctx['has_nan']}, "
            f"has_inf={ctx['has_inf']}, "
            f"min_diagonal_dominance={ctx['min_diagonal_dominance']:.2e}, "
            f"shape={ctx['matrix_shape']}. "
            f"Original error: {str(self.original_error)}"
        )


class EigenEmbedding(nn.Module):
    """Embedding layer using eigenvectors of adjacency matrix.

    Parameters
    ----------
    eigenvalue_reg : float, optional
        Diagonal regularization added before eigendecomposition. This spreads
        eigenvalues apart, improving gradient stability through the eigh
        operation. Default is 0.0 (no regularization). Values around 1e-4 to
        1e-2 help when training produces NaN gradients or unstable loss.
    """

    def __init__(self, eigenvalue_reg: float = 0.0):
        super(EigenEmbedding, self).__init__()
        self.eigenvalue_reg = eigenvalue_reg

    def forward(self, A: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of eigen embedding.

        Args:
            A: Adjacency matrix of shape (batch_size, num_nodes, num_nodes)

        Returns:
            Eigenvector embeddings of shape (batch_size, num_nodes, num_nodes)

        Raises:
            EigenDecompositionError: If eigendecomposition fails with debugging context
        """
        # Apply diagonal regularization if enabled
        if self.eigenvalue_reg > 0:
            n = A.shape[-1]
            eye = torch.eye(n, device=A.device, dtype=A.dtype)
            A = A + self.eigenvalue_reg * eye

        eigenvectors = []
        for i in range(A.shape[0]):
            try:
                # Enforce symmetry before eigh: floating-point operations can
                # introduce small asymmetries, and eigh silently uses only the
                # lower triangle, which can cause inconsistent results
                A_sym = (A[i] + A[i].T) / 2
                _, V = torch.linalg.eigh(A_sym)
                eigenvectors.append(V)
            except torch._C._LinAlgError as e:  # pyright: ignore[reportUnknownMemberType, reportAttributeAccessIssue]
                # Propagate with debugging context
                raise EigenDecompositionError(i, A[i], e)
        return torch.stack(eigenvectors, dim=0)
