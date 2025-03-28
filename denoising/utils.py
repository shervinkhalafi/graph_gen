import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def generate_sbm_adjacency(block_sizes, p, q, rng=None):
    """
    Generate an adjacency matrix for a stochastic block model with variable block sizes.

    Parameters:
    - block_sizes: List of sizes for each block.
    - p: Probability of intra-block edges.
    - q: Probability of inter-block edges.
    - rng: Random number generator (optional).

    Returns:
    - Adjacency matrix as a numpy array.
    """
    if rng is None:
        rng = np.random.default_rng()

    n_blocks = len(block_sizes)
    n = sum(block_sizes)

    # Initialize the adjacency matrix with zeros
    
    adj_matrix = np.zeros((n, n))

    # Calculate the starting index of each block
    block_starts = [0]
    for i in range(n_blocks-1):
        block_starts.append(block_starts[-1] + block_sizes[i])

    for i in range(n_blocks):
        for j in range(i, n_blocks):
            density = p if i == j else q
            block_start_i = block_starts[i]
            block_end_i = block_start_i + block_sizes[i]
            block_start_j = block_starts[j]
            block_end_j = block_start_j + block_sizes[j]

            # Generate random edges within or between blocks
            block_i_size = block_sizes[i]
            block_j_size = block_sizes[j]
            adj_matrix[block_start_i:block_end_i, block_start_j:block_end_j] = (
                rng.random((block_i_size, block_j_size)) < density
            ).astype(int)

            # Make the matrix symmetric (for undirected graphs)
            if i != j:
                adj_matrix[block_start_j:block_end_j, block_start_i:block_end_i] = (
                    adj_matrix[block_start_i:block_end_i, block_start_j:block_end_j].T
                )

    return adj_matrix

# create a random n*n skew-symmetric matrix
def random_skew_symmetric_matrix(n):
    A = np.random.rand(n,n)
    return (A - A.T)/2

def add_rotation_noise(A, eps, skew):
    A = torch.tensor(A, dtype=torch.float32)
    m = 1
    k = A.shape[0]
    l, V = torch.linalg.eigh(A)
    R = expm(eps*skew)
    R_tensor = torch.tensor(R, dtype=torch.float32).to(V.device)
    V_rot = V @ R_tensor

    
    # Initialize tensor of appropriate size
    l_diag = torch.zeros(l.shape[0], l.shape[1], l.shape[1], device=l.device, dtype=l.dtype)


    # Fill diagonal elements for each matrix in the batch
    batch_indices = torch.arange(l.shape[0])
    diag_indices = torch.arange(l.shape[1])
    l_diag[batch_indices[:, None], diag_indices, diag_indices] = l


    result = torch.matmul(torch.matmul(V_rot, l_diag), torch.transpose(V_rot, 1, 2))
    return torch.real(result), torch.real(V_rot), torch.real(l)

def add_gaussian_noise(A, eps):

    A = torch.tensor(A, dtype=torch.float32)
    
    A_noisy = A + eps*torch.randn_like(A)
    l, V = torch.linalg.eigh(A_noisy)
    return torch.tensor(A_noisy, dtype=torch.float32), torch.tensor(V, dtype=torch.float32), torch.tensor(l, dtype=torch.float32)



def add_digress_noise(A, p, rng=None):
    """
    Add noise to an adjacency matrix by flipping edges with probability p.
    
    Parameters:
    - adj_matrix: A 2D numpy array or tensor representing an adjacency matrix (0s and 1s)
    - p: Probability of flipping each element (0 to 1, 1 becomes 0 and 0 becomes 1)
    - rng: Random number generator (optional)
    
    Returns:
    - Noisy adjacency matrix with some edges flipped
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Create a copy of the original matrix to avoid modifying it
    A_noisy = A
    
    # Generate random values for each element
    random_values = torch.rand_like(torch.tensor(A))
    
    # Create a mask for elements to flip (where random value < p)
    flip_mask = random_values < p
    
    # Flip the elements where the mask is True (using XOR operation)
    # XOR with 1 flips 0→1 and 1→0
    A_noisy = torch.where(flip_mask, 1 - torch.tensor(A), torch.tensor(A))
    
    l, V = torch.linalg.eigh(A_noisy)

    return torch.tensor(A_noisy, dtype=torch.float32), torch.tensor(V, dtype=torch.float32), torch.tensor(l, dtype=torch.float32)

class AdjacencyMatrixDataset(Dataset):
    def __init__(self, adj_matrix, num_samples_per_epoch):
        self.adj_matrix = adj_matrix
        self.num_samples_per_epoch = num_samples_per_epoch

    def __len__(self):
        return self.num_samples_per_epoch

    def __getitem__(self, idx):
        # Generate a random permutation of the adjacency matrix
        permuted_matrix = self.permute_matrix(self.adj_matrix)
        return permuted_matrix

    def permute_matrix(self, matrix):
        # Generate a random permutation of indices
        indices = np.random.permutation(matrix.shape[0])
        
        # Apply the permutation to both rows and columns
        permuted_matrix = matrix[indices, :][:, indices]
        
        # Convert to PyTorch tensor
        return torch.tensor(permuted_matrix, dtype=torch.float32)
    
#MODELS
class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in 'Attention Is All You Need' paper.
    
    This implementation supports masked attention and different input/output dimensions.
    """
    
    def __init__(self, d_model, num_heads, d_k=None, d_v=None, dropout=0.0, bias=False):
        """
        Initialize the Multi-Head Attention module.
        
        Parameters:
        - d_model: Model dimension (input and output dimension)
        - num_heads: Number of attention heads
        - d_k: Dimension of keys (default: d_model // num_heads)
        - d_v: Dimension of values (default: d_model // num_heads)
        - dropout: Dropout probability
        """
        super(MultiHeadAttention, self).__init__()
        
        self.num_heads = num_heads
        self.d_model = d_model
        
        # If d_k and d_v are not specified, set them to d_model // num_heads
        self.d_k = d_k if d_k is not None else d_model // num_heads
        self.d_v = d_v if d_v is not None else d_model // num_heads
        
        # Linear projections for queries, keys, and values
        self.W_q = nn.Linear(d_model, num_heads * self.d_k, bias=bias)
        self.W_k = nn.Linear(d_model, num_heads * self.d_k, bias=bias)
        self.W_v = nn.Linear(d_model, num_heads * self.d_v, bias=bias)
        
        # Output projection
        self.W_o = nn.Linear(num_heads * self.d_v, d_model, bias=bias)
        
        # Dropout layer
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization for the output
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Scaling factor for dot product attention
        # self.scale = 1 / math.sqrt(self.d_k)
        self.scale = 1

        # Linear layer to combine attention scores from different heads
        self.score_combination = nn.Linear(num_heads, 1, bias=False)
    
    def forward(self, x, mask=None, residual=None):
        """
        Forward pass of the Multi-Head Attention module.
        
        Parameters:
        - Q: Query tensor of shape (batch_size, seq_len_q, d_model)
        - K: Key tensor of shape (batch_size, seq_len_k, d_model)
        - V: Value tensor of shape (batch_size, seq_len_v, d_model)
        - mask: Optional mask tensor of shape (batch_size, seq_len_q, seq_len_k)
        - residual: Optional residual connection
        
        Returns:
        - output: Output tensor of shape (batch_size, seq_len_q, d_model)
        - attention: Attention weights of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size = x.size(0)
        
        # If residual connection is not provided, use Q as residual
        if residual is None:
            residual = x

        # Linear projections and reshaping for multi-head attention
        # Shape: (batch_size, seq_len, num_heads, d_*)

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)


        q = q.view(batch_size, -1, self.num_heads, self.d_k)
        k = k.view(batch_size, -1, self.num_heads, self.d_k)
        v = v.view(batch_size, -1, self.num_heads, self.d_v)



        # Transpose to shape: (batch_size, num_heads, seq_len, d_*)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        
        # Calculate attention scores
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            # Add an extra dimension for the number of heads
            if mask.dim() == 3:  # (batch_size, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)
            
            # Set masked positions to a large negative value before softmax
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply dropout to attention weights
        attn_weights = self.dropout(attn_weights)
        
        # Calculate weighted sum of values
        # (batch_size, num_heads, seq_len_q, d_v)
        context = torch.matmul(attn_weights, v)
        
        # Transpose and reshape to (batch_size, seq_len_q, num_heads * d_v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_v)
        
        # Apply output projection
        output = self.W_o(context)
        
        # Apply dropout and residual connection
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        # Combine attention scores from different heads using learned weights
        # Transpose scores to have heads dimension last: (batch_size, seq_len_q, seq_len_k, num_heads)
        scores = scores.permute(0, 2, 3, 1)
        # Apply linear combination: (batch_size, seq_len_q, seq_len_k, 1)
        combined_scores = self.score_combination(scores)
        # Remove last singleton dimension
        combined_scores = combined_scores.squeeze(-1)
        
        return output, combined_scores
    



class MultiLayerAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_layers, d_k=None, d_v=None, dropout=0.0, bias=False):
        super().__init__()
        
        # If d_k and d_v are not specified, set them equal to d_model/num_heads
        if d_k is None:
            d_k = d_model // num_heads
        if d_v is None:
            d_v = d_model // num_heads
            
        # Create stack of attention layers
        self.layers = nn.ModuleList([
            MultiHeadAttention(d_model, num_heads=num_heads, d_k=d_k, d_v=d_v, dropout=dropout, bias=bias)
            for _ in range(num_layers)
        ])
        
    def forward(self, x, mask=None):
        # Keep track of attention scores from each layer
        attention_scores = []
        
        # Pass through each attention layer sequentially
        for layer in self.layers:

            x, scores = layer(x, mask=mask)

            attention_scores.append(scores)
            
        return x, attention_scores
