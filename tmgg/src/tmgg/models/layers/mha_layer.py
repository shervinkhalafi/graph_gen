from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention module as described in 'Attention Is All You Need' paper.

    This implementation supports masked attention and different input/output dimensions.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        """
        Initialize the Multi-Head Attention module.

        Args:
            d_model: Model dimension (input and output dimension)
            num_heads: Number of attention heads
            d_k: Dimension of keys (default: d_model // num_heads)
            d_v: Dimension of values (default: d_model // num_heads)
            dropout: Dropout probability
            bias: Whether to use bias in linear layers
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
        self.scale = 1

        # Linear layer to combine attention scores from different heads
        self.score_combination = nn.Linear(num_heads, 1, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        residual: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Multi-Head Attention module.

        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional mask tensor of shape (batch_size, seq_len_q, seq_len_k)
            residual: Optional residual connection

        Returns:
            Tuple of (output, combined_attention_scores)
            - output: Output tensor of shape (batch_size, seq_len, d_model)
            - combined_attention_scores: Attention weights of shape (batch_size, seq_len, seq_len)
        """
        batch_size = x.size(0)

        # If residual connection is not provided, use x as residual
        if residual is None:
            residual = x

        # Linear projections and reshaping for multi-head attention
        q = self.W_q(x).view(batch_size, -1, self.num_heads, self.d_k)
        k = self.W_k(x).view(batch_size, -1, self.num_heads, self.d_k)
        v = self.W_v(x).view(batch_size, -1, self.num_heads, self.d_v)

        # Transpose to shape: (batch_size, num_heads, seq_len, d_*)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Calculate attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply mask if provided
        if mask is not None:
            if mask.dim() == 3:  # (batch_size, seq_len_q, seq_len_k)
                mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Calculate weighted sum of values
        context = torch.matmul(attn_weights, v)

        # Transpose and reshape to (batch_size, seq_len, num_heads * d_v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.d_v)
        )

        # Apply output projection
        output = self.W_o(context)

        # Apply dropout and residual connection
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        # Combine attention scores from different heads using learned weights
        # Use attention weights (after softmax) instead of raw scores for output
        attn_weights_permuted = attn_weights.permute(
            0, 2, 3, 1
        )  # (batch_size, seq_len, seq_len, num_heads)
        combined_scores = self.score_combination(attn_weights_permuted).squeeze(
            -1
        )  # (batch_size, seq_len, seq_len)

        return output, combined_scores
