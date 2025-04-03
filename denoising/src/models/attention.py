import torch
import torch.nn as nn
import torch.nn.functional as F


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