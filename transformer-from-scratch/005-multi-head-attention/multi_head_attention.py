"""
Multi-Head Attention from scratch.

Implements the full MHA mechanism: separate W_Q, W_K, W_V, W_O projections,
concatenation of heads, and scaled dot-product attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention (MHA).

    Instead of performing a single attention function, we project the queries,
    keys, and values H times (H = n_heads) with different learned linear maps.
    Each head learns to attend to different parts of the sequence.

    Steps:
        1. Project x into Q, K, V with three separate linear layers.
        2. Split each into n_heads and reshape to (batch, n_heads, seq, d_head).
        3. Compute scaled dot-product attention for each head.
        4. Concatenate all heads and project with W_O.
    """

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Three separate projections: W_Q, W_K, W_V.
        # Each maps (batch, seq, d_model) -> (batch, seq, d_model).
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)

        # Output projection that combines all heads back to d_model.
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape

        # ---- Project to Q, K, V separately ----
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # ---- Reshape into heads ----
        # After transposing: (batch, n_heads, seq_len, d_head)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # ---- Scaled dot-product attention ----
        scale = math.sqrt(self.d_head)

        # Q @ K^T: for each head, compute attention scores between all positions.
        # Shape: (batch, n_heads, seq_len, seq_len)
        scores = q @ k.transpose(-2, -1) / scale

        # Softmax over the key dimension — each query gets a probability
        # distribution over all keys (positions).
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values: blend information from all positions.
        # Shape: (batch, n_heads, seq_len, d_head)
        attn = attn_weights @ v

        # ---- Concatenate heads and project ----
        # Transpose back to (batch, seq_len, n_heads, d_head),
        # flatten to (batch, seq_len, d_model).
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model)

        # W_O projects back to d_model, mixing information across heads.
        out = self.w_o(attn)
        return out


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Random input: batch=2, seq_len=5, d_model=8, n_heads=2
    batch, seq_len, d_model, n_heads = 2, 5, 8, 2
    x = torch.randn(batch, seq_len, d_model)

    mha = MultiHeadAttention(d_model, n_heads)
    out = mha(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "output shape must match input shape"

    # Inspect per-head attention weights for batch 0, head 0.
    mha.eval()
    with torch.no_grad():
        q = mha.w_q(x)
        k = mha.w_k(x)
        v = mha.w_v(x)

        q = q.view(batch, seq_len, n_heads, d_model // n_heads).transpose(1, 2)
        k = k.view(batch, seq_len, n_heads, d_model // n_heads).transpose(1, 2)
        v = v.view(batch, seq_len, n_heads, d_model // n_heads).transpose(1, 2)

        scale = math.sqrt(d_model // n_heads)
        scores = q @ k.transpose(-2, -1) / scale
        weights = F.softmax(scores, dim=-1)

    print(f"\nAttention weights for batch=0, head=0:")
    print(weights[0, 0])  # (seq_len, seq_len)
    print("\nEach row = query position, columns = key positions.")
    print("Values sum to 1 across each row (probability distribution).")
