"""
Self-Attention from scratch.
Implements the full attention mechanism: Q/K/V projection,
scaled dot-product attention, softmax over scores, and output projection.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SelfAttention(nn.Module):
    """
    Classic self-attention: each token attends to all other tokens.
    Given input x, we project it into Query (Q), Key (K), and Value (V).
    The attention score between token i and token j tells us how much
    token i should "look at" token j when building its representation.
    """

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # We project the input into Q, K, V simultaneously with one linear layer
        # that outputs 3 * d_model. We then split that into Q, K, V.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)

        # Final output projection that mixes the attention outputs back to d_model.
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        batch, seq_len, d_model = x.shape

        # Project to Q, K, V in one shot, then split along the feature dimension.
        # qkv shape: (batch, seq_len, 3 * d_model)
        qkv = self.qkv_proj(x)

        # Reshape into (batch, seq_len, n_heads, 3 * d_head) and transpose
        # to (batch, n_heads, seq_len, 3 * d_head) so heads are the batch dim.
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)  # (batch, n_heads, seq_len, 3 * d_head)

        # Slice out Q, K, V — each shape (batch, n_heads, seq_len, d_head).
        q, k, v = qkv.chunk(3, dim=-1)

        # ---- Scaled dot-product attention ----
        # Attention score between every pair of positions: Q @ K^T.
        # We scale by sqrt(d_head) to keep gradients stable.
        scale = math.sqrt(self.d_head)
        # scores shape: (batch, n_heads, seq_len, seq_len)
        scores = q @ k.transpose(-2, -1) / scale

        # Softmax over the "to" dimension (rows), so each query gets a
        # probability distribution over all keys.
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values: each position aggregates information
        # from all other positions, weighted by attention.
        # attn shape: (batch, n_heads, seq_len, d_head)
        attn = attn_weights @ v

        # Transpose back to (batch, seq_len, n_heads, d_head) and
        # reshape to (batch, seq_len, d_model).
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model)

        # Final linear projection that mixes information across heads.
        out = self.out_proj(attn)
        return out


class CausalSelfAttention(nn.Module):
    """
    Self-attention with a causal (triangular) mask.
    Token i can only attend to tokens 0..i — used in autoregressive models
    so the model can't "peek ahead" at future tokens.
    """

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)

        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale

        # Build a lower-triangular mask (True = masked) and apply it.
        # After masking, those positions will be -inf in scores,
        # so softmax sends their attention weight to ~0.
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model)
        out = self.out_proj(attn)
        return out


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Random input: batch=2, seq_len=4, d_model=8
    batch, seq_len, d_model, n_heads = 2, 4, 8, 2
    x = torch.randn(batch, seq_len, d_model)

    # Standard self-attention
    attn = SelfAttention(d_model, n_heads)
    out = attn(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "output must match input shape"

    # Causal (masked) self-attention
    causal_attn = CausalSelfAttention(d_model, n_heads)
    out_causal = causal_attn(x)
    print(f"Causal output shape: {out_causal.shape}")

    # Verify the causal mask is working: position 3 should NOT attend to position 4+.
    # We can check the attention weights for the last head of batch 0.
    print("\nCausal attention weights for batch=0, head=0:")
    # Re-run forward in eval mode to inspect weights cleanly
    causal_attn.eval()
    # Hook into the softmax weights by re-running
    scale = causal_attn.d_head ** 0.5
    with torch.no_grad():
        qkv = causal_attn.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, causal_attn.n_heads, 3 * causal_attn.d_head)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        scores = q @ k.transpose(-2, -1) / scale
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        weights = F.softmax(scores, dim=-1)
    print(weights[0, 0])  # (seq_len, seq_len)
    print("\nNotice: row i only has non-zero values in columns 0..i (lower triangle).")
