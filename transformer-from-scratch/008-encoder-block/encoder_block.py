"""
Encoder Block from scratch.
MultiHeadAttention → ResidualAdd → LayerNorm → FeedForward → ResidualAdd → LayerNorm

This is the building block of the transformer encoder (BERT, RoBERTa, etc.).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FeedForward(nn.Module):
    """
    Two linear layers with GELU activation.
    Expands from d_model -> d_ff -> d_model.
    d_ff is typically 4 * d_model (BERT default).
    """
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class ResidualAdd(nn.Module):
    """
    Residual connection: output = x + sublayer(x)

    The skip path lets gradients flow directly through the sublayer,
    making deep networks trainable. If the sublayer learns nothing
    (outputs near-zero), the identity pass-through means no degradation.
    """
    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x):
        return x + self.sublayer(x)


class EncoderBlock(nn.Module):
    """
    One transformer encoder block.

    Structure:
        Input x
          ├──► MultiHeadAttention ──► ResidualAdd ──► LayerNorm ──┐
          │                                                         │
          ◄────────────────────────────────────────────────────────┘
                                                                 x
          ├──► FeedForward ──► ResidualAdd ──► LayerNorm ──┐
          │                                              │
          ◄──────────────────────────────────────────────┘

    Both sublayers have input/output shape (batch, seq_len, d_model),
    which is required for the residual add to work.
    """
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # Sublayer 1: multi-head self-attention
        # Q=K=V=x, so each token attends to all tokens
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

        # Sublayer 2: feed-forward network
        self.ffn = FeedForward(d_model, d_ff)

        # Layer norms (one per sublayer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # ---- Sublayer 1: attention + residual + norm ----
        # Self-attention: Q=K=V=input x
        attn_out, _ = self.attention(x, x, x)
        # Residual connection: add input before norm
        x = self.norm1(x + attn_out)

        # ---- Sublayer 2: FFN + residual + norm ----
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Random input: batch=2, seq_len=8, d_model=64
    batch, seq_len, d_model, n_heads = 2, 8, 64, 4
    x = torch.randn(batch, seq_len, d_model)

    encoder = EncoderBlock(d_model, n_heads)
    out = encoder(x)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}")
    assert out.shape == x.shape, "output must match input shape"

    # Verify the residual connections work — gradient path exists
    loss = out.sum()
    loss.backward()
    print("\nBackward pass successful — gradients flow through residual connections.")

    # Print a summary of the block
    num_params = sum(p.numel() for p in encoder.parameters())
    print(f"\nEncoderBlock parameters: {num_params:,}")
    print(f"  d_model={d_model}, n_heads={n_heads}, d_ff={4*d_model}")
