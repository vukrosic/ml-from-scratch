"""
Sublayer pattern: Residual connections and layer norm are the two sublayers.
This file implements the sublayer wrapper explicitly, showing that the
original "Attention is All You Need" paper defines the encoder block
in terms of two parameterized sublayers.

The key insight from the paper:
    LayerNorm(x + Sublayer(x))
    where Sublayer is the function (attention or FFN) that the sublayer applies.

We implement this explicitly with a SublayerWrapper class, then show
how it maps directly to our EncoderBlock.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# The SublayerWrapper — from the original transformer paper
# ---------------------------------------------------------------------------

class SublayerWrapper(nn.Module):
    """
    Wraps any sublayer with a residual connection and layer norm.

    output = LayerNorm(x + Sublayer(x))

    This is the exact formulation from "Attention is All You Need" (2017).
    The paper calls this "Add & Norm" — we implement it explicitly here
    to show the correspondence to our encoder block.
    """
    def __init__(self, d_model, sublayer):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.sublayer = sublayer

    def forward(self, x):
        # Residual: add input to sublayer output, then normalize
        return self.norm(x + self.sublayer(x))


# ---------------------------------------------------------------------------
# The FeedForward sublayer
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


# ---------------------------------------------------------------------------
# EncoderBlock using explicit SublayerWrapper (the paper's formulation)
# ---------------------------------------------------------------------------

class EncoderBlockWithSublayers(nn.Module):
    """
    Encoder block using explicit SublayerWrapper.

    This is functionally identical to our EncoderBlock, but makes
    the paper's formulation explicit:

        LayerNorm(x + Attention(x))
        LayerNorm(x + FFN(x))

    Sublayer 1: Multi-head self-attention
    Sublayer 2: Feed-forward network
    """
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model

        # Self-attention as sublayer 1
        attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.sublayer1 = SublayerWrapper(d_model, attention)

        # FFN as sublayer 2
        ffn = FeedForward(d_model, d_ff)
        self.sublayer2 = SublayerWrapper(d_model, ffn)

    def forward(self, x):
        # Sublayer 1: attention wrapped in residual + norm
        x = self.sublayer1(x)
        # Sublayer 2: FFN wrapped in residual + norm
        x = self.sublayer2(x)
        return x


# ---------------------------------------------------------------------------
# Direct encoder block (same as encoder_block.py) for comparison
# ---------------------------------------------------------------------------

class EncoderBlockDirect(nn.Module):
    """Same encoder block, written directly (no explicit SublayerWrapper)."""
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ---------------------------------------------------------------------------
# Verify both produce the same output
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    batch, seq_len, d_model, n_heads = 2, 8, 64, 4
    x = torch.randn(batch, seq_len, d_model)

    # Both blocks have identical structure — verify forward pass
    block_sublayer = EncoderBlockWithSublayers(d_model, n_heads)
    block_direct = EncoderBlockDirect(d_model, n_heads)

    # Copy weights so we compare identical models
    with torch.no_grad():
        # Copy from sublayer version to direct version
        block_direct.attention = block_sublayer.sublayer1.sublayer
        block_direct.ffn = block_sublayer.sublayer2.sublayer
        block_direct.norm1.load_state_dict(block_sublayer.sublayer1.norm.state_dict())
        block_direct.norm2.load_state_dict(block_sublayer.sublayer2.norm.state_dict())

    out_sublayer = block_sublayer(x)
    out_direct = block_direct(x)

    diff = (out_sublayer - out_direct).abs().max().item()
    print(f"Max |difference| between formulations: {diff:.2e}")
    print(f"Both produce identical output: {diff < 1e-6}")

    # Show that with random weights, both still run fine
    x_random = torch.randn(4, 16, 128)
    out = block_sublayer(x_random)
    print(f"\nSublayerWrapper block output shape: {out.shape}")
    assert out.shape == x_random.shape

    loss = out.sum()
    loss.backward()
    print("Backward pass successful through SublayerWrapper formulation.")

    print("\n" + "-" * 60)
    print("The SublayerWrapper is the 'Add & Norm' from the paper:")
    print("  output = LayerNorm(x + Sublayer(x))")
    print("  where Sublayer is either attention or FFN.")
    print("Both our direct EncoderBlock and SublayerWrapper version")
    print("are mathematically identical — the wrapper is just explicit.")
