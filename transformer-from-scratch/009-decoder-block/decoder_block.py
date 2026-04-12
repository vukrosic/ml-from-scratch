"""
Decoder Block from scratch.

A transformer decoder block has three sublayers:
1. Masked Self-Attention — causal attention over the decoder input
2. Cross Attention — decoder queries attend to encoder keys/values
3. Feed-Forward — position-wise MLP

Each sublayer is wrapped in a residual connection and layer norm:
    output = LayerNorm(x + Sublayer(x))

The full decoder block:
    x → MaskedSelfAttention → ResidualAdd → LayerNorm
      → CrossAttention (with encoder output) → ResidualAdd → LayerNorm
      → FeedForward → ResidualAdd → LayerNorm

This is the building block of GPT, Llama, and other decoder-only transformers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# ---------------------------------------------------------------------------
# LayerNorm (from scratch)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    """LayerNorm: normalize each feature vector independently."""

    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


# ---------------------------------------------------------------------------
# Masked Self-Attention
# ---------------------------------------------------------------------------

class MaskedSelfAttention(nn.Module):
    """
    Self-attention with a causal (triangular) mask.
    Token i can only attend to tokens 0..i — no peeking at future tokens.
    """

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Combined QKV projection (single linear, split into 3)
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        # Project and split into Q, K, V
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)  # (batch, n_heads, seq, 3*d_head)
        q, k, v = qkv.chunk(3, dim=-1)

        # Scaled dot-product attention
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale

        # Causal mask: upper triangle (including diagonal) is masked
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device),
            diagonal=1
        )
        scores = scores.masked_fill(causal_mask, float('-inf'))

        # Softmax and weighted sum
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v

        # Reshape and project output
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model)
        return self.out_proj(attn)


# ---------------------------------------------------------------------------
# Cross Attention
# ---------------------------------------------------------------------------

class CrossAttention(nn.Module):
    """
    Cross attention: decoder queries attend to encoder keys/values.
    Q comes from the decoder, K and V come from the encoder.

    This is how the decoder "looks at" the encoder output to condition
    its generation on the source sequence (e.g., English sentence for translation).
    """

    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Separate projections for Q (from decoder) and K,V (from encoder)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, dec_x, enc_x):
        """
        Args:
            dec_x: decoder input  (batch, dec_seq, d_model)
            enc_x: encoder output (batch, enc_seq, d_model)
        Returns:
            cross-attended output (batch, dec_seq, d_model)
        """
        batch, dec_seq, d_model = dec_x.shape
        enc_seq = enc_x.shape[1]

        # Project decoder input to Q
        q = self.q_proj(dec_x)
        # Project encoder input to K and V
        k = self.k_proj(enc_x)
        v = self.v_proj(enc_x)

        # Reshape into heads: (batch, n_heads, seq, d_head)
        q = q.view(batch, dec_seq, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, enc_seq, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, enc_seq, self.n_heads, self.d_head).transpose(1, 2)

        # Scaled dot-product attention: dec queries attend to enc keys/values
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale

        # No causal mask needed for cross-attention (full encoder sequence visible)
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v

        # Reshape and project output
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, dec_seq, d_model)
        return self.out_proj(attn)


# ---------------------------------------------------------------------------
# Feed-Forward Network
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    Two linear layers with GELU activation in between.
    Expands d_model -> d_ff -> d_model.

    FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
    """

    def __init__(self, d_model, d_ff=None, bias=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


# ---------------------------------------------------------------------------
# Decoder Block
# ---------------------------------------------------------------------------

class DecoderBlock(nn.Module):
    """
    Full transformer decoder block.

    Forward pass:
        x → MaskedSelfAttention → + → LayerNorm
                                  ↓
        x → CrossAttention (enc) → + → LayerNorm
                                  ↓
        x → FeedForward → + → LayerNorm
                          ↓
                      output

    Each arrow is: sublayer_output = LayerNorm(x + Sublayer(x))
    """

    def __init__(self, d_model, n_heads, d_ff=None, bias=True):
        super().__init__()
        self.masked_attn = MaskedSelfAttention(d_model, n_heads, bias=bias)
        self.cross_attn = CrossAttention(d_model, n_heads, bias=bias)
        self.ffn = FeedForward(d_model, d_ff, bias=bias)

        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

    def forward(self, x, enc_x=None):
        """
        Args:
            x: decoder input (batch, dec_seq, d_model)
            enc_x: encoder output (batch, enc_seq, d_model)
                   None for decoder-only models (GPT, Llama)
        Returns:
            decoder block output (batch, dec_seq, d_model)
        """
        # Masked self-attention + residual
        attn_out = self.masked_attn(x)
        x = self.ln1(x + attn_out)

        # Cross-attention (if encoder output provided)
        if enc_x is not None:
            cross_out = self.cross_attn(x, enc_x)
            x = self.ln2(x + cross_out)

        # Feed-forward + residual
        ffn_out = self.ffn(x)
        x = self.ln3(x + ffn_out)

        return x


# ---------------------------------------------------------------------------
# Standalone demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    # Config
    batch, dec_seq, enc_seq = 2, 8, 6
    d_model, n_heads = 128, 4

    # Random inputs
    dec_x = torch.randn(batch, dec_seq, d_model)
    enc_x = torch.randn(batch, enc_seq, d_model)

    # Decoder block (with cross-attention to encoder)
    block_with_cross = DecoderBlock(d_model, n_heads)
    out_cross = block_with_cross(dec_x, enc_x)
    print(f"Decoder input shape:  {dec_x.shape}")
    print(f"Encoder input shape:  {enc_x.shape}")
    print(f"Output shape (cross): {out_cross.shape}")

    # Decoder block (decoder-only, no encoder)
    block_dec_only = DecoderBlock(d_model, n_heads)
    out_dec_only = block_dec_only(dec_x, enc_x=None)
    print(f"Output shape (dec-only): {out_dec_only.shape}")

    assert out_cross.shape == dec_x.shape
    assert out_dec_only.shape == dec_x.shape

    print("\nDecoder block built successfully!")
    print("Shape preserved through all sublayers.")
