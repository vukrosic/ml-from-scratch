"""
model.py — Full encoder-decoder Transformer matching "Attention Is All You Need" specs.

Paper specs used here:
  d_model  = 512   (embedding dimension / residual stream width)
  n_heads  = 8     (parallel attention heads)
  d_ff     = 2048  (feed-forward hidden dimension)
  n_layers = 6    (number of encoder and decoder blocks)
  d_k      = d_model // n_heads = 64  (key/query dimension per head)

We implement:
  1. Scaled dot-product attention (Equation 1 in the paper)
  2. Multi-head attention (MHAttention)
  3. Sinusoidal positional encoding (Section 3.5)
  4. Encoder block (self-attention + feed-forward + residual + layer-norm)
  5. Decoder block (masked self-attention + cross-attention + feed-forward + residual + layer-norm)
  6. Full Encoder-Decoder Transformer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1.  Scaled Dot-Product Attention  (Equation 1)
# ---------------------------------------------------------------------------
# Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
#
# We separate the "separate heads" logic from this scalar so the equation
# is visually clear in its own code block below.

def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    Q, K, V: (batch, n_heads, seq_len, d_k)
    mask:    (batch, 1, seq_len, seq_len) or (batch, 1, 1, seq_len) for causal
    Returns: (batch, n_heads, seq_len, d_k)
    """
    d_k = Q.shape[-1]
    # scores[b, h, i, j] = how much query i attends to key j
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))

    # Softmax along the key dimension (dim=-1) — each query distributes its mass over keys
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights


# ---------------------------------------------------------------------------
# 2.  Multi-Head Attention
# ---------------------------------------------------------------------------
class MultiHeadAttention(nn.Module):
    """
    Projects inputs into n_heads separate Q/K/V spaces, runs attention in
    parallel, then concatenates heads and projects back to d_model.
    """

    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # W_q, W_k, W_v each: (d_model -> d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        # Final projection back to d_model
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        batch = query.shape[0]

        # Linear projections, then split into heads: (B, seq, d_model) -> (B, seq, n_heads, d_k)
        Q = self.W_q(query).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Run scaled dot-product attention on all heads at once
        out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Concatenate heads: (B, n_heads, seq, d_k) -> (B, seq, d_model)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(out), weights


# ---------------------------------------------------------------------------
# 3.  Sinusoidal Positional Encoding  (Section 3.5)
# ---------------------------------------------------------------------------
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# This is a fixed (non-learned) encoding that lets the model reason about
# relative position — the sin/cos frequencies form a geometric progression
# from fine-grained (high-frequency) to coarse (low-frequency) patterns.

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)          # (max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()  # (max_seq_len, 1)

        # div_term = 10000^(2i/d_model) for i = 0, 1, 2, ...
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)    # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)    # odd dimensions
        pe = pe.unsqueeze(0)                             # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)                   # not a learnable parameter

    def forward(self, seq_len):
        # Return the first `seq_len` positions; shape: (1, seq_len, d_model)
        return self.pe[:, :seq_len, :]


# ---------------------------------------------------------------------------
# 4.  Feed-Forward Sub-Layer  (Section 3.3, last paragraph)
# ---------------------------------------------------------------------------
# FFN(x) = max(0, x W_1 + b_1) W_2 + b_2   (ReLU between two linear transforms)
# d_ff = 2048, d_model = 512  =>  expand to 2048 then project back to 512

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 5.  Encoder Block
# ---------------------------------------------------------------------------
class EncoderBlock(nn.Module):
    """
    One encoder layer = self-attention + feed-forward, each with a residual
    connection and layer normalisation (Add & Norm).
    """

    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads)
        self.ffn        = FeedForward(d_model, d_ff)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_out, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = self.norm1(x + attn_out)

        # Feed-forward with residual connection and layer norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ---------------------------------------------------------------------------
# 6.  Decoder Block
# ---------------------------------------------------------------------------
class DecoderBlock(nn.Module):
    """
    One decoder layer = masked self-attention + cross-attention + feed-forward,
    each with residual + layer norm.
    """

    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.self_attn   = MultiHeadAttention(d_model, n_heads)
        self.cross_attn  = MultiHeadAttention(d_model, n_heads)
        self.ffn         = FeedForward(d_model, d_ff)
        self.norm1       = nn.LayerNorm(d_model)
        self.norm2       = nn.LayerNorm(d_model)
        self.norm3       = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None):
        tgt_seq_len = x.shape[1]
        enc_seq_len = enc_output.shape[1]
        device = x.device

        # 1. Masked self-attention (causal — can't see future tokens)
        sa_out, _ = self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        x = self.norm1(x + sa_out)

        # 2. Cross-attention (decoder attends to encoder output)
        # Mask shape: (1, 1, tgt_seq_len, enc_seq_len)
        cross_mask = make_cross_attn_mask(enc_seq_len, tgt_seq_len, device)
        ca_out, _ = self.cross_attn(query=x, key=enc_output, value=enc_output, mask=cross_mask)
        x = self.norm2(x + ca_out)

        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


# ---------------------------------------------------------------------------
# 7.  Full Encoder-Decoder Transformer
# ---------------------------------------------------------------------------
class Transformer(nn.Module):
    """
    Complete Transformer for sequence-to-sequence tasks.

    Architecture (mirrors Figure 1 in the paper):
      Encoder:  Input embedding -> + positional encoding -> N x EncoderBlock
      Decoder:  Output embedding -> + positional encoding -> N x DecoderBlock -> linear -> softmax
    """

    def __init__(self, d_model=512, n_heads=8, d_ff=2048, n_layers=6,
                 src_vocab_size=10000, tgt_vocab_size=10000, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding (fixed sinusoidal)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)

        # Stack of encoder and decoder blocks
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])

        # Final projection to vocabulary
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

        # Tie weights between embedding and projection (optional, reduces params)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        """
        src: (batch, src_seq_len)  — token indices
        Returns: (batch, src_seq_len, d_model) encoded representation
        """
        seq_len = src.shape[1]
        x = self.src_embed(src) * math.sqrt(self.d_model)          # scale embed
        x = x + self.pos_enc(seq_len)                               # add positional info
        for block in self.encoder_blocks:
            x = block(x, mask=src_mask)
        return x

    def decode(self, tgt, enc_output, tgt_mask=None):
        """
        tgt: (batch, tgt_seq_len)  — token indices (teacher forcing at train time)
        enc_output: (batch, src_seq_len, d_model)
        Returns: (batch, tgt_seq_len, tgt_vocab_size) logits
        """
        seq_len = tgt.shape[1]
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = x + self.pos_enc(seq_len)
        for block in self.decoder_blocks:
            # DecoderBlock creates its own cross-attention mask internally
            # based on enc_output.shape[1]
            x = block(x, enc_output, tgt_mask=tgt_mask)
        return self.output_proj(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        device = src.device

        # Encoder self-attention mask: (1, 1, src_seq_len, src_seq_len)
        if src_mask is None:
            src_mask = make_src_mask(src_seq_len, device)
        # Decoder self-attention (causal) mask: (1, 1, tgt_seq_len, tgt_seq_len)
        if tgt_mask is None:
            tgt_mask = make_causal_mask(tgt_seq_len, device)

        enc_output = self.encode(src, src_mask)
        logits = self.decode(tgt, enc_output, tgt_mask)
        return logits


# ---------------------------------------------------------------------------
# 8.  Causal (look-ahead) mask for decoder self-attention
# ---------------------------------------------------------------------------
def make_causal_mask(seq_len, device):
    """
    Returns a (seq_len, seq_len) mask where entry (i, j) is 1 iff j <= i
    (future positions are masked out).
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0)
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, seq_len, seq_len)


def make_src_mask(seq_len, device):
    """All-ones mask for encoder self-attention (no padding assumed)."""
    return torch.ones(1, 1, seq_len, seq_len, device=device)


def make_cross_attn_mask(src_seq_len, tgt_seq_len, device):
    """
    Broadcastable mask for cross-attention.
    Shape: (1, 1, tgt_seq_len, src_seq_len) — query_i attends to all key_j.
    """
    return torch.ones(1, 1, tgt_seq_len, src_seq_len, device=device)


# ---------------------------------------------------------------------------
# 9.  Quick forward test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size  = 2
    src_len     = 10
    tgt_len     = 8
    src_vocab   = 1000
    tgt_vocab   = 1000

    model = Transformer(
        d_model=512, n_heads=8, d_ff=2048, n_layers=6,
        src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab
    )
    model.eval()

    src = torch.randint(0, src_vocab, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

    src_mask = make_src_mask(src_len, src.device)
    tgt_mask = make_causal_mask(tgt_len, tgt.device)

    logits = model(src, tgt, src_mask, tgt_mask)
    print(f"Logits shape : {logits.shape}")      # (batch, tgt_len, tgt_vocab)
    print(f"Expected     : ({batch_size}, {tgt_len}, {tgt_vocab})")

    # Spot-check positional encoding
    pe = model.pos_enc(5)
    print(f"PE shape      : {pe.shape}")          # (1, 5, 512)
    print(f"PE[0, 0, 0]   : {pe[0, 0, 0].item():.4f}  (sin(0)=0)")
    print(f"PE[0, 1, 0]   : {pe[0, 1, 0].item():.4f}  (sin(pos/10000^0))")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params  : {n_params:,}")        # ~39 M as per paper
