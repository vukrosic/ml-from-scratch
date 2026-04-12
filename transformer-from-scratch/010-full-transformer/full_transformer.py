"""
full_transformer.py — Full Encoder-Decoder Transformer from scratch.

This module builds every piece of the Transformer architecture:
  1. Sinusoidal positional encoding
  2. Scaled dot-product attention (single-head)
  3. Multi-head attention (MHA)
  4. Feed-forward network (FFN)
  5. Encoder block (self-attention + FFN + residual + layer-norm)
  6. Decoder block (masked self-attention + cross-attention + FFN + residual + layer-norm)
  7. Full Encoder-Decoder Transformer

Each component is small and standalone so you can see exactly what is happening.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. Sinusoidal Positional Encoding
# ---------------------------------------------------------------------------
# PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
# PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
#
# These fixed encodings give the model a way to reason about token position.
# The sin/cos frequencies form a geometric progression from fine-grained
# (high-frequency) to coarse (low-frequency) patterns.  Unlike learned
# positional encodings, the sinusoidal version works for any sequence length
# up to max_seq_len without retraining.

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)          # (1, max_seq_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]


# ---------------------------------------------------------------------------
# 2. Scaled Dot-Product Attention
# ---------------------------------------------------------------------------
# Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
#
# The scaling factor sqrt(d_k) prevents large dot products from pushing
# softmax into regions with vanishing gradients.

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights


# ---------------------------------------------------------------------------
# 3. Multi-Head Attention
# ---------------------------------------------------------------------------
# Each head gets its own Q/K/V subspace.  Running attention in parallel
# across heads lets the model track different types of relationships
# simultaneously.  All heads are concatenated and projected back to d_model.

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        batch = query.shape[0]

        # Project and split into heads
        Q = self.W_q(query).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)

        # Attention
        out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(out), weights


# ---------------------------------------------------------------------------
# 4. Feed-Forward Network
# ---------------------------------------------------------------------------
# FFN(x) = max(0, x W_1 + b_1) W_2 + b_2
# The FFN applies two linear transforms with a ReLU in between.
# It gives each position a learned non-linear transformation of its
# own representation — separate from what attention learns.

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)


# ---------------------------------------------------------------------------
# 5. Encoder Block
# ---------------------------------------------------------------------------
# Each encoder block applies self-attention across all input tokens,
# then a position-wise feed-forward network.  Both steps use residual
# connections and layer normalisation to stabilise training.

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        # Self-attention with residual connection and layer norm
        attn_out, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = self.norm1(x + attn_out)
        # Feed-forward with residual connection and layer norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


# ---------------------------------------------------------------------------
# 6. Decoder Block
# ---------------------------------------------------------------------------
# Each decoder block has three sub-layers:
#   1. Masked self-attention  — causal mask prevents attending to future tokens
#   2. Cross-attention        — attends to encoder output
#   3. Feed-forward network   — position-wise non-linearity
# All three use residual + layer norm.

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, cross_mask=None):
        # 1. Masked self-attention (causal)
        sa_out, _ = self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        x = self.norm1(x + sa_out)

        # 2. Cross-attention (decoder attends to encoder)
        ca_out, _ = self.cross_attn(query=x, key=enc_output, value=enc_output, mask=cross_mask)
        x = self.norm2(x + ca_out)

        # 3. Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


# ---------------------------------------------------------------------------
# 7. Full Encoder-Decoder Transformer
# ---------------------------------------------------------------------------
# Encoder:  src_embed -> pos_enc -> N x EncoderBlock
# Decoder:  tgt_embed -> pos_enc -> N x DecoderBlock -> linear -> softmax
#
# At training time the decoder receives the full target sequence (teacher forcing).
# At inference time we feed tokens one by one (see generate.py).

class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers,
                 src_vocab_size, tgt_vocab_size, max_seq_len=5000):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers

        # Embeddings
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)

        # Positional encoding
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

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, src, src_mask=None):
        seq_len = src.shape[1]
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = x + self.pos_enc(seq_len)
        for block in self.encoder_blocks:
            x = block(x, mask=src_mask)
        return x

    def decode(self, tgt, enc_output, tgt_mask=None, cross_mask=None):
        seq_len = tgt.shape[1]
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = x + self.pos_enc(seq_len)
        for block in self.decoder_blocks:
            x = block(x, enc_output, tgt_mask=tgt_mask, cross_mask=cross_mask)
        return self.output_proj(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]
        device = src.device

        if src_mask is None:
            src_mask = make_src_mask(src_seq_len, device)
        if tgt_mask is None:
            tgt_mask = make_causal_mask(tgt_seq_len, device)
        if cross_mask is None:
            cross_mask = make_cross_mask(src_seq_len, tgt_seq_len, device)

        enc_output = self.encode(src, src_mask)
        logits = self.decode(tgt, enc_output, tgt_mask, cross_mask)
        return logits


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len, device):
    """Lower-triangular mask: position i can only attend to positions 0..i."""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0)
    return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, seq_len, seq_len)


def make_src_mask(seq_len, device):
    """All-ones mask for encoder self-attention (no padding)."""
    return torch.ones(1, 1, seq_len, seq_len, device=device)


def make_cross_mask(src_seq_len, tgt_seq_len, device):
    """Broadcastable mask for cross-attention (tgt attends to all src)."""
    return torch.ones(1, 1, tgt_seq_len, src_seq_len, device=device)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(42)

    batch_size = 2
    src_len = 10
    tgt_len = 8
    src_vocab = 1000
    tgt_vocab = 1000

    model = Transformer(
        d_model=128, n_heads=4, d_ff=512, n_layers=2,
        src_vocab_size=src_vocab, tgt_vocab_size=tgt_vocab,
    )
    model.eval()

    src = torch.randint(0, src_vocab, (batch_size, src_len))
    tgt = torch.randint(0, tgt_vocab, (batch_size, tgt_len))

    src_mask = make_src_mask(src_len, src.device)
    tgt_mask = make_causal_mask(tgt_len, tgt.device)
    cross_mask = make_cross_mask(src_len, tgt_len, tgt.device)

    logits = model(src, tgt, src_mask, tgt_mask, cross_mask)
    print(f"Logits shape : {logits.shape}")       # (batch, tgt_len, tgt_vocab)
    print(f"Expected     : ({batch_size}, {tgt_len}, {tgt_vocab})")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters   : {n_params:,}")
