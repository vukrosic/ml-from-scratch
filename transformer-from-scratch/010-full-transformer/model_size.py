"""
model_size.py — Count parameters in our Transformer vs standard sizes.

Standard sizes from the literature ("Attention Is All You Need" and follow-up work):
  - Tiny:   d_model=128, n_heads=4,  d_ff=512,  n_layers=2   (~1-2M params)
  - Base:   d_model=512, n_heads=8,  d_ff=2048, n_layers=6   (~39M params, paper)
  - Large:  d_model=1024,n_heads=16, d_ff=4096, n_layers=12  (~250M params)

Our from-scratch implementation matches these dimensions so you can directly
compare parameter counts and see where the memory budget goes.
"""

import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Model definitions (inline for standalone comparison)
# ---------------------------------------------------------------------------

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
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]


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
        Q = self.W_q(query).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.matmul(torch.softmax(scores, dim=-1), V)
        attn = attn.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(attn), None


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


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask=mask)[0])
        x = self.norm2(x + self.ffn(x))
        return x


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, cross_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask=tgt_mask)[0])
        x = self.norm2(x + self.cross_attn(x, enc_output, enc_output, mask=cross_mask)[0])
        x = self.norm3(x + self.ffn(x))
        return x


class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers, src_vocab_size, tgt_vocab_size):
        super().__init__()
        self.d_model = d_model
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        src_len = src.shape[1]
        tgt_len = tgt.shape[1]
        device = src.device
        if src_mask is None:
            src_mask = torch.ones(1, 1, src_len, src_len, device=device)
        if tgt_mask is None:
            tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).unsqueeze(0).unsqueeze(0)
        if cross_mask is None:
            cross_mask = torch.ones(1, 1, tgt_len, src_len, device=device)
        x = self.src_embed(src) * math.sqrt(self.d_model)
        x = x + self.pos_enc(src_len)
        for b in self.encoder_blocks:
            x = b(x, mask=src_mask)
        enc_output = x
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model)
        x = x + self.pos_enc(tgt_len)
        for b in self.decoder_blocks:
            x = b(x, enc_output, tgt_mask=tgt_mask, cross_mask=cross_mask)
        return self.output_proj(x)


# ---------------------------------------------------------------------------
# Per-layer parameter breakdown
# ---------------------------------------------------------------------------

def count_params(module):
    return sum(p.numel() for p in module.parameters())


def param_breakdown(model):
    """Print a table of parameters per layer type."""
    encoders = model.encoder_blocks
    decoders = model.decoder_blocks

    # Per-encoder params
    enc_attn = count_params(encoders[0].self_attn)
    enc_ffn  = count_params(encoders[0].ffn)
    enc_norm = count_params(encoders[0].norm1) + count_params(encoders[0].norm2)

    # Per-decoder params
    dec_sa   = count_params(decoders[0].self_attn)
    dec_ca   = count_params(decoders[0].cross_attn)
    dec_ffn  = count_params(decoders[0].ffn)
    dec_norm = (count_params(decoders[0].norm1) +
                count_params(decoders[0].norm2) +
                count_params(decoders[0].norm3))

    n_layers = len(encoders)

    print(f"{'Component':<35} {'Params per layer':>18} {'Total (all layers)':>22}")
    print("-" * 77)
    print(f"{'Embedding (src + tgt)':<35} {'--':>18} {count_params(model.src_embed) + count_params(model.tgt_embed):>22,}")
    print(f"{'Positional Encoding':<35} {'--':>18} {count_params(model.pos_enc):>22,}")
    print(f"{'Encoder self-attention':<35} {enc_attn:>18,} {enc_attn * n_layers:>22,}")
    print(f"{'Encoder feed-forward':<35} {enc_ffn:>18,} {enc_ffn * n_layers:>22,}")
    print(f"{'Encoder layer norms':<35} {enc_norm:>18,} {enc_norm * n_layers:>22,}")
    print(f"{'Decoder self-attention':<35} {dec_sa:>18,} {dec_sa * n_layers:>22,}")
    print(f"{'Decoder cross-attention':<35} {dec_ca:>18,} {dec_ca * n_layers:>22,}")
    print(f"{'Decoder feed-forward':<35} {dec_ffn:>18,} {dec_ffn * n_layers:>22,}")
    print(f"{'Decoder layer norms':<35} {dec_norm:>18,} {dec_norm * n_layers:>22,}")
    print(f"{'Output projection':<35} {'--':>18} {count_params(model.output_proj):>22,}")
    print("-" * 77)
    total = count_params(model)
    print(f"{'Total':<35} {'':>18} {total:>22,}")
    return total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 77)
    print("Transformer Parameter Counts")
    print("=" * 77)

    configs = [
        ("Tiny (our build)",  128,  4,  512,  2,  10000, 10000),
        ("Base (paper)",      512,  8,  2048, 6,  10000, 10000),
        ("Large (paper)",      1024, 16, 4096, 12, 10000, 10000),
    ]

    for name, d_model, n_heads, d_ff, n_layers, vs, vt in configs:
        print(f"\n{'=' * 77}")
        print(f"{name}")
        model = Transformer(d_model, n_heads, d_ff, n_layers, vs, vt)
        print(f"  d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, n_layers={n_layers}")
        print()
        param_breakdown(model)

    print(f"\n{'=' * 77}")
    print("Parameter count reference (from 'Attention Is All You Need' paper):")
    print("  Base model: ~39M params  |  Big model: ~274M params")
    print("=" * 77)
