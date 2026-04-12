"""
Profile DecoderBlock performance across batch sizes and configurations.

Extended topic: paid Skool video explains the profiling methodology,
memory bandwidth considerations, and how to optimize transformer blocks.

This script profiles:
1. Forward pass latency across batch/sequence/d_model configurations
2. Memory usage estimation
3. Comparison of decoder-only vs encoder-decoder modes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import math
import sys


# ---------------------------------------------------------------------------
# Our DecoderBlock (from decoder_block.py, inlined)
# ---------------------------------------------------------------------------

class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / torch.sqrt(var + self.eps) + self.beta


class MaskedSelfAttention(nn.Module):
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
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        attn = F.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attn)


class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, dec_x, enc_x):
        batch, dec_seq, d_model = dec_x.shape
        enc_seq = enc_x.shape[1]
        q = self.q_proj(dec_x).view(batch, dec_seq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(enc_x).view(batch, enc_seq, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(enc_x).view(batch, enc_seq, self.n_heads, self.d_head).transpose(1, 2)
        scale = math.sqrt(self.d_head)
        attn = F.softmax(q @ k.transpose(-2, -1) / scale, dim=-1) @ v
        attn = attn.transpose(1, 2).contiguous().view(batch, dec_seq, d_model)
        return self.out_proj(attn)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, bias=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, bias=True):
        super().__init__()
        self.masked_attn = MaskedSelfAttention(d_model, n_heads, bias=bias)
        self.cross_attn = CrossAttention(d_model, n_heads, bias=bias)
        self.ffn = FeedForward(d_model, d_ff, bias=bias)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

    def forward(self, x, enc_x=None):
        x = self.ln1(x + self.masked_attn(x))
        if enc_x is not None:
            x = self.ln2(x + self.cross_attn(x, enc_x))
        return self.ln3(x + self.ffn(x))


# ---------------------------------------------------------------------------
# Profiling functions
# ---------------------------------------------------------------------------

def estimate_memory(params_bytes, activations_bytes):
    """Estimate total memory in MB."""
    return (params_bytes + activations_bytes) / 1e6


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def profile_decoder_block(batch, dec_seq, enc_seq, d_model, n_heads, n_warmup=50, n_runs=200):
    """
    Profile a single decoder block configuration.

    Returns dict with timing and memory stats.
    """
    dec_x = torch.randn(batch, dec_seq, d_model)
    enc_x = torch.randn(batch, enc_seq, d_model)

    block = DecoderBlock(d_model, n_heads)
    block.eval()

    n_params = count_parameters(block)

    # Estimate memory for parameters (in bytes, assuming float32)
    param_bytes = n_params * 4

    # Estimate activation memory:
    # - QKV for self-attention: 3 * batch * dec_seq * d_model * 4 bytes
    # - Scores: batch * n_heads * dec_seq^2 * 4 bytes
    # - Attention weights: same as scores
    # - Values: batch * n_heads * dec_seq * d_head * 4 bytes
    # - Cross-attention: similar pattern with enc_seq
    # - FFN activations: batch * dec_seq * d_ff * 4 bytes
    d_ff = 4 * d_model
    activation_bytes = (
        # Self-attention QKV
        3 * batch * dec_seq * d_model * 4 +
        # Self-attention scores + attn weights
        2 * batch * n_heads * dec_seq * dec_seq * 4 +
        # Self-attention values
        batch * n_heads * dec_seq * (d_model // n_heads) * 4 +
        # Cross-attention QKV
        3 * batch * (dec_seq + enc_seq) * d_model * 4 +
        # Cross-attention scores + attn
        2 * batch * n_heads * dec_seq * enc_seq * 4 +
        # FFN
        2 * batch * dec_seq * d_ff * 4
    )

    # Warmup
    for _ in range(n_warmup):
        _ = block(dec_x, enc_x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    # Timed runs
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = block(dec_x, enc_x)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / n_runs * 1000  # ms

    return {
        'batch': batch,
        'dec_seq': dec_seq,
        'enc_seq': enc_seq,
        'd_model': d_model,
        'n_heads': n_heads,
        'n_params': n_params,
        'param_mb': param_bytes / 1e6,
        'activation_mb': activation_bytes / 1e6,
        'time_ms': elapsed
    }


def print_profile_table(results):
    """Print profiling results as a formatted table."""
    print(f"\n{'Config':>40} {'Params(MB)':>10} {'Act(MB)':>10} {'Time(ms)':>10}")
    print("-" * 75)
    for r in results:
        config = f"b={r['batch']},d={r['d_model']},h={r['n_heads']},dec={r['dec_seq']},enc={r['enc_seq']}"
        print(f"{config:>40} {r['param_mb']:>10.2f} {r['activation_mb']:>10.2f} {r['time_ms']:>10.3f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 75)
    print("Decoder Block Profiling")
    print("=" * 75)
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")

    # Configurations to profile
    configs = [
        # (batch, dec_seq, enc_seq, d_model, n_heads)
        (1, 64, 48, 128, 4),
        (1, 128, 96, 256, 8),
        (1, 256, 192, 512, 8),
        (1, 512, 384, 1024, 16),
        (4, 128, 96, 256, 8),
        (8, 128, 96, 256, 8),
        (16, 128, 96, 256, 8),
        (4, 256, 192, 512, 8),
        (1, 2048, 0, 768, 12),  # Decoder-only (no encoder)
    ]

    results = []
    for batch, dec_seq, enc_seq, d_model, n_heads in configs:
        print(f"\nProfiling: batch={batch}, dec_seq={dec_seq}, enc_seq={enc_seq}, "
              f"d_model={d_model}, n_heads={n_heads}")
        r = profile_decoder_block(batch, dec_seq, enc_seq, d_model, n_heads)
        results.append(r)
        print(f"  Params: {r['n_params']:,} ({r['param_mb']:.2f} MB)")
        print(f"  Activations (est): {r['activation_mb']:.2f} MB")
        print(f"  Time: {r['time_ms']:.3f} ms")

    print("\n" + "=" * 75)
    print("Summary Table")
    print("=" * 75)
    print_profile_table(results)

    print("\n" + "=" * 75)
    print("Key observations:")
    print("  - Activation memory grows with O(batch * seq^2) due to attention")
    print("  - Cross-attention adds O(batch * dec_seq * enc_seq) memory")
    print("  - Decoder-only (enc_seq=0) saves significant memory vs encoder-decoder")
    print("=" * 75)
