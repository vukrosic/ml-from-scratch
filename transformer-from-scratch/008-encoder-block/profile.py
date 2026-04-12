"""
Profile: measure time and memory for encoder block across batch sizes,
sequence lengths, and model dimensions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import sys


# ---------------------------------------------------------------------------
# EncoderBlock (standalone copy)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))


class EncoderBlock(nn.Module):
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
# Memory estimation helpers
# ---------------------------------------------------------------------------

def estimate_activation_memory(model, batch, seq_len, d_model):
    """
    Estimate the memory used by activations in a forward pass.
    We trace the sizes of all intermediate tensors.

    Key tensors created during a forward pass through one encoder block:
    - Q, K, V: (batch, n_heads, seq, d_head) each
    - Attention scores: (batch, n_heads, seq, seq)
    - Attention output: (batch, n_heads, seq, d_head)
    - FFN hidden: (batch, seq, d_ff)
    - Total: ~ 3 * batch * n_heads * seq * d_head + batch * n_heads * seq^2
    """
    bytes_per_float = 4  # float32
    n_heads = model.blocks[0].attention.n_heads
    d_head = d_model // n_heads
    d_ff = model.blocks[0].ffn.linear1.out_features

    # Q, K, V
    qkv_mem = 3 * batch * n_heads * seq_len * d_head * bytes_per_float
    # Attention scores (seq x seq matrix per head)
    attn_scores_mem = batch * n_heads * seq_len * seq_len * bytes_per_float
    # Attention output
    attn_out_mem = batch * n_heads * seq_len * d_head * bytes_per_float
    # FFN hidden
    ffn_mem = batch * seq_len * d_ff * bytes_per_float

    per_block_mb = (qkv_mem + attn_scores_mem + attn_out_mem + ffn_mem) / 1024 / 1024
    return per_block_mb


# ---------------------------------------------------------------------------
# Timing benchmark
# ---------------------------------------------------------------------------

def benchmark_forward(model, x, n_warmup=50, n_runs=200):
    for _ in range(n_warmup):
        model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_runs):
        model(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    return elapsed


# ---------------------------------------------------------------------------
# Profile table
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 80)
    print("EncoderBlock Profiling: time and memory across configurations")
    print("=" * 80)

    header = f"{'config':<35} {'depth':>5} {'params(M)':>8} {'act(MB)':>8} {'fwd(ms)':>8}"
    print(header)
    print("-" * 80)

    configs = [
        # (batch, seq_len, d_model, n_heads, n_blocks)
        (1, 64, 128, 4, 1),
        (1, 64, 128, 4, 6),
        (1, 64, 256, 8, 6),
        (1, 128, 256, 8, 6),
        (1, 128, 512, 8, 12),
        (4, 64, 256, 8, 6),
        (4, 128, 512, 8, 12),
        (8, 64, 512, 16, 12),
    ]

    for batch, seq_len, d_model, n_heads, n_blocks in configs:
        torch.manual_seed(0)

        model = EncoderBlock(d_model, n_heads)
        # Stack N blocks manually
        class StackedEncoder(nn.Module):
            def __init__(self, n_blocks, d_model, n_heads):
                super().__init__()
                self.blocks = nn.ModuleList([
                    EncoderBlock(d_model, n_heads)
                    for _ in range(n_blocks)
                ])
            def forward(self, x):
                for b in self.blocks:
                    x = b(x)
                return x

        stacked = StackedEncoder(n_blocks, d_model, n_heads)
        stacked.eval()

        x = torch.randn(batch, seq_len, d_model)

        # Parameter count
        n_params = sum(p.numel() for p in stacked.parameters()) / 1e6

        # Activation memory estimate per block
        act_per_block = estimate_activation_memory(stacked, batch, seq_len, d_model)
        total_act_mb = act_per_block * n_blocks

        # Forward time
        fwd_ms = benchmark_forward(stacked, x)

        config_str = f"b={batch} seq={seq_len} d={d_model} h={n_heads}"
        print(f"{config_str:<35} {n_blocks:>5} {n_params:>8.2f} {total_act_mb:>8.1f} {fwd_ms:>8.2f}")

    print("-" * 80)

    print("\n" + "=" * 80)
    print("Memory breakdown (per encoder block, d_model=512, n_heads=8):")
    print("=" * 80)

    # Detailed breakdown
    for batch in [1, 4, 8]:
        for seq_len in [64, 128, 256]:
            d_model, n_heads = 512, 8
            d_head = d_model // n_heads
            d_ff = 4 * d_model
            bytes_per = 4

            qkv = 3 * batch * n_heads * seq_len * d_head * bytes_per / 1024 / 1024
            attn = batch * n_heads * seq_len * seq_len * bytes_per / 1024 / 1024
            attn_out = batch * n_heads * seq_len * d_head * bytes_per / 1024 / 1024
            ffn = batch * seq_len * d_ff * bytes_per / 1024 / 1024
            total = qkv + attn + attn_out + ffn

            print(f"  batch={batch}, seq={seq_len}: "
                  f"QKV={qkv:5.1f}MB  attn_scores={attn:6.1f}MB  "
                  f"ffn={ffn:6.1f}MB  total={total:6.1f}MB")

    print("\n" + "=" * 80)
    print("Key insights:")
    print("  - Attention scores memory grows as O(seq_len^2) — the bottleneck")
    print("  - FFN memory grows as O(d_ff * seq_len) — large but manageable")
    print("  - QKV projection memory grows as O(batch * n_heads * seq_len * d_head)")
    print("  - For long sequences, activation memory dominates over parameters")
    print("=" * 80)
