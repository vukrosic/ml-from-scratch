"""
Benchmark: our EncoderBlock vs nn.TransformerEncoderLayer.
Measures forward pass latency across multiple configurations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


# ---------------------------------------------------------------------------
# Our from-scratch EncoderBlock
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


class PyTorchEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )

    def forward(self, x):
        return self.layer(x)


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmark(fn, x, n_warmup=5, n_runs=10):
    """Time the average forward pass in milliseconds."""
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    elapsed = (time.perf_counter() - start) / n_runs * 1000
    return elapsed


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 60)
    print("EncoderBlock Benchmark")
    print("=" * 60)

    configs = [
        # (batch, seq_len, d_model, n_heads)
        (1, 32, 128, 4),
    ]

    for batch, seq_len, d_model, n_heads in configs:
        print(f"\n--- batch={batch}, seq_len={seq_len}, "
              f"d_model={d_model}, n_heads={n_heads} ---")

        x = torch.randn(batch, seq_len, d_model)

        # Our from-scratch implementation
        ours = EncoderBlock(d_model, n_heads)
        ours.eval()

        # PyTorch reference
        pytorch_block = PyTorchEncoderBlock(d_model, n_heads)
        pytorch_block.eval()

        # Warmup and benchmark
        t_ours = benchmark(lambda: ours(x), x)
        t_pytorch = benchmark(lambda: pytorch_block(x), x)

        print(f"  Ours time:       {t_ours:7.3f} ms")
        print(f"  PyTorch time:    {t_pytorch:7.3f} ms")
        print(f"  Ratio (ours/pytorch): {t_ours/t_pytorch:.3f}x")

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("Note: Our implementation uses the same architecture as")
    print("      nn.TransformerEncoderLayer with batch_first=True.")
    print("      Speed differences reflect implementation details.")
    print("=" * 60)
