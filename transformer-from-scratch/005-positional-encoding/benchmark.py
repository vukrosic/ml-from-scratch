"""
Benchmark: sinusoidal vs learned vs no-PE forward pass speed.

We simulate a small transformer layer with and without positional encoding,
measuring throughput across different sequence lengths.
"""

import time
import math
import torch
import torch.nn as nn
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# PE implementations (reproduced for standalone use)
# ---------------------------------------------------------------------------

def sinusoidal_pe_vectorized(seq_len, d_model):
    positions = torch.arange(seq_len).unsqueeze(1)
    freqs_idx = torch.arange(0, d_model, 2).float()
    freqs = torch.exp(-math.log(10000.0) * freqs_idx / d_model)
    angles = positions * freqs
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe


class LearnedPositionalEmbedding(nn.Module):
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(seq_len, d_model)
        nn.init.normal_(self.pe.weight, mean=0.0, std=0.02)

    def forward(self, x):
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions)


# ---------------------------------------------------------------------------
# Simple transformer layer for benchmarking
# ---------------------------------------------------------------------------

class TinyTransformerLayer(nn.Module):
    """
    A minimal transformer-like layer for benchmarking the overhead of
    different positional encoding schemes.
    """
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        attn = torch.softmax(scores, dim=-1) @ v
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attn)


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def benchmark_layer(layer, x, n_warmup=10, n_runs=50):
    """Time average forward pass in milliseconds."""
    for _ in range(n_warmup):
        layer(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.perf_counter()
    for _ in range(n_runs):
        layer(x)
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return (time.perf_counter() - start) / n_runs * 1000


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    d_model, n_heads = 128, 4
    seq_lens = [32, 64, 128, 256, 512]
    batch = 2

    results = {"no_pe": [], "sinusoidal": [], "learned": []}

    print(f"{'seq_len':>8} | {'no_pe':>8} | {'sinusoidal':>10} | {'learned':>8}")
    print("-" * 45)

    for seq_len in seq_lens:
        x = torch.randn(batch, seq_len, d_model)

        # No PE
        layer_no_pe = TinyTransformerLayer(d_model, n_heads)
        t_no_pe = benchmark_layer(layer_no_pe, x)

        # Sinusoidal: precompute PE and add
        sin_pe = sinusoidal_pe_vectorized(seq_len, d_model)
        layer_sin = TinyTransformerLayer(d_model, n_heads)
        def forward_sin(_x): return layer_sin(_x + sin_pe)
        t_sin = benchmark_layer(forward_sin, x)

        # Learned: PE is part of the model
        layer_learned = TinyTransformerLayer(d_model, n_heads)
        learned_pe = LearnedPositionalEmbedding(seq_len, d_model)
        def forward_learned(_x): return layer_learned(learned_pe(_x))
        t_learned = benchmark_layer(forward_learned, x)

        results["no_pe"].append(t_no_pe)
        results["sinusoidal"].append(t_sin)
        results["learned"].append(t_learned)

        print(f"{seq_len:>8} | {t_no_pe:8.3f} | {t_sin:10.3f} | {t_learned:8.3f}")

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(seq_lens, results["no_pe"],      label="no PE", marker="o")
    plt.plot(seq_lens, results["sinusoidal"], label="sinusoidal", marker="s")
    plt.plot(seq_lens, results["learned"],    label="learned", marker="^")
    plt.xlabel("sequence length")
    plt.ylabel("forward pass (ms)")
    plt.title("Positional Encoding Overhead")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("pe_benchmark.png", dpi=150)
    plt.close()
    print("\nSaved pe_benchmark.png")

    print("\n--- Observations ---")
    print("Sinusoidal PE overhead is negligible (precomputed and added).")
    print("Learned PE adds Embedding lookup + addition overhead.")
    print("Both are minor compared to the attention computation itself.")
