"""
Benchmark MultiHeadAttention across batch sizes and sequence lengths.
Profiles both our from-scratch implementation and torch.nn.MultiheadAttention.
"""

import torch
import torch.nn as nn
import time
import math


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        attn_weights = torch.softmax(scores, dim=-1)
        attn = attn_weights @ v
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model)
        return self.w_o(attn)


def benchmark(fn, n_warmup=50, n_runs=200):
    """Time average forward pass in milliseconds."""
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 65)
    print("Multi-Head Attention Benchmark")
    print("=" * 65)
    print(f"{'batch':>6} {'seq_len':>8} {'d_model':>8} {'heads':>6} "
          f"{'ours ms':>10} {'torch ms':>10} {'ratio':>8}")
    print("-" * 65)

    configs = [
        # (batch, seq_len, d_model, n_heads)
        (1, 32, 128, 4),
        (1, 64, 128, 4),
        (1, 128, 256, 8),
        (1, 256, 512, 8),
        (1, 512, 512, 8),
        (4, 32, 128, 4),
        (4, 128, 256, 8),
        (4, 256, 512, 8),
        (8, 64, 256, 8),
        (8, 256, 512, 8),
    ]

    for batch, seq_len, d_model, n_heads in configs:
        x = torch.randn(batch, seq_len, d_model)

        ours = MultiHeadAttention(d_model, n_heads)
        ours.eval()

        pytorch_mha = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        pytorch_mha.eval()

        # PyTorch stores Q/K/V weights concatenated in in_proj_weight.
        with torch.no_grad():
            pytorch_mha.in_proj_weight.copy_(
                torch.cat([ours.w_q.weight.T, ours.w_k.weight.T, ours.w_v.weight.T], dim=0)
            )
            pytorch_mha.out_proj.weight.copy_(ours.w_o.weight.T)

        t_ours = benchmark(lambda: ours(x))
        t_torch = benchmark(lambda: pytorch_mha(x, x, x))

        ratio = t_ours / t_torch if t_torch > 0 else float('inf')

        print(f"{batch:>6} {seq_len:>8} {d_model:>8} {n_heads:>6} "
              f"{t_ours:>10.3f} {t_torch:>10.3f} {ratio:>8.3f}x")

    print("=" * 65)
    print("Note: Both implementations use the same W_Q, W_K, W_V, W_O architecture.")
    print("      Speed differences reflect implementation details in PyTorch's C++ backend.")
