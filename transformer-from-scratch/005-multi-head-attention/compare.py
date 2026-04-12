"""
Compare our MultiHeadAttention vs torch.nn.MultiheadAttention.
Verifies numerical equivalence and measures forward pass latency.
"""

import torch
import torch.nn as nn
import time
import math


# ---------------------------------------------------------------------------
# From-scratch MultiHeadAttention (copied so this file is fully standalone)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------

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


def verify_close(ours, theirs, name="output", atol=1e-5, rtol=1e-4):
    """Check numerical equivalence within tolerance."""
    diff = (ours - theirs).abs()
    passed = diff.max().item() < atol
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name} max|diff|: {diff.max().item():.2e}")
    return passed


# ---------------------------------------------------------------------------
# Main comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 60)
    print("Multi-Head Attention: From Scratch vs torch.nn.MultiheadAttention")
    print("=" * 60)

    configs = [
        # (batch, seq_len, d_model, n_heads)
        (1, 32, 128, 4),
        (1, 128, 256, 8),
        (1, 256, 512, 8),
        (4, 64, 256, 8),
        (4, 256, 512, 8),
    ]

    all_passed = True

    for batch, seq_len, d_model, n_heads in configs:
        print(f"\n--- batch={batch}, seq_len={seq_len}, "
              f"d_model={d_model}, n_heads={n_heads} ---")

        x = torch.randn(batch, seq_len, d_model)

        # ---- Our implementation ----
        ours = MultiHeadAttention(d_model, n_heads)
        ours.eval()

        # ---- PyTorch implementation ----
        # torch.nn.MultiheadAttention stores Q/K/V weights concatenated in
        # in_proj_weight (shape 3*d_model x d_model) and in_proj_bias.
        # We copy our W_Q, W_K, W_V into the corresponding slices.
        pytorch_mha = nn.MultiheadAttention(
            d_model, n_heads, batch_first=True
        )
        with torch.no_grad():
            pytorch_mha.in_proj_weight.copy_(
                torch.cat([ours.w_q.weight.T, ours.w_k.weight.T, ours.w_v.weight.T], dim=0)
            )
            if ours.w_q.bias is not None:
                pytorch_mha.in_proj_bias.copy_(
                    torch.cat([ours.w_q.bias, ours.w_k.bias, ours.w_v.bias], dim=0)
                )
            pytorch_mha.out_proj.weight.copy_(ours.w_o.weight.T)
            if ours.w_o.bias is not None:
                pytorch_mha.out_proj.bias.copy_(ours.w_o.bias)
        pytorch_mha.eval()

        # ---- Numerical equivalence ----
        with torch.no_grad():
            out_ours = ours(x)
            out_pytorch, _ = pytorch_mha(x, x, x)

        passed = verify_close(out_ours, out_pytorch, "forward output")
        if not passed:
            all_passed = False

        # ---- Speed benchmark ----
        t_ours = benchmark(lambda: ours(x))
        t_pytorch = benchmark(lambda: pytorch_mha(x, x, x))

        print(f"  Ours time:      {t_ours:8.3f} ms")
        print(f"  PyTorch time:   {t_pytorch:7.3f} ms")
        print(f"  Ratio:          {t_ours / t_pytorch:.3f}x")

    print("\n" + "=" * 60)
    if all_passed:
        print("All numerical equivalence checks PASSED.")
    else:
        print("Some checks FAILED — review differences above.")
    print("=" * 60)
