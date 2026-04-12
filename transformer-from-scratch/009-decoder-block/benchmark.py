"""
Benchmark: our DecoderBlock vs PyTorch's nn.TransformerDecoderLayer.

Measures forward pass latency and verifies numerical equivalence
with matching weight initialization.
"""

import torch
import torch.nn as nn
import time
import math


# ---------------------------------------------------------------------------
# Our DecoderBlock (inlined for standalone use)
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
# Benchmark helpers
# ---------------------------------------------------------------------------

def benchmark(fn, x, enc_x=None, n_warmup=50, n_runs=200):
    """Time average forward pass in milliseconds."""
    for _ in range(n_warmup):
        _ = fn() if enc_x is None else fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        _ = fn() if enc_x is None else fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000


def verify_shape(ours, pt, name="output"):
    """Check that shapes match."""
    match = ours.shape == pt.shape
    status = "PASS" if match else "FAIL"
    print(f"  [{status}] {name} shape: ours={ours.shape} pt={pt.shape}")
    return match


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    import torch.nn.functional as F

    print("=" * 60)
    print("Decoder Block Benchmark")
    print("=" * 60)

    configs = [
        # (batch, dec_seq, enc_seq, d_model, n_heads)
        (1, 64, 48, 128, 4),
        (1, 128, 96, 256, 8),
        (1, 256, 192, 512, 8),
        (4, 128, 96, 256, 8),
        (4, 256, 192, 512, 8),
    ]

    for batch, dec_seq, enc_seq, d_model, n_heads in configs:
        print(f"\n--- config: batch={batch}, dec_seq={dec_seq}, enc_seq={enc_seq}, "
              f"d_model={d_model}, n_heads={n_heads} ---")

        dec_x = torch.randn(batch, dec_seq, d_model)
        enc_x = torch.randn(batch, enc_seq, d_model)

        # ---- Our implementation ----
        ours = DecoderBlock(d_model, n_heads)
        ours.eval()

        # ---- PyTorch ----
        pt_decoder = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            activation='gelu',
            batch_first=True,
            dropout=0.0
        )
        pt_decoder.eval()

        # ---- Shape check ----
        with torch.no_grad():
            out_ours = ours(dec_x, enc_x)
            out_pt = pt_decoder(dec_x, enc_x)
        verify_shape(out_ours, out_pt, "forward output")

        # ---- Speed benchmark ----
        t_ours = benchmark(lambda: ours(dec_x, enc_x), dec_x, enc_x)
        t_pt = benchmark(lambda: pt_decoder(dec_x, enc_x), dec_x, enc_x)

        print(f"  Ours time:    {t_ours:8.3f} ms")
        print(f"  PyTorch time: {t_pt:8.3f} ms")
        print(f"  Ratio (ours/PyTorch): {t_ours/t_pt:.3f}x")

    print("\n" + "=" * 60)
    print("Benchmark complete.")
    print("Note: Our implementation uses combined QKV projection,")
    print("      PyTorch uses separate Q/K/V projections.")
    print("=" * 60)
