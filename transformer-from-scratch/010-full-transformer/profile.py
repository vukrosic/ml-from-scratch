"""
profile.py — Profile the Transformer's forward pass across batch sizes.

We use torch.profiler to identify where time is spent (CPU vs CUDA,
forward vs backward, which module).  This is useful for understanding
bottlenecks before optimising.

Run with:
  python profile.py            # summary table
  python -m torch.profiler.profile --export-flamegraph profile_output/ profile.py
"""

import time
import torch
import torch.nn as nn
import math


# ---------------------------------------------------------------------------
# Inline model (same as full_transformer.py)
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
# Profiling helpers
# ---------------------------------------------------------------------------

def make_masks(src_len, tgt_len, device):
    src_mask = torch.ones(1, 1, src_len, src_len, device=device)
    tgt_mask = torch.tril(torch.ones(tgt_len, tgt_len, device=device)).unsqueeze(0).unsqueeze(0)
    cross_mask = torch.ones(1, 1, tgt_len, src_len, device=device)
    return src_mask, tgt_mask, cross_mask


def profile_forward(model, src, tgt, device, n_warmup=10, n_runs=50):
    """Profile forward pass with CPU timers."""
    src_mask, tgt_mask, cross_mask = make_masks(src.shape[1], tgt.shape[1], device)

    src = src.to(device)
    tgt = tgt.to(device)
    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)
    cross_mask = cross_mask.to(device)

    for _ in range(n_warmup):
        model(src, tgt, src_mask, tgt_mask, cross_mask)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        model(src, tgt, src_mask, tgt_mask, cross_mask)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1000)

    return times


def profile_with_profiler(model, src, tgt, device, output_path="profile_output"):
    """Use torch.profiler for detailed breakdown (CPU and CUDA)."""
    import os
    os.makedirs(output_path, exist_ok=True)

    src_mask, tgt_mask, cross_mask = make_masks(src.shape[1], tgt.shape[1], device)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        model(src, tgt, src_mask, tgt_mask, cross_mask)

    print("\n=== Top 20 operations by CPU time ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    if torch.cuda.is_available():
        print("\n=== Top 20 operations by CUDA time ===")
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    prof.export_chrome_trace(f"{output_path}/trace.json")
    print(f"\nChrome trace saved to {output_path}/trace.json")
    print("View at: chrome://tracing")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    configs = [
        # (name, d_model, n_heads, d_ff, n_layers, batch, src_len, tgt_len)
        ("Tiny / short seq",   128,  4, 512,  2,   8,  32,  32),
        ("Tiny / long seq",    128,  4, 512,  2,   8, 128, 128),
        ("Base / short seq",   512,  8, 2048, 6,   4,  64,  64),
        ("Base / long seq",    512,  8, 2048, 6,   4, 256, 256),
    ]

    for name, d_model, n_heads, d_ff, n_layers, batch, src_len, tgt_len in configs:
        model = Transformer(d_model, n_heads, d_ff, n_layers, src_vocab=10000, tgt_vocab=10000)
        model.eval()
        if device == "cuda":
            model = model.cuda()

        src = torch.randint(1, 10000, (batch, src_len))
        tgt = torch.randint(1, 10000, (batch, tgt_len))

        print(f"--- {name} | batch={batch}, src={src_len}, tgt={tgt_len} ---")
        times = profile_forward(model, src, tgt, device)
        print(f"  Mean: {sum(times)/len(times):.2f} ms  |  Min: {min(times):.2f} ms  |  Max: {max(times):.2f} ms")
        print(f"  Params: {sum(p.numel() for p in model.parameters()):,}")

        if device == "cuda":
            try:
                profile_with_profiler(model, src, tgt, device)
            except Exception as e:
                print(f"  (Profiler output unavailable: {e})")
        print()
