"""
benchmark.py — Compare our from-scratch Transformer vs torch.nn.Transformer.

We measure forward pass latency across a range of batch/sequence sizes
on the encoder-decoder task.  Both models receive the same random input.

torch.nn.Transformer uses a different internal design (shared Q/K/V projections
in some paths, different mask handling) but performs the same mathematical
operation.  The benchmark is primarily a consistency check.
"""

import time
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Our Transformer (copied inline so this file is fully standalone)
# ---------------------------------------------------------------------------

import math


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
# Benchmarking helpers
# ---------------------------------------------------------------------------

def benchmark(fn, n_warmup=30, n_runs=100):
    for _ in range(n_warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(n_runs):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / n_runs * 1000   # ms


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 65)
    print("Transformer Forward Pass Benchmark")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    configs = [
        # (d_model, n_heads, d_ff, n_layers, batch, src_len, tgt_len)
        (128,  4, 512,  2,    1,  32,  32),
        (128,  4, 512,  2,    4,  32,  32),
        (256,  8, 1024, 4,    1,  64,  64),
        (256,  8, 1024, 4,    4,  64,  64),
        (512,  8, 2048, 6,    1, 128, 128),
    ]

    src_vocab = 1000
    tgt_vocab = 1000

    for d_model, n_heads, d_ff, n_layers, batch, src_len, tgt_len in configs:
        print(f"--- d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, "
              f"n_layers={n_layers}, batch={batch}, src={src_len}, tgt={tgt_len} ---")

        src = torch.randint(1, src_vocab, (batch, src_len))
        tgt = torch.randint(1, tgt_vocab, (batch, tgt_len))

        # ---- Our from-scratch model ----
        ours = Transformer(d_model, n_heads, d_ff, n_layers, src_vocab, tgt_vocab)
        ours.eval()
        if device == "cuda":
            ours = ours.cuda()
            src = src.cuda()
            tgt = tgt.cuda()

        t_ours = benchmark(lambda: ours(src, tgt))

        # ---- PyTorch nn.Transformer ----
        theirs = nn.Transformer(d_model=d_model, nhead=n_heads,
                                num_encoder_layers=n_layers,
                                num_decoder_layers=n_layers,
                                dim_feedforward=d_ff,
                                batch_first=True)
        theirs.eval()
        if device == "cuda":
            theirs = theirs.cuda()

        t_theirs = benchmark(lambda: theirs(src, tgt))

        print(f"  Ours (from scratch) : {t_ours:8.2f} ms")
        print(f"  PyTorch nn.Trans.   : {t_theirs:7.2f} ms")
        print(f"  Ratio (ours/pytorch): {t_ours/t_theirs:8.3f}x")
        print()

    print("=" * 65)
    print("Note: nn.Transformer uses its own internal design")
    print("      (e.g. shared Q/K/V in some paths). Numerical")
    print("      equivalence is not expected. Focus is on")
    print("      understanding and educational clarity.")
    print("=" * 65)
