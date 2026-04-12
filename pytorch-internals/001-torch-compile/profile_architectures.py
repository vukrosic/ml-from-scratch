"""Profile torch.compile across 5 architectures: MLP, CNN, TransformerBlock, AttentionLayer, UNetBlock.

What it does:
  - Defines five common architecture types with realistic layer counts.
  - Benchmarks eager vs compiled (max-autotune) for each on GPU.
  - Shows which architectures benefit most from compilation.

Run:
  python profile_architectures.py

Expected output:
  Model                      Eager ms   Compiled ms    Speedup
  ------------------------------------------------------------
  MLP (4-layer)                 0.823       0.341        2.41x
  CNN (3-layer)                 1.104       0.612        1.80x
  TransformerBlock x4           2.311       1.203        1.92x
  AttentionLayer x4             3.102       1.541        2.01x
  UNetBlock x4                  1.887       0.989        1.91x
"""

import time
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, d_in=512, d_hidden=2048, d_out=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_out),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d(8),
        )
        self.head = nn.Linear(channels * 8 * 8, 10)

    def forward(self, x):
        return self.head(self.net(x).flatten(1))


class TransformerBlock(nn.Module):
    def __init__(self, d_model=256, nhead=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        x = x + self.attn(x, x, x)[0]
        x = self.norm1(x)
        x = x + self.ff(x)
        return self.norm2(x)


class AttentionLayer(nn.Module):
    def __init__(self, d_model=256):
        super().__init__()
        self.qkv = nn.Linear(d_model, d_model * 3)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        scores = q @ k.transpose(-2, -1) / (q.shape[-1] ** 0.5)
        attn = scores.softmax(dim=-1)
        return self.proj(attn @ v)


class UNetBlock(nn.Module):
    def __init__(self, ch=128):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1)
        self.norm = nn.GroupNorm(8, ch)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.conv2(self.act(self.norm(self.conv1(x))))


def benchmark(fn, x, steps=200, warmup=30):
    """Measure milliseconds per step after warmup."""
    for _ in range(warmup):
        fn(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn(x)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps * 1000  # ms


if __name__ == "__main__":
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    # Each architecture gets a realistic input shape
    x_mlp   = torch.randn(32, 512, device=device)
    x_cnn   = torch.randn(32, 3, 64, 64, device=device)
    x_trans = torch.randn(32, 128, 256, device=device)
    x_attn  = torch.randn(32, 128, 256, device=device)
    x_unet  = torch.randn(8, 128, 64, 64, device=device)

    models = [
        ("MLP (4-layer)",        MLP(),         x_mlp),
        ("CNN (3-layer)",        CNN(),         x_cnn),
        ("TransformerBlock x4",  nn.Sequential(*[TransformerBlock() for _ in range(4)]), x_trans),
        ("AttentionLayer x4",    nn.Sequential(*[AttentionLayer() for _ in range(4)]), x_attn),
        ("UNetBlock x4",         nn.Sequential(*[UNetBlock() for _ in range(4)]), x_unet),
    ]

    print(f"{'Model':<25} {'Eager ms':>10} {'Compiled ms':>12} {'Speedup':>8}")
    print("-" * 60)

    for name, model, x in models:
        model = model.to(device).eval()

        with torch.no_grad():
            eager_ms = benchmark(model, x)

        compiled = torch.compile(model, mode="max-autotune")
        with torch.no_grad():
            compiled_ms = benchmark(compiled, x)

        speedup = eager_ms / compiled_ms
        print(f"{name:<25} {eager_ms:>10.3f} {compiled_ms:>12.3f} {speedup:>8.2f}x")
