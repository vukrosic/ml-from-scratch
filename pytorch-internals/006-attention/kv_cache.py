"""KV caching benchmark for autoregressive decoding.

Run:
    python kv_cache.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import time


class KVCache:
    """Key/value cache for autoregressive decoding.

    Stores K and V tensors for each computed position.
    On each step, appends new K/V and optionally trims to max_len.
    """
    def __init__(self, max_len=1024):
        self.max_len = max_len
        self.keys = []
        self.values = []

    def update(self, k, v):
        """Append new key/value tensors.

        k, v: (batch, 1, d_head) — single position
        Returns: (batch, seq, d_head) for all cached keys and values
        """
        self.keys.append(k)
        self.values.append(v)

        # Trim if over max_len
        if len(self.keys) > self.max_len:
            self.keys = self.keys[-self.max_len:]
            self.values = self.values[-self.max_len:]

        return torch.cat(self.keys, dim=1), torch.cat(self.values, dim=1)

    def reset(self):
        self.keys = []
        self.values = []


class MultiHeadAttention(nn.Module):
    """Minimal MHA for benchmarking."""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, k_cache=None, v_cache=None):
        """Forward with optional KV cache.

        x: (batch, seq, d_model) — the input to compute Q/K/V from
        k_cache, v_cache: KVCache instances, or None
        Returns: (batch, seq, d_model)
        """
        batch, seq, _ = x.shape

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Update cache if provided
        if k_cache is not None and v_cache is not None:
            K, V = k_cache.update(K, V)

        # Reshape for multi-head
        Q = Q.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, -1, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, -1, self.num_heads, self.d_head).transpose(1, 2)

        attn = F.scaled_dot_product_attention(Q, K, V)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.d_model)
        return self.W_o(attn)


def benchmark_no_cache(model, prompt_len, gen_len):
    """Time autoregressive generation without KV cache.

    In a real no-cache scenario, each step would recompute Q, K, V for
    the entire accumulated sequence. We simulate this by re-running
    the full projection on an accumulated input.
    """
    d_model = model.d_model
    batch = 1

    # Accumulated sequence (prompt + generated so far)
    x = torch.randn(batch, prompt_len, d_model, device='cuda')

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(gen_len):
        # Get the last token
        x_step = x[:, -1:, :]

        # Full recompute each step: project fresh
        Q = model.W_q(x_step)
        K = model.W_k(x_step)
        V = model.W_v(x_step)

        # Reshape
        Q = Q.view(batch, 1, model.num_heads, model.d_head).transpose(1, 2)
        K = K.view(batch, -1, model.num_heads, model.d_head).transpose(1, 2)
        V = V.view(batch, -1, model.num_heads, model.d_head).transpose(1, 2)

        # Attention over all accumulated tokens
        attn = F.scaled_dot_product_attention(Q, K, V)
        attn = attn.transpose(1, 2).contiguous().view(batch, 1, model.d_model)
        out = model.W_o(attn)

        # Append to sequence (simulates full recompute)
        new_token = out
        x = torch.cat([x, new_token], dim=1)

    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / gen_len


def benchmark_with_cache(model, prompt_len, gen_len):
    """Time autoregressive generation with KV cache."""
    d_model = model.d_model
    batch = 1

    # Start with prompt
    x = torch.randn(batch, prompt_len, d_model, device='cuda')
    k_cache = KVCache()
    v_cache = KVCache()

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    for step in range(gen_len):
        # Get only the last token
        x_step = x[:, -1:, :]

        # Forward with cache
        _ = model(x_step, k_cache=k_cache, v_cache=v_cache)

        # Simulate getting the next token (we just append a random tensor)
        x = torch.cat([x, torch.randn(batch, 1, d_model, device='cuda')], dim=1)

    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / gen_len


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")

    d_model, num_heads = 512, 8
    model = MultiHeadAttention(d_model, num_heads).cuda()

    prompt_len = 128
    gen_len = 64

    print(f"Prompt length: {prompt_len}, Generation length: {gen_len}")
    print(f"\nNote: without cache, the full sequence grows each step.")
    print(f"      with cache, each step only processes one new token.")

    # Warmup
    _ = benchmark_with_cache(model, prompt_len, gen_len)
    torch.cuda.synchronize()

    # Benchmark
    no_cache_time = benchmark_no_cache(model, prompt_len, gen_len)
    with_cache_time = benchmark_with_cache(model, prompt_len, gen_len)

    print(f"\nWithout KV cache: ~{no_cache_time*1000:.2f} ms/token")
    print(f"With KV cache:    ~{with_cache_time*1000:.2f} ms/token")
    print(f"Speedup:          ~{no_cache_time/with_cache_time:.1f}x")

    # Scaling with sequence length
    print("\n--- KV Cache speedup vs. sequence length ---")
    for prompt_len in [64, 128, 256, 512]:
        gen_len = 32
        torch.cuda.synchronize()
        no_cache_t = benchmark_no_cache(model, prompt_len, gen_len)
        torch.cuda.synchronize()
        with_cache_t = benchmark_with_cache(model, prompt_len, gen_len)
        print(f"  prompt={prompt_len:3d}: no_cache=~{no_cache_t*1000:.2f}ms, "
              f"cached=~{with_cache_t*1000:.2f}ms, speedup=~{no_cache_t/with_cache_t:.1f}x")


if __name__ == "__main__":
    main()
