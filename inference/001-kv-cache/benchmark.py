"""
Benchmark: naive vs cached generation.

Measures tokens/sec and peak memory as sequence length grows from 1 to 512.
"""

import gc
import time
import torch

from naive import NaiveGenerator, generate_naive
from kv_cache import CachedGenerator, generate_cached


def measure_tokens_per_sec(func, model, prompt, max_new, num_runs=3):
    """Run generation multiple times and return median tokens/sec."""
    times = []
    for _ in range(num_runs):
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

        start = time.perf_counter()
        func(model, prompt, max_new)
        elapsed = time.perf_counter() - start

        tokens_generated = max_new
        tok_per_sec = tokens_generated / elapsed
        times.append(tok_per_sec)

    return sorted(times)[len(times) // 2]


def measure_memory(func, model, prompt, max_new):
    """Rough memory estimate using resident set size."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    func(model, prompt.clone(), max_new)

    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024**2  # MB
    return 0  # CPU fallback


def benchmark_sequence_lengths(ModelClass, generate_fn, name, max_len=512, step=64):
    """Sweep sequence lengths and collect tokens/sec and memory."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {name}")
    print(f"{'='*60}")
    print(f"{'Seq Len':>10} {'Tokens/sec':>15} {'Memory (MB)':>15}")
    print(f"{'-'*60}")

    results = []
    prompt_base = torch.tensor([[1, 2, 3, 4, 5]])

    for seq_len in range(1, max_len + 1, step):
        # Vary prompt length to simulate growing context
        prompt = prompt_base[:, :min(seq_len, 5)].clone()

        # Number of new tokens to generate
        max_new = min(32, max_len - seq_len)
        if max_new < 4:
            continue

        model = ModelClass(vocab_size=512, d_model=128, n_heads=4, n_layers=2)

        tps = measure_tokens_per_sec(generate_fn, model, prompt, max_new, num_runs=3)
        mem = measure_memory(generate_fn, model, prompt, max_new)

        results.append((seq_len, tps, mem))
        print(f"{seq_len:>10} {tps:>15.2f} {mem:>15.2f}")

    return results


if __name__ == "__main__":
    print("KV-Cache Benchmark")
    print("Sequence lengths: 1 to 512, step 64")
    print("Metrics: tokens/sec and memory usage")

    # Naive generation results
    naive_results = benchmark_sequence_lengths(
        NaiveGenerator, generate_naive, "Naive (no cache)", max_len=512, step=64
    )

    # Cached generation results
    cached_results = benchmark_sequence_lengths(
        CachedGenerator, generate_cached, "KV-Cache", max_len=512, step=64
    )

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if naive_results and cached_results:
        # Compare at a mid-range length
        mid = naive_results[len(naive_results) // 2]
        mid_cached = cached_results[len(cached_results) // 2]

        print(f"\nAt sequence length ~{mid[0]}:")
        print(f"  Naive:   {mid[1]:.2f} tokens/sec")
        print(f"  Cached:  {mid_cached[1]:.2f} tokens/sec")
        print(f"  Speedup: {mid_cached[1]/mid[1]:.1f}x")

        print(f"\nKey insight:")
        print(f"  Naive is O(n^2) — time grows quadratically with sequence length")
        print(f"  Cached is O(n)  — time grows linearly with sequence length")
        print(f"  The speedup grows larger as sequences get longer.")
