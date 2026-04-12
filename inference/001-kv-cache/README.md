# KV-Cache From Scratch

Why autoregressive generation is O(n²) without cache and O(n) with it.

## What we build
- Naive generation (recompute all keys/values each step)
- KV-cache implementation
- Memory vs speed tradeoff visualization
- Benchmark: tokens/sec with and without cache

## Files (planned)
- `naive.py` — no cache
- `kv_cache.py` — with cache
- `benchmark.py`
