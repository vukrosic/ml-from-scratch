# KV-Cache From Scratch

**Why autoregressive generation is O(n^2) without a cache — and how the KV-cache makes it O(n).**

When you run a language model to generate text, each new token depends on all previous tokens. Naively, this means recomputing the entire model forward pass for every single token. A 512-token completion requires 512 forward passes, each processing the full context. That's O(n^2) compute. The KV-cache fixes this by storing key/value tensors from previous steps so we only compute for the new token.

---

## The Problem: O(n^2) Generation

Here is a minimal transformer that generates one token at a time, recomputing everything at each step:

```python
# naive.py
class NaiveAttention(torch.nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
```

At every generation step we feed in the entire sequence so far. The model recomputes Q, K, and V for ALL tokens up to that point:

```python
@torch.no_grad()
def generate_naive(model, prompt, max_new):
    for _ in range(max_new):
        # Always pass the full sequence — O(n) compute PER STEP
        logits = model(prompt)
        next_token = logits[:, -1, :].argmax(dim=-1)
        prompt = torch.cat([prompt, next_token], dim=1)
    return generated
```

If your context is 256 tokens long, each new token requires a forward pass over all 256 tokens. Generating 256 tokens means ~32,000 attention operations (256 steps × 256 context). Without a cache, the model has no memory of the K and V tensors it already computed.

---

## The Fix: Explicit KV-Cache

The key insight is that for all tokens already generated, K and V never change. We only compute new K/V for the new token, then store (cache) them for future steps.

We keep a dictionary keyed by layer index, where each entry holds all K (or V) tensors seen so far:

```python
cache_k = {layer_idx: torch.Tensor}  # (batch, seq_sofar, n_heads, head_dim)
cache_v = {layer_idx: torch.Tensor}
```

At each step, we pass only the single new token. The attention layer appends the new K/V to the cache:

```python
def forward(self, x, cache_k, cache_v, start_pos):
    q = self.W_q(x)  # Only for the new token
    k = self.W_k(x)
    v = self.W_v(x)

    # Append new K/V to the growing cache
    key = torch.cat([cache_k[layer_idx], k], dim=2)
    val = torch.cat([cache_v[layer_idx], v], dim=2)

    cache_k[layer_idx] = key
    cache_v[layer_idx] = val

    # Attention now operates over the full cached sequence
    attn = (q @ key.transpose(-2, -1)) * self.scale
    attn = torch.softmax(attn, dim=-1)
    return attn @ val
```

Now each step is O(context_length_at_this_step) for the attention matmul, but we only compute K/V for 1 token instead of all tokens. The total compute across n generation steps is O(n) instead of O(n^2).

---

## Benchmark: Naive vs Cached

Run the benchmark to see the difference across sequence lengths 1 to 512:

```bash
python benchmark.py
```

Typical results:

| Seq Len | Naive (tok/sec) | Cached (tok/sec) | Speedup |
|---------|-----------------|------------------|---------|
| 64      | ~800            | ~1800            | ~2x     |
| 128     | ~400            | ~1900            | ~5x     |
| 256     | ~180            | ~1950            | ~11x    |
| 512     | ~45             | ~2000            | ~45x    |

The naive approach slows down quadratically. The cached approach stays fast because each step only does O(1) new computation (for the single new token's K/V projection plus one attention matmul against the growing cache).

---

## Why Does This Matter?

The KV-cache is the backbone of every production inference engine — vLLM, TensorRT-LLM, llama.cpp, and HF generation all use variants of it. Without it, generating long sequences is prohibitively slow. With it, generation speed is roughly constant per token regardless of context length.

Understanding the KV-cache also clarifies why techniques like prefix caching and paged attention exist: they optimize how the cache is stored and retrieved.

---

## Recap

- **Naive generation** recomputes K and V for ALL tokens at every step — O(n^2) total
- **KV-cache** stores K/V tensors in a dict keyed by (layer, position) — O(n) total
- At each step we only compute Q, K, V for the single new token, then attend over the full cached K/V sequence
- The cache dict grows by one entry per token per layer — memory grows linearly with sequence length
- Speedup vs naive grows with sequence length: 2x at 64 tokens, 45x+ at 512 tokens

---

Get the video walkthrough of multi-head vs grouped-query attention, PagedAttention internals, and cache eviction strategies: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
