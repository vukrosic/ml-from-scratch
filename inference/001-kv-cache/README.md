# KV-Cache From Scratch

**Why autoregressive generation is O(n^2) without a cache — and how the KV-cache makes it O(n).**

Every time ChatGPT prints a word, behind the scenes a transformer runs a
forward pass. Without the KV-cache, the model would redo *all* the work it
already did for every previous token — every single step. At 512 tokens that
means roughly 131,000 redundant matrix multiplications. The KV-cache is a
dead-simple idea (store the results you already computed) that turns a
quadratic nightmare into linear-time generation. It is the single most
important optimization in every production LLM serving stack: vLLM,
TensorRT-LLM, llama.cpp, HuggingFace generate — all of them depend on it.

This tutorial builds the cache from scratch so you can see exactly what is
stored, when, and why.

---

## 1. The O(n^2) Problem — Visualized

In autoregressive generation the model produces one token at a time. Each
new token's logits depend on *all* tokens before it. Naively, the model
recomputes Q, K, and V projections for the entire sequence at every step.

Here is what that looks like over 8 generation steps:

```
Step  Tokens processed this step         Work
────  ─────────────────────────────────  ────
  1   [t1]                                 1
  2   [t1 t2]                              2
  3   [t1 t2 t3]                           3
  4   [t1 t2 t3 t4]                        4
  5   [t1 t2 t3 t4 t5]                     5
  6   [t1 t2 t3 t4 t5 t6]                  6
  7   [t1 t2 t3 t4 t5 t6 t7]              7
  8   [t1 t2 t3 t4 t5 t6 t7 t8]           8
                                     ─────────
                              Total = 1+2+...+8 = 36
```

The total work is the triangular number n(n+1)/2. For real sequence lengths:

```
  n       n(n+1)/2       Relative to n
─────   ───────────     ───────────────
   64         2,080           32x
  128         8,256           64x
  256        32,896          128x
  512       131,328          256x
 2048     2,098,176        1,024x
```

At n = 2048 the model does over 2 million token-level forward computations
to generate 2048 tokens. Most of that work is *identical* to work already
done on prior steps — we keep recomputing K and V for tokens we have seen
before.

The fix: compute K and V for each token *once*, store them, reuse them.

---

## 2. The Naive Implementation

Here is a minimal attention module that does NOT use a cache. Every call
processes the full sequence from scratch.

### Piece 1 — Naive attention module

```python
# naive.py
import torch
import torch.nn as nn
import math

class NaiveAttention(nn.Module):
    """Single-head attention with no cache. Recomputes everything."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        B, S, _ = x.shape

        q = self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        # q, k, v: (batch, n_heads, seq_len, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale   # (B, H, S, S)
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v                                    # (B, H, S, head_dim)

        return self.W_o(out.transpose(1, 2).contiguous().view(B, S, -1))
```

Every time `forward` is called with the growing sequence, it re-projects K
and V for *all* tokens, including those it already projected last step.

### Piece 2 — Naive generation loop

```python
@torch.no_grad()
def generate_naive(model, prompt_ids, max_new_tokens):
    """Generate tokens one at a time. Full forward pass every step."""
    seq = prompt_ids  # (1, prompt_len)

    for _ in range(max_new_tokens):
        logits = model(seq)                          # forward over ENTIRE seq
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_id], dim=1)       # seq grows by 1

    return seq
```

At step t the model processes a sequence of length (prompt_len + t). The
attention matmul alone is O(seq_len^2) per layer. Summed over all steps,
total compute is O(n^3) for the attention blocks and O(n^2) for the linear
projections. Either way, it is catastrophically wasteful.

---

## 3. The Fix: Explicit KV-Cache — Step by Step

The key insight: **for tokens already generated, K and V never change.**
The Q/K/V projections are deterministic linear transforms of fixed
embeddings. Once we compute K_3 and V_3 for token 3, they are the same
forever. So we store them.

### What the cache looks like

```
cache = {
    layer_0: { "k": Tensor(B, H, seq_so_far, D),
               "v": Tensor(B, H, seq_so_far, D) },
    layer_1: { "k": Tensor(B, H, seq_so_far, D),
               "v": Tensor(B, H, seq_so_far, D) },
    ...
    layer_N: { "k": ..., "v": ... },
}
```

Each layer has its own independent K and V cache. The `seq_so_far`
dimension grows by 1 at every generation step.

### Walking through 4 generation steps

Assume d_model=64, n_heads=4, head_dim=16, 2 layers. Prompt is 3 tokens.

```
STEP 0 — Prefill (process the 3-token prompt all at once)
──────────────────────────────────────────────────────────
  Input to model:  x = (1, 3, 64)      ← all 3 prompt tokens

  For each layer:
    Q = W_q(x) → (1, 4, 3, 16)         ← 3 queries
    K = W_k(x) → (1, 4, 3, 16)         ← 3 keys
    V = W_v(x) → (1, 4, 3, 16)         ← 3 values

    cache[layer]["k"] = K               ← store:  (1, 4, 3, 16)
    cache[layer]["v"] = V               ← store:  (1, 4, 3, 16)

    attn = Q @ K^T → (1, 4, 3, 3)      ← 3x3 attention matrix
    out  = attn @ V → (1, 4, 3, 16)

  Output logits: (1, 3, vocab)
  Sample from logits[:, -1, :] → token t4


STEP 1 — Decode (generate token t4)
────────────────────────────────────
  Input to model:  x = (1, 1, 64)      ← ONLY the new token t4

  For each layer:
    q_new = W_q(x) → (1, 4, 1, 16)     ← 1 query
    k_new = W_k(x) → (1, 4, 1, 16)     ← 1 key
    v_new = W_v(x) → (1, 4, 1, 16)     ← 1 value

    K = cat(cache[layer]["k"], k_new)   ← (1, 4, 4, 16)  was 3, now 4
    V = cat(cache[layer]["v"], v_new)   ← (1, 4, 4, 16)

    cache[layer]["k"] = K               ← update cache
    cache[layer]["v"] = V

    attn = q_new @ K^T → (1, 4, 1, 4)  ← 1 query attends to 4 keys
    out  = attn @ V     → (1, 4, 1, 16)

  Sample → token t5


STEP 2 — Decode (generate token t5)
────────────────────────────────────
  Input:  x = (1, 1, 64)               ← ONLY token t5

  For each layer:
    q_new → (1, 4, 1, 16)
    k_new → (1, 4, 1, 16)
    v_new → (1, 4, 1, 16)

    K = cat(cache, k_new)              ← (1, 4, 5, 16)  was 4, now 5
    V = cat(cache, v_new)              ← (1, 4, 5, 16)

    attn = q_new @ K^T → (1, 4, 1, 5)  ← 1 query, 5 keys
    out  = attn @ V     → (1, 4, 1, 16)

  Sample → token t6


STEP 3 — Decode (generate token t6)
────────────────────────────────────
  Input:  x = (1, 1, 64)               ← ONLY token t6

  K grows to (1, 4, 6, 16), V same.
  attn shape: (1, 4, 1, 6)             ← 1 query, 6 keys
  Sample → token t7
```

Notice the pattern: at every decode step the model only computes Q, K, V
for **one** token. The attention matmul is (1 x seq_so_far) instead of
(seq_so_far x seq_so_far). The K and V projections (the expensive linear
layers) run on a single token instead of the full sequence.

---

## 4. The Code — Cached Attention

### Piece 3 — Attention module with KV-cache

```python
# cached.py
class CachedAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, cache_k, cache_v, start_pos):
        """
        x:         (batch, seq_len, d_model)   seq_len=1 during decode
        cache_k:   (batch, n_heads, pos, head_dim) or None on first call
        cache_v:   same
        start_pos: int — how many tokens are already in the cache
        """
        B, S, _ = x.shape

        # Project ONLY the new token(s)
        q = self.W_q(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        # Append new K/V to cache
        if cache_k is not None:
            k = torch.cat([cache_k, k], dim=2)   # (B, H, start_pos + S, D)
            v = torch.cat([cache_v, v], dim=2)

        # Attention: q has S rows, k has (start_pos + S) rows
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = attn @ v

        out = out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.W_o(out), k, v   # return updated cache
```

Key differences from the naive version:
- `x` is only the new token(s), not the full sequence
- We concatenate new K/V onto the existing cache
- We return the updated K and V so the caller can store them
- `start_pos` tracks how far into the sequence we are (needed for
  rotary position embeddings in real models)

### Piece 4 — Multi-layer cache management

```python
class CachedTransformer(nn.Module):
    def __init__(self, n_layers, d_model, n_heads, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            CachedAttention(d_model, n_heads) for _ in range(n_layers)
        ])
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.n_layers = n_layers

    def forward(self, token_ids, cache, start_pos):
        """
        cache: list of dicts, one per layer
               cache[i] = {"k": Tensor or None, "v": Tensor or None}
        """
        x = self.embed(token_ids)

        for i, layer in enumerate(self.layers):
            x, new_k, new_v = layer(
                x,
                cache[i]["k"],
                cache[i]["v"],
                start_pos
            )
            cache[i]["k"] = new_k
            cache[i]["v"] = new_v

        return self.head(x)
```

The cache is a list of length `n_layers`. Each entry holds that layer's
accumulated K and V tensors. Every layer independently grows its cache by
one token per decode step.

### Piece 5 — Generation loop with cache

```python
@torch.no_grad()
def generate_cached(model, prompt_ids, max_new_tokens):
    """Generate tokens using the KV-cache."""
    # Initialize empty cache for each layer
    cache = [{"k": None, "v": None} for _ in range(model.n_layers)]

    # PREFILL: process entire prompt at once
    logits = model(prompt_ids, cache, start_pos=0)
    next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    seq = torch.cat([prompt_ids, next_id], dim=1)
    start_pos = prompt_ids.shape[1]

    # DECODE: one token at a time, using cache
    for step in range(1, max_new_tokens):
        logits = model(next_id, cache, start_pos=start_pos + step - 1)
        next_id = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        seq = torch.cat([seq, next_id], dim=1)

    return seq
```

Compare to the naive loop: the cached version passes `next_id` (1 token)
to the model instead of `seq` (the full growing sequence). The model
touches each layer's K/V projection for just one token, then multiplies
against the cached keys and values.

---

## 5. Prefill vs Decode — Two Very Different Phases

Real inference has two distinct phases with different compute
characteristics:

```
         PREFILL                              DECODE
  ┌─────────────────────┐           ┌─────────────────────┐
  │  Process all prompt  │           │  Generate one token  │
  │  tokens in parallel  │           │  at a time, using    │
  │                      │   ───►    │  the cache           │
  │  Compute-bound       │           │  Memory-bound        │
  │  (big matmuls)       │           │  (reading cache)     │
  └─────────────────────┘           └─────────────────────┘
       prompt_len tokens                  1 token per step
```

**Prefill** processes the entire prompt (e.g., 2000 tokens of system
prompt + user message) in a single forward pass. The attention matmul is
(2000 x 2000) — large and compute-bound. This fills the KV-cache for all
prompt tokens at once. GPUs love this: the big matrix multiplications
saturate the compute units.

**Decode** generates one token per step. The attention matmul is
(1 x seq_so_far) — tiny. The bottleneck shifts from compute to memory
bandwidth: we need to read the entire KV-cache from GPU memory to compute
attention, but we only produce one output vector. The arithmetic intensity
is very low. This is why decode is often called "memory-bound."

This asymmetry explains several engineering decisions:
- Batching multiple requests together during decode improves GPU
  utilization (more queries share the same memory-read cost)
- Prefill and decode are sometimes run on different hardware or scheduled
  in separate phases (disaggregated serving)
- Quantizing the KV-cache (e.g., FP16 to INT8) helps decode more than
  prefill because decode is memory-bandwidth limited

---

## 6. Memory Analysis — How Big Is the Cache?

The KV-cache stores two tensors (K and V) per layer. The memory formula:

```
Cache memory = 2 * n_layers * seq_len * n_heads * head_dim * bytes_per_param
               ─   ────────   ───────   ───────   ────────   ──────────────
               K+V  depth     context   width     per head   FP16 = 2 bytes
```

### Example: LLaMA 7B at 2048 context length (FP16)

```
Model parameters:
  n_layers  = 32
  n_heads   = 32
  head_dim  = 128    (d_model=4096, 4096/32=128)
  seq_len   = 2048
  bytes     = 2      (FP16)

Cache memory = 2 * 32 * 2048 * 32 * 128 * 2
             = 2 * 32 * 2048 * 32 * 128 * 2
             = 536,870,912 bytes
             = 512 MB  (per request!)
```

That is 512 MB of GPU memory *per concurrent request*. On a 24 GB GPU,
you can fit roughly 46 concurrent KV-caches at 2048 context — before
accounting for model weights (~14 GB in FP16) and activations. In
practice:

```
24 GB total GPU memory
 - 14 GB model weights (FP16)
 - ~1 GB activations / overhead
─────────────────────────────────
   9 GB available for KV-caches
   9 GB / 0.5 GB per cache ≈ 18 concurrent requests at 2048 ctx
```

At longer contexts the cache dominates memory:

```
Context    Cache/req    Fit in 9 GB (LLaMA 7B FP16)
───────    ─────────    ────────────────────────────
   512      128 MB         70 requests
  2048      512 MB         18 requests
  4096     1024 MB          9 requests
  8192     2048 MB          4 requests
 32768     8192 MB          1 request
```

This is why KV-cache memory management is such a critical problem in
production serving and why techniques like quantized KV-cache, paged
attention, and cache eviction exist.

---

## 7. PagedAttention (vLLM's Approach)

The naive cache implementation pre-allocates a contiguous tensor for the
maximum sequence length. This wastes memory: a request that only generates
50 tokens still reserves memory for 2048 tokens.

**PagedAttention** (the key innovation behind vLLM) borrows the idea of
virtual memory paging from operating systems:

```
 Traditional KV-cache             PagedAttention
 ─────────────────────           ──────────────────────
 ┌─────────────────┐             ┌──────┐  ┌──────┐
 │  request 1      │             │ pg 0 │  │ pg 1 │  ← request 1
 │  [==used==|waste]│             └──────┘  └──────┘
 ├─────────────────┤
 │  request 2      │             ┌──────┐
 │  [=used|waste===]│             │ pg 0 │             ← request 2
 ├─────────────────┤             └──────┘
 │  request 3      │
 │  [===used|waste=]│             ┌──────┐  ┌──────┐  ┌──────┐
 └─────────────────┘             │ pg 0 │  │ pg 1 │  │ pg 2 │ ← req 3
                                  └──────┘  └──────┘  └──────┘

 Each request gets max_len      Pages allocated on demand.
 contiguous block. Internal      No internal fragmentation.
 fragmentation wastes memory.    Pages can be non-contiguous.
```

How it works:
1. The KV-cache is divided into fixed-size **blocks** (pages), typically
   holding 16 tokens each.
2. Each request has a **block table** that maps logical token positions to
   physical block addresses (like a page table in an OS).
3. New blocks are allocated only when needed — no up-front reservation.
4. A custom attention kernel gathers K/V vectors from scattered physical
   blocks using the block table at attention-computation time.

Benefits:
- Near-zero memory waste (only the last block can be partially empty)
- Requests with different lengths share the physical memory pool efficiently
- Enables much higher batch sizes = higher throughput
- Blocks can be shared across requests (enables prefix caching, see below)

---

## 8. Prefix Caching — Reusing Shared Prompts

Many production workloads send the same system prompt to every request:

```
Request 1: [SYSTEM PROMPT] + "What is the capital of France?"
Request 2: [SYSTEM PROMPT] + "Explain quantum computing."
Request 3: [SYSTEM PROMPT] + "Write a haiku about rain."
                ▲
                │
    Same 500-token prefix, computed 3 times
```

**Prefix caching** computes the KV-cache for the shared system prompt
once and reuses it across all requests:

```
                 ┌──────────────────────────┐
                 │  Shared KV-cache for      │
                 │  system prompt (500 tok)   │
                 └────────┬─────────────────┘
                    ┌─────┼──────┐
                    ▼     ▼      ▼
                  Req 1  Req 2  Req 3
                  (own   (own   (own
                  tail)  tail)  tail)
```

Implementation with PagedAttention: the shared prefix maps to the same
physical blocks. Each request's block table points to those shared blocks
for positions 0..499, then to private blocks for the unique suffix. The
shared blocks use reference counting — they are freed only when all
requests using them have finished.

Savings: for a 500-token system prompt with 100 concurrent requests, you
store 500 tokens of K/V once instead of 100 times. At LLaMA 7B FP16
that is ~125 MB saved per request, or ~12.5 GB total.

---

## 9. Common Mistakes

### Mistake 1: Forgetting to update the position index

Rotary position embeddings (RoPE) encode *absolute* position into Q and
K. When using the cache, the new token must get position `start_pos`, not
position 0:

```python
# WRONG — new token always gets position 0
freqs = self.rotary_emb(seq_len=1)
q = apply_rotary(q, freqs)

# RIGHT — new token gets its actual position in the sequence
freqs = self.rotary_emb(seq_len=start_pos + 1)
q = apply_rotary(q, freqs[start_pos:start_pos+1])
k = apply_rotary(k, freqs[start_pos:start_pos+1])
```

If you get this wrong, the model will attend as if every token is at
position 0 and produce garbage after the first few tokens.

### Mistake 2: Cache shape mismatches

The cache tensors must have shape `(batch, n_heads, seq, head_dim)`. A
common bug is transposing the wrong dimensions:

```python
# WRONG — head_dim and seq dimensions swapped
k = k.view(B, S, self.n_heads, self.head_dim)  # (B, S, H, D)
cache_k = torch.cat([cache_k, k], dim=1)        # cat along S... but
                                                  # cache is (B, H, S, D)!

# RIGHT — transpose before caching
k = k.view(B, S, self.n_heads, self.head_dim).transpose(1, 2)  # (B, H, S, D)
cache_k = torch.cat([cache_k, k], dim=2)  # cat along dim=2 (the S dim)
```

### Mistake 3: Not pre-allocating cache memory

Using `torch.cat` to grow the cache at every step is simple but slow — it
allocates a new tensor and copies all data each time:

```python
# SLOW — O(n^2) total memory copies across n steps
cache_k = torch.cat([cache_k, k_new], dim=2)

# FAST — pre-allocate, then fill slices
# At init:
cache_k = torch.zeros(B, H, max_seq_len, D, device=device)
# At each step:
cache_k[:, :, start_pos:start_pos+1, :] = k_new
```

Pre-allocation avoids repeated copies and memory fragmentation. Every
serious implementation does this. The `torch.cat` version is fine for
tutorials but not for production.

### Mistake 4: Applying causal mask incorrectly with cache

During prefill you need a causal mask (each token can only attend to
earlier tokens). During decode with the cache, the single query token
should attend to ALL cached positions — no causal mask needed since there
is only one query position:

```python
if seq_len > 1:
    # Prefill: apply causal mask
    mask = torch.triu(torch.ones(S, S), diagonal=1).bool()
    attn.masked_fill_(mask, float("-inf"))
# Decode (seq_len == 1): no mask needed — single query attends to all
```

---

## 10. Benchmark: Naive vs Cached

Run the benchmark to see the difference across sequence lengths:

```bash
python benchmark.py
```

Typical results on a single GPU:

```
Tokens/sec vs Sequence Length
─────────────────────────────────────────────────────────────

2000 ┤                                              ●─────── ●
     │                                  ●──────●
     │                        ●────●
1500 ┤              ●────●                              Cached
     │     ●───●
     │
1000 ┤ ●
     │ ●
     │
 500 ┤     ●
     │          ●
     │               ●                                  Naive
 200 ┤                    ●
     │                         ●
 100 ┤                              ●
  50 ┤                                   ●
     │                                        ●
  25 ┤                                              ●──────── ●
     └──┬────┬────┬────┬────┬────┬────┬────┬────┬────┬────┬──
       32   64   96  128  192  256  320  384  448  512  576

       Sequence Length (tokens)
```

Numerical results:

```
 Seq Len │ Naive (tok/s) │ Cached (tok/s) │ Speedup
─────────┼───────────────┼────────────────┼────────
      32 │         ~1000 │          ~1100 │    1.1x
      64 │          ~800 │          ~1500 │    1.9x
     128 │          ~400 │          ~1800 │    4.5x
     256 │          ~180 │          ~1900 │   10.6x
     384 │           ~90 │          ~1950 │   21.7x
     512 │           ~45 │          ~2000 │   44.4x
     576 │           ~30 │          ~2000 │   66.7x
```

The naive approach slows down quadratically — halving the speed each time
we double the sequence length. The cached approach stays nearly constant
because each decode step does O(1) new projection work plus one small
matmul against the growing cache (which is memory-bandwidth bound, not
compute bound, so it barely changes speed).

At 512 tokens the cache gives a **45x speedup**. At 2048 tokens (common
for real workloads) the speedup would be ~500x. This is not a micro-
optimization; it is the difference between usable and unusable.

---

## 11. Summary

```
Concept              What it does                              Complexity
───────────────────  ────────────────────────────────────────  ──────────
Naive generation     Recompute Q, K, V for ALL tokens/step     O(n^2)
KV-cache             Store K, V; only compute new token/step   O(n)
Prefill              Process full prompt, fill cache            Compute-bound
Decode               1 token/step, read from cache             Memory-bound
Pre-allocation       Reserve max_seq_len up front              Avoids copies
PagedAttention       Non-contiguous memory pages for cache     Better batching
Prefix caching       Share cache for common system prompts     Memory savings
```

The KV-cache is conceptually simple — just save your intermediate results
— but it has deep implications for every aspect of LLM serving: memory
management, batching strategy, hardware utilization, and system
architecture. Every production inference engine is built around it.

---

Get the video walkthrough of multi-head vs grouped-query attention, PagedAttention internals, and cache eviction strategies: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
