# KV-Cache — Extended Deep Dive

This document covers the deeper topics NOT in the free lesson: PagedAttention, multi-head vs grouped-query attention, cache eviction, and prefix caching. The free lesson implements a simple dict-based cache. Production systems need much more.

---

## 1. PagedAttention (vLLM-style)

The naive cache as a dict works but has a critical flaw: **memory fragmentation**. When you allocate `torch.Tensor` for each new position, you request a new GPU memory allocation. GPU memory allocations are expensive, and you end up with many small non-contiguous tensors that can't be efficiently processed in parallel.

vLLM's PagedAttention solves this by analogy to virtual memory paging. Instead of one large contiguous tensor per layer, we allocate a fixed-size **page** (e.g., 16 tokens) and fill it like a book. The "page table" tracks which pages hold which token positions.

### Block-wise storage

```
Page 0: [token_0, token_1, ..., token_15]   # 16 tokens per page
Page 1: [token_16, token_17, ..., token_31]
...
```

The KV-cache is a list of page references. A page table maps (layer, physical_page_idx) -> page. This allows:

1. **Non-contiguous storage**: pages don't need to be adjacent in memory
2. **Memory sharing**: when two sequences share a prefix (e.g., system prompt), they can share the same physical pages for that prefix
3. **Efficient eviction**: when GPU memory is full, evict least-recently-used pages to CPU RAM or disk (-swizzle)

### PagedAttention kernel

The attention matmul becomes a gather-scatter operation over pages. Instead of `attn = (q @ k.T)`, we iterate over pages:

```python
# Simplified paged attention (not actual Triton code)
for page_idx in range(num_pages):
    page_k = cache_k[page_table[page_idx]]          # Gather page
    block_attn = q @ page_k.transpose(-2, -1)       # Attention over page
    acc_attn += block_attn                           # Accumulate
    block_v = cache_v[page_table[page_idx]]
    acc_out += block_attn @ block_v
```

The actual vLLM implementation uses FlashAttention with custom page-aware tiling, running as a single fused Triton kernel. The key property: **no materialization of the full KV tensor**. Pages are streamed through SRAM.

### Memory efficiency gains

With naive allocation: a 70B model with 2048 context needs ~1.6TB for KV-cache (layers × 2 × seq_len × batch × heads × head_dim × dtype).

With PagedAttention at 16 tokens/page: you only allocate what's needed. A batch of diverse-length sequences wastes minimal memory — pages are filled as needed, and partially filled last pages are compact.

---

## 2. Multi-Head Attention vs Grouped-Query Attention

Standard **Multi-Head Attention (MHA)** has separate K and V projections for every head:

```
n_heads_q query heads
n_heads_kv key heads (same as query in MHA)
n_heads_v value heads (same as query in MHA)
```

For d_model=512, n_heads=8, head_dim=64: each of 8 heads has its own W_k, W_v, W_q, W_o. Total K/V compute: 8 × 64 × 512 = 262K parameters.

**Grouped-Query Attention (GQA)** (used in Llama 2, Mistral, and most modern models) reduces the number of K/V heads to fewer than Q heads. For example, Llama 2 70B uses n_heads=8 Q heads but only n_heads_kv=1 key/value head:

```
n_heads_q = 8   (query heads — full computation)
n_heads_kv = 1  (shared key/value heads — much cheaper)
```

### KV-cache size difference

The cache stores tensors of shape `(batch, seq_len, n_heads_kv, head_dim)`. With MHA, n_heads_kv = n_heads_q. With GQA, n_heads_kv << n_heads_q.

For the 70B model above:
- MHA cache per layer: `(batch, seq, 8, 128)` = 8 heads
- GQA cache per layer: `(batch, seq, 1, 128)` = 1 head (8x smaller KV-cache!)

This is why GQA dramatically reduces KV-cache memory. For long-context models (32K+ tokens), GQA is essential.

### Implementation difference

In the code, GQA changes the shape mismatch in attention:

```python
# MHA: q, k, v all have same n_heads
q: (batch, n_heads, seq, head_dim)
k: (batch, n_heads, seq, head_dim)

# GQA: k/v have fewer heads than q
q: (batch, n_heads_q, seq, head_dim)
k: (batch, n_heads_kv, seq, head_dim)  # n_heads_kv < n_heads_q

# Need to expand k/v before attention
k = k.repeat_interleave(n_heads_q // n_heads_kv, dim=1)
v = v.repeat_interleave(n_heads_q // n_heads_kv, dim=1)
```

The repeat_interleave broadcasts the shared K/V heads across query groups. This is a cheap operation (just a view, no memory copy) but changes the FLOPs profile.

---

## 3. Cache Eviction for Long Sequences

Even with GQA, very long sequences exhaust GPU memory. Cache eviction policies decide which tokens to keep and which to drop.

### Eviction strategies

**1. Naive sliding window**: keep only the last W tokens (e.g., last 4096). Simple but destroys long-range dependencies.

**2. Learned eviction**: a small policy network learns which past tokens to retain. Used in Huang et al. "H2O: Heavy-Hitter Oracle" and "FastV" — these score tokens by their attention "importance" (heavy-hitter heuristic) and evict low-importance tokens first.

**3. Priority-based eviction**: assign each token a priority score (e.g., last recv time, attention rank) and evict lowest-priority when memory is full. vLLM uses this with its page table — partially filled pages are evicted first.

**4. Cascade memory hierarchy**: keep recent pages on GPU, older pages in CPU RAM, oldest on NVMe SSD. "PagedAttention v2" (vLLM 0.3+) supports KV-cache offloading to CPU/NVMe with automatic prefetching.

### The Eviction API (conceptual)

```python
class EvictionPolicy:
    def select_victim(self, cache_pages: list[Page], access_log: list[Access]) -> Page:
        """Return the page to evict."""
        raise NotImplementedError

class LRUPolicy(EvictionPolicy):
    def select_victim(self, cache_pages, access_log):
        return min(cache_pages, key=lambda p: p.last_access_time)

class H2OPolicy(EvictionPolicy):
    def select_victim(self, cache_pages, heavy_hitter_scores):
        return min(cache_pages, key=lambda p: heavy_hitter_scores[p.token_start_idx])
```

Production systems (vLLM, TensorRT-LLM) implement custom eviction kernels that update the page table and copy evicted pages to slower storage in the same kernel stream as generation.

---

## 4. Prefix Caching Optimization

In many inference scenarios, multiple requests share the same **prefix** — a system prompt, an instruction template, or a RAG context. Without prefix caching, each request independently recomputes K/V for the shared prefix.

**Prefix caching** exploits the fact that the KV-cache for a prefix is identical across requests. Instead of recomputing, we cache the prefix's KV-cache pages and reuse them.

### How it works

Each KV-cache page is checksummed (e.g., a hash of the token IDs and model weights). When a new request arrives:

1. Hash its prefix tokens → lookup in a prefix cache
2. If found: reuse those pages directly, only compute K/V for the request-specific suffix
3. If not found: compute K/V for the full sequence, populate cache

```
Request A: [system_prompt=100 tokens] [user_query=20 tokens]
Request B: [system_prompt=100 tokens] [different_query=20 tokens]

Without prefix caching: compute 120 tokens twice
With prefix caching:    compute 100 tokens once + 20 tokens twice
```

### Prefix caching in vLLM

vLLM implements this via **automatic prefix caching** (APC). When a new request enters the scheduler:

1. vLLM hashes the entire token sequence
2. Looks up the hash in a RadixTree (prefix tree) stored in GPU memory
3. If a prefix match exists, reuses those cache pages immediately
4. The cache is persistent across requests until evicted by LRU

```python
# Conceptual vLLM prefix cache lookup
def get_or_compute(prefix_tokens, model, cache):
    prefix_hash = hash_tokens(prefix_tokens)
    cached_pages = cache.radix_tree.get(prefix_hash)
    if cached_pages is not None:
        return cached_pages  # Reuse without recompute
    # Compute and cache
    pages = compute_kv_pages(prefix_tokens, model)
    cache.radix_tree.put(prefix_hash, pages)
    return pages
```

### Memory overhead

The prefix cache is stored alongside the request-level cache. The total KV-cache memory is the sum of all unique prefixes plus request-specific suffixes. In workloads with many requests sharing a long system prompt (common in RAG and agentic pipelines), prefix caching can reduce compute by 30-70%.

### Limits of prefix caching

- Cache is tied to model weights. Any weight change (fine-tuning, speculative decoding draft model) invalidates the cache.
- Long-tail queries with unique prefixes see no benefit.
- The radix tree itself consumes memory (~a few MB for millions of entries).

---

## Summary

The simple dict-based cache from the free lesson captures the core idea. Production inference requires much more:

- **PagedAttention** replaces allocation-per-token with fixed-size pages → eliminates fragmentation, enables memory sharing, allows hierarchical storage
- **GQA** reduces KV-cache size 4-8x vs MHA by sharing K/V heads across query groups → essential for long-context models
- **Cache eviction** handles memory pressure via LRU, heavy-hitter scores, or hierarchical GPU/CPU/NVMe storage
- **Prefix caching** exploits shared prefixes across requests → 30-70% compute reduction in common workloads
