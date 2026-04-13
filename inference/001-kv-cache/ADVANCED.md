# KV-Cache: Advanced Topics

Beyond basic caching — PagedAttention, prefix caching, cache quantization, sliding windows, speculative decoding, continuous batching, and multi-GPU KV-cache strategies.

The core lesson covers why we cache keys and values and how it eliminates redundant computation during autoregressive generation. This companion digs into the systems-level innovations that make LLM serving fast and memory-efficient at scale.

---

## 1. PagedAttention: virtual memory for KV-cache

The core problem: KV-cache sizes are dynamic (each request has a different sequence length) and grow during generation. Pre-allocating maximum-length buffers wastes 60-80% of GPU memory. vLLM's PagedAttention (Kwon et al., 2023) solves this with an OS-inspired virtual memory approach.

```
Traditional KV-cache allocation:
  Request 1: [allocated: 2048 tokens] [actually used: 150 tokens] -> 93% waste
  Request 2: [allocated: 2048 tokens] [actually used: 1800 tokens] -> 12% waste
  Request 3: [allocated: 2048 tokens] [actually used: 50 tokens]  -> 98% waste
  
  Average utilization: ~30%

PagedAttention:
  Physical memory split into fixed-size pages (e.g., 16 tokens each)
  Request 1: 150 tokens -> 10 pages allocated (160 token slots, 94% used)
  Request 2: 1800 tokens -> 113 pages allocated (1808 slots, 99.6% used)
  Request 3: 50 tokens -> 4 pages allocated (64 slots, 78% used)
  
  Average utilization: ~95%
```

### Page table structure

```
                    Virtual               Physical GPU Memory
                    Page Table            (non-contiguous pages)
                    ┌─────────┐
  Request 1  ──>   │ VP0 -> P7 │────────> [Page 7:  tokens 0-15 ]
                    │ VP1 -> P2 │────────> [Page 2:  tokens 16-31]
                    │ VP2 -> P15│────────> [Page 15: tokens 32-47]
                    └─────────┘
                    ┌─────────┐
  Request 2  ──>   │ VP0 -> P1 │────────> [Page 1:  tokens 0-15 ]
                    │ VP1 -> P9 │────────> [Page 9:  tokens 16-31]
                    │ VP2 -> P4 │────────> [Page 4:  tokens 32-47]
                    └─────────┘

  Pages are NOT contiguous in physical memory.
  Virtual page table maps logical token positions to physical pages.
```

```python
import torch
from dataclasses import dataclass, field

@dataclass
class KVPage:
    """A fixed-size page holding KV-cache data for PAGE_SIZE tokens."""
    page_id: int
    key_data: torch.Tensor    # (PAGE_SIZE, n_heads, head_dim)
    value_data: torch.Tensor  # (PAGE_SIZE, n_heads, head_dim)
    num_filled: int = 0       # how many token slots are actually used

PAGE_SIZE = 16  # tokens per page

class PagedKVCache:
    def __init__(self, max_pages, n_layers, n_kv_heads, head_dim, device='cuda'):
        self.page_size = PAGE_SIZE
        self.n_layers = n_layers
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim

        # Pre-allocate all physical pages
        self.key_pool = torch.zeros(
            max_pages, n_layers, PAGE_SIZE, n_kv_heads, head_dim,
            device=device, dtype=torch.float16
        )
        self.value_pool = torch.zeros_like(self.key_pool)

        # Free page list
        self.free_pages = list(range(max_pages))

        # Per-request page tables: request_id -> list of physical page indices
        self.page_tables: dict[int, list[int]] = {}

    def allocate_page(self, request_id):
        """Allocate one physical page for a request."""
        if not self.free_pages:
            raise MemoryError("No free pages — need to evict or reject request")
        page_id = self.free_pages.pop()
        if request_id not in self.page_tables:
            self.page_tables[request_id] = []
        self.page_tables[request_id].append(page_id)
        return page_id

    def free_request(self, request_id):
        """Return all pages for a request to the free pool."""
        pages = self.page_tables.pop(request_id, [])
        self.free_pages.extend(pages)

    def append_token(self, request_id, layer_idx, key, value):
        """Append one token's KV to the cache, allocating pages as needed."""
        pages = self.page_tables.get(request_id, [])

        # Check if current last page is full
        if not pages:
            self.allocate_page(request_id)
            pages = self.page_tables[request_id]

        last_page = pages[-1]
        # Count tokens in last page
        token_offset = self._count_tokens(request_id) % self.page_size

        if token_offset == 0 and self._count_tokens(request_id) > 0:
            # Last page is full, allocate new one
            self.allocate_page(request_id)
            pages = self.page_tables[request_id]
            last_page = pages[-1]
            token_offset = 0

        # Write KV data
        self.key_pool[last_page, layer_idx, token_offset] = key
        self.value_pool[last_page, layer_idx, token_offset] = value

    def get_keys_values(self, request_id, layer_idx):
        """Gather all keys/values for a request across its pages."""
        pages = self.page_tables[request_id]
        keys = torch.cat([self.key_pool[p, layer_idx] for p in pages], dim=0)
        values = torch.cat([self.value_pool[p, layer_idx] for p in pages], dim=0)
        n_tokens = self._count_tokens(request_id)
        return keys[:n_tokens], values[:n_tokens]

    def _count_tokens(self, request_id):
        pages = self.page_tables.get(request_id, [])
        if not pages:
            return 0
        # This is simplified — production systems track this explicitly
        return len(pages) * self.page_size  # approximate
```

---

## 2. Prefix caching (RadixAttention)

Many requests share a common prefix (system prompt, few-shot examples, shared document). Prefix caching avoids recomputing KV-cache for shared prefixes.

```
Without prefix caching:
  Request 1: [SYSTEM PROMPT (500 tokens)] + "What is Python?"   -> compute 500+5 KV
  Request 2: [SYSTEM PROMPT (500 tokens)] + "What is Rust?"     -> compute 500+5 KV
  Request 3: [SYSTEM PROMPT (500 tokens)] + "What is Go?"       -> compute 500+5 KV
  Total: 1515 tokens computed

With prefix caching:
  Cache:     [SYSTEM PROMPT (500 tokens)]  -> compute once, cache KV
  Request 1: cache hit + "What is Python?" -> compute 5 KV
  Request 2: cache hit + "What is Rust?"   -> compute 5 KV
  Request 3: cache hit + "What is Go?"     -> compute 5 KV
  Total: 515 tokens computed  (3x reduction)
```

### RadixAttention: tree-structured prefix cache

SGLang's RadixAttention organizes cached prefixes as a radix tree, enabling efficient prefix matching and sharing.

```
Radix tree of cached prefixes:

                     [ROOT]
                       |
              [System prompt tokens]
                    /        \
           [few-shot ex 1]   [few-shot ex 2]
              /      \              |
         [user Q1] [user Q2]  [user Q3]

Each node stores the KV-cache for its token span.
Matching a new request = walking down the tree until mismatch.
```

```python
class RadixNode:
    """Node in a radix tree for prefix caching."""
    def __init__(self):
        self.children: dict[tuple, 'RadixNode'] = {}
        self.kv_cache_ref = None  # reference to cached KV data
        self.last_access_time = 0
        self.ref_count = 0

class PrefixCache:
    def __init__(self, max_cache_tokens):
        self.root = RadixNode()
        self.max_cache_tokens = max_cache_tokens
        self.current_tokens = 0

    def match_prefix(self, token_ids):
        """Find longest cached prefix match. Returns (matched_length, kv_cache)."""
        node = self.root
        matched = 0

        # Walk down tree matching token chunks
        remaining = tuple(token_ids)
        while remaining:
            found = False
            for edge_tokens, child in node.children.items():
                if remaining[:len(edge_tokens)] == edge_tokens:
                    matched += len(edge_tokens)
                    remaining = remaining[len(edge_tokens):]
                    node = child
                    found = True
                    break
            if not found:
                break

        return matched, node.kv_cache_ref

    def insert(self, token_ids, kv_cache):
        """Insert a prefix and its KV-cache into the tree."""
        node = self.root
        remaining = tuple(token_ids)

        while remaining:
            found = False
            for edge_tokens, child in node.children.items():
                # Find common prefix with this edge
                common_len = 0
                for i in range(min(len(edge_tokens), len(remaining))):
                    if edge_tokens[i] == remaining[i]:
                        common_len += 1
                    else:
                        break

                if common_len > 0:
                    if common_len < len(edge_tokens):
                        # Split the edge
                        self._split_edge(node, edge_tokens, common_len)
                    remaining = remaining[common_len:]
                    node = node.children[edge_tokens[:common_len]]
                    found = True
                    break

            if not found:
                # Create new edge
                new_node = RadixNode()
                node.children[remaining] = new_node
                new_node.kv_cache_ref = kv_cache
                return

        node.kv_cache_ref = kv_cache
```

---

## 3. KV-cache quantization

Full-precision KV-cache is a memory hog. Quantizing cached keys and values to int8 or int4 cuts memory 2-4x with minimal quality loss.

```
Memory per token per layer (32 KV heads, head_dim=128):

  FP16:  2 * 32 * 128 * 2 bytes = 16,384 bytes
  INT8:  2 * 32 * 128 * 1 byte  =  8,192 bytes  (2x savings)
  INT4:  2 * 32 * 128 * 0.5     =  4,096 bytes  (4x savings)
```

```python
class QuantizedKVCache:
    """KV-cache with per-channel int8 quantization."""

    def __init__(self, max_seq_len, n_layers, n_kv_heads, head_dim, device='cuda'):
        # Store quantized data as int8
        self.k_cache = torch.zeros(
            n_layers, max_seq_len, n_kv_heads, head_dim,
            dtype=torch.int8, device=device
        )
        self.v_cache = torch.zeros_like(self.k_cache)

        # Per-head, per-token scale factors (needed for dequantization)
        self.k_scales = torch.zeros(
            n_layers, max_seq_len, n_kv_heads, 1,
            dtype=torch.float16, device=device
        )
        self.v_scales = torch.zeros_like(self.k_scales)

    def store(self, layer_idx, pos, key, value):
        """Quantize and store KV pair.
        key, value: (n_kv_heads, head_dim) in float16
        """
        # Per-head absmax quantization
        k_scale = key.abs().amax(dim=-1, keepdim=True) / 127.0
        v_scale = value.abs().amax(dim=-1, keepdim=True) / 127.0

        self.k_cache[layer_idx, pos] = (key / k_scale.clamp(min=1e-10)).round().to(torch.int8)
        self.v_cache[layer_idx, pos] = (value / v_scale.clamp(min=1e-10)).round().to(torch.int8)
        self.k_scales[layer_idx, pos] = k_scale
        self.v_scales[layer_idx, pos] = v_scale

    def get(self, layer_idx, start, end):
        """Dequantize and return KV data for positions [start, end)."""
        k_quant = self.k_cache[layer_idx, start:end].float()
        v_quant = self.v_cache[layer_idx, start:end].float()

        k = k_quant * self.k_scales[layer_idx, start:end].float()
        v = v_quant * self.v_scales[layer_idx, start:end].float()

        return k.half(), v.half()
```

### Key insight: keys need more precision than values

Research (Hooper et al., 2024, KIVI) shows that keys are more sensitive to quantization than values because attention scores are computed as Q @ K^T — quantization error in K directly distorts attention weights.

```
Perplexity impact of KV quantization (LLaMA-2-7B):
  FP16 K, FP16 V:  5.47 (baseline)
  INT8 K, INT8 V:   5.49 (+0.02, negligible)
  INT4 K, INT8 V:   5.55 (+0.08, acceptable)
  INT8 K, INT4 V:   5.51 (+0.04, fine)
  INT4 K, INT4 V:   5.68 (+0.21, some degradation)
  INT2 K, INT4 V:   6.31 (+0.84, significant)

Strategy: use higher precision for keys than values.
KIVI uses INT2 K (per-channel) + INT2 V (per-token) with residual quantization.
```

---

## 4. Sliding window cache (Mistral)

Mistral uses a sliding window attention pattern: each token only attends to the last W tokens (typically W=4096). This means the KV-cache never grows beyond W entries.

```
Full attention (seq_len grows indefinitely):
  Step 1000: cache has 1000 KV pairs
  Step 5000: cache has 5000 KV pairs
  Step 20000: cache has 20000 KV pairs -> OOM

Sliding window (W=4096):
  Step 1000:  cache has 1000 KV pairs
  Step 5000:  cache has 4096 KV pairs (old ones evicted)
  Step 20000: cache has 4096 KV pairs (constant memory!)
```

```python
class SlidingWindowKVCache:
    """Fixed-size circular buffer KV-cache for sliding window attention."""
    def __init__(self, window_size, n_layers, n_kv_heads, head_dim, device='cuda'):
        self.window_size = window_size
        self.k_cache = torch.zeros(
            n_layers, window_size, n_kv_heads, head_dim,
            dtype=torch.float16, device=device
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.pos = 0  # current write position (circular)

    def append(self, layer_idx, key, value):
        """Append new KV pair, overwriting oldest if full."""
        slot = self.pos % self.window_size
        self.k_cache[layer_idx, slot] = key
        self.v_cache[layer_idx, slot] = value
        # Only increment pos in layer 0 (once per token, not per layer)
        if layer_idx == 0:
            self.pos += 1

    def get_all(self, layer_idx):
        """Get all cached KV pairs in chronological order."""
        if self.pos <= self.window_size:
            return self.k_cache[layer_idx, :self.pos], \
                   self.v_cache[layer_idx, :self.pos]
        else:
            # Circular buffer: reorder so oldest is first
            start = self.pos % self.window_size
            k = torch.cat([
                self.k_cache[layer_idx, start:],
                self.k_cache[layer_idx, :start]
            ], dim=0)
            v = torch.cat([
                self.v_cache[layer_idx, start:],
                self.v_cache[layer_idx, :start]
            ], dim=0)
            return k, v
```

### How sliding window still captures long-range dependencies

Even with a window of 4096, information propagates across layers:

```
Layer 0: token at position 8000 sees tokens 3905-8000 (window of 4096)
Layer 1: token at position 8000 sees tokens 3905-8000,
         BUT those tokens' representations already saw tokens 0-3905 in layer 0

Effective receptive field after L layers with window W:
  L * W tokens

Mistral 7B: 32 layers * 4096 window = 131,072 effective context
(in practice, information degrades with distance through layers)
```

---

## 5. Speculative decoding

Speculative decoding uses a small, fast "draft" model to propose multiple tokens, then the large model verifies them in parallel. This can provide 2-3x speedup for autoregressive generation.

```
Standard autoregressive (1 token per forward pass):
  Step 1: Large model -> token A  (100ms)
  Step 2: Large model -> token B  (100ms)
  Step 3: Large model -> token C  (100ms)
  Step 4: Large model -> token D  (100ms)
  Total: 400ms for 4 tokens

Speculative decoding:
  Step 1: Draft model proposes [A, B, C, D]  (10ms, small model is fast)
  Step 2: Large model verifies all 4 in parallel (100ms, single forward pass)
  Step 3: Large model accepts A, B, C but rejects D, samples D' instead
  Total: 110ms for 4 tokens (3.6x speedup!)
```

```python
import torch
import torch.nn.functional as F

def speculative_decode(
    target_model,    # large model (e.g., 70B)
    draft_model,     # small model (e.g., 7B, same tokenizer)
    input_ids,       # current context
    n_speculative=5, # number of tokens to draft
    temperature=1.0,
):
    """Speculative decoding with rejection sampling."""

    # Step 1: Draft model generates n tokens autoregressively (fast)
    draft_tokens = []
    draft_probs = []
    draft_input = input_ids.clone()

    for _ in range(n_speculative):
        with torch.no_grad():
            logits = draft_model(draft_input).logits[:, -1, :]
        probs = F.softmax(logits / temperature, dim=-1)
        token = torch.multinomial(probs, num_samples=1)
        draft_tokens.append(token.item())
        draft_probs.append(probs[0, token.item()].item())
        draft_input = torch.cat([draft_input, token.unsqueeze(0)], dim=-1)

    # Step 2: Target model scores ALL draft tokens in one forward pass
    full_input = torch.cat([
        input_ids,
        torch.tensor([draft_tokens], device=input_ids.device)
    ], dim=-1)

    with torch.no_grad():
        target_logits = target_model(full_input).logits

    # Step 3: Rejection sampling — accept/reject each draft token
    accepted_tokens = []
    n_input = input_ids.shape[-1]

    for i, (draft_token, draft_prob) in enumerate(zip(draft_tokens, draft_probs)):
        target_probs = F.softmax(
            target_logits[:, n_input + i - 1, :] / temperature, dim=-1
        )
        target_prob = target_probs[0, draft_token].item()

        # Accept with probability min(1, target_prob / draft_prob)
        acceptance_ratio = min(1.0, target_prob / draft_prob)

        if torch.rand(1).item() < acceptance_ratio:
            accepted_tokens.append(draft_token)
        else:
            # Reject: sample from adjusted distribution
            adjusted = torch.clamp(target_probs - draft_probs_tensor, min=0)
            adjusted = adjusted / adjusted.sum()
            new_token = torch.multinomial(adjusted, num_samples=1)
            accepted_tokens.append(new_token.item())
            break  # stop accepting after first rejection

    # If all draft tokens accepted, sample one more from target
    if len(accepted_tokens) == n_speculative:
        bonus_probs = F.softmax(
            target_logits[:, -1, :] / temperature, dim=-1
        )
        bonus = torch.multinomial(bonus_probs, num_samples=1)
        accepted_tokens.append(bonus.item())

    return accepted_tokens
```

### Acceptance rate depends on draft-target alignment

```
Draft model quality vs acceptance rate:

  Draft = same architecture, 10x fewer params:
    Acceptance rate: ~70-80%
    Effective speedup: 2-3x

  Draft = same architecture, 100x fewer params:
    Acceptance rate: ~40-50%
    Effective speedup: 1.3-1.5x

  Draft = n-gram model (trivial):
    Acceptance rate: ~20-30% (on repetitive text)
    Effective speedup: 1.1-1.2x (barely worth it)

Sweet spot: draft model is 5-15x smaller than target.
  LLaMA-70B + LLaMA-7B draft -> ~2.5x speedup
  Gemma-27B + Gemma-2B draft  -> ~2.8x speedup
```

---

## 6. Continuous batching

Static batching wastes GPU cycles because shorter sequences finish early and sit idle. Continuous batching (also called "iteration-level batching") immediately fills empty slots with new requests.

```
Static batching:
  Time ->  1  2  3  4  5  6  7  8  9  10
  Req A:  [=  =  =  =  =  DONE  .  .  .  .]   <- idle for 4 steps
  Req B:  [=  =  =  =  =  =  =  =  =  DONE]
  Req C:  [=  =  =  DONE  .  .  .  .  .  .]   <- idle for 6 steps
  GPU utilization: ~60%

Continuous batching:
  Time ->  1  2  3  4  5  6  7  8  9  10
  Req A:  [=  =  =  =  =  DONE]
  Req B:  [=  =  =  =  =  =  =  =  =  DONE]
  Req C:  [=  =  =  DONE]
  Req D:           [=  =  =  =  =  DONE]       <- fills C's slot at step 4
  Req E:                    [=  =  =  =  DONE]  <- fills A's slot at step 6
  GPU utilization: ~95%
```

```python
class ContinuousBatcher:
    """Simple continuous batching scheduler."""
    def __init__(self, model, max_batch_size, max_seq_len):
        self.model = model
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.active_requests = {}  # request_id -> RequestState
        self.waiting_queue = []

    def add_request(self, request_id, input_ids):
        self.waiting_queue.append((request_id, input_ids))

    def step(self):
        """Run one decoding step for all active requests."""
        # Fill empty slots with waiting requests
        while (len(self.active_requests) < self.max_batch_size
               and self.waiting_queue):
            req_id, input_ids = self.waiting_queue.pop(0)
            self.active_requests[req_id] = {
                'input_ids': input_ids,
                'generated': [],
                'kv_cache': None,
            }

        if not self.active_requests:
            return {}

        # Batch forward pass for all active requests
        # (In practice, this requires careful padding/packing)
        results = {}
        finished = []

        for req_id, state in self.active_requests.items():
            # Single-token forward pass with KV-cache
            next_token = self._decode_one_token(state)
            state['generated'].append(next_token)

            if next_token == self.model.eos_token_id or \
               len(state['generated']) >= self.max_seq_len:
                results[req_id] = state['generated']
                finished.append(req_id)

        # Remove finished requests (frees their KV-cache slots)
        for req_id in finished:
            del self.active_requests[req_id]

        return results
```

---

## 7. Multi-GPU tensor parallel KV-cache

When a model is split across GPUs with tensor parallelism, the KV-cache is also split. Each GPU stores KV data for its subset of attention heads.

```
8-GPU tensor parallelism for 64-head model:
  GPU 0: heads  0- 7 -> KV-cache for 8 heads
  GPU 1: heads  8-15 -> KV-cache for 8 heads
  GPU 2: heads 16-23 -> KV-cache for 8 heads
  ...
  GPU 7: heads 56-63 -> KV-cache for 8 heads

Each GPU stores 1/8 of the KV-cache.
All-reduce needed after attention output projection.
```

```python
class TensorParallelKVCache:
    """KV-cache for tensor-parallel attention."""
    def __init__(self, n_layers, n_total_heads, head_dim, tp_rank, tp_size,
                 max_seq_len, device):
        # Each GPU handles a subset of heads
        assert n_total_heads % tp_size == 0
        self.n_local_heads = n_total_heads // tp_size
        self.tp_rank = tp_rank

        self.k_cache = torch.zeros(
            n_layers, max_seq_len, self.n_local_heads, head_dim,
            dtype=torch.float16, device=device
        )
        self.v_cache = torch.zeros_like(self.k_cache)
        self.seq_len = 0

    def append(self, layer_idx, key, value):
        """key, value: (n_local_heads, head_dim) — already sharded."""
        self.k_cache[layer_idx, self.seq_len] = key
        self.v_cache[layer_idx, self.seq_len] = value
        if layer_idx == 0:
            self.seq_len += 1

    def memory_per_gpu(self, n_layers, seq_len):
        """Memory footprint per GPU."""
        per_token = 2 * self.n_local_heads * self.k_cache.shape[-1] * 2  # fp16
        return per_token * n_layers * seq_len
```

### GQA + tensor parallelism interaction

With GQA, the number of KV heads may be less than the number of GPUs. In this case, KV heads are replicated rather than sharded:

```
Model: 64 Q heads, 8 KV heads, 8 GPUs

Option 1: Each GPU gets 1 KV head (no replication needed)
  GPU 0: 8 Q heads, 1 KV head
  GPU 1: 8 Q heads, 1 KV head
  ...

Option 2: 4 GPUs, each gets 2 KV heads
  GPU 0: 16 Q heads, 2 KV heads
  GPU 1: 16 Q heads, 2 KV heads
  ...

If n_kv_heads < n_gpus (e.g., 4 KV heads, 8 GPUs):
  Each KV head is replicated across 2 GPUs
  GPU 0: 8 Q heads, KV head 0 (replicated)
  GPU 1: 8 Q heads, KV head 0 (replicated)
  GPU 2: 8 Q heads, KV head 1 (replicated)
  ...
```

---

## 8. KV-cache sizing cheat sheet

```
Model          Params  n_layers  n_kv_heads  head_dim  KV per token  4K context
─────────────────────────────────────────────────────────────────────────────────
Mistral 7B     7.2B      32         8          128      64 KB        256 MB
LLaMA-2 7B     6.7B      32        32          128     256 KB        1.0 GB
LLaMA-2 13B   13.0B      40        40          128     400 KB        1.6 GB
LLaMA-2 70B   65.2B      80         8          128     160 KB        640 MB*
LLaMA-3 8B     8.0B      32         8          128      64 KB        256 MB
LLaMA-3 70B   70.6B      80         8          128     160 KB        640 MB
DeepSeek 67B   67.0B      95         1          128      24 KB         96 MB**

* GQA makes 70B cache smaller than 13B cache!
** MQA makes 67B cache tiny.

At 128K context:
  LLaMA-2 7B (MHA):    32 GB  (won't fit on single GPU)
  LLaMA-3 8B (GQA):     8 GB  (fits on 24GB GPU)
  DeepSeek 67B (MQA):   3 GB  (negligible)
```

---

Get the video walkthrough: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
