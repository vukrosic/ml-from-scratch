# Attention From Scratch

Attention is the mechanism that lets transformer models weigh the importance of different positions when processing a sequence. It is the core operation that makes transformers powerful — and it is expensive. Understanding attention from the ground up reveals why memory efficiency matters at long sequences and how KV caching cuts that cost dramatically.

---

## What Problem Attention Solves

### The Problem: O(n²) Memory

Every position in a sequence needs to attend to every other position. Computing this requires storing an n×n matrix, where n is the sequence length. For a sequence of 4096 tokens, that is 16 million values — and that is before batching.

```
Attention matrix size:  n²
Sequence length 512:    262K elements  (~1 MB)
Sequence length 2048:   4.2M elements  (~16 MB)
Sequence length 4096:   16.8M elements (~64 MB)
```

The compute scales the same way: a 4096×4096 matmul is ~16× more expensive than 1024×1024. The O(n²) cost is the central challenge of attention.

### The Fix: Scaled Dot-Product Attention

Scaled dot-product attention computes three things from the input: queries Q, keys K, and values V. The formula is:

```
attention(Q, K, V) = softmax(Q @ K.T / sqrt(d)) @ V
```

Where d is the dimension of each head. Q and K are both (batch, seq, d), so Q @ K.T is (batch, seq, seq) — the attention matrix. The division by sqrt(d) prevents the softmax from saturating when d is large.

### Why Divide by sqrt(d)?

When d is large, the dot products Q @ K.T grow large in magnitude. The softmax exponential saturates at 1.0, making all attention weights nearly uniform — the model cannot distinguish positions. Dividing by sqrt(d) keeps the dot products in a reasonable range.

---

## The Attention Operation

### Step 1: Compute Q, K, V from the input

From an input tensor of shape (batch, seq, d_model), three linear projections produce Q, K, and V:

```
Q = x @ W_q   # (batch, seq, d_model) -> (batch, seq, d_head)
K = x @ W_k
V = x @ W_v
```

### Step 2: Compute the attention matrix

```
scores = Q @ K.T                     # (batch, seq, seq)
scores = scores / sqrt(d_head)       # scale
weights = softmax(scores, dim=-1)    # normalize along key dimension
```

### Step 3: Apply attention to values

```
output = weights @ V                 # (batch, seq, seq) @ (batch, seq, d_head)
```

The attention matrix weights each position's value by how relevant it is to the query position.

---

## Naive Attention vs. PyTorch Built-In

PyTorch provides `torch.nn.functional.scaled_dot_product_attention` (SDPA), which fuses the entire operation and handles masking efficiently. Here is a direct comparison.

### The naive implementation

```python
import torch
import torch.nn.functional as F

def naive_attention(Q, K, V, scale=None):
    """Scaled dot-product attention, computed naively.

    Q, K, V: (batch, seq, d_head)
    Returns: (batch, seq, d_head)
    """
    d = Q.shape[-1]
    if scale is None:
        scale = d ** -0.5

    # Compute attention scores
    scores = Q @ K.transpose(-2, -1)   # (batch, seq, seq)
    scores = scores * scale            # scale

    # Softmax over key dimension
    weights = F.softmax(scores, dim=-1)

    # Apply to values
    return weights @ V
```

### The built-in version

```python
def fast_attention(Q, K, V):
    """Uses PyTorch's fused SDPA kernel.

    Q, K, V: (batch, seq, d_head)
    Returns: (batch, seq, d_head)
    """
    return F.scaled_dot_product_attention(Q, K, V)
```

### Benchmark: naive vs. SDPA

```python
import time

def benchmark(fn, *args, steps=200, warmup=50):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / steps

batch, seq, d_head = 4, 1024, 64
Q = torch.randn(batch, seq, d_head, device='cuda')
K = torch.randn(batch, seq, d_head, device='cuda')
V = torch.randn(batch, seq, d_head, device='cuda')

naive_time = benchmark(naive_attention, Q, K, V)
fast_time = benchmark(fast_attention, Q, K, V)

print(f"Naive:   {naive_time*1000:.2f} ms")
print(f"SDPA:    {fast_time*1000:.2f} ms")
print(f"Speedup: {naive_time/fast_time:.2f}x")
```

Run `attention.py` on your GPU to see the numbers. SDPA is typically ~2-4× faster and uses less memory because it avoids materializing the full (batch, seq, seq) attention matrix explicitly.

---

## Masking: Causal and Arbitrary

Autoregressive models must not attend to future positions. A causal mask ensures each position only attends to itself and earlier positions.

### Causal mask

A causal mask zeros out the upper triangle (above the diagonal) of the attention matrix:

```
Causal mask for seq=4:
[[1, 0, 0, 0],
 [1, 1, 0, 0],
 [1, 1, 1, 0],
 [1, 1, 1, 1]]
```

### Naive masking

```python
def causal_mask(scores, seq_len):
    """Create a causal mask and apply it to attention scores.

    scores: (batch, seq, seq)
    Returns: (batch, seq, seq) with causal masking applied
    """
    mask = torch.triu(torch.ones(seq_len, seq_len, device=scores.device), diagonal=1).bool()
    scores = scores.masked_fill(mask, float('-inf'))
    return scores
```

### Arbitrary mask

You can apply any mask by adding it to the scores before softmax:

```python
def apply_mask(scores, mask):
    """Apply an arbitrary mask to attention scores.

    scores: (batch, seq, seq)
    mask: (batch, seq, seq) of booleans — True means mask out (set to -inf)
    """
    return scores.masked_fill(mask, float('-inf'))
```

Common masks include padding masks (to ignore padding tokens) and document masks (to prevent cross-document attention in long-context models).

### Masking with SDPA

PyTorch's SDPA accepts a mask directly:

```python
def masked_attention(Q, K, V, mask=None):
    """SDPA with optional masking.

    Q, K, V: (batch, seq, d_head)
    mask: (batch, seq, seq) or (seq, seq), True = mask out
    """
    return F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)
```

Passing `attn_mask=False` or `None` uses no mask. Pass a boolean tensor to apply arbitrary masking.

---

## Flash Attention: Tiling to Avoid O(n²) Memory

The naive attention implementation materializes the full (batch, seq, seq) attention matrix. At sequence length 4096 with batch 1 and d_head 64, this is 16.8M float32 elements — ~64 MB. For batch 8, it is ~512 MB just for the attention matrix.

Flash Attention reformulates the computation to avoid materializing this matrix entirely. It works by **tiling**: instead of computing all n×n attention at once, it processes the sequence in blocks.

### How tiling works

Instead of:
```
S = Q @ K.T          # full (seq, seq) matrix
P = softmax(S)       # full (seq, seq) matrix
O = P @ V            # (seq, seq) @ (seq, d)
```

Flash Attention processes in blocks:
```
# Partition Q, K, V into row blocks of size Br and column blocks of size Bc
# For each row block i:
#   Load Q_i, K, V
#   For each column block j:
#       Load K_j, V_j
#       Compute S_ij = Q_i @ K_j.T
#       Apply mask to S_ij
#       P_ij = softmax(S_ij)
#       O_i += P_ij @ V_j
```

The key insight: O_i can be updated incrementally block by block. The softmax is computed correctly because Flash Attention tracks the row-wise max and normalization constant as it processes each block.

### Simulating Flash Attention tiling

Here is a simplified version that shows the block-by-block computation without the full softmax stabilization math:

```python
def flash_attention_simulated(Q, K, V, block_size=128):
    """Simulate Flash Attention's tiling approach.

    Does NOT include the online softmax stabilization — for illustration only.
    Shows how attention can be computed block-by-block to avoid O(n²) memory.

    Q, K, V: (batch, seq, d_head)
    Returns: (batch, seq, d_head)
    """
    batch, seq, d = Q.shape
    O = torch.zeros_like(Q)

    # Process each row block
    for i in range(0, seq, block_size):
        # Get the block of queries
        Q_block = Q[:, i:i+block_size, :]  # (batch, Br, d)

        # Process each column block
        for j in range(0, seq, block_size):
            # Get the block of keys and values
            K_block = K[:, j:j+block_size, :]  # (batch, Bc, d)
            V_block = V[:, j:j+block_size, :]  # (batch, Bc, d)

            # Compute attention for this block
            # S_block: (batch, Br, Bc)
            S_block = Q_block @ K_block.transpose(-2, -1)
            S_block = S_block / (d ** 0.5)

            # Compute softmax for this block
            P_block = torch.softmax(S_block, dim=-1)

            # Update output: add contribution from this block
            # O_block: (batch, Br, d)
            O[:, i:i+block_size, :] += P_block @ V_block

    return O
```

The real Flash Attention does three things this simulation skips:
1. **Online softmax stabilization:** Tracks row-wise max and normalization incrementally.
2. **No HBM access for the full matrix:** Loads K and V blocks from HBM for each block computation.
3. **Fused kernel:** Everything runs on GPU without intermediate materialization.

### Memory comparison

| Implementation | Memory for seq=4096 |
|---------------|---------------------|
| Naive attention | ~64 MB (attention matrix) |
| Flash Attention (tiling) | ~Br×d + Bc×d + O(d²) per block |

Flash Attention reduces memory from O(n²) to O(n) for the attention matrix itself. The actual speedup comes from keeping the working set in fast on-chip SRAM rather than fetching from HBM.

Run `flash_attention.py` to see the memory difference between naive and tiled computation.

---

## Multi-Head Attention

Instead of one attention head, multi-head attention runs h attention heads in parallel, each with its own Q, K, V projections. The outputs are concatenated and projected back to d_model.

```
x -> Q, K, V (each split into h heads)
Each head: softmax(Q_h @ K_h.T / sqrt(d_h)) @ V_h
Concat all heads: [head_1, head_2, ..., head_h]  (seq, h*d_h)
Final projection: concat @ W_o  (seq, d_model)
```

### Multi-head attention module

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """Multi-head attention from scratch.

    Args:
        d_model: model dimension
        num_heads: number of attention heads (must divide d_model evenly)
    """
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads

        # Q, K, V projections — one per head
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        """Forward pass.

        x: (batch, seq, d_model)
        mask: optional attention mask
        Returns: (batch, seq, d_model)
        """
        batch, seq, _ = x.shape

        # Project to Q, K, V
        Q = self.W_q(x)  # (batch, seq, d_model)
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head: (batch, seq, num_heads, d_head) -> (batch, num_heads, seq, d_head)
        Q = Q.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        K = K.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)
        V = V.view(batch, seq, self.num_heads, self.d_head).transpose(1, 2)

        # Compute attention with PyTorch's fused SDPA
        # attn: (batch, num_heads, seq, d_head)
        attn = F.scaled_dot_product_attention(Q, K, V, attn_mask=mask)

        # Reshape back: (batch, num_heads, seq, d_head) -> (batch, seq, d_model)
        attn = attn.transpose(1, 2).contiguous().view(batch, seq, self.d_model)

        # Final projection
        return self.W_o(attn)
```

### Why split into multiple heads?

A single attention head can only capture one type of relationship at a time. With multiple heads, the model can attend to different aspects of the sequence in parallel — one head might focus on syntactic relationships, another on semantic proximity, another on positional patterns. The concatenation and final projection let the model mix information across heads.

Run `multihead.py` to test the module.

---

## KV Caching: Avoiding Recomputation in Autoregressive Decoding

During autoregressive decoding, each new token depends on all previous tokens. Without caching, recomputing the full Q, K, V for every step is O(n²) in total — generating n tokens costs O(n²) instead of O(n).

### The caching insight

K and V for previous tokens never change. Once computed, they can be stored and reused. Only the new Q for the current token needs to be computed fresh.

```
Without cache at step t:
  Need: Q_t, K_1:t, V_1:t  (recompute all K/V)

With cache at step t:
  Stored: K_1:(t-1), V_1:(t-1)
  Fresh:  Q_t, K_t, V_t
  Combine: cached + fresh -> full K/V
```

### KV cache implementation

```python
class KVCache:
    """KV cache for autoregressive decoding.

    Stores key and value tensors for all computed positions.
    On each step, appends new K and V, and optionally trims to max_len.
    """
    def __init__(self, max_len=1024):
        self.max_len = max_len
        self.keys = []    # list of (batch, 1, d_head) tensors
        self.values = []

    def update(self, k, v):
        """Append new key/value tensors.

        k, v: (batch, 1, d_head) — single position
        Returns: (batch, seq, d_head) for both keys and values
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
```

### Benchmark: with and without KV cache

```python
def benchmark_autoregressive(model, prompt_len, gen_len, use_cache=True):
    """Time autoregressive generation with and without KV cache.

    model: the attention module
    prompt_len: length of the prompt
    gen_len: number of tokens to generate
    use_cache: whether to use KV caching
    """
    cache = KVCache() if use_cache else None

    # Setup: run prompt through once
    x = torch.randn(1, prompt_len, d_model, device='cuda')
    mask = None  # or causal mask

    if use_cache:
        # Encode the prompt and populate the cache
        # (In practice, this encodes all prompt tokens at once)
        pass
    else:
        # Without cache: full recomputation each step
        pass

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Generate gen_len tokens
    for _ in range(gen_len):
        # Get current token representation (last position)
        x_step = x[:, -1:, :]  # (batch, 1, d_model)

        # Compute Q, K, V
        Q = model.W_q(x_step)
        K = model.W_k(x_step)
        V = model.W_v(x_step)

        if use_cache:
            # Update cache and get full K, V
            K_full, V_full = cache.update(K, V)
        else:
            # Without cache: would need full recomputation
            K_full = K  # placeholder — in reality would recompute all
            V_full = V

        # SDPA with the full K/V
        Q = Q.view(1, 1, model.num_heads, model.d_head).transpose(1, 2)
        K_full = K_full.view(1, -1, model.num_heads, model.d_head).transpose(1, 2)
        V_full = V_full.view(1, -1, model.num_heads, model.d_head).transpose(1, 2)

        attn = F.scaled_dot_product_attention(Q, K_full, V_full, attn_mask=mask)
        # ... continue with rest of generation

    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / gen_len
```

Without KV caching, each generation step recomputes K and V for all previous tokens. With a cache, each step is O(seq) in the cache size — you only compute Q for the new token and fetch the cached K and V.

Run `kv_cache.py` to benchmark.

---

## Recap

- **Scaled dot-product attention:** `softmax(Q @ K.T / sqrt(d)) @ V`
- **The O(n²) problem:** Both the attention matrix (n×n) and the values matrix (n×d) scale quadratically with sequence length.
- **Masking:** Apply before softmax with `masked_fill(mask, -inf)`. Causal mask uses `torch.triu`.
- **SDPA:** `F.scaled_dot_product_attention` is fused and handles masking efficiently.
- **Flash Attention:** Tiles the computation to avoid materializing the full attention matrix — O(n) memory instead of O(n²).
- **Multi-head attention:** Split into h heads, attend independently, concatenate, project.
- **KV caching:** Store K and V for previous tokens; only compute fresh Q each step.

---

## Going Further

For real benchmark numbers across sequence lengths, memory profiling for the attention matrix, KV cache speedup measurements, and profiling tools for attention — see [ADVANCED.md](./ADVANCED.md).

Sources:
- https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
- https://arxiv.org/abs/2205.14135 — Flash Attention paper
- https://pytorch.org/blog/optimizing-cuda-algorithms-with-flash-attention-2

---

Get the video walkthrough of flash attention internals, KV-cache memory profiling, and attention pattern analysis: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
