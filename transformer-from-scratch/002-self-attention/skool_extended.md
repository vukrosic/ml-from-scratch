# Self-Attention — Extended Notebook

*Extended content for the $49 tier. Not available on YouTube or GitHub.*

This notebook walks through everything in the free lesson, then goes deeper:
multi-head attention internals, grouped-query attention (GQA), flash attention intuition,
and attention pattern analysis across real transformer layers.

---

## Multi-Head Attention: Why Multiple Heads?

The single-head attention we built in the lesson computes one set of Q/K/V projections
and produces one weighted sum. Multi-head attention runs `n_heads` attention operations
in parallel, then concatenates and projects the results.

### What each head learns

In practice, different heads specialize. A well-known finding (Vaswani et al., 2017;
Clark et al., 2019) shows that some heads learn:

- **Syntactic relationships**: subject-verb agreement, coreference
- **Positional patterns**: nearby tokens, sequential structure
- **Semantic clusters**: topic coherence across long ranges

No head is truly "doing the same thing" — the diversity is the point.

### Multi-head formula

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) @ W_O

where head_i = Attention(Q @ W_Q_i, K @ W_K_i, V @ W_V_i)
```

In code, this is exactly what our `SelfAttention` class does — we reshape Q/K/V into
`n_heads` slices, run attention on each slice independently, then reshape and project.

### Per-head attention visualization

In the main notebook we show all heads separately for the same input.
You can clearly see different attention patterns: some heads are "broader" (attending
across the whole sequence), others are "narrower" (focused on local patterns).

---

## Grouped-Query Attention (GQA)

Standard multi-head attention uses the same number of K and Q heads.
GQA (Ainslie et al., 2023, used in Llama 2) reduces the number of K heads to fewer
than Q heads. For example, in a model with 8 Q heads and 2 K heads (groups of 4 Q heads
per K head), each K head is shared across 4 Q heads.

### Why this works

- **Memory savings**: Keys and values are stored in the KV cache during autoregressive
  generation. Fewer K heads = smaller KV cache = less memory.
- **Quality tradeoff is small**: Different Q heads often attend to similar things anyway,
  so sharing K across groups doesn't hurt much.

### GQA vs MHA vs MQA

| Variant       | Q heads | K heads | V heads |
|---------------|---------|---------|---------|
| MHA (standard)| H       | H       | H       |
| MQA           | H       | 1       | 1       |
| GQA           | H       | G < H   | G       |

MQA (Multi-Query Attention, Shazeer 2019) shares a single K and V head across all Q heads.
GQA generalizes this by allowing G K/V heads where 1 < G < H.

### GQA implementation sketch

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_q_heads, n_kv_heads):
        super().__init__()
        assert n_q_heads % n_kv_heads == 0
        self.n_q_heads = n_q_heads
        self.n_kv_heads = n_kv_heads
        self.n_kv_groups = n_q_heads // n_kv_heads  # Q heads per KV head
        self.d_head = d_model // n_q_heads

        self.q_proj = nn.Linear(d_model, n_q_heads * self.d_head)
        self.kv_proj = nn.Linear(d_model, 2 * n_kv_heads * self.d_head)
        self.out_proj = nn.Linear(n_q_heads * self.d_head, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape

        q = self.q_proj(x)
        q = q.view(batch, seq_len, self.n_q_heads, self.d_head).transpose(1, 2)

        kv = self.kv_proj(x)
        kv = kv.view(batch, seq_len, self.n_kv_heads, 2 * self.d_head).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)

        # Broadcast K and V across Q groups: (batch, n_kv_heads, seq, d_head)
        # -> (batch, n_q_heads, seq, d_head)
        k = k.repeat_interleave(self.n_kv_groups, dim=1)
        v = v.repeat_interleave(self.n_kv_groups, dim=1)

        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, self.n_q_heads * self.d_head)
        return self.out_proj(attn)
```

---

## Flash Attention Intuition

Standard attention computes the full `seq_len × seq_len` attention matrix and stores it.
For a sequence of length 4096, that's 16 million entries — ~64 MB per head in float32.
For 32 heads, that's 2 GB just for the attention matrix, and it grows quadratically.

Flash Attention (Dao et al., 2022) computes attention **without materializing the full N×N matrix**.
It does this in tiles — small blocks of the attention computation that fit in SRAM —
reading and writing to HBM (GPU memory) only once per tile.

### The algorithm

Flash Attention processes the attention computation in two nested loops:

1. **Outer loop**: iterates over blocks of K and V (loaded into SRAM).
2. **Inner loop**: iterates over blocks of Q, computing partial attention scores.

It maintains running row-wise maxima and normalisation terms (the "online softmax" trick)
so that the final result is exactly the same as the standard algorithm, but with
- **O(N) memory instead of O(N²)**
- **Faster execution** due to better memory access patterns

### Why this matters

Flash Attention made it possible to train transformers with context windows of 32k+
tokens. Without it, the O(N²) memory of standard attention would be prohibitive.

The actual implementation uses CUDA kernels written in Triton or C++ and is what
`torch.nn.functional.scaled_dot_product_attention` calls when available.

---

## Attention Patterns Across Real Transformer Layers

In real models (GPT, BERT, T5), attention patterns change systematically across layers.
Researchers have characterized several recurring patterns:

### Layer 0 (Embedding layer): diverse, local

The first attention layer often shows high attention to adjacent tokens — the model
is learning low-level patterns: punctuation, word boundaries, simple co-occurrences.

### Middle layers: syntactic heads

Layers in the middle of the network often show strong attention between syntactic
pairs: subject-verb, modifier-modified, pronoun-noun. These are the "syntax" heads.

### Late layers: semantic, long-range

By the last few layers, attention often becomes more diffuse — spreading across the
full sequence — as the model assembles task-specific representations.

### Head specialization is not fixed

A striking finding (MERIT, HETH et al., 2022): the same head can show different
patterns depending on the input. Some heads that appear "uniform" on average
actually specialize conditionally on the content.

### Visualization with real embeddings

The notebook demonstrates attention pattern visualization using real token embeddings
(from a pretrained model) rather than random vectors. This shows how attention
actually groups semantically related tokens together — something you don't see
with random inputs.

---

## Gradient Analysis

Let's verify the attention gradient flow.

### Jacobian of the attention operation

For a single head with scalar output (for clarity):

```
dL/dQ = dL/dA @ dA/dS @ dS/dQ
```

where A = softmax(S) and S = Q @ K^T / √d.

The Jacobian `dA/dS` is the Jacobian of the softmax function:
`dA_i/dS_j = A_i * (δ_ij - A_j)`.

This has an elegant form: the Jacobian is a diagonal matrix scaled by the
attention weights, minus an outer product of attention weights.

```python
import torch

def softmax_jacobian(weights):
    """Compute Jacobian of softmax: d(softmax)/d(scores)."""
    # weights shape: (seq, seq) — already softmaxed over rows.
    # Jacobian[i, j] = weights[i] * (delta[i,j] - weights[j])
    diag = torch.diag(weights)
    outer = weights.unsqueeze(-1) * weights.unsqueeze(-2)
    return diag - outer

seq = 4
weights = torch.rand(seq, seq).softmax(dim=-1)
J = softmax_jacobian(weights)
print(f"Jacobian shape: {J.shape}")  # (seq, seq, seq, seq) — 4D!
print(f"Sum of Jacobian rows (should = 0, softmax sum-to-1 property): {J.sum(dim=-1)}")
```

The Jacobian confirms that softmax is a row-stochastic operation:
`dA_i/dS_j` summed over j gives zero for each i — which makes sense because
increasing one score decreases the probability mass on others.

---

## Numerical Stability Across dtypes

Softmax is notoriously sensitive to overflow when scores are large.
We use `scores / sqrt(d_head)` specifically to keep values bounded.

| dtype    | max exp value | notes |
|----------|--------------|-------|
| float32  | ~89          | safe for most d_head |
| float16  | ~11          | d_head > 128 risks overflow |
| bfloat16 | ~3e38        | same exponent range as float32 |

When implementing attention in float16 or bfloat16, the `sqrt(d_head)` scaling
is even more critical. Modern training (A100, H100) uses bfloat16 for this reason.

---

## Profiling: Memory + Speed Across Batch Sizes

The benchmark notebook includes a full profiling sweep showing:

- **Forward pass latency** as a function of batch size and sequence length.
- **Memory usage** of the attention weights matrix: `batch × n_heads × seq² × 4 bytes` (float32).
- **KV cache size** for autoregressive generation: `2 × batch × n_kv_heads × seq_cache × d_head × 4 bytes`.

At seq_len=4096, n_heads=32, float32: the attention weights alone consume
`32 × 4096² × 4 ≈ 2 GB` — this is why Flash Attention matters in practice.

---

*End of extended content. Get the full runnable notebook at:*
**https://www.skool.com/opensuperintelligencelab**
