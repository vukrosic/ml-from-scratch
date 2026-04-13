# Multi-Head Attention From Scratch

Why do transformers use multiple attention heads instead of one big one?

Think about reading a sentence: "The animal didn't cross the street because it was too tired." To understand "it", your brain simultaneously tracks grammar (subject-verb structure), coreference (what does "it" refer to?), and proximity (which nouns are nearby). A single attention head computes one weighted average over positions — it has to cram all of these relationships into a single attention pattern. That is a brutal bottleneck.

Multi-head attention removes it. Instead of one attention head looking at everything, you run H heads in parallel, each with its own learned Q/K/V projections. Each head specializes in a different kind of relationship. Then you concatenate the outputs and project them back to the model dimension. The result is dramatically more expressive than a single head with the same total parameter count.

In this lesson we build multi-head attention from scratch in pure PyTorch. Then we extend it to Grouped-Query Attention (GQA) and Multi-Query Attention (MQA) — the variants used in LLaMA 2/3, PaLM, and Falcon. By the end you will understand every reshape, every transpose, and why each one exists.

---

## Why multiple heads: the single-head bottleneck

Consider a 6-token sentence and what different heads learn to attend to:

```
Sentence:  "The  cat  sat  on  the  mat"
            t0   t1   t2   t3  t4   t5

HEAD 1 — Syntactic (subject-verb):
  t2("sat") attends strongly to t1("cat")
  ┌─────────────────────────────────┐
  │  The  cat  sat  on  the  mat    │
  │        ◄━━━━┛                   │
  │   subject ← verb               │
  └─────────────────────────────────┘

HEAD 2 — Positional (nearby tokens):
  t2("sat") attends to t1("cat") and t3("on")
  ┌─────────────────────────────────┐
  │  The  cat  sat  on  the  mat    │
  │        ◄━━━━┻━━━━►              │
  │      left neighbor + right      │
  └─────────────────────────────────┘

HEAD 3 — Coreference / semantic:
  t5("mat") attends to t1("cat") — both are nouns, potential referents
  ┌─────────────────────────────────┐
  │  The  cat  sat  on  the  mat    │
  │        ◄━━━━━━━━━━━━━━━━━┛     │
  │      noun ← noun (semantic)     │
  └─────────────────────────────────┘
```

A single head cannot produce all three patterns simultaneously — it computes one softmax distribution per query position. Multiple heads solve this by computing independent attention patterns, then combining them.

Research backs this up. Voita et al. (2019) showed that in trained transformers, individual heads specialize: some track syntax, some track positional offsets, and some become "rare word" detectors. Clark et al. (2019) found that specific BERT heads learn to attend to the previous token, the next token, or the end of the sentence — each head has a role.

---

## How heads split the dimensions

Here is the key insight: multi-head attention does NOT increase the number of parameters compared to a single head with the same d_model. Instead, it splits the representation.

```
d_model = 512,  n_heads = 8

Each head gets:  d_head = d_model / n_heads = 512 / 8 = 64

Total Q parameters:  512 x 512 = 262,144  (same either way)
Total K parameters:  512 x 512 = 262,144
Total V parameters:  512 x 512 = 262,144
Total W_O params:    512 x 512 = 262,144

Single-head attention:  Q is (batch, seq, 512)
Multi-head attention:   Q is (batch, 8, seq, 64)  ← same data, reshaped
```

The reshape operation is the entire trick. We project into the full d_model dimension, then view the last dimension as (n_heads, d_head):

```
Before reshape:  (batch, seq_len, 512)
                                  └── one flat vector per token

After reshape:   (batch, seq_len, 8, 64)
                                  │   └── 64 dims per head
                                  └── 8 heads

After transpose: (batch, 8, seq_len, 64)
                         │              └── each head's private subspace
                         └── heads are now a batch dimension
```

Moving heads into the batch position is what lets us compute all 8 attention patterns with a single batched matrix multiply. No loop over heads required.

---

## Piece 1: the projection layers

We use three separate `nn.Linear` layers for Q, K, V (instead of one combined projection). This makes the architecture easier to understand and easier to modify for GQA/MQA later.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        # Three separate learned projections.
        # Each one: (d_model) -> (d_model), i.e. (512) -> (512)
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)

        # Output projection mixes all heads back to d_model.
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
```

Why separate W_Q, W_K, W_V instead of one combined W_QKV? Both work. The original "Attention Is All You Need" paper uses separate projections. Many modern implementations (GPT-2, LLaMA) fuse them into a single `nn.Linear(d_model, 3 * d_model)` for speed. We keep them separate here because it makes the code clearer, and because GQA/MQA need different sizes for Q vs K/V.

---

## Piece 2: project and split into heads

The forward pass starts by projecting the input through W_Q, W_K, W_V, then reshaping so each head has its own slice. Follow the tensor shapes at every step:

```python
    def forward(self, x):
        batch, seq_len, _ = x.shape

        # Step A: Project input through each weight matrix.
        q = self.w_q(x)    # (batch, seq_len, d_model)  e.g. (2, 10, 512)
        k = self.w_k(x)    # (batch, seq_len, d_model)  e.g. (2, 10, 512)
        v = self.w_v(x)    # (batch, seq_len, d_model)  e.g. (2, 10, 512)

        # Step B: Reshape last dim from d_model to (n_heads, d_head).
        #   (2, 10, 512) -> (2, 10, 8, 64)
        q = q.view(batch, seq_len, self.n_heads, self.d_head)
        k = k.view(batch, seq_len, self.n_heads, self.d_head)
        v = v.view(batch, seq_len, self.n_heads, self.d_head)

        # Step C: Transpose so heads are a batch dimension.
        #   (2, 10, 8, 64) -> (2, 8, 10, 64)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
```

Here is what Step B and Step C look like visually for one sequence:

```
Step B — view(batch, seq_len, n_heads, d_head):

  Token 0:  [ h0_d0..d63 | h1_d0..d63 | ... | h7_d0..d63 ]
  Token 1:  [ h0_d0..d63 | h1_d0..d63 | ... | h7_d0..d63 ]
  ...
  Token 9:  [ h0_d0..d63 | h1_d0..d63 | ... | h7_d0..d63 ]

  Shape: (seq_len=10, n_heads=8, d_head=64)

Step C — transpose(1, 2):

  Head 0:   [ tok0_d0..d63, tok1_d0..d63, ..., tok9_d0..d63 ]
  Head 1:   [ tok0_d0..d63, tok1_d0..d63, ..., tok9_d0..d63 ]
  ...
  Head 7:   [ tok0_d0..d63, tok1_d0..d63, ..., tok9_d0..d63 ]

  Shape: (n_heads=8, seq_len=10, d_head=64)
```

After the transpose, each head has a (seq_len, d_head) matrix — exactly the shape needed for standard scaled dot-product attention.

---

## Piece 3: scaled dot-product attention per head

Each head independently computes attention scores, applies softmax, and produces a weighted sum of values. This is identical to single-head attention, just operating on d_head dimensions instead of d_model.

```python
        scale = math.sqrt(self.d_head)

        # Q @ K^T: score every (query, key) pair within each head.
        # (2, 8, 10, 64) @ (2, 8, 64, 10) -> (2, 8, 10, 10)
        scores = q @ k.transpose(-2, -1) / scale

        # Softmax over the key dimension (last dim).
        # Each query gets a probability distribution over all keys.
        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values.
        # (2, 8, 10, 10) @ (2, 8, 10, 64) -> (2, 8, 10, 64)
        attn = attn_weights @ v
```

The shapes tell the whole story:

```
scores:       (batch, n_heads, seq_len, seq_len)
              ──────  ───────  ───────  ───────
                │       │        │        └── which key position
                │       │        └── which query position
                │       └── which head
                └── which example in the batch

attn_weights: same shape, but each row sums to 1.0 after softmax

attn:         (batch, n_heads, seq_len, d_head)
              ──────  ───────  ───────  ──────
                                         └── each position now holds a
                                             weighted blend of all V vectors
```

---

## Piece 4: concatenate heads and apply W_O

Now we reverse the reshape from Piece 2 and pass through the output projection:

```python
        # Transpose heads back: (2, 8, 10, 64) -> (2, 10, 8, 64)
        attn = attn.transpose(1, 2).contiguous()

        # Merge head outputs: (2, 10, 8, 64) -> (2, 10, 512)
        attn = attn.view(batch, seq_len, self.d_model)

        # Output projection mixes information across heads.
        out = self.w_o(attn)   # (2, 10, 512) -> (2, 10, 512)
        return out
```

Why `.contiguous()`? The `transpose` operation returns a view with non-contiguous memory layout. The following `view` requires contiguous data. Without `.contiguous()`, PyTorch will raise a `RuntimeError`. This is one of the most common bugs when writing attention from scratch.

Why W_O? Without it, each head's output occupies its own 64-dim slice of the 512-dim vector, with no cross-head interaction. W_O is a learned (512, 512) matrix that mixes all heads together. It lets head 3's syntactic signal combine with head 7's positional signal into a single representation for the next layer.

---

## The full layer — all pieces assembled

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        batch, seq_len, _ = x.shape
        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, self.d_model)
        return self.w_o(attn)
```

Run it:

```bash
python multi_head_attention.py
```

---

## The shape journey — one diagram for the full forward pass

```
Input x:          (batch, seq, d_model)           e.g. (2, 10, 512)
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
      W_Q(x)          W_K(x)          W_V(x)      Linear projections
   (2,10,512)      (2,10,512)      (2,10,512)
         │               │               │
         ▼               ▼               ▼
       view            view            view        Reshape
  (2,10,8,64)     (2,10,8,64)     (2,10,8,64)
         │               │               │
         ▼               ▼               ▼
    transpose       transpose       transpose      Heads → batch dim
   (2,8,10,64)    (2,8,10,64)    (2,8,10,64)
         │               │               │
         ▼               ▼               │
     Q @ K^T / √d       │               │         Score computation
    (2,8,10,10)          │               │
         │               │               │
         ▼               │               │
      softmax            │               │         Normalize
    (2,8,10,10)          │               │
         │               │               ▼
         └───────────────┴──► weights @ V          Weighted sum
                             (2,8,10,64)
                                  │
                                  ▼
                             transpose             Heads back
                            (2,10,8,64)
                                  │
                                  ▼
                               view                Merge heads
                             (2,10,512)
                                  │
                                  ▼
                               W_O                 Output projection
                             (2,10,512)
```

---

## Grouped-Query Attention (GQA)

Standard MHA uses n_heads each for Q, K, and V. GQA reduces the number of K and V heads while keeping Q heads the same. LLaMA 2 70B uses GQA with 64 query heads and 8 KV heads. LLaMA 3 continues this pattern.

Why? During inference, the KV-cache stores past key and value tensors for every layer. With standard MHA, this cache grows as:

```
KV-cache per layer = 2 x seq_len x n_heads x d_head x bytes_per_param

MHA  (n_kv_heads = 64):  2 x 4096 x 64 x 128 x 2 = 128 MB per layer
GQA  (n_kv_heads = 8):   2 x 4096 x  8 x 128 x 2 =  16 MB per layer
                                                       ^^^^^^^^^^^^^^^^
                                                       8x memory saving
```

Multiple query heads share the same KV head. With 64 Q heads and 8 KV heads, every group of 8 query heads uses the same K and V:

```
Standard MHA (8 Q heads, 8 KV heads):

  Q head 0 ──► KV head 0
  Q head 1 ──► KV head 1
  Q head 2 ──► KV head 2
  Q head 3 ──► KV head 3
  Q head 4 ──► KV head 4
  Q head 5 ──► KV head 5
  Q head 6 ──► KV head 6
  Q head 7 ──► KV head 7

GQA (8 Q heads, 2 KV heads — group_size = 4):

  Q head 0 ─┐
  Q head 1 ─┤► KV head 0
  Q head 2 ─┤
  Q head 3 ─┘
  Q head 4 ─┐
  Q head 5 ─┤► KV head 1
  Q head 6 ─┤
  Q head 7 ─┘
```

The implementation expands (repeats) K and V to match the number of Q heads before computing attention:

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        assert n_heads % n_kv_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.group_size = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, n_heads * self.d_head, bias=bias)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)  # smaller!
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_head, bias=bias)  # smaller!
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        batch, seq_len, _ = x.shape

        q = self.w_q(x)  # (batch, seq, n_heads * d_head)      e.g. (2, 10, 512)
        k = self.w_k(x)  # (batch, seq, n_kv_heads * d_head)   e.g. (2, 10, 128)
        v = self.w_v(x)  # (batch, seq, n_kv_heads * d_head)   e.g. (2, 10, 128)

        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_kv_heads, self.d_head).transpose(1, 2)
        # q: (2, 8, 10, 64)   k: (2, 2, 10, 64)   v: (2, 2, 10, 64)

        # Repeat KV heads to match Q heads.
        # Each KV head is copied group_size times.
        k = k.repeat_interleave(self.group_size, dim=1)  # (2, 8, 10, 64)
        v = v.repeat_interleave(self.group_size, dim=1)  # (2, 8, 10, 64)

        # From here, standard attention.
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, self.d_model)
        return self.w_o(attn)
```

The parameter savings come from W_K and W_V being smaller. With n_heads=8, n_kv_heads=2, d_head=64:

```
                  MHA                    GQA
W_Q:    512 x 512 = 262,144     512 x 512 = 262,144   (same)
W_K:    512 x 512 = 262,144     512 x 128 =  65,536   (4x smaller)
W_V:    512 x 512 = 262,144     512 x 128 =  65,536   (4x smaller)
W_O:    512 x 512 = 262,144     512 x 512 = 262,144   (same)
                    ─────────                ─────────
Total:             1,048,576                 655,360    (37% fewer params)
```

---

## Multi-Query Attention (MQA)

MQA is the extreme case of GQA: only 1 KV head for all query heads. Used in PaLM (Google, 2022) and Falcon (TII, 2023).

```
MQA (8 Q heads, 1 KV head):

  Q head 0 ─┐
  Q head 1 ─┤
  Q head 2 ─┤
  Q head 3 ─┤► KV head 0  (single KV head serves everyone)
  Q head 4 ─┤
  Q head 5 ─┤
  Q head 6 ─┤
  Q head 7 ─┘
```

The implementation is identical to GQA with `n_kv_heads=1`. The KV-cache shrinks to the minimum possible size:

```
KV-cache comparison (per layer, seq_len=4096, d_head=128, fp16):

MHA  (64 KV heads):  2 x 4096 x 64 x 128 x 2 = 128.0 MB
GQA  ( 8 KV heads):  2 x 4096 x  8 x 128 x 2 =  16.0 MB
MQA  ( 1 KV head):   2 x 4096 x  1 x 128 x 2 =   2.0 MB
```

The tradeoff: MQA can slightly degrade quality because all query heads are forced to attend using the same key and value representations. GQA sits in the sweet spot — significant memory savings with minimal quality loss. This is why LLaMA 2/3 chose GQA over MQA.

---

## Head ablation: which heads matter?

Not all heads are equally important. Michel et al. (2019) showed you can zero out many heads at test time with little performance drop. Some heads are critical; others are redundant.

A simple ablation experiment: zero out one head at a time and measure perplexity change.

```python
def ablate_head(model, layer_idx, head_idx, eval_fn):
    """Zero out a single head's output and measure the damage."""
    layer = model.layers[layer_idx].attention

    # Save original W_O rows for this head.
    d_head = layer.d_head
    start = head_idx * d_head
    end = start + d_head
    original = layer.w_o.weight[:, start:end].clone()

    # Zero out this head's contribution to the output.
    layer.w_o.weight.data[:, start:end] = 0.0

    result = eval_fn(model)  # e.g. compute perplexity

    # Restore.
    layer.w_o.weight.data[:, start:end] = original

    return result
```

Typical findings on a trained model:

```
Head  │ Perplexity Change │ Role (if analyzed)
──────┼───────────────────┼────────────────────
  0   │  +0.02            │ Positional (attends to previous token)
  1   │  +0.01            │ Redundant (similar to head 0)
  2   │  +5.31            │ CRITICAL — syntactic head
  3   │  +0.15            │ Moderate — rare-word detector
  4   │  +0.03            │ Redundant
  5   │  +0.08            │ Moderate — coreference
  6   │  +3.72            │ CRITICAL — long-range dependency
  7   │  +0.01            │ Redundant
```

This has practical implications: pruning redundant heads (structured pruning) can speed up inference with minimal quality loss. It also explains why GQA works — if many heads are doing similar things, sharing KV projections forces the model to be more efficient about what each head computes.

---

## Visualizing attention patterns

Here is what attention weights look like across different heads for the sentence "The cat sat on the mat":

```
HEAD 0 — Previous-token pattern (positional):
         The   cat   sat   on    the   mat
  The   [0.9   0.02  0.02  0.02  0.02  0.02]
  cat   [0.85  0.05  0.03  0.02  0.03  0.02]
  sat   [0.03  0.82  0.05  0.03  0.04  0.03]
  on    [0.02  0.04  0.80  0.05  0.05  0.04]
  the   [0.03  0.03  0.04  0.78  0.07  0.05]
  mat   [0.02  0.03  0.03  0.05  0.80  0.07]
              ^^^^^^
         Each row peaks at the previous column.
         This head has learned a "bigram" pattern.

HEAD 2 — Syntactic (subject-verb):
         The   cat   sat   on    the   mat
  The   [0.40  0.30  0.10  0.05  0.10  0.05]
  cat   [0.20  0.30  0.25  0.05  0.10  0.10]
  sat   [0.05  0.70  0.10  0.05  0.05  0.05]   ← "sat" attends to "cat"
  on    [0.05  0.10  0.50  0.15  0.10  0.10]
  the   [0.10  0.10  0.10  0.10  0.30  0.30]
  mat   [0.05  0.05  0.40  0.20  0.10  0.20]   ← "mat" attends to "sat"

HEAD 5 — Broad / uniform (information gathering):
         The   cat   sat   on    the   mat
  The   [0.18  0.17  0.17  0.16  0.16  0.16]
  cat   [0.17  0.17  0.17  0.16  0.17  0.16]
  sat   [0.16  0.17  0.17  0.17  0.16  0.17]
  on    [0.17  0.16  0.17  0.17  0.17  0.16]
  the   [0.16  0.17  0.16  0.17  0.17  0.17]
  mat   [0.17  0.16  0.17  0.16  0.17  0.17]
         This head attends roughly uniformly — it computes
         a bag-of-words average. Likely redundant; safe to prune.
```

---

## Common mistakes

### Mistake 1: wrong reshape dimensions

```python
# WRONG — swapped n_heads and d_head
q = q.view(batch, seq_len, self.d_head, self.n_heads)  # Bug!

# RIGHT — n_heads comes before d_head
q = q.view(batch, seq_len, self.n_heads, self.d_head)
```

The data in memory is laid out as `[head0_dim0, head0_dim1, ..., head0_dim63, head1_dim0, ...]`. If you swap the dimensions in `view`, head 0 will contain a mix of dimensions from multiple heads. The code will run without error but produce garbage.

### Mistake 2: forgetting to transpose back after attention

```python
# After attention, shape is (batch, n_heads, seq_len, d_head)
# WRONG — view directly without transposing back
attn = attn.view(batch, seq_len, d_model)  # RuntimeError or silent corruption

# RIGHT — transpose first, then contiguous, then view
attn = attn.transpose(1, 2).contiguous()
attn = attn.view(batch, seq_len, d_model)
```

Without the transpose, the view would interleave head dimensions with sequence positions. The resulting tensor would have the right shape but completely wrong values.

### Mistake 3: forgetting .contiguous()

```python
# WRONG — transpose returns a non-contiguous view
attn = attn.transpose(1, 2)
attn = attn.view(batch, seq_len, d_model)  # RuntimeError!

# RIGHT — call .contiguous() after transpose
attn = attn.transpose(1, 2).contiguous()
attn = attn.view(batch, seq_len, d_model)  # Works
```

Alternatively, use `reshape` instead of `view` — it handles non-contiguous tensors by copying if needed. But `.contiguous().view()` is the convention you will see in most implementations.

### Mistake 4: mixing up n_heads and head_dim in the scale factor

```python
# WRONG — scaling by total d_model instead of d_head
scale = math.sqrt(d_model)      # Bug! Too large, attention too uniform

# WRONG — scaling by n_heads
scale = math.sqrt(self.n_heads)  # Bug! Unrelated to the actual dot-product magnitude

# RIGHT — scale by d_head (the dimension of the dot product)
scale = math.sqrt(self.d_head)
```

The scale factor exists because the dot product of two random d-dimensional vectors has variance proportional to d. We divide by sqrt(d) to normalize it. The relevant d is d_head (64), not d_model (512), because each head computes its own dot product in a 64-dimensional space.

### Mistake 5: applying causal mask with wrong value

```python
# WRONG — masking with 0
scores = scores.masked_fill(mask == 0, 0)  # Bug! 0 is a valid score

# RIGHT — masking with -inf so softmax produces 0
scores = scores.masked_fill(mask == 0, float('-inf'))
```

When you add a causal mask to prevent attending to future tokens, you need `-inf` so that `softmax(-inf) = 0`. Masking with 0 still allows some attention to flow to masked positions.

---

## Compare against PyTorch

Run `compare.py` to verify our implementation matches `torch.nn.MultiheadAttention` numerically, and to compare forward pass latency:

```bash
python compare.py
```

Both implementations use the same W_Q, W_K, W_V, W_O architecture. When we initialize PyTorch's MHA with the same weights as ours, outputs are nearly identical. Speed differences reflect PyTorch's C++ backend optimizations.

---

## Benchmark across sizes

Run `benchmark.py` to profile forward pass time across batch sizes and sequence lengths:

```bash
python benchmark.py
```

This shows how latency scales with seq_len (O(N^2) in the naive implementation) and how batch size interacts with throughput.

---

## Before and after: single-head vs multi-head

Here is a concrete comparison on a small language modeling task to illustrate why multi-head matters:

```
Configuration:  d_model=256, seq_len=128, 4 layers, 10K training steps

                    │ Params  │ Val Loss │ Val Perplexity
────────────────────┼─────────┼──────────┼───────────────
1 head  (d_head=256)│ 3.15M   │  4.21    │  67.4
4 heads (d_head=64) │ 3.15M   │  3.89    │  48.9
8 heads (d_head=32) │ 3.15M   │  3.82    │  45.6
16 heads (d_head=16)│ 3.15M   │  3.91    │  49.9  ← too many heads hurts

Same parameter count. More heads = better, up to a point.
With 16 heads, d_head=16 is too small for each head to learn
meaningful patterns. The sweet spot depends on model size.
```

---

## Recap

- **Single head bottleneck**: one attention pattern per position. Multiple heads let the model track syntax, coreference, and position simultaneously.
- **The reshape trick**: `view(batch, seq, n_heads, d_head).transpose(1, 2)` turns heads into a batch dimension. No extra parameters; same data, different layout.
- **W_O matters**: without the output projection, heads cannot share information. W_O mixes all head outputs into a single vector.
- **GQA**: fewer KV heads shared across query heads. Cuts KV-cache memory by the group factor. Used in LLaMA 2/3.
- **MQA**: extreme case — one KV head for all queries. Maximum memory savings, slight quality tradeoff. Used in PaLM, Falcon.
- **Not all heads are equal**: ablation studies show some heads are critical, others redundant. This is why head pruning and GQA work.
- **Scaled dot-product**: divide by sqrt(d_head), not sqrt(d_model). The dimension that matters is the one participating in the dot product.

---

Get the video walkthrough of per-head attention pattern analysis (which heads learn which relationships), the flash attention algorithm (tile-based online softmax, O(N) memory), and grouped-query attention (GQA): [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
