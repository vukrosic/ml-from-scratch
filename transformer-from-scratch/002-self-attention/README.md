# Self-Attention From Scratch

Q, K, V — what they actually compute and why.

Every transformer model runs on self-attention. It is the mechanism that lets
each token look at every other token and decide what is relevant. Before GPT,
before BERT, before Llama — there is attention. Understanding it from scratch
means nothing is hidden.

In this lesson we build self-attention from raw matrices in pure PyTorch. No
black boxes. By the end you will know exactly what Q, K, and V are, how the
attention score is computed, why we scale it, and how masking works for
autoregressive (causal) models.

---

## Why attention exists

Recurrent Neural Networks (RNNs) process sequences one token at a time, left
to right. Token 5 must wait for tokens 1-4 to finish before it can begin.
This creates two serious problems:

```
RNN processing (sequential — slow):

  tok_0 --> tok_1 --> tok_2 --> tok_3 --> tok_4
  [____]   [____]   [____]   [____]   [____]
   step 1   step 2   step 3   step 4   step 5

  Each step depends on the previous one.
  Token 4's representation of token 0 has passed through
  4 sequential compression bottlenecks.
```

**Problem 1: No parallelism.** Each step depends on the hidden state of the
previous step, so you cannot process them simultaneously on a GPU. Training is
slow.

**Problem 2: Long-range forgetting.** By the time the RNN reaches token 50,
the information from token 0 has been compressed through 50 sequential steps.
In practice, RNNs struggle to remember things more than ~20-30 tokens back.

Self-attention solves both problems in one shot:

```
Self-attention processing (parallel — fast):

  tok_0   tok_1   tok_2   tok_3   tok_4
  [____]  [____]  [____]  [____]  [____]
    |  \   / | \   / | \   / |  \  / |
    |   \ /  |  \ /  |  \ /  |   \/  |
    |    X   |   X   |   X   |   /\  |
    |   / \  |  / \  |  / \  |  /  \ |
    v  v   v v v   v v v   v v v    vv
  [____]  [____]  [____]  [____]  [____]
  out_0   out_1   out_2   out_3   out_4

  Every token attends to every other token
  in a single parallel operation.
  Token 4 sees token 0 directly — no compression.
```

Every token can see every other token in one step. No sequential bottleneck,
no forgetting. The cost is O(n^2) in sequence length (every token looks at
every other), but for moderate sequence lengths this is massively faster than
the O(n) sequential passes of an RNN because each of those O(n^2) operations
runs in parallel on a GPU.

The transformer architecture — and every large language model built on it —
exists because attention works at scale.

But attention is not magic. It is three linear projections and a softmax.
Let us see exactly how.

---

## The Q/K/V intuition

Self-attention projects each token embedding three ways — into a **Query (Q)**,
a **Key (K)**, and a **Value (V)** vector. The best way to understand these is
with a library analogy.

Imagine you walk into a library. You have a question in your head — that is
your **Query**. Each book on the shelf has a label describing what it contains —
that is its **Key**. The actual content inside each book is its **Value**.

The process:
1. You compare your question (Query) against every book label (Key).
2. The labels that match your question get high scores.
3. You read from the matching books (Values), in proportion to how well they matched.

```
Q/K/V analogy:

  Query  = "What am I looking for?"
           (each token asks this question)

  Key    = "What do I contain?"
           (each token advertises this about itself)

  Value  = "What information do I give you if you attend to me?"
           (the actual content a token contributes)

  Token "cat" might ask:  Q = "looking for an adjective that describes me"
  Token "black" responds: K = "I am an adjective about color"  --> high match!
  Token "the" responds:   K = "I am a determiner"              --> low match
  Token "black" delivers: V = [its learned representation]
```

Critically, Q, K, and V are all learned linear projections. The network
learns what to ask for, what to advertise, and what to give — during training.

We produce all three with a single linear layer that projects `d_model`
features into `3 * d_model`, then split the result into three equal parts:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
```

---

## Piece 1: The projection layer

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # One linear layer outputs Q, K, V concatenated.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
```

Why a single combined projection instead of three separate ones? Fewer kernel
launches on the GPU. One large matrix multiply is faster than three small ones.
The output is `3 * d_model` wide — we will split it into Q, K, V later.

---

## Piece 2: Reshaping into heads

Multi-head attention splits the model dimension into `n_heads` independent
attention mechanisms. Each head gets its own slice of Q, K, V and can learn to
attend to different relationships — one head might track syntax, another
coreference, another positional patterns.

```python
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        # Project and split Q, K, V in one shot.
        qkv = self.qkv_proj(x)                    # (batch, seq, 3*d_model)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)                  # (batch, n_heads, seq, 3*d_head)
        q, k, v = qkv.chunk(3, dim=-1)            # each (batch, n_heads, seq, d_head)
```

Let us trace the shapes with concrete numbers. Suppose `batch=2, seq_len=4,
d_model=8, n_heads=2`, so `d_head=4`:

```
Step by step shape trace (batch=2, seq_len=4, d_model=8, n_heads=2, d_head=4):

  x                                 (2, 4, 8)
       |
       v  qkv_proj (Linear 8 -> 24)
       |
  qkv                              (2, 4, 24)       = (batch, seq, 3*d_model)
       |
       v  view
       |
  qkv                              (2, 4, 2, 12)    = (batch, seq, n_heads, 3*d_head)
       |
       v  transpose(1, 2)
       |
  qkv                              (2, 2, 4, 12)    = (batch, n_heads, seq, 3*d_head)
       |
       v  chunk(3, dim=-1)
       |
  q, k, v  each                    (2, 2, 4, 4)     = (batch, n_heads, seq, d_head)
```

After the transpose, `n_heads` sits in position 1, so each head is an
independent batch dimension. PyTorch's batched matrix multiply handles
the rest.

---

## Piece 3: Scaled dot-product attention

This is the core of the mechanism. We compute how much token i should attend
to token j by taking the dot product of token i's query with token j's key.

```python
        scale = math.sqrt(self.d_head)
        # Q @ K^T gives us a score for every (query, key) pair.
        scores = q @ k.transpose(-2, -1) / scale  # (batch, n_heads, seq, seq)
```

The shape arithmetic:

```
Matrix multiply: Q @ K^T

  Q shape:    (batch, n_heads, seq, d_head)    = (2, 2, 4, 4)
  K^T shape:  (batch, n_heads, d_head, seq)    = (2, 2, 4, 4)
                                                        ^    ^
                                                      d_head  seq
  Result:     (batch, n_heads, seq, seq)        = (2, 2, 4, 4)

  scores[b][h][i][j] = dot product of query_i with key_j in head h of batch b
```

The output is a `(seq, seq)` matrix for each head in each batch. Entry `[i][j]`
tells us how much token i wants to attend to token j. Higher dot product means
the query and key are more aligned — a better match.

---

## Why we scale by sqrt(d_k)

This is not just a nice-to-have. Without scaling, attention breaks.

The dot product of two random vectors with `d_head` dimensions has variance
proportional to `d_head`. As `d_head` grows, the dot products get larger in
magnitude, and softmax saturates — most of the probability mass concentrates
on one or two tokens, and the gradients for all other positions approach zero.

Here are actual numbers. With `d_head=64` (standard for many models), two
random unit-variance vectors:

```
Without scaling (d_head = 64):

  E[q . k] = 0           (mean is fine)
  Var[q . k] = 64        (variance grows with d_head!)
  std[q . k] = 8.0

  Typical dot products might look like:  [-12.3,  8.7,  -5.1,  15.2]

  softmax([-12.3, 8.7, -5.1, 15.2]) = [0.000, 0.001, 0.000, 0.999]
                                        ^^^^^^^^^^^^^^^^^^^^   ^^^^^
                                        gradient ~ 0           winner takes all

With scaling (divide by sqrt(64) = 8):

  Scaled dot products:                   [-1.54, 1.09, -0.64,  1.90]

  softmax([-1.54, 1.09, -0.64, 1.90]) = [0.027, 0.375, 0.066, 0.532]
                                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                          gradients flow to all positions
```

Dividing by `sqrt(d_head)` normalizes the variance back to ~1, keeping softmax
in its useful range where gradients are non-zero for multiple positions. This
is why the original "Attention Is All You Need" paper calls it **scaled**
dot-product attention.

---

## Piece 4: Softmax over attention scores

We pass the scores through softmax to get a probability distribution. Each row
sums to 1 — each query distributes its "attention budget" across all keys.

```python
        # Softmax over the key dimension — each query gets a distribution.
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq, seq)
```

The `dim=-1` is critical. We softmax over the last dimension (the key
dimension), so each **row** of the attention matrix becomes a probability
distribution. This means: for each query position, the weights over all key
positions sum to 1.

```
Attention weight matrix (seq_len=4, one head):

  attn_weights[i][j] = "how much does token i attend to token j"

              Key_0   Key_1   Key_2   Key_3
  Query_0  [ 0.35    0.25    0.15    0.25  ]  <- sums to 1.0
  Query_1  [ 0.10    0.40    0.30    0.20  ]  <- sums to 1.0
  Query_2  [ 0.20    0.15    0.45    0.20  ]  <- sums to 1.0
  Query_3  [ 0.05    0.30    0.25    0.40  ]  <- sums to 1.0
                                                  ^
                                            each row sums to 1
```

---

## Piece 5: Weighted sum of values

Finally, each output token is a weighted combination of all value vectors,
where the weights are the attention probabilities we just computed.

```python
        # Each output position blends all values, weighted by attention.
        attn = attn_weights @ v                    # (batch, n_heads, seq, d_head)
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model) # (batch, seq, d_model)
        return self.out_proj(attn)
```

The matrix multiply `attn_weights @ v`:

```
  attn_weights:  (batch, n_heads, seq, seq)     = (2, 2, 4, 4)
  v:             (batch, n_heads, seq, d_head)   = (2, 2, 4, 4)
  result:        (batch, n_heads, seq, d_head)   = (2, 2, 4, 4)

  output[b][h][i] = sum_j ( attn_weights[b][h][i][j] * v[b][h][j] )

  Token i's new representation is a mixture of ALL tokens' values,
  weighted by how much token i attended to each.
```

After the attention operation, we transpose heads back and reshape to
`(batch, seq_len, d_model)`. The final `out_proj` linear layer mixes
information across heads — without it, the heads would be completely
independent.

---

## The full self-attention layer

Putting all five pieces together:

```python
class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model)
        return self.out_proj(attn)
```

Run it:

```bash
python attention.py
```

The output confirms that the input and output shapes match — `(batch, seq_len,
d_model)` in, `(batch, seq_len, d_model)` out. The attention layer transforms
representations without changing their shape.

---

## Causal (masked) attention

In autoregressive models like GPT, token i must not attend to any token after
position i. The model generates tokens left to right — if token 3 could see
token 4 during training, the model would learn to cheat by copying the answer
instead of predicting it.

We enforce this with a lower-triangular boolean mask. Before softmax, we set
all "future" positions to `-inf`. When softmax sees `-inf`, it sends that
position's weight to ~0.

```python
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
```

Here is what the mask and the resulting scores look like for `seq_len=4`:

```
The causal mask (True = allowed, False = blocked):

                 Key_0  Key_1  Key_2  Key_3
  Query_0    [   True   False  False  False ]   <- can only see itself
  Query_1    [   True   True   False  False ]   <- sees 0 and itself
  Query_2    [   True   True   True   False ]   <- sees 0, 1, and itself
  Query_3    [   True   True   True   True  ]   <- sees everything up to here

Scores BEFORE masking:

              Key_0   Key_1   Key_2   Key_3
  Query_0  [  1.2     0.8    -0.3     2.1  ]
  Query_1  [  0.5     1.7     0.9    -0.4  ]
  Query_2  [ -0.1     0.3     1.5     0.7  ]
  Query_3  [  0.4     1.1     0.2     1.8  ]

Scores AFTER masking (blocked positions set to -inf):

              Key_0   Key_1   Key_2   Key_3
  Query_0  [  1.2    -inf    -inf    -inf   ]
  Query_1  [  0.5     1.7    -inf    -inf   ]
  Query_2  [ -0.1     0.3     1.5    -inf   ]
  Query_3  [  0.4     1.1     0.2     1.8   ]

After softmax (each row sums to 1, -inf positions become 0):

              Key_0   Key_1   Key_2   Key_3
  Query_0  [  1.00    0.00    0.00    0.00  ]
  Query_1  [  0.23    0.77    0.00    0.00  ]
  Query_2  [  0.14    0.21    0.65    0.00  ]
  Query_3  [  0.10    0.20    0.08    0.62  ]
```

This is the only difference from standard self-attention. The `CausalSelfAttention`
class in `attention.py` is identical except for these two lines.

---

## Attention patterns visualization

What does the attention matrix actually look like? Here is a schematic for
standard vs. causal attention on a 6-token sequence:

```
Standard self-attention             Causal self-attention
(every token sees every token)      (each token sees only past + self)

     K0  K1  K2  K3  K4  K5            K0  K1  K2  K3  K4  K5
Q0 [ ##  .   ##  .   .   .  ]    Q0 [ ##  --  --  --  --  -- ]
Q1 [ .   ##  .   .   ##  .  ]    Q1 [ .   ##  --  --  --  -- ]
Q2 [ ##  .   ##  .   .   ## ]    Q2 [ ##  .   ##  --  --  -- ]
Q3 [ .   .   .   ##  .   .  ]    Q3 [ .   .   .   ##  --  -- ]
Q4 [ .   ##  .   .   ##  .  ]    Q4 [ .   ##  .   .   ##  -- ]
Q5 [ .   .   ##  .   .   ## ]    Q5 [ .   .   ##  .   .   ## ]

## = high attention weight           -- = masked (zero weight)
.  = low attention weight            Upper triangle is always zero
```

In the standard version, attention is distributed freely — token Q2 might
attend strongly to K0 and K5 if they are relevant. In the causal version,
the upper triangle is always zero — Q2 can never see K3, K4, or K5.

Run `visualize.py` to generate real heatmaps:

```bash
python visualize.py
```

It produces:
- `attention_weights.png` — mean attention across all heads, standard vs. causal
- `attention_weights_standard_heads.png` — per-head patterns for standard attention
- `attention_weights_causal_heads.png` — per-head patterns for causal attention

Different heads learn different patterns. In trained models, you will often see
heads that specialize: one head attends to the previous token, another to the
first token, another to semantically related tokens. With random weights (as
in our demo), the patterns are arbitrary but the lower-triangular structure of
causal masking is already clearly visible.

---

## Common mistakes and gotchas

Building attention from scratch is a rite of passage, but there are several
sharp edges that trip almost everyone the first time.

### Mistake 1: Wrong transpose dimensions

```python
# WRONG: transposes batch and seq dimensions
scores = q @ k.transpose(0, 1) / scale

# RIGHT: transpose the last two dims (seq and d_head)
scores = q @ k.transpose(-2, -1) / scale
```

You need `K^T` to be `(batch, n_heads, d_head, seq)` so the matmul produces
`(batch, n_heads, seq, seq)`. Transposing the wrong dimensions will either
crash with a shape mismatch or silently compute garbage.

### Mistake 2: Softmax on the wrong axis

```python
# WRONG: softmax over the query dimension (columns sum to 1)
attn_weights = F.softmax(scores, dim=-2)

# RIGHT: softmax over the key dimension (rows sum to 1)
attn_weights = F.softmax(scores, dim=-1)
```

If you softmax over `dim=-2`, each **column** sums to 1 instead of each row.
This means the attention weights answer "how much does each query attend to
this key" instead of "how much does this query attend to each key." The model
may still train but will converge much slower and to worse loss.

### Mistake 3: Forgetting the mask (or applying it after softmax)

```python
# WRONG: mask after softmax (too late — probabilities already computed)
attn_weights = F.softmax(scores, dim=-1)
attn_weights = attn_weights.masked_fill(~causal_mask, 0.0)

# RIGHT: mask before softmax (so -inf -> 0 probability through softmax)
scores = scores.masked_fill(~causal_mask, float('-inf'))
attn_weights = F.softmax(scores, dim=-1)
```

Applying the mask after softmax zeroes out the forbidden positions, but the
remaining weights no longer sum to 1. The model effectively "wastes" attention
budget on positions that are then zeroed, reducing the effective signal.

### Mistake 4: Forgetting .contiguous() before view

```python
# WRONG: view after transpose without contiguous
attn = attn.transpose(1, 2)
attn = attn.view(batch, seq_len, d_model)  # RuntimeError!

# RIGHT: call .contiguous() first
attn = attn.transpose(1, 2).contiguous()
attn = attn.view(batch, seq_len, d_model)
```

`transpose()` returns a view with non-contiguous memory layout. `view()`
requires contiguous memory. The fix is a single `.contiguous()` call. You can
also use `reshape()` instead, which handles this automatically but may copy.

### Mistake 5: Not dividing d_model evenly by n_heads

```python
# WRONG: d_model=768, n_heads=10 -> 768/10 = 76.8 (??)
# RIGHT: d_model=768, n_heads=8  -> 768/8  = 96 (clean split)
```

If `d_model` is not evenly divisible by `n_heads`, the Q/K/V split produces
heads of unequal size. Our implementation catches this with an assert. Common
models use d_model=768 with 12 heads (64 per head), or d_model=1024 with 16
heads (64 per head).

---

## Benchmark: from scratch vs PyTorch

PyTorch ships `nn.MultiheadAttention` which uses a slightly different internal
design (separate Q and K projection layers). The mathematical operation is
identical. Run `benchmark.py` to compare:

```bash
python benchmark.py
```

The output shows forward pass latency across several batch/sequence/model-size
configurations. Our from-scratch version uses a single combined QKV projection,
which can be faster for small models because it does one fewer matrix multiply.

For production, PyTorch's `F.scaled_dot_product_attention` (added in PyTorch
2.0) dispatches to FlashAttention or memory-efficient attention kernels
automatically. Our from-scratch version is for understanding — the production
path is to call the optimized kernel.

---

## What the toy task demonstrates

The demo in `attention.py` runs a minimal sanity check:

1. **Shape preservation.** A random input `(batch=2, seq_len=4, d_model=8)`
   goes in, and the same shape comes out. Self-attention transforms
   representations without changing their dimensionality.

2. **Causal mask correctness.** We extract the attention weights from the
   causal layer and print them. The matrix is lower-triangular — row i has
   non-zero values only in columns 0 through i. This proves the mask is
   working: no token can attend to future positions.

3. **Head specialization potential.** Even with random weights, each head
   produces different attention patterns. After training, these patterns
   would specialize to capture different linguistic relationships.

This is not a training task — there is no loss function or optimizer. The
purpose is to verify that the forward pass is mechanically correct before
we compose attention into a full transformer block (next lesson).

---

## Recap

```
The full self-attention pipeline:

  Input x: (batch, seq_len, d_model)
       |
       v
  [QKV Projection]  Linear(d_model -> 3*d_model)
       |
       v
  Split into Q, K, V  (each: batch, n_heads, seq_len, d_head)
       |
       v
  [Attention Scores]  Q @ K^T / sqrt(d_head)   -> (batch, n_heads, seq, seq)
       |
       v
  [Optional Mask]     set future positions to -inf  (causal models only)
       |
       v
  [Softmax]           normalize each row to sum to 1
       |
       v
  [Weighted Sum]      attn_weights @ V          -> (batch, n_heads, seq, d_head)
       |
       v
  [Reshape + Project] concat heads, Linear(d_model -> d_model)
       |
       v
  Output: (batch, seq_len, d_model)
```

Key takeaways:

- **Q, K, V** are three learned linear projections. Q is "what am I looking
  for?", K is "what do I contain?", V is "what do I contribute?"
- **Attention score** = `Q @ K^T / sqrt(d_head)`. The scaling keeps gradients
  stable by preventing softmax saturation.
- **Softmax** converts scores to a probability distribution over keys for each
  query. Apply it on `dim=-1`.
- **Output** = `attn_weights @ V`. Each position blends information from all
  positions it is allowed to see.
- **Causal mask** sets attention to future tokens to `-inf` before softmax,
  enforcing autoregressive ordering.
- The **output projection** mixes information across heads — without it, heads
  are completely independent.

---

Get the video walkthrough of multi-head attention internals, grouped-query attention (GQA), flash attention intuition, and attention pattern analysis: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
