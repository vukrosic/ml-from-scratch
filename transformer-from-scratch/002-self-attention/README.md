# Self-Attention From Scratch

Q, K, V — what they actually compute and why.

Every transformer model runs on self-attention. It's the mechanism that lets each token look at every other token and decide what's relevant. Before GPT, before BERT, before Llama — there is attention. Understanding it from scratch means nothing is hidden.

In this lesson we build self-attention from raw matrices in pure PyTorch. No black boxes. By the end you'll know exactly what Q, K, and V are, how the attention score is computed, why we scale it, and how masking works for autoregressive (causal) models.

---

## Hook: why attention matters

RNNs process sequences token by token — they literally cannot look at future tokens. This makes them slow to train and hard to parallelize. Self-attention solves both: every token attends to every other token simultaneously, in one parallelizable operation. The transformer architecture — and every large language model built on it — exists because attention works at scale.

But attention isn't magic. It's three linear projections and a softmax. Let's see exactly how.

---

## Step 1: Q, K, V projection

Given a sequence of tokens, we first embed each token into a vector of size `d_model`. Self-attention projects each embedding three times — into a **Query (Q)**, a **Key (K)**, and a **Value (V)** vector. Think of it like a retrieval system:

- The **Query** is what I'm looking for.
- The **Key** is what I contain — used to score whether the query matches me.
- The **Value** is what I actually contribute if you attend to me.

We do this with a single linear layer that projects `d_model` features into `3 * d_model`, then split the result.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # One linear layer outputs Q, K, V concatenated.
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
```

---

## Step 2: Reshaping into heads

We split the model into `n_heads` parallel attention heads. Each head gets its own slice of the Q/K/V vectors. This lets different heads attend to different aspects of the sequence — one head might focus on syntax, another on coreference.

```python
    def forward(self, x):
        batch, seq_len, d_model = x.shape
        # Project and split Q, K, V in one shot.
        qkv = self.qkv_proj(x)                    # (batch, seq, 3*d_model)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head)
        qkv = qkv.transpose(1, 2)                  # (batch, n_heads, seq, 3*d_head)
        q, k, v = qkv.chunk(3, dim=-1)            # each (batch, n_heads, seq, d_head)
```

---

## Step 3: Scaled dot-product attention score

Now we compute how much token i should attend to token j. The score is the dot product of their query and key vectors — higher dot product means better match.

We divide by `sqrt(d_head)` to keep the gradient signal stable. Without this scaling, large `d_head` values make the dot products huge, pushing softmax into regions with near-zero gradients.

```python
        scale = math.sqrt(self.d_head)
        # Q @ K^T gives us a score for every (query, key) pair.
        scores = q @ k.transpose(-2, -1) / scale  # (batch, n_heads, seq, seq)
```

---

## Step 4: Softmax over attention scores

We pass the scores through softmax to get a probability distribution over keys for each query. Every query now has normalized attention weights that sum to 1 across all keys.

```python
        # Softmax over the key dimension (rows) — each query gets a distribution.
        attn_weights = F.softmax(scores, dim=-1)  # (batch, n_heads, seq, seq)
```

---

## Step 5: Weighted sum of values

Finally, each output is the weighted sum of all values, where the weights are the attention probabilities. Token i's new representation is a mixture of all tokens' values, mixed by how much i attended to each.

```python
        # Each output position blends all values, weighted by attention.
        attn = attn_weights @ v                    # (batch, n_heads, seq, d_head)
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model) # (batch, seq, d_model)
        return self.out_proj(attn)
```

---

## The full layer

Putting it together, here's the complete `SelfAttention` class:

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

---

## Causal (masked) attention

In autoregressive models like GPT, token i must not attend to any token > i. We enforce this with a lower-triangular mask. Before softmax, we set forbidden positions to `-inf`. Softmax then sends their weight to ~0.

```python
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool))
        scores = scores.masked_fill(~causal_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
```

This is the only difference from standard self-attention. Everything else is identical.

---

## Visualizing attention weights

Run `visualize.py` to generate a heatmap of the attention weights:

```bash
python visualize.py
```

It produces `attention_weights.png` showing:
- Standard self-attention: every token attends to every other token.
- Causal self-attention: each token only attends to itself and previous tokens.

You can also see per-head patterns — different heads specialize to different relationships in the sequence.

---

## Benchmark: from scratch vs PyTorch

PyTorch's `nn.MultiheadAttention` uses a slightly different internal design (separate Q and K projection layers), but the mathematical operation is identical. Run `benchmark.py` to compare:

```bash
python benchmark.py
```

The output shows forward pass latency across several batch/sequence/model-size configurations. Our from-scratch version uses a single combined QKV projection, which can be faster for small models because it does one less matrix multiply.

---

## Recap

- **Q, K, V** are three learned linear projections of the input. Q is "what am I looking for?", K is "what do I contain?", V is "what do I contribute?".
- **Attention score** = Q @ K^T / sqrt(d_head). The scaling keeps gradients stable.
- **Softmax** converts scores to a probability distribution over keys for each query.
- **Output** = attention_weights @ V. Each position blends information from all positions.
- **Causal mask** sets attention to future tokens to `-inf` before softmax, enforcing autoregressive masking.
- The full `SelfAttention` layer adds an output projection to mix the result back to `d_model`.

---

## Get the extended notebook with multi-head attention internals, grouped-query attention (GQA), flash attention intuition, and attention pattern analysis across real transformer layers:

**https://www.skool.com/opensuperintelligencelab**
