# Multi-Head Attention From Scratch

Why do transformers use multiple attention heads instead of one?

A single attention head can only learn one kind of relationship between tokens at a time. With multiple heads, different heads can specialize: one head might learn syntactic relationships (subject-verb agreement), another tracks coreference (which "it" refers to), another focuses on positional proximity. The network learns which patterns to delegate to which head, then combines their outputs.

In this lesson we build multi-head attention (MHA) from scratch — the same mechanism inside every transformer model. By the end you'll understand why MHA uses separate W_Q, W_K, W_V projections, what concatenation and the output projection W_O actually do, and why the computation stays numerically stable.

---

## Hook: the limitation of single-head attention

Single-head attention computes one set of Q, K, V projections and one attention weight matrix. If the model needs to track both syntactic structure and semantic similarity, a single head must compromise — learning a weighted average of both patterns. This limits expressiveness.

Multi-head attention solves this by running H attention heads in parallel, each with its own Q/K/V projections. The outputs are concatenated and projected back to d_model. Each head learns different relationships; the final projection mixes them.

---

## Step 1: separate W_Q, W_K, W_V projections

Single-head attention used a single combined QKV projection. Multi-head attention uses three separate linear layers — one for Q, one for K, one for V. Each maps from d_model to d_model.

The three outputs are independent: Q asks "what am I looking for?", K says "what do I contain?", V says "what do I contribute if you attend to me?".

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Three separate learned projections.
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)

        # Output projection mixes all heads back to d_model.
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
```

---

## Step 2: reshape into heads

After projecting x into Q, K, V (each shape batch x seq_len x d_model), we split each into n_heads and reshape so heads are the batch dimension. This lets us compute attention for all heads simultaneously with a single batched matrix multiply.

```python
    def forward(self, x):
        batch, seq_len, d_model = x.shape

        q = self.w_q(x)
        k = self.w_k(x)
        v = self.w_v(x)

        # Split each into n_heads and move head dimension to batch position.
        # After transpose: (batch, n_heads, seq_len, d_head)
        q = q.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(batch, seq_len, self.n_heads, self.d_head).transpose(1, 2)
```

---

## Step 3: scaled dot-product attention per head

For each head, we compute the standard scaled dot-product attention: scores = Q @ K^T / sqrt(d_head), then softmax over rows. The scaling factor prevents vanishing gradients when d_head is large.

```python
        scale = math.sqrt(self.d_head)

        # Q @ K^T: attention score between every pair of positions.
        # Shape: (batch, n_heads, seq_len, seq_len)
        scores = q @ k.transpose(-2, -1) / scale

        attn_weights = F.softmax(scores, dim=-1)

        # Weighted sum of values: each position blends all others.
        attn = attn_weights @ v  # (batch, n_heads, seq_len, d_head)
```

---

## Step 4: concatenate heads and project with W_O

Each head produces an output of shape (batch, n_heads, seq_len, d_head). We transpose back to (batch, seq_len, n_heads, d_head), reshape to (batch, seq_len, d_model), then pass through W_O. This final projection mixes information across heads so the next layer can use a combined representation.

```python
        # Transpose and reshape: back to (batch, seq_len, d_model).
        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_len, d_model)

        # W_O mixes outputs from all heads.
        out = self.w_o(attn)
        return out
```

---

## The full layer

Putting all the steps together:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
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
        attn = attn.view(batch, seq_len, d_model)
        return self.w_o(attn)
```

Run it:

```bash
python multi_head_attention.py
```

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

## Recap

- **Separate W_Q, W_K, W_V** projections allow Q, K, V to represent different aspects of each token independently.
- **Multiple heads** let the model learn different relational patterns simultaneously — syntax, coreference, proximity.
- **Concatenation + W_O** combines per-head outputs into a single d_model-dimensional vector for the next layer.
- **Scaled dot-product** (divide by sqrt(d_head)) keeps gradients stable during training.
- Multi-head attention is the core operation in every transformer layer, repeated H times in parallel.

---

Get the video walkthrough of per-head attention pattern analysis (which heads learn which relationships), the flash attention algorithm (tile-based online softmax, O(N) memory), and grouped-query attention (GQA): [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
