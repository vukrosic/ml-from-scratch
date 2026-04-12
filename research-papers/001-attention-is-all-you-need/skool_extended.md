# Attention Is All You Need — Extended Notebook Content
## Skool $49 — Open SuperIntelligence Lab

This document goes beyond the free lesson with deeper analysis, annotated equations,
ablation studies, and comparisons to modern improvements.  The companion Jupyter
notebook (`extended_notebook.ipynb`) runs all experiments top-to-bottom.

---

## 1. Annotated Key Equations

### Equation 1 — Scaled Dot-Product Attention

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

**What it means:** Each query vector Q (shape `(seq_len, d_k)`) computes a dot product
with every key vector K, producing a `(seq_len, seq_len)` score matrix.  Dividing by
`sqrt(d_k)` prevents the softmax from saturating when d_k is large.  Softmax normalises
each row of scores into a probability distribution over all positions.  The final output
is a weighted sum of value vectors V, where the weights are the attention probabilities.

**Why `sqrt(d_k)`?** If Q and K have independent entries with mean 0 and variance 1,
their dot product has mean 0 and variance d_k.  Without the sqrt, the softmax inputs
grow with d_k, pushing the output toward a one-hot distribution (very small gradients).
With `sqrt(d_k)` scaling, the variance stays constant at 1 regardless of d_k.

**Code verification:**
```python
d_k = 64
Q = torch.randn(10, d_k)          # 10 queries, d_k=64
K = torch.randn(10, d_k)
scores = torch.matmul(Q, K.T)    # (10, 10)
var_before = scores.var()
scores_scaled = scores / math.sqrt(d_k)
var_after = scores_scaled.var()
print(f"Variance before scaling: {var_before:.2f}")   # ~64
print(f"Variance after scaling : {var_after:.2f}")   # ~1
```

**Gradient flow:** During backpropagation, gradients flow through three paths:
- Through the softmax weights (attention weights → upstream activations)
- Through the value projection (gradient × attention weights)
- Through the key/query projections

A well-known numerical issue: if scores become `inf` (e.g., from large products),
`softmax` produces `nan`.  This is the primary cause of training instability in
large Transformers without careful initialization (e.g., GPT-3 uses modified init).

---

### Equation 2 — Multi-Head Attention

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W_O
head_i = Attention(Q W_q^i, K W_k^i, V W_v^i)
```

**What it means:** Instead of doing attention once in a d_model=512 space, the model
splits the space into H=8 independent heads of dimension d_k=64.  Each head has its
own W_q, W_k, W_v projections.  After running attention in each head, outputs are
concatenated (8 × 64 = 512) and passed through a final linear projection W_O.

**Why split?** A single attention head can only learn one type of relationship at
a time due to the bottleneck of d_k.  With 8 heads, the model can simultaneously
track: syntactic subject-verb agreement, coreference chains, positional patterns,
semantic similarity, and more.  This is an empirical finding — visualisation of
attention heads in trained models consistently shows specialisation.

**Memory complexity:** Multi-head attention stores O(b × h × n²) attention weights.
For batch_size=32, n_heads=8, seq_len=512: 32×8×512² ≈ 64M floats ≈ 256 MB.
This is a key bottleneck for long sequences (see Flash Attention for a fix).

---

### Sinusoidal Positional Encoding — Section 3.5

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**What it means:** Position `pos` is encoded as a vector of length `d_model` where
even indices use sine, odd indices use cosine, at frequencies that form a geometric
progression from `1/10000^0` to `1/10000^1`.  The highest frequency (i=0) is
`sin(pos)`, `cos(pos)` — fine-grained position within a single token.  The lowest
frequency (i=d_model/2) varies over 10000 tokens — coarse position over the whole
sequence.

**Relative position intuition:** Because `sin(a+b) = sin(a)cos(b) + cos(a)sin(b)`,
the encoding for position `pos + delta` can be expressed as a linear combination
of the encoding for position `pos`.  This gives the model an algebraic way to reason
about relative distances, not just absolute positions.

**Code to inspect the encoding:**
```python
import matplotlib.pyplot as plt
d_model = 512
max_len = 200
pe = SinusoidalPositionalEncoding(d_model, max_len).pe[0]  # (200, 512)

# Plot heatmap of positional encoding
plt.imshow(pe.T, aspect='auto', cmap='RdBu_r')
plt.xlabel('Position'); plt.ylabel('Dimension')
plt.title('Sinusoidal Positional Encoding')
plt.colorbar()
```

---

## 2. Ablation Study

We remove one component at a time and measure the impact on validation perplexity
and BLEU score.  This tells us which architectural choices actually matter for this
task.

### Ablation: Remove Positional Encoding

If we add positional encodings and then set them to zero, we can measure the cost
of positional information:

```python
class PositionalEncodingAblation(nn.Module):
    def __init__(self, d_model, max_seq_len=5000, zero=False):
        super().__init__()
        self.zero = zero
        self.pe = SinusoidalPositionalEncoding(d_model, max_seq_len)

    def forward(self, seq_len):
        out = self.pe(seq_len)
        return torch.zeros_like(out) if self.zero else out
```

**Expected observation:** Without positional encoding, the model cannot distinguish
token order — "hello world" and "world hello" produce identical embeddings.  Perplexity
increases by 20-40% on this task because the decoder loses word order information.

### Ablation: Reduce from 8 to 1 Attention Head

```python
model_1head = Transformer(n_heads=1, ...)  # instead of n_heads=8
```

**Expected observation:** Single-head attention forces all query-key-value relationships
to compete in one 512-dimensional space.  Perplexity typically increases by 10-20%.
However, on very small tasks (like ours), 1 head may sometimes perform comparably because
the dataset is too small to benefit from head specialisation.

### Ablation: Remove Residual Connections

Residual connections are the highway for gradient flow in deep networks.  Without them,
backpropagating through 6 layers of encoder + 6 layers of decoder becomes extremely
difficult (the vanishing gradient problem).

```python
class EncoderBlockNoResidual(nn.Module):
    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = self.norm1(x + attn_out)  # only residual on LayerNorm, not attention
        # Remove: x = x + attn_out
        ...
```

**Expected observation:** Training becomes unstable or diverges entirely after 2-3 epochs.
Validation loss remains high.  This is one of the most impactful ablations.

### Ablation: Remove LayerNorm (replace with Identity)

```python
self.norm1 = nn.Identity()  # instead of nn.LayerNorm(d_model)
```

**Expected observation:** Training diverges.  LayerNorm normalises activations to
mean=0, variance=1, preventing intermediate activations from exploding in magnitude
through deep stacks.  Without it, residual branches accumulate unnormalised scale.

### Summary Table (approximate on our toy task)

| Ablation | Val PPL increase | Notes |
|---|---|---|
| Remove positional encoding | +30-50% | Loses word order entirely |
| Reduce to 1 head | +15-25% | Cannot specialise attention patterns |
| Remove residual connections | diverges | Gradient flow breaks down |
| Remove LayerNorm | diverges | Activation scale explodes |
| d_ff=512 (instead of 2048) | +20% | Feed-forward capacity too low |
| n_layers=2 (instead of 6) | +40% | Model too shallow for compositionality |

---

## 3. Modern Improvements Since the Paper

The original Transformer has been extended in several important ways.  We compare
the most impactful changes.

### RoPE — Rotary Positional Embedding (Su et al., 2022)

**Problem with sinusoidal/learned positional encodings:** They are added to token
embeddings, making position information interact with token content in an unstructured
way.  Attention scores QK^T mix position and content information in ways that are
hard to extrapolate beyond the training sequence length.

**RoPE solution:** Instead of adding positional encodings to embeddings, rotate the
Q and K vectors within each attention head so that the dot product naturally encodes
relative position.  Specifically, for dimension pair (2i, 2i+1) in a d_k-dimensional
vector, apply a rotation by angle `theta = pos / 10000^(2i/d_k)`:

```
Q_rotated = Q * cos(theta) + rotate_half(Q) * sin(theta)
K_rotated = K * cos(theta) + rotate_half(K) * sin(theta)
Attention = (Q_rotated)(K_rotated)^T
```

The dot product `Q_rotated · K_rotated` now contains `cos(theta_i - theta_j)` —
an explicit function of relative position `i-j`, completely decoupled from content.

**Why it matters:** RoPE scales better to long sequences because the position
information is baked into the attention computation itself, not just added to
embeddings.  Llama, PaLM-2, and Mistral all use RoPE.

```python
def rotate_half(x):
    """Rotate half the dimensions."""
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    """Apply rotary positional embedding to Q and K."""
    # q, k: (batch, n_heads, seq_len, d_k)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)
```

### ALiBi — Attention with Linear Biases (Press et al., 2021)

**Problem:** Learned and sinusoidal positional encodings both require the model to
learn to ignore position (when doing relative position reasoning) and extrapolate to
longer sequences than seen during training.

**ALiBi solution:** Add a linear bias to the attention scores, proportional to the
distance between query and key positions.  No learned positional embeddings at all.

```
score_{i,j} = Q_i · K_j - |i - j| * m
```

where `m` is a head-specific slope (different for each head).  ALiBi naturally
extrapolates to longer sequences because the distance penalty applies at any distance.

**Comparison:** ALiBi performs comparably to RoPE on most benchmarks.  Both are
strict improvements over sinusoidal encoding for long-context tasks.  Neither is
clearly "better" — both are used in production models.

### Flash Attention (Dao et al., 2022)

**Problem:** Standard attention stores the full `(seq_len, seq_len)` attention matrix
in GPU HBM (high-bandwidth memory).  For seq_len=2048, that's 2048² = 4M floats per
head, 32M floats for 8 heads.  This is memory-bound and slow.

**Flash Attention solution:** Compute attention in tiles that fit in SRAM, materialising
the full matrix only implicitly.  Uses the "tiling" and "online softmax" tricks.
Delivers 2-4× speedup and logarithmic memory usage with respect to sequence length.

Flash Attention does not change the mathematical output of attention — it's a
numerical implementation optimisation.  All modern large models use it.

### Comparison Table

| Technique | What it replaces | Key benefit | Used in |
|---|---|---|---|
| RoPE | Sinusoidal/Learned PE | Better long-context extrapolation | Llama, Mistral, PaLM-2 |
| ALiBi | Sinusoidal/Learned PE | Simple, no extra params, extrapolates | MPT, Bloom |
| Flash Attention | Standard attention impl | 2-4× speedup, memory efficient | GPT-4, Llama 2, Claude |
| GLU Variants | Standard FFN ReLU | Better performance on language tasks | PaLM, Llama, Mistral |

---

## 4. Scaling Behavior Discussion

The original paper (2017) trained on WMT translation with d_model=512, n_layers=6.
Modern large language models scale to d_model=4096+, n_layers=32-96, and train on
trillions of tokens.  Understanding scaling laws helps predict performance.

### The Chinchilla Scaling Laws (Hoffmann et al., 2022)

Given a fixed compute budget C (in FLOPs), the optimal model size and training tokens
scale as:

```
N_opt ≈ C^0.50    (parameters)
D_opt ≈ C^0.50    (training tokens)
```

The original GPT-3 paper (2020) used N^0.73 / D^0.27 — overallocating to parameters
and underallocating to data.  Chinchilla showed this was suboptimal.  Llama 2 and
Mistral follow the Chinchilla scaling more closely.

### How Parameters Scale with Architecture

For a Transformer with n_layers, d_model, n_heads, d_ff:

```
Embedding params:   2 * vocab_size * d_model
Encoder params:     n_layers * (4 * d_model^2 + 4 * d_model * d_ff)
Decoder params:     n_layers * (4 * d_model^2 + 4 * d_model * d_ff)
Output proj params: d_model * vocab_size
Total ≈ 40M for base Transformer (d_model=512, d_ff=2048, n_layers=6)
```

For GPT-3 scale (d_model=12288, n_layers=96, d_ff=4*d_model=49152, vocab=50257):
```
≈ 175B parameters
```

### Critical Scaling Observations

**1. Compute grows quadratically with sequence length**
Each attention head computes an n×n matrix.  Doubling sequence length quadruples
attention compute.  This is why most LLMs use 2048-4096 context windows — beyond
that, the attention computation dominates.  Flash Attention reduces this to O(n).

**2. Feed-forward dominates at large scale**
The FFN has 4× more parameters than the attention projections (d_ff ≈ 4*d_model).
At GPT-3 scale: FFN ≈ 120B params, attention ≈ 30B params.  Most of the model's
knowledge is stored in the FFN layers.

**3. Batch size scaling**
Large models benefit from larger batch sizes (up to millions of tokens per batch).
The optimal batch size scales with model size: GPT-3 (175B) used batch size 3.2M tokens.
Smaller models (our 40M) typically use 16K-128K tokens per batch.

**4. Precision considerations**
- float32 (32-bit): full precision, most stable
- float16 (16-bit): 2× memory/bandwidth, requires loss-scaling to avoid underflow
- bfloat16: Google's format, better dynamic range than float16, used in TPU and A100
- int8/fp8 quantization: Post-training or QAT to reduce memory further

---

## 5. Additional Experiments (run in the notebook)

### Experiment: Vary d_model and plot convergence

```python
configs = [
    {"d_model": 128, "n_heads": 4, "d_ff": 512,  "n_layers": 4},
    {"d_model": 256, "n_heads": 8, "d_ff": 1024, "n_layers": 6},
    {"d_model": 512, "n_heads": 8, "d_ff": 2048, "n_layers": 6},
]
# Train each for 60 epochs, plot validation loss curves on the same axes
# Expected: larger models converge faster and to lower loss
```

### Experiment: Attention weight heatmap per layer

```python
# Hook into the attention weights of each head at each layer
# Plot as a grid of heatmaps: (n_layers, n_heads)
# Expected: early layers show local patterns; later layers show longer-range dependencies
```

### Experiment: Gradient magnitude per layer

```python
# Log gradient norm of each parameter tensor after each step
# Expected: gradients are largest in shallower layers; vanish in deep layers without residuals
```

---

## 6. Further Reading

- Vaswani et al. (2017) — "Attention Is All You Need" — [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
- Dosovitskiy et al. (2021) — "An Image is Worth 16x16 Words: Transformers for Image Recognition" — ViT
- Su et al. (2022) — "RoFormer: Enhanced Transformer with Rotary Position Embedding" — [arXiv:2104.09864](https://arxiv.org/abs/2104.09864)
- Press et al. (2021) — "Train Short, Test Long: Attention with Linear Biases" — [arXiv:2108.12409](https://arxiv.org/abs/2108.12409)
- Dao et al. (2022) — "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness" — [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)
- Hoffmann et al. (2022) — "Training Compute-Optimal Large Language Models" (Chinchilla) — [arXiv:2203.15556](https://arxiv.org/abs/2203.15556)
