# LayerNorm From Scratch

Every transformer layer has a normalization step. Without it, activations explode or collapse as signals pass through dozens of layers, and training becomes impossible. LayerNorm is the most common fix — it normalizes each token's feature vector independently, keeping numbers in a healthy range no matter how deep the network goes.

In this lesson we build LayerNorm from raw PyTorch operations, understand why it replaced BatchNorm for transformers, implement RMSNorm (what LLaMA and Mistral actually use), and explore the design choices that determine whether a 96-layer model trains smoothly or diverges on step 1.

## What we build
- LayerNorm from scratch (mean + variance normalization)
- RMSNorm from scratch (variance-only normalization)
- Correctness verification against PyTorch's `nn.LayerNorm`
- Forward/backward benchmarks
- Memory and compute profiling across shapes and dtypes

## Files
- `layer_norm.py` — our LayerNorm implementation
- `rms_norm.py` — our RMSNorm implementation
- `compare.py` — numerical comparison against `nn.LayerNorm` and `nn.GroupNorm`
- `benchmark.py` — forward/backward timing
- `profile.py` — memory and compute profiling across shapes
- `warmup_layernorm.py` — experimental warmup statistics tracking

---

## Why normalization is essential

Imagine a simple network: 50 layers, each multiplying by a weight matrix. If the average singular value of those matrices is even slightly above 1.0 — say 1.05 — then after 50 layers:

```
1.05^50 = 11.47       # activations grow 11x
1.1^50  = 117.39      # activations grow 117x
1.5^50  = 637621.5    # activations are in the millions
2.0^50  = 1.13e15     # activations hit 1e15 — overflow in float16
```

If the average is below 1.0:

```
0.95^50 = 0.0769      # activations shrink to ~8%
0.9^50  = 0.00515     # activations shrink to 0.5%
0.5^50  = 8.88e-16    # activations are effectively zero
```

This is the **exploding/vanishing activation problem**. The exact same thing happens to gradients during backpropagation — they flow through the same chain of multiplications in reverse.

Normalization fixes this by resetting the scale at every layer:

```
Layer 1 output:  mean=3.2, std=14.7   -->  LayerNorm  -->  mean~0, std~1
Layer 2 output:  mean=-1.8, std=22.1  -->  LayerNorm  -->  mean~0, std~1
Layer 3 output:  mean=0.4, std=0.03   -->  LayerNorm  -->  mean~0, std~1
   ...
Layer 50 output: mean=7.1, std=31.4   -->  LayerNorm  -->  mean~0, std~1
```

No matter how much the weights amplify or shrink the signal, normalization brings it back. Training stays stable because every layer sees inputs in a predictable range.

---

## Hook: why LayerNorm instead of BatchNorm?

BatchNorm normalizes across the batch dimension — it needs enough samples to compute meaningful statistics. LayerNorm normalizes across the feature dimension — each sample gets its own normalization, no batch required. This matters for transformers because:

- Transformers process variable-length sequences where batch statistics can be noisy
- At inference time, batch size is often 1 — BatchNorm statistics are meaningless
- LayerNorm is the only choice for autoregressive models (GPT, LLaMA, etc.)

---

## BatchNorm vs LayerNorm vs RMSNorm

These three normalizations differ in **which dimensions** they compute statistics over. Here is a tensor with shape `(batch=3, seq=4, d_model=5)`:

```
BatchNorm: normalize over batch + seq dimensions (each feature independently)
           Stats computed: one mean and one variance PER FEATURE

           Batch 0:  [ x x x x x ]     Compute mean/var
           Batch 1:  [ x x x x x ]  <-- down this column
           Batch 2:  [ x x x x x ]     for each of d_model features
                       |         |
                     feat_0   feat_4
           Stats shape: (d_model,) = (5,)

LayerNorm: normalize over d_model dimension (each token independently)
           Stats computed: one mean and one variance PER TOKEN

           Token [2,1]:  [ x  x  x  x  x ]
                           |--------------|
                           mean/var across
                           these 5 features
           Stats shape: (batch, seq) = (3, 4) --> 12 separate normalizations

RMSNorm:   same dimensions as LayerNorm, but only computes RMS (no mean)
           Stats computed: one RMS value PER TOKEN

           Token [2,1]:  [ x  x  x  x  x ]
                           |--------------|
                           sqrt(mean(x^2))
                           across features
           Stats shape: (batch, seq) = (3, 4) --> 12 separate normalizations
```

The key difference: **BatchNorm couples samples in a batch** (every sample's normalization depends on every other sample). **LayerNorm treats each token independently** — the normalization of token [2,1] depends only on its own 5 feature values, not on any other token in the batch.

| Property          | BatchNorm              | LayerNorm          | RMSNorm            |
|-------------------|------------------------|--------------------|--------------------|
| Normalizes over   | batch + spatial dims   | feature dim        | feature dim        |
| Stats per         | feature                | token              | token              |
| Centers to zero   | yes                    | yes                | **no**             |
| Scales to unit    | yes (variance)         | yes (variance)     | yes (RMS)          |
| Running stats     | yes (inference mode)   | no                 | no                 |
| Batch dependency  | **yes**                | no                 | no                 |
| Learnable params  | gamma, beta            | gamma, beta        | gamma only         |
| Used in           | CNNs (ResNet, etc.)    | Original Transformer, BERT | LLaMA, Mistral, Gemma |

---

## The LayerNorm formula

For each feature vector x (shape: `d_model`), LayerNorm computes:

```
mean  = mean(x)
var   = var(x)
x_hat = (x - mean) / sqrt(var + eps)
y     = gamma * x_hat + beta
```

Where:
- `gamma` (scale) starts at 1, learned during training
- `beta` (shift) starts at 0, learned during training
- `eps` prevents division by zero (typically 1e-5)

The key is that mean and variance are computed over the feature dimension only — not across the batch.

---

## Piece 1: compute mean and variance

```python
mean = x.mean(dim=-1, keepdim=True)   # mean over feature axis (last dim)
var  = x.var(dim=-1, keepdim=True, unbiased=False)  # variance, not sample var
```

`dim=-1` means we always normalize over the feature dimension, whatever the batch shape. `unbiased=False` uses the population variance (divides by N), which is what PyTorch's `nn.LayerNorm` uses by default.

**Why population variance (N) and not sample variance (N-1)?** Because we are not estimating the variance of a population from a sample. We have the complete feature vector — all N values — and we want the variance of exactly those N numbers. Using `unbiased=True` (the PyTorch default for `.var()`) would divide by N-1 and give slightly different results than `nn.LayerNorm`.

---

## Piece 2: normalize

```python
x_norm = (x - mean) / torch.sqrt(var + eps)
```

This centers the features around 0 and scales them to unit variance. After this step, each token's feature vector has mean approximately 0 and std approximately 1 (before gamma and beta are applied).

---

## Piece 3: scale and shift

```python
return self.gamma * x_norm + self.beta
```

The learned gamma and beta allow the model to undo the normalization if it wants. If gamma equals the original std and beta equals the original mean, the output is identical to the input. The optimizer can discover whatever scale and shift work best.

**Why bother normalizing if the model can just undo it?** Because the normalization makes the optimization landscape smoother. Even if the final learned gamma and beta partially undo the centering, the gradients flow much more cleanly through the normalized path during training.

---

## Piece 4: the full LayerNorm class

```python
class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta
```

Run it:

```bash
python layer_norm.py
```

```
Input shape:  torch.Size([4, 8])
Output shape: torch.Size([4, 8])
Output mean per sample (should be ~0): tensor([ 3.7253e-08, -1.4901e-08, ...])
Output std  per sample (should be ~1): tensor([1.0690, 1.0690, ...])
```

The means are essentially zero (floating-point noise). The stds are close to 1 — not exactly 1 because `std()` uses N-1 by default, while we normalized with N.

---

## Compare against PyTorch

```bash
python compare.py
```

This verifies that our output is numerically identical to `nn.LayerNorm` across multiple shapes and dtypes (float32, float16, bfloat16). It also checks `nn.GroupNorm(1)` — GroupNorm with 1 group is mathematically equivalent to LayerNorm over the feature dimension.

```
Comparing our LayerNorm vs nn.LayerNorm
============================================================

dtype: torch.float32 (tol=1e-05)
  shape: (2, 4, 8)
    ours vs nn.LayerNorm: max_diff=0.00e+00  [PASS]
  shape: (1, 16, 32)
    ours vs nn.LayerNorm: max_diff=0.00e+00  [PASS]
  ...
```

---

## Pre-norm vs post-norm

The original "Attention Is All You Need" transformer uses **post-norm**: the normalization comes *after* the sublayer (attention or feedforward):

```
POST-NORM (original transformer, 2017)

  x -----> [ Attention ] ----> [ Add ] ----> [ LayerNorm ] ----> output
  |                              ^
  |______________________________|
          residual connection

  In code:  output = LayerNorm(x + Attention(x))
```

Modern LLMs (GPT-2, GPT-3, LLaMA, Mistral, Gemma) use **pre-norm**: the normalization comes *before* the sublayer:

```
PRE-NORM (modern LLMs)

  x -----> [ LayerNorm ] ----> [ Attention ] ----> [ Add ] ----> output
  |                                                  ^
  |__________________________________________________|
                   residual connection

  In code:  output = x + Attention(LayerNorm(x))
```

Why pre-norm wins for deep models:

```
POST-NORM residual stream:
  x0 -> LN(x0 + attn(x0)) -> LN(prev + ff(prev)) -> ...
  The residual connection is INSIDE the LayerNorm. Gradients must flow
  through the normalization at every layer. After 96 layers, gradients
  get distorted.

PRE-NORM residual stream:
  x0 -> x0 + attn(LN(x0)) -> prev + ff(LN(prev)) -> ...
  The residual connection is a CLEAN ADDITION. Gradients flow straight
  through the addition without touching any normalization. The gradient
  highway is unobstructed.
```

The practical difference: post-norm requires careful learning rate warmup and can diverge for deep models (48+ layers). Pre-norm trains stably even at 96 or 128 layers with minimal warmup. This is why every modern LLM uses pre-norm.

---

## Piece 5: RMSNorm implementation

LLaMA, Mistral, and Gemma all use RMSNorm instead of LayerNorm. The insight: the mean subtraction in LayerNorm mostly just centers the features around zero. The important part is the *scaling* — keeping feature magnitudes under control. RMSNorm drops the mean subtraction entirely:

```
LayerNorm:  y = (x - mean(x)) / sqrt(var(x) + eps) * gamma + beta
RMSNorm:    y = x / sqrt(mean(x^2) + eps) * gamma
```

Two simplifications:
1. No mean subtraction (saves one reduction operation)
2. No beta parameter (one fewer learnable parameter per layer)

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        # RMS = sqrt(mean(x^2)) over feature dimension
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x_norm = x / rms
        return self.gamma * x_norm
```

Run it:

```bash
python rms_norm.py
```

```
Output RMS per sample (should be ~1): tensor([1.0000, 1.0000, 1.0000, 1.0000])
Note: output mean is NOT forced to 0 (that's RMSNorm, not LayerNorm)
```

**How much faster is RMSNorm?** It saves one reduction (`mean`) and one elementwise subtraction per normalization call. In a 32-layer model with both attention and FFN norms, that is 64 norm calls per forward pass. The saving is roughly 10-15% on the normalization step, which translates to a 1-3% speedup on total model forward time. Small per layer, but it compounds.

---

## Gradient flow analysis

Normalization does not just help activations — it fundamentally changes how gradients flow. Consider what happens during backpropagation without normalization:

```
WITHOUT LayerNorm:
  Layer 50 gradient: ||g|| = 1.0
  Layer 40 gradient: ||g|| = 0.13    (decaying)
  Layer 30 gradient: ||g|| = 0.017
  Layer 20 gradient: ||g|| = 0.002
  Layer 10 gradient: ||g|| = 0.0003
  Layer  1 gradient: ||g|| = 0.00004  <-- 25000x smaller than layer 50!

  Early layers barely learn. Training is slow and unstable.
```

```
WITH LayerNorm (pre-norm):
  Layer 50 gradient: ||g|| = 1.0
  Layer 40 gradient: ||g|| = 0.95
  Layer 30 gradient: ||g|| = 0.91
  Layer 20 gradient: ||g|| = 0.87
  Layer 10 gradient: ||g|| = 0.82
  Layer  1 gradient: ||g|| = 0.78   <-- only 1.3x smaller than layer 50

  All layers learn at roughly the same rate. Training is stable.
```

Why does this happen? LayerNorm acts as a gradient "normalizer" during backpropagation too. The Jacobian of the LayerNorm operation has a bounded spectral norm — it cannot amplify or shrink gradients by more than a constant factor. This is the key mathematical property that makes deep transformer training possible.

With pre-norm specifically, the residual connection creates a **gradient highway**:

```
d(loss)/d(x_layer_1) = d(loss)/d(x_layer_50) * [ I + small_corrections ]

The identity matrix I means gradients pass through almost unchanged.
The small_corrections come from each layer's contribution.
Neither the identity path nor the corrections go through LayerNorm.
```

---

## The eps parameter

The `eps` (epsilon) parameter prevents division by zero when the variance is near zero. It defaults to `1e-5` in most implementations.

**When does near-zero variance happen?**

```
Scenario 1: A token whose features are all nearly identical
  x = [3.001, 3.002, 2.999, 3.001, 3.000]
  var(x) = 0.0000012
  sqrt(var(x)) = 0.0011
  sqrt(var(x) + 1e-5) = 0.0011  <-- eps barely matters

Scenario 2: A token whose features are exactly identical (padding tokens, etc.)
  x = [0.0, 0.0, 0.0, 0.0, 0.0]
  var(x) = 0.0
  sqrt(var(x)) = 0.0              <-- division by zero!
  sqrt(var(x) + 1e-5) = 0.00316   <-- eps saves us
```

**What if eps is too large?** If you set `eps=1.0` (absurdly large), the normalization becomes too weak:

```
x = [10.0, -5.0, 3.0, 8.0, -2.0]
var(x) = 30.16

With eps=1e-5:  sqrt(30.16 + 1e-5) = 5.49  --> proper normalization
With eps=1.0:   sqrt(30.16 + 1.0)  = 5.58  --> slightly weaker (OK here)

x = [0.01, -0.02, 0.03, 0.01, -0.01]
var(x) = 0.000296

With eps=1e-5:  sqrt(0.000296 + 1e-5) = 0.0175  --> proper normalization
With eps=1.0:   sqrt(0.000296 + 1.0)  = 1.0001  --> normalization does nothing!
```

With too-large eps, small-variance tokens pass through essentially unnormalized, defeating the purpose. The standard `1e-5` for float32 and `1e-6` for float64 work well. Some bfloat16 implementations use `1e-6` or `1e-8` — check what your framework defaults to.

---

## Fused kernel optimization

Our pure-Python LayerNorm does this:

```
Step 1: Read x from memory, compute mean, write mean        (read x once)
Step 2: Read x from memory, read mean, compute var, write var  (read x again)
Step 3: Read x, mean, var from memory, compute normalized, write result (read x a third time)
Step 4: Read normalized, read gamma/beta, compute output, write output (read/write again)
```

That is **4 memory reads** of the full tensor `x`. On a GPU, memory bandwidth is the bottleneck — the actual arithmetic is nearly free by comparison.

A fused kernel does everything in one pass:

```
Fused: Read x once, compute mean + var + normalize + scale/shift, write output
       (read x ONCE, write output ONCE)
```

This is why NVIDIA's Apex `FusedLayerNorm` and PyTorch's built-in `nn.LayerNorm` (which calls a C++/CUDA kernel) are significantly faster than our pure-PyTorch version:

```
Memory reads/writes comparison:
  Pure PyTorch:    4 reads of x + intermediate writes = ~8 memory ops
  Fused kernel:    1 read of x + 1 write of output    = 2 memory ops
  Speedup:         ~2-4x (memory-bound operations)
```

For large models, NVIDIA provides `apex.normalization.FusedLayerNorm` and `apex.normalization.FusedRMSNorm` as drop-in replacements. PyTorch 2.0+ also supports `torch.compile()` which can automatically fuse these operations.

---

## Benchmark: our vs nn.LayerNorm

```bash
python benchmark.py
```

```
d_model=512, seq_len=64
 batch  ours fwd (us)  nn.LN fwd (us)  ours bwd (us)  nn.LN bwd (us)
---------------------------------------------------------------------------
     1          42.15           18.73          98.42           45.21  (fwd 2.25x, bwd 2.18x)
     4         112.34           52.18         287.51          131.42  (fwd 2.15x, bwd 2.19x)
    16         438.72          198.45        1142.31          523.18  (fwd 2.21x, bwd 2.18x)
```

Our pure-PyTorch version runs within ~2-3x of the highly optimized C++ implementation. The gap comes from the fused kernel advantage described above. The goal is understanding, not beating highly optimized C++ kernels.

---

## Profile across shapes and dtypes

```bash
python profile.py
```

This compares LayerNorm, GroupNorm(1), and RMSNorm across multiple tensor shapes and dtypes (float32, float16). Key observations:

- LayerNorm and GroupNorm(1) have identical compute cost (same formula)
- RMSNorm saves one mean-reduction, but the difference is small in pure PyTorch
- Memory is dominated by the input tensor itself, not normalization overhead
- float16 is roughly 2x faster than float32 for large tensors on CPU; on GPU the gap is even larger

---

## Common mistakes

### Mistake 1: Wrong dimension for normalization

```python
# WRONG: normalizes over batch dimension (like BatchNorm, not LayerNorm)
mean = x.mean(dim=0, keepdim=True)

# WRONG: normalizes over all dimensions at once
mean = x.mean()

# CORRECT: normalizes over feature dimension (last dim)
mean = x.mean(dim=-1, keepdim=True)
```

If you normalize over the wrong dimension, the model might still train (badly), and the bug is very hard to catch because the shapes often still work out.

### Mistake 2: Biased vs unbiased variance

```python
# WRONG: PyTorch .var() defaults to unbiased=True (divides by N-1)
var = x.var(dim=-1, keepdim=True)  # unbiased=True by default!

# CORRECT: use population variance (divides by N) to match nn.LayerNorm
var = x.var(dim=-1, keepdim=True, unbiased=False)
```

Using `unbiased=True` gives slightly different results than `nn.LayerNorm`. For `d_model=768`, the difference is tiny (768 vs 767 in the denominator). For `d_model=4`, it is significant (4 vs 3 in the denominator). Always use `unbiased=False` to match the standard.

### Mistake 3: Forgetting that gamma and beta are learnable

```python
# WRONG: hardcoded scale/shift (not learnable)
self.gamma = torch.ones(d_model)
self.beta = torch.zeros(d_model)

# CORRECT: nn.Parameter so they get gradients and update during training
self.gamma = nn.Parameter(torch.ones(d_model))
self.beta = nn.Parameter(torch.zeros(d_model))
```

Without `nn.Parameter`, the optimizer never sees gamma and beta. The model still runs (with fixed scale=1 and shift=0), but it cannot learn to adjust the normalization — you lose representational power.

### Mistake 4: Normalizing the wrong tensor shape

```python
# Input: (batch, seq_len, d_model) = (2, 128, 768)

# WRONG: LayerNorm(128) — normalizing over seq_len
ln = nn.LayerNorm(128)

# CORRECT: LayerNorm(768) — normalizing over d_model
ln = nn.LayerNorm(768)

# ALSO CORRECT: LayerNorm([128, 768]) — normalizing over both seq and features
# This is valid but uncommon in transformers
ln = nn.LayerNorm([128, 768])
```

The `normalized_shape` argument to `nn.LayerNorm` must match the trailing dimensions of your input. For transformers, this is always `d_model`.

### Mistake 5: Applying LayerNorm before the residual connection (in post-norm)

```python
# WRONG (common in post-norm): normalizing the sublayer output, then adding residual
output = x + LayerNorm(Attention(x))

# CORRECT post-norm: add residual first, then normalize the sum
output = LayerNorm(x + Attention(x))

# CORRECT pre-norm: normalize first, then sublayer, then add residual
output = x + Attention(LayerNorm(x))
```

Getting the order wrong in post-norm means the residual connection bypasses normalization, which changes the training dynamics significantly.

---

## Recap

- **Why normalize**: Without it, activations explode/collapse through deep layers. Values hit 1e15 or 1e-15 after just 50 layers.
- **LayerNorm formula**: `y = gamma * (x - mean) / sqrt(var + eps) + beta` — normalizes over feature dim, each token independently.
- **BatchNorm vs LayerNorm**: BatchNorm couples samples in a batch and needs running stats at inference. LayerNorm is per-token and batch-independent.
- **RMSNorm**: Drops mean subtraction. Used by LLaMA, Mistral, Gemma. Simpler, ~10-15% faster on the norm step.
- **Pre-norm vs post-norm**: Pre-norm (modern LLMs) puts LayerNorm before the sublayer, creating a clean gradient highway through residual connections.
- **eps**: Prevents division by zero. 1e-5 is standard. Too large defeats normalization for low-variance tokens.
- **Fused kernels**: Read the tensor once instead of 4 times. 2-4x faster. Use `nn.LayerNorm`, Apex FusedLayerNorm, or `torch.compile()`.
- **gamma and beta are learnable** — they let the model adjust normalization. Without `nn.Parameter`, they are frozen constants.

---

Get the video walkthrough of RMSNorm internals, warmup statistics tracking, memory profiling across shapes and dtypes, and the gradient derivation for LayerNorm: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
