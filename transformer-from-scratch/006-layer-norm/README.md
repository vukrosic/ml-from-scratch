# LayerNorm From Scratch

Every transformer layer has a normalization step. LayerNorm is the most common — it normalizes each token's feature vector independently, without touching the batch dimension. Understanding it from scratch means understanding exactly what happens inside every transformer block.

In this lesson we build LayerNorm from raw PyTorch operations: compute mean and variance over the feature dimension, normalize, then apply a learned scale (gamma) and shift (beta). We'll also compare against PyTorch's built-in `nn.LayerNorm` and `nn.GroupNorm` to verify correctness.

---

## Hook: why LayerNorm instead of BatchNorm?

BatchNorm normalizes across the batch dimension — it needs enough samples to compute meaningful statistics. LayerNorm normalizes across the feature dimension — each sample gets its own normalization, no batch required. This matters for transformers because:

- Transformers process variable-length sequences where batch statistics can be noisy
- Recurrent models (RNNs, LSTMs) have no batch dimension in the same sense
- LayerNorm is the only choice for autoregressive models (GPT, Llama, etc.)

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

## Step 1: compute mean and variance

```python
mean = x.mean(dim=-1, keepdim=True)   # mean over feature axis (last dim)
var  = x.var(dim=-1, keepdim=True, unbiased=False)  # variance, not sample var
```

`dim=-1` means we always normalize over the feature dimension, whatever the batch shape. `unbiased=False` uses the population variance (divides by N), which is what PyTorch's `nn.LayerNorm` uses by default.

---

## Step 2: normalize

```python
x_norm = (x - mean) / torch.sqrt(var + eps)
```

This centers the features around 0 and scales them to unit variance. After this step, each feature has mean ≈ 0 and std ≈ 1 (approximately — the scale is also affected by gamma).

---

## Step 3: scale and shift

```python
return self.gamma * x_norm + self.beta
```

The learned gamma and beta allow the model to undo the normalization if it wants. If gamma equals the std and beta equals the mean, the output is identical to the input. The optimizer can discover whatever scale and shift work best.

---

## The full LayerNorm class

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

---

## Compare against PyTorch

```bash
python compare.py
```

This verifies that our output is numerically identical to `nn.LayerNorm` and `nn.GroupNorm(1)` (GroupNorm with 1 group is mathematically equivalent to LayerNorm over the feature dimension).

---

## Benchmark: our vs nn.LayerNorm

```bash
python benchmark.py
```

LayerNorm's cost is dominated by two reductions (mean + variance) and a few elementwise ops. For typical transformer dimensions (d_model >= 512), the cost is modest. The benchmark shows our pure-PyTorch version runs within ~1-3x of the highly optimized C++ implementation.

---

## Recap

- LayerNorm normalizes over the feature dimension only — each sample is independent
- Formula: `y = gamma * (x - mean) / sqrt(var + eps) + beta`
- gamma (scale) and beta (shift) are learnable parameters
- `eps` (usually 1e-5) prevents division by zero when variance is near zero
- nn.GroupNorm with `num_groups=1` is mathematically identical to LayerNorm
- RMSNorm (used in Llama, Mistral) removes the mean subtraction for a small speedup

---

Get the video walkthrough of RMSNorm internals, warmup statistics tracking, memory profiling across shapes and dtypes, and the gradient derivation for LayerNorm: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
