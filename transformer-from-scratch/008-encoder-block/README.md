# Encoder Block From Scratch

An encoder block is the building block of the transformer encoder. Stack N of them, add input embeddings and positional encoding, and you have a BERT model. Understanding this one module is the final step before the full transformer.

In this lesson we build an encoder block from the components we already have: multi-head attention and a feedforward network. The two new ideas are **residual connections** and **layer normalization** — and together they are what make deep transformers trainable.

---

## Hook: why residual connections and layer norm matter

A plain 50-layer CNN degrades accuracy not because the network is too expressive, but because the gradient vanishes through repeated multiplication. The same problem hit transformers: add more attention layers and training got harder, not easier.

Residual connections (ResNet, 2015) and layer normalization (Ba, Kiros, Hinton 2016) independently solved this. The transformer encoder block combines both. Without them, depth is unreachable. With them, you can stack 12, 24, even 100+ layers and train stably.

---

## The structure of an encoder block

An encoder block has two **sublayers** — each one wrapped in a residual connection and layer norm:

```
x → MultiHeadAttention → ResidualAdd(x, attn) → LayerNorm → FeedForward → ResidualAdd(x, ffn) → LayerNorm → output
```

Sublayer 1: **Multi-head self-attention** — each token attends to all other tokens.
Sublayer 2: **Feed-forward network** — two linear layers with GELU activation.

Both sublayers have the same input/output shape `(batch, seq_len, d_model)`, which is what makes residual connections work.

---

## Step 1: The FeedForward sublayer

Before attention, let's define the feedforward network. It is two linear layers with a GELU activation in between. The hidden dimension is typically 4x the model dimension.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FeedForward(nn.Module):
    """
    Two linear layers with GELU activation.
    Expands from d_model -> d_ff -> d_model.
    d_ff is typically 4 * d_model.
    """
    def __init__(self, d_model, d_ff=4 * d_model):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.gelu(self.linear1(x)))
```

The GELU (Gaussian Error Linear Unit) is a smooth, probabilistic activation — roughly like `0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))`. It is used in nearly every modern transformer (BERT, GPT, ViT) instead of ReLU.

---

## Step 2: ResidualAdd

A residual connection adds the input of a sublayer to its output. This creates a "skip path" that gradients can flow through directly, bypassing the sublayer entirely. The sublayer still learns — but training is stable even when it learns near-zero.

```python
class ResidualAdd(nn.Module):
    """
    Residual connection: output = x + sublayer(x)
    The skip path lets gradients flow unchanged through the sublayer.
    """
    def __init__(self, sublayer):
        super().__init__()
        self.sublayer = sublayer

    def forward(self, x):
        return x + self.sublayer(x)
```

The key insight: the sublayer output and the input `x` must have the **same shape** for addition to work. Multi-head attention and feedforward both preserve shape, so this is satisfied automatically.

---

## Step 3: LayerNorm

Layer normalization normalizes over the feature dimension (not the batch). For each token (each row in the `(batch, seq, d_model)` tensor), it computes mean and variance over the `d_model` features, then shifts and scales with learnable parameters.

```python
class LayerNorm(nn.Module):
    """
    Normalize over the feature dimension (last axis).
    For each (batch, seq) position: compute mean and std over d_model,
    then affine-transform with learnable gamma and beta.
    """
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        # x shape: (batch, seq, d_model)
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta
```

`eps` prevents division by zero when the variance is very small. `unbiased=False` uses the population variance (dividing by N), which is the standard convention for layer norm in transformers.

---

## Step 4: EncoderBlock — putting it together

Now we compose the full encoder block. Each sublayer is wrapped in `ResidualAdd` + `LayerNorm`:

```python
class EncoderBlock(nn.Module):
    """
    One transformer encoder block.
    Sublayer 1: Multi-head self-attention
    Sublayer 2: Feed-forward network
    Both are wrapped in: ResidualAdd → LayerNorm
    """
    def __init__(self, d_model, n_heads, d_ff=None):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        # Sublayer 1: attention with residual and norm
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Sublayer 2: feedforward with residual and norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x
```

Run it:

```bash
python encoder_block.py
```

---

## Why this architecture is trainable

The residual skip path is the critical design. Without it, backpropagation through N encoder blocks means N sequential multiplications — gradients explode or vanish exponentially. With it, gradients can flow directly from the output loss to any earlier layer, essentially "skipping" all intermediate layers.

Layer norm acts element-wise on each token's feature vector. Unlike batch norm, it doesn't depend on batch statistics, making it stable across variable sequence lengths and batch sizes.

Together: **residual connections handle gradient flow, layer norm handles activation statistics.**

---

## Compare: our EncoderBlock vs PyTorch's

PyTorch's `nn.TransformerEncoderLayer` and `nn.TransformerEncoder` implement the same idea. Run `compare.py` to verify our implementation matches PyTorch's output numerically:

```bash
python compare.py
```

The comparison checks that our `EncoderBlock` produces the same output as `nn.TransformerEncoderLayer` for random inputs, across multiple configurations.

---

## Benchmark: speed

```bash
python benchmark.py
```

Measures forward pass latency of our `EncoderBlock` against `nn.TransformerEncoderLayer` across several batch/model-size configurations.

---

## Recap

- **Encoder block** = two sublayers, each wrapped in `ResidualAdd → LayerNorm`
- **Sublayer 1**: Multi-head self-attention (Q=K=V=input)
- **Sublayer 2**: Feed-forward network (d_model → d_ff → d_model with GELU)
- **ResidualAdd**: `output = x + sublayer(x)` — creates a skip path for gradient flow
- **LayerNorm**: normalizes over features per token, with learnable scale and shift
- Stacking N encoder blocks gives you a full transformer encoder (BERT architecture)

---

Get the video walkthrough of sublayer wrapper pattern, deep encoder stacks (depth vs training stability), and profiling across model sizes: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)