# FeedForward Networks From Scratch

Every transformer layer has two parts: attention and the feedforward network. The FFN is what gives the model its capacity to learn complex functions — it's two linear layers with a non-linear activation in between. Despite its simplicity, it accounts for roughly two-thirds of a transformer's parameters.

In this lesson we build a FFN from scratch in pure PyTorch. No `nn.Sequential`. We'll implement the GELU activation from its mathematical definition, then stack Linear → GELU → Linear, exactly as GPT and BERT do.

---

## Hook: why the FFN matters

A transformer layer is: `output = attention(input) + FFN(input)`. The attention block lets tokens talk to each other. The FFN lets each token think for itself. Without the FFN, a transformer would just be a complicated weighted sum — powerful for retrieval, but limited for computation.

The FFN is also where most of the model's memory and parameters live. In a typical transformer, the FFN accounts for about 2/3 of the total parameters. When you scale up a model, most of the new capacity goes to the FFN.

---

## GELU: the activation transformers use

Before we build the FFN, we need an activation function. GPT-2, BERT, and most modern transformers use **GELU** (Gaussian Error Linear Unit), not ReLU. Unlike ReLU which is zero for negative inputs, GELU smoothly gates negative values based on their magnitude.

The exact formula is `GELU(x) = x * Φ(x)` where Φ is the CDF of the standard normal distribution. Computing Φ(x) is expensive, so we use a standard approximation:

```python
import math

def gelu(x):
    # GELU(x) = x * Phi(x) where Phi is the standard normal CDF.
    # Approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    cdf = 0.5 * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))
    return x * cdf
```

The approximation was derived in the GELU paper (Hendrycks & Gimpel, 2016) and is what PyTorch uses internally. For negative inputs, GELU suppresses the value rather than zeroing it outright — this gives the network more expressive power.

```python
x = 1.0
gelu(x)   # 0.841...
x = -1.0
gelu(x)   # -0.158...  (not zero, unlike ReLU)
```

---

## The FFN layer

The standard transformer FFN is:

```
FFN(x) = Linear2(GELU(Linear1(x)))
```

Where both Linear layers have shape `(d_model, d_ff)` and `(d_ff, d_model)`. The hidden dimension `d_ff` is typically 4x the model dimension (so `d_ff = 4 * d_model`). This expansion-and-compression is what gives the FFN its capacity.

```python
import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = gelu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

The first linear expands from `d_model` to `d_ff` (usually 4x larger). The activation breaks linearity. The second linear projects back from `d_ff` to `d_model`.

Note that we apply dropout between the two linear layers — not after. This is the standard arrangement. The dropout randomly zeros some of the hidden activations during training, which acts as a regularizer.

---

## Running the FFN

```python
batch, seq_len, d_model = 2, 10, 128
d_ff = 4 * d_model  # 512
x = torch.randn(batch, seq_len, d_model)
ffn = FeedForward(d_model, d_ff)
output = ffn(x)
output.shape  # torch.Size([2, 10, 128])
```

The FFN operates position-wise — every token in the sequence goes through the same two linear transformations independently. This is why the FFN adds minimal computational overhead relative to attention (which has quadratic seq_len scaling).

---

## Compare: our FFN vs PyTorch

Let's verify that our implementation produces numerically identical results to PyTorch's built-in FFN. We'll compare against `nn.Sequential(nn.Linear, nn.GELU, nn.Dropout, nn.Linear)`.

```python
import torch.nn as nn

def gelu(x):
    cdf = 0.5 * (1.0 + torch.tanh(torch.sqrt(torch.tensor(2.0 / math.pi)) * (x + 0.044715 * x ** 3))))
    return x * cdf

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(gelu(self.linear1(x))))
```

PyTorch's `nn.GELU` uses the same approximation internally. We can confirm they match:

```python
# our FFN
ffn_ours = FeedForward(d_model, d_ff)
# PyTorch equivalent
ffn_torch = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(0.0), nn.Linear(d_ff, d_model))
# verify
x = torch.randn(4, 20, d_model)
torch.allclose(ffn_ours(x), ffn_torch(x))  # True
```

---

## Benchmark: from scratch vs PyTorch

The FFN's computational cost is dominated by the two matrix multiplications. Our from-scratch version does exactly the same operations as PyTorch — there is no overhead from writing it ourselves.

Run `benchmark.py` to compare forward pass latency:

```bash
python benchmark.py
```

---

## Recap

- The FFN (FeedForward Network) is `Linear → Activation → Linear`, operating position-wise on token embeddings
- **GELU** (`x * Φ(x)`) is the standard activation in modern transformers, approximated by `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715*x^3)))`
- The FFN expands from `d_model` to `d_ff` (typically 4x) then projects back — this expansion is where most transformer parameters live
- Dropout is applied between the two linear layers, not after the second
- Our from-scratch FFN is numerically identical to `nn.Sequential(nn.Linear, nn.GELU, nn.Dropout, nn.Linear)`

---

Get the video walkthrough of SwiGLU activation (used in Llama, PaLM), ReGLU variant, FFN architecture comparison across GPT-2 models, and memory/speed profiling: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)