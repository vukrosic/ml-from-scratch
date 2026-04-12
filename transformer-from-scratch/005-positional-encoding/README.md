# Positional Encoding From Scratch

Attention is permutation-invariant: swap two tokens and the output is identical. "cat" attending to "dog" looks exactly the same as "dog" attending to "cat". Without some signal about position, a transformer cannot tell the difference between "the cat sat on the mat" and "the mat sat on the cat". Positional encoding is the fix — injecting position information directly into the model.

This lesson builds the two classic schemes: sinusoidal (from the original Attention Is All You Need paper) and learned embeddings (used in BERT, GPT-2, and most modern models). We implement both from scratch, compare them, and see why neither generalises perfectly to sequences longer than what they were trained on.

---

## Hook: why position matters

Words get their meaning partly from what surrounds them. The word "bank" could be a financial institution or the edge of a river — you only know from context. Transformers compute relationships between tokens through attention, but attention has no built-in notion of sequence order. "Dog bites man" and "man bites dog" are identical to the model if we only look at tokens.

The solution is positional encoding: add a second signal to every token that encodes where it sits in the sequence. The model then has both the token's content (from the embedding) and its position (from the positional encoding).

---

## Sinusoidal positional encoding (Attention Is All You Need)

The original transformer paper uses fixed sine and cosine waves of different frequencies. Even dimensions get `sin(pos / 10000^(2i/d_model))`, odd dimensions get `cos(pos / 10000^(2i/d_model))`. The intuition is that each position becomes a unique vector, and nearby positions have similar encodings.

### The formula

For position `pos` and dimension `i`:
- If `i` is even: `PE(pos, i) = sin(pos / 10000^(i/d_model))`
- If `i` is odd:  `PE(pos, i) = cos(pos / 10000^((i-1)/d_model))`

This gives each position a distinct signature across all dimensions.

### Computing the power terms

```python
import math

def compute_freqs(d_model, max_seq=100):
    """
    Precompute the frequency terms for each dimension.
    dim i uses: 10000^(-i/d_model)  for even i
                10000^(-(i-1)/d_model) for odd i
    """
    freqs = []
    for i in range(d_model):
        if i % 2 == 0:
            freqs.append(1.0 / (10000 ** (i / d_model)))
        else:
            freqs.append(1.0 / (10000 ** ((i - 1) / d_model)))
    return freqs
```

The denominator grows exponentially with dimension index, so high-frequency oscillations appear in early dimensions and low-frequency waves in later dimensions.

### Building the full PE matrix

```python
def sinusoidal_pe(seq_len, d_model):
    """
    Create a (seq_len, d_model) matrix of sinusoidal positional encodings.
    Even dims: sin. Odd dims: cos.
    """
    freqs = compute_freqs(d_model, seq_len)
    pe = []
    for pos in range(seq_len):
        row = [math.sin(pos * freq) if i % 2 == 0
               else math.cos(pos * freq)
               for i, freq in enumerate(freqs)]
        pe.append(row)
    return pe
```

Run it and inspect the values:

```python
pe = sinusoidal_pe(seq_len=8, d_model=16)
import torch
pe_t = torch.tensor(pe)
print(pe_t.shape)  # torch.Size([8, 16])
print(pe_t)
```

The first few rows look different, and the last few rows are also distinct — but rows in the middle can look similar, which is by design: the model learns to distinguish absolute positions, and the cosine on odd dims helps with relative position too.

### Adding to token embeddings

```python
def add_pe(x, pe):
    """Add positional encoding to token embeddings. x and pe are torch Tensors."""
    return x + pe
```

Addition works because both are the same shape: `(batch, seq_len, d_model)`. The model learns to separate the token-content signal from the position signal through training.

### The vector view

A cleaner vectorised version using stack and tile:

```python
import torch

def sinusoidal_pe_vectorized(seq_len, d_model):
    """
    Vectorised sinusoidal PE. Produces (seq_len, d_model).
    Even dims: sin, odd dims: cos.
    """
    positions = torch.arange(seq_len).unsqueeze(1)          # (seq, 1)
    freqs    = torch.arange(0, d_model, 2).unsqueeze(0)     # (1, d_model//2)
    freqs    = torch.exp(-math.log(10000) * freqs / d_model) # (1, d_model//2)

    # shape: (seq_len, d_model//2)
    angles = positions * freqs

    # Stack sin and cos: even cols = sin, odd cols = cos
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(angles)
    pe[:, 1::2] = torch.cos(angles)
    return pe
```

```python
pe_t = sinusoidal_pe_vectorized(8, 16)
print(pe_t.shape)  # torch.Size([8, 16])
assert torch.allclose(torch.tensor(sinusoidal_pe(8, 16)), pe_t, atol=1e-5)
print("Vectorised version matches analytical version.")
```

---

## Learned positional embeddings

Instead of using fixed sine waves, we let the model learn the best positional encoding. We create a lookup table `nn.Embedding(seq_len, d_model)` — position `pos` indexes row `pos` of the table. The model adjusts these values during training to suit its needs.

### The module

```python
import torch
import torch.nn as nn

class LearnedPositionalEmbedding(nn.Module):
    """
    Learned positional encoding: a trainable lookup table.
    Position p is represented by the p-th row of a (seq_len, d_model) matrix.
    """
    def __init__(self, seq_len, d_model):
        super().__init__()
        self.pe = nn.Embedding(seq_len, d_model)

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        seq_len = x.shape[1]
        positions = torch.arange(seq_len, device=x.device)
        return x + self.pe(positions)  # broadcast: (batch, seq, d_model)
```

During training the model learns to represent position in whatever way minimises loss — which often looks quite different from sinusoidal patterns.

---

## Comparing learned vs sinusoidal

Run the comparison:

```bash
python compare.py
```

The script generates visualisations showing both PE schemes. It also demonstrates the generalisation problem: when you ask a learned PE for position 500 but it only trained on positions 0–99, it extrapolates arbitrarily — the embedding values are nonsense. Sinusoidal PE handles this more gracefully because the mathematical pattern continues naturally.

### Why learned PE doesn't generalise

Learned PE is a lookup table. If you train on positions 0–99, you only have entries for those positions. There's no mechanism to predict what position 150 "should" look like — it just grabs whatever row 150 contains, which was never trained on. Sinusoidal PE uses a fixed mathematical function, so it generates a valid encoding for any integer position.

This is one reason RoPE and ALiBi were invented — they avoid absolute position entirely and encode relative position instead.

---

## Recap

- Attention is permutation-invariant: without positional encoding the model cannot distinguish token order.
- Sinusoidal PE uses `sin(pos * 10000^(-i/d_model))` for even dims and `cos(...)` for odd dims — a fixed mathematical pattern valid for any position.
- Learned PE uses `nn.Embedding(seq_len, d_model)` — a trainable lookup table that adapts to the training data.
- Neither naive scheme generalises perfectly to sequences longer than training: learned PE extrapolates arbitrarily, sinusoidal PE extrapolates mathematically but the learned weights won't have seen those positions.
- Relative position encodings (RoPE, ALiBi) sidestep this by not having absolute positions at all.

---

Get the video walkthrough of RoPE (rotary position embedding), ALiBi (attention with linear biases), and inference speed/quality benchmarks across sequence lengths: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
