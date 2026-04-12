# Decoder Block From Scratch

The decoder block is the building block of GPT, Llama, and every other autoregressive language model. It has three sublayers that work together: masked self-attention to keep the model causal, cross-attention to condition on the encoder (in encoder-decoder models), and a feed-forward network to transform features.

In this lesson we build a full decoder block from scratch in pure PyTorch. No nn.TransformerDecoderLayer — every line is explicit.

---

## Hook: why decoder blocks matter

GPT, Llama, Mistral — every large language model is just decoder blocks stacked on top of each other. The original "Attention Is All You Need" paper stacked 6 decoder blocks. GPT-2 uses 12, GPT-3 uses 96. The entire model is this pattern repeated: masked self-attention, cross-attention (optional), feed-forward, all wrapped in residual connections.

Understanding the decoder block means understanding the core of how modern LLMs work.

---

## Step 1: The decoder block architecture

A decoder block has three sublayers in sequence. Each sublayer is wrapped in a residual connection (skip connection) and layer normalization:

```
x → MaskedSelfAttention → + → LayerNorm
                          ↓
x → CrossAttention (enc) → + → LayerNorm
                          ↓
x → FeedForward → + → LayerNorm
                  ↓
              output
```

The residual connection adds the input to the output of the sublayer before layer normalization. This helps gradients flow through deep networks and stabilizes training.

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, bias=True):
        super().__init__()
        self.masked_attn = MaskedSelfAttention(d_model, n_heads, bias=bias)
        self.cross_attn = CrossAttention(d_model, n_heads, bias=bias)
        self.ffn = FeedForward(d_model, d_ff, bias=bias)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)
```

---

## Step 2: Masked self-attention (causal attention)

The first sublayer is masked self-attention. This enforces causality — token i can only attend to tokens 0..i. We do this with a causal (lower-triangular) mask applied before softmax.

The mask uses `torch.triu` with `diagonal=1` to create a mask where the diagonal and everything above is `True` (masked):

```python
def create_causal_mask(seq_len, device='cpu'):
    # torch.triu: upper triangle (including diagonal) becomes 1
    # We invert: True = masked position
    mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)
    return mask
```

And apply it before softmax:

```python
def apply_causal_mask(scores, mask):
    scores = scores.masked_fill(mask, float('-inf'))
    return scores
```

The full masked self-attention computes attention scores, applies the causal mask, softmaxes, and blends values:

```python
class MaskedSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, x):
        batch, seq_len, d_model = x.shape
        qkv = self.qkv_proj(x)
        qkv = qkv.view(batch, seq_len, self.n_heads, 3 * self.d_head).transpose(1, 2)
        q, k, v = qkv.chunk(3, dim=-1)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float('-inf'))
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v
        attn = attn.transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.out_proj(attn)
```

---

## Step 3: Cross-attention (decoder attends to encoder)

The second sublayer is cross-attention. This is what allows the decoder to "look at" the encoder output. Unlike self-attention where Q, K, V all come from the same source, cross-attention takes Q from the decoder and K, V from the encoder.

In a translation model, this is how the decoder attends to the English sentence while generating the French translation.

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads, bias=True):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Separate projections for Q (decoder) and K,V (encoder)
        self.q_proj = nn.Linear(d_model, d_model, bias=bias)
        self.k_proj = nn.Linear(d_model, d_model, bias=bias)
        self.v_proj = nn.Linear(d_model, d_model, bias=bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

    def forward(self, dec_x, enc_x):
        batch, dec_seq, d_model = dec_x.shape
        enc_seq = enc_x.shape[1]
        # Q from decoder, K and V from encoder
        q = self.q_proj(dec_x).view(batch, dec_seq, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(enc_x).view(batch, enc_seq, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(enc_x).view(batch, enc_seq, self.n_heads, self.d_head).transpose(1, 2)
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v
        attn = attn.transpose(1, 2).contiguous().view(batch, dec_seq, d_model)
        return self.out_proj(attn)
```

Note: there is no causal mask in cross-attention — the decoder can attend to all encoder positions.

---

## Step 4: Feed-forward network

The third sublayer is a position-wise feed-forward network. Two linear layers with GELU activation in between, expanding d_model to d_ff and back.

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None, bias=True):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)

    def forward(self, x):
        return self.w2(F.gelu(self.w1(x)))
```

The expansion ratio 4x is standard (from the original transformer paper). GELU is the smooth, probabilistic alternative to ReLU.

---

## Step 5: The full forward pass

Putting it all together with residual connections and layer norm:

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, bias=True):
        super().__init__()
        self.masked_attn = MaskedSelfAttention(d_model, n_heads, bias=bias)
        self.cross_attn = CrossAttention(d_model, n_heads, bias=bias)
        self.ffn = FeedForward(d_model, d_ff, bias=bias)
        self.ln1 = LayerNorm(d_model)
        self.ln2 = LayerNorm(d_model)
        self.ln3 = LayerNorm(d_model)

    def forward(self, x, enc_x=None):
        # Masked self-attention + residual + layer norm
        attn_out = self.masked_attn(x)
        x = self.ln1(x + attn_out)

        # Cross-attention (if encoder output provided)
        if enc_x is not None:
            cross_out = self.cross_attn(x, enc_x)
            x = self.ln2(x + cross_out)

        # Feed-forward + residual + layer norm
        ffn_out = self.ffn(x)
        x = self.ln3(x + ffn_out)

        return x
```

Run it:

```bash
python decoder_block.py
```

---

## Benchmark: from scratch vs PyTorch

PyTorch's `nn.TransformerDecoderLayer` uses the same architecture but with separate Q/K/V projection layers and different internal ordering. Run `benchmark.py` to compare:

```bash
python benchmark.py
```

The benchmark shows forward pass latency across several batch/sequence/model-size configurations.

---

## Recap

- **Decoder block** stacks three sublayers: masked self-attention, cross-attention, and feed-forward — each wrapped in residual + layer norm.
- **Masked self-attention** uses a lower-triangular mask (`torch.triu(..., diagonal=1)`) to enforce causality — token i cannot attend to future tokens.
- **Cross-attention** takes Q from the decoder and K, V from the encoder — this is how the decoder "reads" the encoder output.
- **Feed-forward** is two linear layers with GELU: expands to d_ff (usually 4x d_model), then projects back.
- **Residual connections** (`x + sublayer(x)`) help gradients flow through deep networks and are essential for stable training.

---

Get the video walkthrough of cross-attention head analysis, profiling the decoder block across batch sizes, and understanding encoder-decoder vs decoder-only architectures: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
