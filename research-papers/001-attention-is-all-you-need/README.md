# Attention Is All You Need — ML From Scratch

**Paper:** Vaswani et al. (2017) — "Attention Is All You Need"
**Series:** ML From Scratch — [vukrosic](https://github.com/vukrosic/ml-from-scratch)

---

## Hook

In 2017, every major translation system used recurrent networks (LSTMs, GRUs) — slow, sequential, and unable to handle long dependencies. Then Vaswani et al. asked a simple question: what if we threw out recurrence entirely and just used attention? The result was the Transformer: faster to train, parallelisable across GPUs, and the foundation of every large language model today (GPT, BERT, T5, Llama, Gemini...). This lesson rebuilds it from scratch, exactly matching the paper's specs, and trains it on a real translation task.

---

## What we build

A **full encoder-decoder Transformer** with the exact hyperparameters from the paper:

| Parameter | Value |
|---|---|
| d_model (embedding size) | 512 |
| n_heads (attention heads) | 8 |
| d_ff (feed-forward hidden) | 2048 |
| n_layers (blocks) | 6 |

We train it on character-level English -> French translation and measure quality with BLEU score.

---

## Files

- `model.py` — the full Transformer architecture, every component labelled with the paper section it comes from
- `data.py` — character-level vocabulary and parallel corpus loader
- `train.py` — training loop with the paper's label smoothing and warmup LR schedule
- `evaluate.py` — BLEU scoring + greedy decoding to show actual translations

---

## 1. Scaled Dot-Product Attention (Equation 1)

The core of the Transformer is this single equation:

```
Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V
```

Q (queries), K (keys), and V (values) are all linear projections of the same input. Each query "queries" all keys via a dot product, gets normalised by the square-root of the key dimension (preventing vanishing gradients at large d_k), softmaxes over all positions to get attention weights, and returns a weighted sum of values.

```python
import math
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float("-inf"))
    attn_weights = F.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, V), attn_weights
```

The `sqrt(d_k)` scaling is critical: without it, as d_k grows, the dot products become large in magnitude, pushing the softmax into regions with extremely small gradients.

---

## 2. Multi-Head Attention

Running only one attention head limits what the model can attend to. Multi-head attention runs H attention heads in parallel, each with a reduced key/query dimension (d_k = d_model / H = 64), then concatenates and projects the results.

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_H) W_O
where head_i = Attention(Q W_q^i, K W_k^i, V W_v^i)
```

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, query, key, value, mask=None):
        batch = query.shape[0]
        Q = self.W_q(query).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        out, weights = scaled_dot_product_attention(Q, K, V, mask=mask)
        out = out.transpose(1, 2).contiguous().view(batch, -1, self.d_model)
        return self.W_o(out), weights
```

Each head can specialise: one head might track syntactic dependencies, another coreference, another positional relationships.

---

## 3. Sinusoidal Positional Encoding (Section 3.5)

Attention is permutation-invariant — swapping two tokens changes nothing. But order matters in language. The original paper uses fixed sinusoidal encodings rather than learned embeddings:

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

These frequencies form a geometric progression from 1/10000^0 up to 1/10000^1, giving the model a way to represent any position up to max_seq_len as a unique pattern.

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]
```

The encoding is added to the token embeddings before entering the first encoder block. It never changes during training — no gradient flows through it.

---

## 4. Encoder Block

Each encoder block has two sub-layers, each wrapped in a residual connection and LayerNorm:

```
x = x + Sublayer(LayerNorm(x))
```

SubLayer 1: Multi-head self-attention (all tokens attend to all tokens).
SubLayer 2: Feed-forward network (two linear layers with ReLU, d_model=512 → d_ff=2048 → d_model=512).

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn       = FeedForward(d_model, d_ff)
        self.norm1     = nn.LayerNorm(d_model)
        self.norm2     = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(query=x, key=x, value=x, mask=mask)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x
```

Residual connections let gradients flow directly through the sub-layer without intermediate matrix multiplications, enabling training of deeper networks. LayerNorm stabilises training by normalising over the feature dimension.

---

## 5. Decoder Block

The decoder has three sub-layers:

1. **Masked self-attention** — causal mask prevents attending to future tokens
2. **Cross-attention** — queries from the decoder attend to keys/values from the encoder output
3. **Feed-forward** — identical to the encoder's FFN

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn        = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, src_mask=None, tgt_mask=None):
        # 1. Masked self-attention (causal)
        sa_out, _ = self.self_attn(query=x, key=x, value=x, mask=tgt_mask)
        x = self.norm1(x + sa_out)
        # 2. Cross-attention (decoder attends to encoder)
        ca_out, _ = self.cross_attn(query=x, key=enc_output, value=enc_output, mask=src_mask)
        x = self.norm2(x + ca_out)
        # 3. Feed-forward
        x = self.norm3(x + self.ffn(x))
        return x
```

The causal mask is a lower-triangular matrix of 1s — token at position i can only attend to positions 0..i.

---

## 6. The Full Transformer

Stacking 6 encoder blocks and 6 decoder blocks with shared embeddings and positional encodings gives the complete model. At inference time, we use greedy decoding (or beam search in production) to generate tokens one by one.

```python
class Transformer(nn.Module):
    def __init__(self, d_model=512, n_heads=8, d_ff=2048, n_layers=6,
                 src_vocab_size=10000, tgt_vocab_size=10000):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc   = SinusoidalPositionalEncoding(d_model)
        self.encoder_blocks = nn.ModuleList([EncoderBlock() for _ in range(n_layers)])
        self.decoder_blocks = nn.ModuleList([DecoderBlock() for _ in range(n_layers)])
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)

    def encode(self, src, src_mask=None):
        x = self.src_embed(src) * math.sqrt(self.d_model) + self.pos_enc(src.shape[1])
        for block in self.encoder_blocks:
            x = block(x, mask=src_mask)
        return x

    def decode(self, tgt, enc_output, src_mask=None, tgt_mask=None):
        x = self.tgt_embed(tgt) * math.sqrt(self.d_model) + self.pos_enc(tgt.shape[1])
        for block in self.decoder_blocks:
            x = block(x, enc_output, src_mask=src_mask, tgt_mask=tgt_mask)
        return self.output_proj(x)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        return self.decode(tgt, self.encode(src, src_mask), src_mask, tgt_mask)
```

---

## 7. Training: Label Smoothing and Warmup LR

The paper uses two key regularisation techniques:

**Label smoothing (epsilon=0.1):** Instead of hard one-hot labels, the target distribution is mixed with a uniform distribution. This prevents the model from becoming overconfident and improves generalisation.

**Learning rate warmup:** For the first 400 steps, the learning rate increases linearly from 0 to the base LR, then decreases with an inverse-square-root schedule. Warmup prevents the large parameter updates that would occur early in training when gradients are noisy.

```python
class WarmupScheduler:
    def step(self):
        self.step_count += 1
        scale = self.d_model ** -0.5
        warmup = min(self.step_count ** -0.5,
                     self.step_count * self.warmup_steps ** -1.5)
        lr = scale * warmup * self.base_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr
```

Train with:

```bash
python train.py
```

Expected output: training loss should drop from ~3.5 to ~0.5 over ~60 epochs, with validation perplexity falling from ~15 to ~2-3 on this tiny corpus.

---

## 8. Evaluation: BLEU Score

We evaluate translation quality with BLEU (Bilingual Evaluation Understudy), the same metric used in the paper. BLEU measures the geometric mean of modified n-gram precision scores, penalising both under- and over-generation.

```python
def sentence_bleu(reference, hypothesis, max_n=4):
    bp = brevity_penalty(reference, hypothesis)
    p_log_sum = 0.0
    for n in range(1, max_n + 1):
        p = max(modified_precision(reference, hypothesis, n), 1e-10)
        p_log_sum += math.log(p)
    return bp * math.exp(p_log_sum / max_n)
```

Run evaluation:

```bash
python evaluate.py
```

---

## Recap

- **Scaled dot-product attention** (Equation 1) computes attention weights as softmax(QK^T / sqrt(d_k))V — all tokens attend to all other tokens in parallel
- **Multi-head attention** runs 8 attention heads in parallel, each with d_k=64, letting the model track different dependency types simultaneously
- **Sinusoidal positional encoding** injects position information without learned parameters, using a geometric progression of frequencies
- **Residual connections + LayerNorm** (Add & Norm) enable deep stacks by providing a gradient highway around each sub-layer
- **The full Transformer** (encoder 6 layers + decoder 6 layers) achieves ~39M parameters and is trained with label smoothing + warmup LR
- **BLEU score** correctly ranks translation quality even on this toy-scale dataset

---

Get the extended notebook with ablation studies, attention weight visualisations, annotated paper equations, RoPE vs sinusoidal comparison, and hyperparameter scaling plots:
**https://www.skool.com/opensuperintelligencelab**
