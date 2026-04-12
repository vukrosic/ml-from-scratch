# Full Transformer From Scratch

Every large language model today — GPT, BERT, T5, Llama — is built on the Transformer architecture. It is the engine that made scale work. This lesson builds the complete encoder-decoder Transformer from scratch in pure PyTorch: positional encoding, multi-head attention, feed-forward networks, encoder layers, decoder layers, and the full forward pass. No black boxes.

By the end you will understand exactly what happens when a Transformer translates a sentence, generates the next token, or answers a question.

---

## Hook: why the Transformer changed everything

RNNs process sequences token by token — slow and unable to look ahead. CNNs can parallelise but struggle with long-range dependencies. The Transformer solves both: every token attends to every other token in a single parallel operation, making it fast to train and capable of capturing relationships across any distance.

The key insight from "Attention Is All You Need" (2017) was that you do not need recurrence at all. Attention alone, stacked in layers, learns to route information across the entire sequence. Everything that followed — BERT, GPT-2, T5, Llama — is a variation on this theme.

---

## Positional encoding: giving the model a sense of order

Attention is inherently permutation-invariant: shuffle the input tokens and the attention scores are the same. To let the model reason about position, we add a fixed encoding to each token's embedding before they enter the network.

Sinusoidal positional encoding uses sine and cosine functions at different frequencies:

```python
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=5000):
        super().__init__()
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)   # even dimensions
        pe[:, 1::2] = torch.cos(position * div_term)   # odd dimensions
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, seq_len):
        return self.pe[:, :seq_len, :]
```

Each dimension corresponds to a different frequency — low dimensions encode fine-grained position (high frequency), high dimensions encode coarse position (low frequency). This fixed encoding works for any sequence length up to `max_seq_len` without retraining. The model learns to read position from these patterns.

---

## Multi-head attention: attending to many things at once

Attention computes how much each token should attend to each other token. Multi-head attention runs this computation in parallel across `n_heads` heads, each with its own learned Q/K/V subspace. Different heads can specialize: one might track syntax, another coreference, another long-range dependencies.

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch = query.shape[0]
        Q = self.W_q(query).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attn = torch.matmul(torch.softmax(scores, dim=-1), V)
        attn = attn.transpose(1, 2).contiguous().view(batch, -1, d_model)
        return self.W_o(attn)
```

The scaling by `sqrt(d_k)` keeps the dot products in a reasonable range — without it, large `d_k` makes softmax collapse to near-one-hot distributions with vanishing gradients.

---

## Feed-forward network: per-token non-linearity

Between attention layers, each position passes through a two-layer MLP with a ReLU:

```python
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        return self.net(x)
```

The FFN gives each token an independent, learned non-linear transformation. It is applied position-wise (the same weights at every position), so it does not mix information across tokens — that is the job of attention. In the base Transformer, `d_ff = 4 * d_model`.

---

## Encoder block: self-attention + FFN

An encoder block applies multi-head self-attention across all source tokens, then a feed-forward network. Both steps use residual connections (add the input to the output) and layer normalisation to stabilise training.

```python
class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask=mask))
        x = self.norm2(x + self.ffn(x))
        return x
```

The residual connection lets gradients flow directly through the block, enabling deep stacks. Layer normalisation stabilises activations across the batch dimension.

---

## Decoder block: masked self-attention + cross-attention + FFN

The decoder is more complex because it must generate tokens autoregressively. It has three sub-layers:

1. **Masked self-attention** — each token can only attend to previous tokens (causal mask)
2. **Cross-attention** — attends to the encoder output
3. **Feed-forward network** — same as in the encoder

```python
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, n_heads)
        self.cross_attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x, enc_output, tgt_mask=None, cross_mask=None):
        x = self.norm1(x + self.self_attn(x, x, x, mask=tgt_mask))   # causal
        x = self.norm2(x + self.cross_attn(x, enc_output, enc_output, mask=cross_mask))
        x = self.norm3(x + self.ffn(x))
        return x
```

The causal mask is a lower-triangular matrix that sets attention to future tokens to `-inf` before softmax, effectively sending their weight to zero.

---

## The full Transformer

Putting it together:

```python
class Transformer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, n_layers,
                 src_vocab_size, tgt_vocab_size, max_seq_len=5000):
        super().__init__()
        self.src_embed = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embed = nn.Embedding(tgt_vocab_size, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_seq_len)
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
        ])
        self.output_proj = nn.Linear(d_model, tgt_vocab_size)
```

The forward pass encodes the source, then decodes the target:

```python
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, cross_mask=None):
        enc_output = self.encode(src, src_mask)
        return self.decode(tgt, enc_output, tgt_mask, cross_mask)
```

The output projection maps from `d_model` to `tgt_vocab_size` — each dimension is a logit for one vocabulary token. The softmax of these logits gives the probability distribution for the next token.

---

## Training: teacher forcing

At training time the decoder receives the full target sequence at once (teacher forcing). The model predicts each token given the source and all previous target tokens. The loss is cross-entropy on next-token prediction.

```python
tgt_input  = tgt[:, :-1]   # decoder input: all tokens except last
tgt_labels = tgt[:, 1:]    # labels:      all tokens except first
logits = model(src, tgt_input)
loss = F.cross_entropy(logits.reshape(-1, vocab_size), tgt_labels.reshape(-1))
```

---

## Inference: greedy decoding

At inference time we do not have the full target. We generate one token at a time:

1. Encode the source once.
2. Start with an SOS token.
3. Decode one step, take the argmax (most likely next token).
4. Append to the decoder input and repeat until EOS or max length.

```python
def greedy_decode(model, src, max_len, sos_token, eos_token, device):
    model.eval()
    enc_output = model.encode(src)
    tgt = torch.tensor([[sos_token]], device=device)
    for _ in range(max_len):
        logits = model.decode(tgt, enc_output)
        next_token = logits[:, -1, :].argmax(dim=-1)
        if next_token.item() == eos_token:
            break
        tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
    return tgt[0].tolist()
```

This is called greedy because it always picks the single most likely token. Beam search (see `beam_search.py`) keeps multiple candidates and picks the best complete sequence.

---

## Benchmark: our Transformer vs torch.nn.Transformer

Run `benchmark.py` to compare forward pass latency:

```bash
python benchmark.py
```

The output shows mean latency across multiple batch/sequence/model-size configurations. Our implementation uses a single combined QKV projection, while `nn.Transformer` uses a different internal design — numerical equivalence is not expected. The goal is educational clarity, not matching PyTorch internals.

---

## Recap

- **Sinusoidal positional encoding** adds fixed, interpretable position signals to token embeddings — works for any sequence length up to `max_seq_len`
- **Multi-head attention** runs `n_heads` parallel attention operations, each with its own Q/K/V subspace; outputs are concatenated and projected back to `d_model`
- **The FFN** is a position-wise two-layer MLP with ReLU, applied independently at every position after attention
- **Encoder blocks** = self-attention + FFN, both with residual + layer norm
- **Decoder blocks** = masked self-attention (causal) + cross-attention (attends to encoder) + FFN
- **Teacher forcing** at training: decoder receives the full target at once; cross-entropy loss on next-token prediction
- **Greedy decoding** at inference: generate one token at a time, always picking the argmax; beam search considers multiple paths
- A base Transformer (~39M params) stacks 6 encoder and 6 decoder layers with `d_model=512, n_heads=8, d_ff=2048`

---

Get the video walkthrough of beam search decoding, model parameter count breakdown across tiny/base/large sizes, and torch.profiler flamegraph analysis: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
