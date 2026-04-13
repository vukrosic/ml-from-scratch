# Self-Attention: Advanced Topics

Beyond the basics — the mechanisms that make modern attention work at scale.

The core lesson covers Q, K, V projections, scaled dot-product attention, and causal masking. This companion digs into the ideas that separate textbook attention from production attention: position-aware scoring, efficient windowed computation, attention pathologies, cross-sequence attention, and subquadratic approximations.

---

## 1. Attention as soft dictionary lookup

A Python dictionary maps a key to a value. You give it an exact key, you get exactly one value back.

```python
phonebook = {"Alice": "555-0100", "Bob": "555-0200", "Carol": "555-0300"}
result = phonebook["Bob"]  # exact match -> "555-0200"
```

Self-attention does the same thing, but *soft*. Instead of an exact key match, every key gets a similarity score against the query. Instead of returning one value, it returns a weighted blend of all values.

```
Hard lookup:   query -> find exact key match   -> return that value
Soft lookup:   query -> score ALL keys         -> return weighted sum of ALL values
```

Concretely:

```python
# Hard dictionary lookup (discrete)
def hard_lookup(query, keys, values):
    for i, key in enumerate(keys):
        if key == query:
            return values[i]

# Soft dictionary lookup (differentiable) — this IS attention
def soft_lookup(query, keys, values):
    scores = [dot(query, key) for key in keys]     # similarity to each key
    weights = softmax(scores)                       # normalize to distribution
    return sum(w * v for w, v in zip(weights, values))  # blend values
```

The critical insight: because `soft_lookup` uses continuous similarity and differentiable softmax, gradients flow through the entire operation. The model *learns* what queries to emit and what keys to expose. It learns how to route information between tokens.

This is why attention is sometimes called a **differentiable hash table**. The Q projection learns to generate lookup queries. The K projection learns to generate keys that match relevant queries. The V projection learns what information to return when matched.

Every head in multi-head attention is a separate soft dictionary — different heads learn to index by different properties (syntax, coreference, position, semantic similarity).

---

## 2. Relative position encodings

Vanilla attention is permutation-invariant. Without positional information, the model cannot distinguish "the cat sat on the mat" from "mat the on sat cat the." Absolute position embeddings (learned or sinusoidal) solve this but have a fixed maximum sequence length.

Modern models use **relative** position encodings that inject position information directly into the attention score computation. Two dominant approaches: ALiBi and RoPE.

### ALiBi: Attention with Linear Biases

ALiBi is dead simple. After computing `Q @ K^T`, subtract a linear penalty proportional to the distance between tokens. Closer tokens get higher scores. Farther tokens get penalized.

```python
def alibi_bias(seq_len, n_heads):
    """Each head gets a different slope — head 0 has the steepest penalty,
    last head has the gentlest. This lets different heads attend at different ranges."""
    # Slopes are geometric: 2^(-8/n_heads), 2^(-16/n_heads), ...
    slopes = torch.pow(2, -8 * torch.arange(1, n_heads + 1) / n_heads)

    # Distance matrix: position i to position j
    positions = torch.arange(seq_len)
    distance = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (seq, seq)

    # Each head applies its slope to the distance matrix
    bias = -slopes.view(-1, 1, 1) * distance.unsqueeze(0)  # (n_heads, seq, seq)
    return bias  # Add this to attention scores before softmax

# Usage inside attention:
# scores = q @ k.transpose(-2, -1) / scale
# scores = scores + alibi_bias(seq_len, n_heads).to(scores.device)
# attn_weights = F.softmax(scores, dim=-1)
```

ALiBi trains at one context length and generalizes to longer contexts at inference because the bias is a pure function of distance — no learned position embeddings to extrapolate.

### RoPE: Rotary Position Embeddings

RoPE encodes position by *rotating* the Q and K vectors in 2D subspaces. When you take the dot product of a rotated Q at position `m` and a rotated K at position `n`, the rotation angles partially cancel, leaving a term that depends only on the relative distance `m - n`.

The core idea: split each d-dimensional vector into d/2 pairs of coordinates. Rotate each pair by an angle proportional to the position.

```python
def precompute_rope_freqs(d_head, max_seq_len, theta=10000.0):
    """Precompute the rotation frequencies for RoPE."""
    # Each pair of dimensions gets a different base frequency
    freqs = 1.0 / (theta ** (torch.arange(0, d_head, 2).float() / d_head))
    # Outer product with positions gives the rotation angle for each (position, dim_pair)
    positions = torch.arange(max_seq_len)
    angles = positions.unsqueeze(1) * freqs.unsqueeze(0)  # (seq, d_head/2)
    return torch.cos(angles), torch.sin(angles)

def apply_rope(x, cos, sin):
    """Apply rotary embeddings to Q or K.
    x: (batch, n_heads, seq, d_head)
    cos, sin: (seq, d_head/2) — precomputed frequencies
    """
    d_half = x.shape[-1] // 2
    x1 = x[..., :d_half]      # first half of each pair
    x2 = x[..., d_half:]      # second half of each pair

    # 2D rotation: [cos, -sin; sin, cos] applied to each (x1, x2) pair
    cos = cos[:x.shape[2]].unsqueeze(0).unsqueeze(0)  # broadcast to (1, 1, seq, d/2)
    sin = sin[:x.shape[2]].unsqueeze(0).unsqueeze(0)

    rotated_x1 = x1 * cos - x2 * sin
    rotated_x2 = x1 * sin + x2 * cos
    return torch.cat([rotated_x1, rotated_x2], dim=-1)
```

The rotation matrix for a single 2D pair at position `m`:

```
R(m, i) = [ cos(m * theta_i)   -sin(m * theta_i) ]
          [ sin(m * theta_i)    cos(m * theta_i)  ]
```

When you compute `dot(R(m) @ q, R(n) @ k)`, the rotation matrices combine as `R(m)^T @ R(n) = R(n - m)`. The dot product now depends on `n - m` — the relative position. This is the mathematical elegance of RoPE: relative position encoding falls out of the rotation group structure for free.

Usage in the attention forward pass:

```python
# Precompute once at init
cos_cached, sin_cached = precompute_rope_freqs(d_head, max_seq_len)

# In forward(), after computing q, k:
q = apply_rope(q, cos_cached, sin_cached)
k = apply_rope(k, cos_cached, sin_cached)
# Then proceed with standard scaled dot-product attention
scores = q @ k.transpose(-2, -1) / scale
```

RoPE is used in Llama, Mistral, Qwen, and most modern open-weight LLMs. It extrapolates to longer sequences better than learned absolute embeddings, though not as effortlessly as ALiBi. Techniques like NTK-aware scaling and YaRN further improve RoPE's length generalization.

---

## 3. Sliding window attention

Standard self-attention computes scores between all pairs of tokens. For a sequence of length `n`, that is `O(n^2)` memory and compute. At 128K context, the attention matrix has 16 billion entries per head.

Mistral introduced **sliding window attention** (SWA): each token only attends to the previous `W` tokens (the "window"). This reduces memory from `O(n^2)` to `O(n * W)`.

```python
def sliding_window_mask(seq_len, window_size):
    """Create a mask where token i attends only to tokens [i-W+1, i].
    Also enforces causality (no attending to future tokens)."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        start = max(0, i - window_size + 1)
        mask[i, start:i + 1] = True
    return mask

# Visualize the mask pattern for seq_len=8, window_size=3:
#
#   K: 0 1 2 3 4 5 6 7
# Q:
# 0 [ 1 0 0 0 0 0 0 0 ]   <- token 0 attends to itself only
# 1 [ 1 1 0 0 0 0 0 0 ]   <- token 1 attends to 0,1
# 2 [ 1 1 1 0 0 0 0 0 ]   <- token 2 attends to 0,1,2  (window fills up)
# 3 [ 0 1 1 1 0 0 0 0 ]   <- token 3 attends to 1,2,3  (0 drops out)
# 4 [ 0 0 1 1 1 0 0 0 ]   <- token 4 attends to 2,3,4
# 5 [ 0 0 0 1 1 1 0 0 ]   <- ...
# 6 [ 0 0 0 0 1 1 1 0 ]
# 7 [ 0 0 0 0 0 1 1 1 ]

# Usage in attention:
# mask = sliding_window_mask(seq_len, window_size)
# scores = scores.masked_fill(~mask, float('-inf'))
# attn_weights = F.softmax(scores, dim=-1)
```

But wait — if token 7 can only see tokens 5, 6, 7 directly, how does it access information from token 0? Through **stacking layers**. After layer 1, token 5 contains information about tokens 3, 4, 5. After layer 2, token 7 (attending to 5, 6, 7) indirectly accesses information from token 3. After `L` layers with window size `W`, the effective receptive field is `L * W` tokens.

Mistral 7B uses `W = 4096` with 32 layers, giving an effective receptive field of 131,072 tokens — far beyond the 32K training context.

Some architectures mix sliding window layers with full attention layers. A few global-attention layers every N layers ensure long-range information can flow without depending solely on the stacking trick.

---

## 4. Attention sink phenomenon

In 2023, researchers (Xiao et al., "Efficient Streaming Language Models with Attention Sinks") discovered something strange: in trained LLMs, the first token in the sequence receives disproportionately high attention across almost all heads and layers — regardless of what that token actually is.

```
Typical attention pattern for a mid-layer head:

Token:  [BOS]  The   cat   sat   on    the   mat
Weight:  0.35  0.02  0.05  0.30  0.03  0.02  0.23
              ^---- the first token gets 35% of attention
```

Why does this happen? The model needs a "no-op" target. When a head has nothing useful to attend to for a given query (the relevant information simply is not in the context), it still must produce a valid probability distribution. Attending uniformly would inject noise. Instead, the model learns to dump excess attention onto the first token — which, during training, develops a value vector that contributes minimally to the output. It becomes an **attention sink**: a safe default target.

This has practical consequences:

1. **Streaming inference**: if you drop the first token from the KV cache to save memory, perplexity explodes. The model loses its sink. StreamingLLM solves this by always keeping the first few "sink" tokens in the cache, even when evicting middle tokens.

2. **Attention analysis**: if you see a head with massive weight on position 0, that is not a meaningful syntactic pattern. It is the model using its garbage-collection mechanism.

3. **Training**: some architectures now add explicit sink tokens (learnable dummy tokens prepended to the sequence) so the model does not have to hijack the BOS token.

```python
def detect_attention_sinks(attn_weights, threshold=0.3):
    """Flag heads where the first token gets more than `threshold` of total attention.
    attn_weights: (batch, n_heads, seq, seq) after softmax
    """
    # Average attention to position 0 across all query positions
    avg_attn_to_first = attn_weights[:, :, :, 0].mean(dim=-1)  # (batch, n_heads)
    sinks = avg_attn_to_first > threshold
    return sinks  # True for heads exhibiting the sink pattern
```

---

## 5. Attention score analysis

Attention weights reveal what the model is doing internally. The key tool: **entropy** of the attention distribution.

- **Low entropy**: the head focuses sharply on one or two tokens. This usually means the head has found a specific syntactic or semantic relationship (e.g., "this pronoun refers to that noun").
- **High entropy**: the head spreads attention broadly across many tokens. Either it is computing a global average (valid), or it has not learned a useful pattern (dead head).

```python
import torch
import math

def attention_entropy(attn_weights, eps=1e-8):
    """Compute per-head entropy of attention distributions.
    attn_weights: (batch, n_heads, seq_q, seq_k) — output of softmax
    Returns: (batch, n_heads, seq_q) — entropy for each query position in each head
    """
    # H = -sum(p * log(p)) over the key dimension
    log_weights = torch.log(attn_weights + eps)
    entropy = -(attn_weights * log_weights).sum(dim=-1)
    return entropy

def analyze_attention(model, input_ids, tokenizer):
    """Extract and analyze attention patterns from a model forward pass."""
    # Run forward pass, collecting attention weights
    with torch.no_grad():
        outputs = model(input_ids, output_attentions=True)

    tokens = [tokenizer.decode(t) for t in input_ids[0]]

    for layer_idx, layer_attn in enumerate(outputs.attentions):
        # layer_attn: (batch, n_heads, seq, seq)
        ent = attention_entropy(layer_attn)  # (batch, n_heads, seq)
        mean_ent = ent[0].mean(dim=-1)       # (n_heads,) average over positions

        max_possible = math.log(layer_attn.shape[-1])  # log(seq_len)
        normalized_ent = mean_ent / max_possible        # 0 = focused, 1 = uniform

        print(f"\nLayer {layer_idx}:")
        for head_idx in range(mean_ent.shape[0]):
            bar = "#" * int(normalized_ent[head_idx] * 40)
            label = "focused" if normalized_ent[head_idx] < 0.3 else \
                    "diffuse" if normalized_ent[head_idx] > 0.7 else "mixed"
            print(f"  Head {head_idx:2d}: {normalized_ent[head_idx]:.3f} {bar} ({label})")
```

What to look for in the output:

- **Consistently low-entropy heads** in early layers often track local syntax (attend to previous token, attend to matching bracket).
- **Low-entropy heads** in later layers often handle coreference or semantic retrieval.
- **High-entropy heads** that persist across inputs may be candidates for pruning — the model may not need them.
- **Entropy that varies dramatically across inputs** indicates the head is data-dependent, which is healthy.

You can also extract the actual attention matrix and plot which tokens attend to which:

```python
import matplotlib.pyplot as plt

def plot_attention_head(attn_weights, tokens, layer, head):
    """Plot one attention head as a heatmap.
    attn_weights: full model attention output (list of per-layer tensors)
    """
    attn = attn_weights[layer][0, head].cpu().numpy()  # (seq, seq)
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(attn, cmap='Blues')
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=90, fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)
    ax.set_xlabel("Key (attended to)")
    ax.set_ylabel("Query (attending from)")
    ax.set_title(f"Layer {layer}, Head {head}")
    plt.tight_layout()
    plt.savefig(f"attn_L{layer}_H{head}.png", dpi=150)
    plt.close()
```

---

## 6. Cross-attention

Self-attention: Q, K, and V all come from the same sequence. Cross-attention: Q comes from one sequence, K and V come from another.

This is the mechanism that connects two different modalities or representations:
- **Encoder-decoder transformers** (T5, BART): the decoder queries attend to encoder output keys and values.
- **Vision-language models** (Flamingo, LLaVA): text tokens query into image patch representations.
- **Diffusion models** (Stable Diffusion): the U-Net queries attend to text-encoder embeddings of the prompt.

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        # Q comes from the "decoder" / primary sequence
        self.q_proj = nn.Linear(d_model, d_model)
        # K, V come from the "encoder" / secondary sequence
        self.kv_proj = nn.Linear(d_model, 2 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, context):
        """
        x:       (batch, seq_q, d_model)  — the sequence generating queries
        context: (batch, seq_k, d_model)  — the sequence providing keys and values
        """
        batch, seq_q, d_model = x.shape
        seq_k = context.shape[1]

        # Project queries from x, keys and values from context
        q = self.q_proj(x)                              # (batch, seq_q, d_model)
        kv = self.kv_proj(context)                      # (batch, seq_k, 2*d_model)

        # Reshape into heads
        q = q.view(batch, seq_q, self.n_heads, self.d_head).transpose(1, 2)
        kv = kv.view(batch, seq_k, self.n_heads, 2 * self.d_head).transpose(1, 2)
        k, v = kv.chunk(2, dim=-1)

        # Standard scaled dot-product attention
        scale = math.sqrt(self.d_head)
        scores = q @ k.transpose(-2, -1) / scale       # (batch, n_heads, seq_q, seq_k)

        # No causal mask — queries can attend to any context position
        attn_weights = F.softmax(scores, dim=-1)
        attn = attn_weights @ v                         # (batch, n_heads, seq_q, d_head)

        attn = attn.transpose(1, 2).contiguous()
        attn = attn.view(batch, seq_q, d_model)
        return self.out_proj(attn)
```

The key differences from self-attention:

1. **Separate projections.** Q is projected from `x`, while K and V are projected from `context`. In self-attention, all three come from the same input.
2. **No causal mask.** The decoder query at position `i` can attend to *all* encoder positions. There is no notion of "future" in the encoder output.
3. **Asymmetric sequence lengths.** `seq_q` and `seq_k` can differ. An image might have 256 patches while the text has 50 tokens.

In a typical encoder-decoder transformer, each decoder layer has three sub-layers:
1. Causal self-attention (decoder attends to itself)
2. Cross-attention (decoder queries attend to encoder output)
3. Feed-forward network

The cross-attention layer is what lets the decoder "read" the encoder's representation.

---

## 7. Linear attention

Standard attention computes `softmax(Q @ K^T) @ V`. The matrix `Q @ K^T` is `(seq, seq)` — quadratic in sequence length. Linear attention eliminates this bottleneck using the **kernel trick**.

The insight: if we can decompose the softmax kernel into a product of feature maps, we can reorder the computation.

Standard attention for a single query at position `i`:

```
output_i = sum_j [ softmax(q_i @ k_j / sqrt(d)) * v_j ]
```

Linear attention replaces `softmax(q @ k^T)` with `phi(q) @ phi(k)^T`, where `phi` is a feature map (typically `elu(x) + 1` or a random feature approximation):

```
output_i = phi(q_i) @ [ sum_j phi(k_j)^T @ v_j ] / [ phi(q_i) @ sum_j phi(k_j)^T ]
```

The crucial trick: `sum_j phi(k_j)^T @ v_j` does not depend on `i`. Compute it once as a `(d_phi, d_v)` matrix, then multiply each query against it. This flips the computation from `O(n^2 * d)` to `O(n * d^2)` — linear in sequence length.

```python
def linear_attention(q, k, v, eps=1e-6):
    """Linear attention using ELU+1 feature map.
    q, k, v: (batch, n_heads, seq, d_head)
    Returns: (batch, n_heads, seq, d_head)
    """
    # Feature map: elu(x) + 1 ensures non-negative "similarities"
    phi_q = F.elu(q) + 1    # (batch, n_heads, seq, d_head)
    phi_k = F.elu(k) + 1    # (batch, n_heads, seq, d_head)

    # Key-value summary: phi(K)^T @ V — computed once, O(n * d^2)
    kv = phi_k.transpose(-2, -1) @ v       # (batch, n_heads, d_head, d_head)

    # Each query retrieves from the summary: O(n * d^2)
    numerator = phi_q @ kv                   # (batch, n_heads, seq, d_head)

    # Normalizer: sum of phi(K) for each query
    denominator = phi_q @ phi_k.transpose(-2, -1).sum(dim=-1, keepdim=True)
    # Simpler: sum phi_k over sequence, dot with each phi_q
    denominator = (phi_q * phi_k.sum(dim=-2, keepdim=True)).sum(dim=-1, keepdim=True)

    return numerator / (denominator + eps)

def causal_linear_attention(q, k, v, eps=1e-6):
    """Causal version: use cumulative sums instead of global sums.
    Each position only aggregates keys/values from previous positions."""
    phi_q = F.elu(q) + 1
    phi_k = F.elu(k) + 1

    # Cumulative KV matrix: at position i, contains sum of phi(k_j)^T @ v_j for j <= i
    # We compute this with a running sum
    batch, heads, seq, d = phi_k.shape
    kv_cumsum = torch.zeros(batch, heads, d, d, device=q.device)
    k_cumsum = torch.zeros(batch, heads, 1, d, device=q.device)
    outputs = []

    for i in range(seq):
        ki = phi_k[:, :, i:i+1, :]            # (batch, heads, 1, d)
        vi = v[:, :, i:i+1, :]                # (batch, heads, 1, d)
        qi = phi_q[:, :, i:i+1, :]            # (batch, heads, 1, d)

        kv_cumsum = kv_cumsum + ki.transpose(-2, -1) @ vi   # running (d, d) matrix
        k_cumsum = k_cumsum + ki                              # running (1, d) vector

        num = qi @ kv_cumsum                                  # (batch, heads, 1, d)
        den = (qi * k_cumsum).sum(dim=-1, keepdim=True)       # (batch, heads, 1, 1)
        outputs.append(num / (den + eps))

    return torch.cat(outputs, dim=2)  # (batch, heads, seq, d)
```

### Why linear attention underperforms in practice

The theory is clean but the results disappoint. Softmax attention has a property that linear attention lacks: **sharp, selective retrieval**. Softmax can put 99% of attention on a single key. The feature-map approximation spreads weight more diffusely — it cannot achieve the same spiky distributions.

This matters most for tasks requiring precise recall: "What was the name mentioned 3000 tokens ago?" Softmax attention can laser-focus on that token. Linear attention smears the signal across many tokens, degrading retrieval quality.

Recent work (RetNet, Mamba, RWKV, Based) attempts to close this gap with more expressive recurrent or state-space formulations that maintain linear complexity while preserving sharper selection. The field is actively evolving — linear attention is not dead, but naive kernel substitution is not enough.

---

## Summary

| Technique | What it does | Complexity | Used in |
|-----------|-------------|-----------|---------|
| ALiBi | Linear distance penalty on attention scores | O(n^2) | BLOOM, MPT |
| RoPE | Rotates Q, K so dot product encodes relative position | O(n^2) | Llama, Mistral, Qwen |
| Sliding window | Restrict attention to last W tokens | O(n*W) | Mistral, Mixtral |
| Attention sinks | First token absorbs excess attention as no-op target | — | Observed in most LLMs |
| Cross-attention | Q from one sequence, K/V from another | O(n_q * n_k) | T5, Stable Diffusion |
| Linear attention | Replace softmax with kernel feature maps | O(n * d^2) | Linear Transformer, RWKV-like |

---

Get deep-dive video walkthroughs on RoPE internals, FlashAttention mechanics, GQA vs MQA tradeoffs, and live attention pattern analysis on real models: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
