# Multi-Head Attention: Advanced Topics

Beyond standard MHA — head pruning, grouped-query attention, sparse patterns, and the memory trade-offs that define modern LLM architectures.

The core lesson covers multi-head attention mechanics, parallel heads, and concatenation. This companion explores what happens when you remove heads, share KV projections, analyze head specialization, and optimize for inference memory.

---

## 1. Head pruning: most heads are redundant

A landmark finding from Voita et al. (2019) and Michel et al. (2019): you can remove 20-40% of attention heads with minimal loss in accuracy. Some heads are critical; many are passengers.

```python
import torch
import torch.nn as nn

def compute_head_importance(model, dataloader, n_layers, n_heads):
    """Compute importance score for each head using gradient-based method.

    Importance = E[|gradient of loss w.r.t. head output|]
    Heads with small gradients contribute little to reducing loss.
    """
    head_importance = torch.zeros(n_layers, n_heads)

    for batch in dataloader:
        loss = model(batch).loss
        loss.backward()

        for layer_idx in range(n_layers):
            attn = model.layers[layer_idx].self_attn
            # Each head's output projection gradient
            # W_o shape: (d_model, d_model), reshaped to (n_heads, d_head, d_model)
            grad = attn.o_proj.weight.grad.view(n_heads, -1, attn.d_model)
            param = attn.o_proj.weight.data.view(n_heads, -1, attn.d_model)
            # Sensitivity: how much does the loss change if we zero this head?
            head_importance[layer_idx] += (grad * param).abs().sum(dim=(1, 2))

    head_importance /= len(dataloader)
    return head_importance

def prune_heads(model, head_importance, prune_fraction=0.3):
    """Zero out the least important heads."""
    n_layers, n_heads = head_importance.shape
    n_prune = int(n_heads * prune_fraction)

    # Global ranking across all layers
    flat = head_importance.flatten()
    threshold = flat.sort()[0][n_prune * n_layers]

    pruned_count = 0
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            if head_importance[layer_idx, head_idx] < threshold:
                # Zero out this head's output projection columns
                attn = model.layers[layer_idx].self_attn
                d_head = attn.head_dim
                start = head_idx * d_head
                end = start + d_head
                attn.o_proj.weight.data[:, start:end] = 0
                pruned_count += 1

    print(f"Pruned {pruned_count}/{n_layers * n_heads} heads "
          f"({100 * pruned_count / (n_layers * n_heads):.0f}%)")
```

### Which heads survive pruning?

Research consistently finds certain head "roles" are critical:

```
Head roles (from mechanistic interpretability):

1. Positional heads      - attend to fixed relative positions
                          (e.g., always attend to previous token)
                          CRITICAL: pruning these breaks syntax

2. Rare-word heads       - activate strongly on infrequent tokens
                          Important for factual recall

3. Syntactic heads       - track dependency structure
                          (subject-verb agreement across distance)
                          Moderately important

4. Duplicate/redundant   - multiple heads learn similar patterns
                          SAFE TO PRUNE: removing one changes nothing

5. Positional backup     - secondary positional attention
                          SAFE TO PRUNE: primary positional head suffices
```

---

## 2. Grouped-Query Attention (GQA): the LLaMA 2 approach

Standard MHA gives each head its own Q, K, and V projections. GQA shares K and V across groups of heads, dramatically reducing KV-cache memory.

```
MHA  (Multi-Head Attention):     n_heads Q, n_heads K, n_heads V
GQA  (Grouped-Query Attention):  n_heads Q, n_kv_heads K, n_kv_heads V
MQA  (Multi-Query Attention):    n_heads Q, 1 K, 1 V

Example with 32 heads:
  MHA:  32 Q heads, 32 KV heads    (32 KV pairs in cache)
  GQA:  32 Q heads,  8 KV heads    ( 8 KV pairs in cache -> 4x less memory)
  MQA:  32 Q heads,  1 KV head     ( 1 KV pair  in cache -> 32x less memory)
```

```
                    MHA                              GQA (4 groups)
          Q  K  V   Q  K  V   Q  K  V         Q  Q  K  V   Q  Q  K  V
          |  |  |   |  |  |   |  |  |         |  |  |  |   |  |  |  |
          H1 H1 H1  H2 H2 H2  H3 H3 H3       H1 H2 G1 G1  H3 H4 G2 G2
                                               └──┘  └──┘   └──┘  └──┘
          Each head has its own K,V         2 Q heads share 1 K,V pair
```

### GQA implementation (LLaMA 2 style)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        assert n_heads % n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.n_groups = n_heads // n_kv_heads  # Q heads per KV group
        self.head_dim = d_model // n_heads

        # Q has full n_heads projections
        self.wq = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        # K and V have fewer projections
        self.wk = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(d_model, n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(n_heads * self.head_dim, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape

        # Project Q, K, V
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim)
        k = self.wk(x).view(B, L, self.n_kv_heads, self.head_dim)
        v = self.wv(x).view(B, L, self.n_kv_heads, self.head_dim)

        # Transpose to (B, heads, L, head_dim)
        q = q.transpose(1, 2)  # (B, n_heads, L, head_dim)
        k = k.transpose(1, 2)  # (B, n_kv_heads, L, head_dim)
        v = v.transpose(1, 2)  # (B, n_kv_heads, L, head_dim)

        # Expand K, V to match Q heads by repeating within groups
        # (B, n_kv_heads, L, d) -> (B, n_heads, L, d)
        k = k.repeat_interleave(self.n_groups, dim=1)
        v = v.repeat_interleave(self.n_groups, dim=1)

        # Standard scaled dot-product attention
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)

        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)

# LLaMA 2 configurations:
#   7B:  n_heads=32, n_kv_heads=32  (standard MHA — no GQA)
#   13B: n_heads=40, n_kv_heads=40  (standard MHA — no GQA)
#   34B: n_heads=48, n_kv_heads=8   (GQA with 6 groups)
#   70B: n_heads=64, n_kv_heads=8   (GQA with 8 groups)
```

### Converting MHA checkpoints to GQA

You can convert a pre-trained MHA model to GQA by averaging KV heads within each group, then fine-tuning briefly:

```python
def convert_mha_to_gqa(mha_layer, n_kv_heads):
    """Convert MHA weights to GQA by averaging KV heads per group."""
    n_heads = mha_layer.n_heads
    n_groups = n_heads // n_kv_heads
    head_dim = mha_layer.head_dim

    # Average K weights within each group
    k_weight = mha_layer.wk.weight.data.view(n_heads, head_dim, -1)
    k_grouped = k_weight.view(n_kv_heads, n_groups, head_dim, -1).mean(dim=1)
    # k_grouped shape: (n_kv_heads, head_dim, d_model)

    # Same for V
    v_weight = mha_layer.wv.weight.data.view(n_heads, head_dim, -1)
    v_grouped = v_weight.view(n_kv_heads, n_groups, head_dim, -1).mean(dim=1)

    return k_grouped.reshape(-1, k_weight.shape[-1]), \
           v_grouped.reshape(-1, v_weight.shape[-1])
```

---

## 3. Multi-Query Attention (MQA): PaLM and Falcon

MQA is the extreme case of GQA: a single KV head shared across all query heads. Used by PaLM (540B) and Falcon.

```python
class MultiQueryAttention(nn.Module):
    """MQA: all query heads share one K and one V projection."""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)       # n_heads projections
        self.wk = nn.Linear(d_model, self.head_dim, bias=False)  # 1 projection
        self.wv = nn.Linear(d_model, self.head_dim, bias=False)  # 1 projection
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, L, _ = x.shape
        q = self.wq(x).view(B, L, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, L, 1, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, L, 1, self.head_dim).transpose(1, 2)

        # Broadcast k, v across all heads (no explicit repeat needed for matmul)
        # k: (B, 1, L, d) broadcasts with q: (B, n_heads, L, d)
        scale = self.head_dim ** -0.5
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # v broadcasts: (B, 1, L, d)
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.wo(out)
```

MQA quality is slightly worse than MHA but inference is much faster because the KV-cache is 32x smaller (for 32-head models).

---

## 4. Head specialization analysis

Different heads learn to attend to different things. You can analyze this by measuring attention entropy per head.

```python
def analyze_head_entropy(model, dataloader, n_layers, n_heads):
    """Low entropy = focused/specialized head. High entropy = diffuse/uniform."""
    head_entropies = torch.zeros(n_layers, n_heads)
    count = 0

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            outputs = model(batch, output_attentions=True)
            for layer_idx, attn_weights in enumerate(outputs.attentions):
                # attn_weights: (B, n_heads, L, L)
                # Entropy of each head's attention distribution
                # H = -sum(p * log(p))
                p = attn_weights.clamp(min=1e-10)
                entropy = -(p * p.log()).sum(dim=-1).mean(dim=(0, 2))
                # entropy shape: (n_heads,)
                head_entropies[layer_idx] += entropy.cpu()
            count += 1

    head_entropies /= count

    # Print analysis
    for layer in range(n_layers):
        for head in range(n_heads):
            e = head_entropies[layer, head].item()
            label = "FOCUSED" if e < 1.0 else "MIXED" if e < 3.0 else "DIFFUSE"
            print(f"  Layer {layer:2d} Head {head:2d}: entropy={e:.2f} [{label}]")

    return head_entropies
```

### Typical head specialization patterns

```
Layer 0 (early):
  Head 0: entropy=0.3  [FOCUSED]   <- "previous token" head
  Head 1: entropy=0.5  [FOCUSED]   <- positional head (attend to position 0)
  Head 5: entropy=4.2  [DIFFUSE]   <- uniform attention (bag-of-words)

Layer 15 (middle):
  Head 3: entropy=1.2  [MIXED]     <- syntactic head (subject-verb)
  Head 7: entropy=0.8  [FOCUSED]   <- induction head (copy pattern)

Layer 31 (late):
  Head 2: entropy=2.5  [MIXED]     <- semantic head (coreference)
  Head 6: entropy=0.4  [FOCUSED]   <- "last token" head (for next-token pred)
```

Induction heads (identified by Olsson et al., 2022) are especially important. They implement the pattern: "if A followed B earlier, and A appears again, predict B." These emerge at a phase transition during training.

```python
def detect_induction_heads(attn_weights, tokens):
    """An induction head attends to the token AFTER a previous occurrence
    of the current token.

    Pattern: ...A B ... A [head attends to B]
    """
    B, n_heads, L, L2 = attn_weights.shape
    induction_scores = torch.zeros(n_heads)

    for b in range(B):
        for pos in range(2, L):
            current_token = tokens[b, pos]
            # Find previous occurrences of current_token
            for prev_pos in range(pos - 1):
                if tokens[b, prev_pos] == current_token:
                    # Induction head should attend to prev_pos + 1
                    target = prev_pos + 1
                    if target < L:
                        induction_scores += attn_weights[b, :, pos, target]

    return induction_scores
```

---

## 5. Sparse attention patterns

Full attention is O(n^2). For long sequences (4K, 16K, 128K tokens), sparse patterns reduce this dramatically.

### Longformer: sliding window + global attention

```
Full attention (every token attends to every token):
  [X X X X X X X X]
  [X X X X X X X X]
  [X X X X X X X X]
  [X X X X X X X X]

Longformer (window=3, global tokens marked G):
  [G X X . . . . .]     G = global (attends to everything)
  [X X X X . . . .]     X = local window
  [X X X X X . . .]     . = not computed (saves memory)
  [. X X X X X . .]
  [. . X X X X X .]
  [. . . X X X X X]
  [. . . . X X X X]
  [G . . . . X X X]     <- another global token
```

```python
def create_longformer_mask(seq_len, window_size, global_positions):
    """Create sparse attention mask for Longformer pattern."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)

    # Sliding window: each token attends to window_size neighbors
    for i in range(seq_len):
        start = max(0, i - window_size // 2)
        end = min(seq_len, i + window_size // 2 + 1)
        mask[i, start:end] = True

    # Global tokens: attend to and are attended by everything
    for pos in global_positions:
        mask[pos, :] = True  # global token attends to all
        mask[:, pos] = True  # all tokens attend to global

    return mask
```

### BigBird: window + random + global

BigBird adds random attention connections to Longformer's pattern:

```
BigBird attention pattern:
  [G X X r . . r .]     G = global
  [X X X X . r . .]     X = local window
  [X X X X X . . r]     r = random
  [r X X X X X . .]
  [. . X X X X X .]
  [. r . X X X X X]
  [r . . . X X X X]
  [G . r . . X X X]

Random connections ensure any token can reach any other
token in O(log n) hops through the attention graph.
```

---

## 6. KV-cache memory: MHA vs GQA vs MQA at scale

The KV-cache stores keys and values for all previous tokens during autoregressive generation. Its size determines how many concurrent requests a GPU can serve.

```
KV-cache memory per token per layer:
  MHA:  2 * n_heads * head_dim * dtype_bytes
  GQA:  2 * n_kv_heads * head_dim * dtype_bytes
  MQA:  2 * 1 * head_dim * dtype_bytes

For a 7B model (32 layers, 32 heads, head_dim=128, fp16):

  MHA KV-cache per token:
    2 * 32 * 128 * 2 bytes = 16,384 bytes per layer
    * 32 layers = 524,288 bytes = 0.5 MB per token

  For 4096-token context:
    0.5 MB * 4096 = 2 GB just for KV-cache of ONE sequence

  GQA (8 KV heads) KV-cache per token:
    2 * 8 * 128 * 2 bytes = 4,096 bytes per layer
    * 32 layers = 131,072 bytes = 0.125 MB per token
    4096 tokens: 512 MB  (4x reduction)

  MQA (1 KV head):
    2 * 1 * 128 * 2 bytes = 512 bytes per layer
    * 32 layers = 16,384 bytes = 0.016 MB per token
    4096 tokens: 64 MB  (32x reduction)
```

```python
def kv_cache_memory(n_layers, n_kv_heads, head_dim, seq_len,
                    batch_size=1, dtype_bytes=2):
    """Calculate KV-cache memory in bytes."""
    per_token_per_layer = 2 * n_kv_heads * head_dim * dtype_bytes
    total = per_token_per_layer * n_layers * seq_len * batch_size
    return total

# Compare at 7B scale
configs = {
    "LLaMA-2-7B (MHA)":   dict(n_layers=32, n_kv_heads=32, head_dim=128),
    "LLaMA-2-70B (GQA)":  dict(n_layers=80, n_kv_heads=8,  head_dim=128),
    "Falcon-7B (MQA)":    dict(n_layers=32, n_kv_heads=1,  head_dim=64),
}

for name, cfg in configs.items():
    mem = kv_cache_memory(**cfg, seq_len=4096)
    print(f"{name:25s}: {mem / 1e9:.2f} GB for 4K context")
    mem_32k = kv_cache_memory(**cfg, seq_len=32768)
    print(f"{'':25s}  {mem_32k / 1e9:.2f} GB for 32K context")
```

### Concurrent request throughput

```
RTX 4090 (24 GB VRAM):
  Model weights (7B fp16): ~14 GB
  Remaining for KV-cache:  ~10 GB

  MHA (0.5 MB/token): 10 GB / (0.5 MB * 4096) = ~5 concurrent 4K requests
  GQA (0.125 MB/token): 10 GB / (0.125 MB * 4096) = ~19 concurrent requests
  MQA (0.016 MB/token): 10 GB / (0.016 MB * 4096) = ~152 concurrent requests

GQA gives ~4x more throughput than MHA.
MQA gives ~30x more throughput than MHA.
```

This is why every major production model (LLaMA 3, Gemma 2, Mistral, DeepSeek) uses GQA. The quality loss is negligible but the serving cost reduction is massive.

---

## 7. Practical head count guidelines

```
Model size    d_model    Typical n_heads    head_dim    n_kv_heads (GQA)
───────────────────────────────────────────────────────────────────────────
  125M         768          12               64          12 (no GQA)
  350M        1024          16               64          16 (no GQA)
  1.3B        2048          16              128           4
  7B          4096          32              128           8
  13B         5120          40              128           8
  34B         8192          48              128           8
  70B         8192          64              128           8

Rules of thumb:
- head_dim=64 or 128 (128 preferred for RoPE compatibility)
- GQA ratio of 4-8 Q heads per KV head works well
- Below 1B parameters, GQA savings are marginal — use standard MHA
- n_kv_heads=8 seems to be a sweet spot across scales
```

---

Get the video walkthrough: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
