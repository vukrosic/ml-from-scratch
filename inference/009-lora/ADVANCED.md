# LoRA — Advanced

This document covers QLoRA, DoRA, LoRA+, multi-adapter serving, rank selection, LoRA beyond linear layers, adapter merging, and training stability.

---

## QLoRA Deep Dive

QLoRA combines 4-bit quantization of the base model with standard LoRA adapters. The key insight: you can backpropagate through a frozen quantized model because gradients only flow to the LoRA parameters, not the quantized weights.

### 4-bit NormalFloat (NF4)

NF4 is an information-theoretically optimal quantization dtype for normally distributed weights. Neural network weights are approximately Gaussian, so NF4 places quantization levels at the quantiles of a standard normal distribution.

```python
# NF4 quantile computation (what bitsandbytes does internally)
from scipy.stats import norm
import numpy as np

def compute_nf4_levels():
    """16 quantization levels placed at quantiles of N(0,1)."""
    n = 16
    boundaries = norm.ppf(np.linspace(0, 1, n + 1))
    levels = [(boundaries[i] + boundaries[i+1]) / 2 for i in range(n)]
    levels[0], levels[-1] = norm.ppf(0.5/n), norm.ppf(1 - 0.5/n)
    return levels
# Result: [-1.05, -0.76, ..., 0.76, 1.05] — denser near zero
```

Compared to uniform INT4, NF4 achieves lower quantization error for Gaussian-distributed values because it allocates more levels near the center where most weights cluster.

### Double Quantization

QLoRA applies a second round of quantization to the quantization constants themselves. Each block of 64 weights shares one FP32 scale factor (absmax). With 8-bit quantization of these constants:

```
Memory per parameter:
  4 bits (weight) + 32/64 bits (scale, per block of 64) = 4.5 bits/param

With double quantization:
  4 bits (weight) + 8/64 bits (quantized scale) + 32/256 bits (second-level scale)
  = 4.127 bits/param
```

### Memory Breakdown for a 7B Model

```
Component                          Memory
----------------------------------------------
Base model (NF4, 4.127 bits/param) ~3.6 GB
LoRA adapters (rank=16, FP16)      ~20 MB
Optimizer states (AdamW, FP32)     ~160 MB  (only for LoRA params)
Activations (batch=1, seq=512)     ~400 MB
Gradient computation buffers       ~200 MB
----------------------------------------------
Total                              ~4.4 GB  (fits on a single 24GB GPU)

Compare to full fine-tuning:
Base model (FP16)                  ~14 GB
Optimizer states (AdamW, FP32)     ~28 GB
Gradients (FP16)                   ~14 GB
----------------------------------------------
Total                              ~56 GB   (needs 4x A100 40GB)
```

### QLoRA Forward Pass

```python
def qlora_forward(x, W_nf4, scale, lora_A, lora_B, alpha, rank):
    # Dequantize on the fly (computed in tiles, not fully materialized)
    W_fp16 = dequantize_nf4(W_nf4, scale)
    base_out = x @ W_fp16.T              # frozen, no grad
    lora_out = (x @ lora_A) @ lora_B     # trainable low-rank path
    return base_out + (alpha / rank) * lora_out
```

### Paged Optimizers

QLoRA uses CUDA unified memory to page optimizer states to CPU RAM automatically under memory pressure:

```python
import bitsandbytes as bnb
optimizer = bnb.optim.PagedAdamW(model.parameters(), lr=2e-4)
```

---

## DoRA: Weight-Decomposed Low-Rank Adaptation

DoRA (2024) decomposes a pretrained weight matrix into magnitude and direction, then applies LoRA only to the direction component. This mimics the learning pattern of full fine-tuning more closely.

### Mathematical Formulation

Full fine-tuning learns W' = W + dW. DoRA instead decomposes:

```
W = m * (V / ||V||_c)

where:
  m = ||W||_c          (column-wise magnitude, a vector)
  V = W                (direction matrix)
  ||V||_c = column norms of V
```

During DoRA training:

```
W' = (m + dm) * ((V + BA) / ||V + BA||_c)

where:
  dm:  trainable magnitude adjustment (vector, not matrix)
  B, A: standard LoRA low-rank matrices applied to direction V
```

### Why This Works Better

Full fine-tuning changes both magnitude and direction of weight columns. Standard LoRA entangles these two updates into a single low-rank matrix BA, which limits expressiveness. DoRA separates them:

```python
class DoRALinear(torch.nn.Module):
    def __init__(self, base_layer, rank=16, alpha=32):
        super().__init__()
        w = base_layer.weight
        w.requires_grad_(False)
        self.weight = w
        self.scaling = alpha / rank
        self.magnitude = torch.nn.Parameter(w.norm(dim=1, keepdim=True))
        self.lora_A = torch.nn.Parameter(torch.randn(w.shape[1], rank) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, w.shape[0]))

    def forward(self, x):
        adapted = self.weight + self.scaling * (self.lora_A @ self.lora_B).T
        direction = adapted / adapted.norm(dim=1, keepdim=True)
        return x @ (self.magnitude * direction).T
```

DoRA matches full fine-tuning at rank 8 where standard LoRA needs rank 32+, cutting trainable parameters by 4x for the same accuracy.

---

## LoRA+: Differential Learning Rates for A and B

LoRA+ (2024) observes that the A and B matrices have different gradient dynamics and should use different learning rates. The key finding: **B should have a much higher learning rate than A**.

### The Intuition

B is initialized to zero, A with Kaiming uniform. The gradients are asymmetric: dL/dB scales with ||A|| (which is O(1)) while dL/dA scales with ||B|| (which starts at 0). B needs a higher learning rate to compensate.

### Implementation

```python
def get_lora_plus_param_groups(model, lr_A=1e-5, lr_B=1e-3):
    """Separate param groups with different LRs. Ratio lr_B/lr_A ~ 16-100."""
    groups = []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        lr = lr_B if 'lora_B' in name else lr_A
        groups.append({'params': [p], 'lr': lr})
    return groups

optimizer = torch.optim.AdamW(get_lora_plus_param_groups(model))
```

LoRA+ achieves 1-2% accuracy improvements on commonsense reasoning benchmarks and converges faster (up to 2x fewer steps to reach the same loss).

---

## Multi-Adapter Serving: Punica and S-LoRA

Serving many LoRA adapters efficiently is critical for multi-tenant deployments where each user has a personalized fine-tune.

### The Problem

Naive approach: load N separate models. For 100 users with 7B models, that is 1.4 TB. Better: share the frozen base and swap LoRA adapters per request. But each request in a batch might need a different adapter, breaking batched GEMM.

### Punica: Multi-LoRA CUDA Kernels

Punica (2023) introduces SGMV (Segmented Gather Matrix-Vector), a custom CUDA kernel that applies different LoRA adapters to different requests in a single kernel launch using segment pointers.

```python
class MultiLoRAServer:
    def __init__(self, base_model, adapter_pool):
        self.base = base_model                    # shared, frozen
        self.adapter_buffer = pack_adapters(adapter_pool)  # contiguous GPU buffer

    def batched_forward(self, inputs, adapter_ids):
        base_out = self.base(inputs)
        lora_out = sgmv_kernel(inputs, self.adapter_buffer, adapter_ids)
        return base_out + lora_out
```

### S-LoRA: Scalable Serving

S-LoRA extends Punica with:

1. **Adapter memory management**: Pages adapters between GPU and CPU memory using a custom memory pool, similar to PagedAttention for KV caches
2. **Adapter scheduling**: Batches requests that share adapters together when possible, falling back to SGMV for mixed batches
3. **Unified paging**: Both KV cache pages and adapter weight pages managed by the same allocator

S-LoRA can serve **thousands of adapters** on a single GPU with minimal overhead compared to serving the base model alone:

```
Throughput comparison (A100 80GB, LLaMA-7B, 100 adapters):
  Base model only:           ~1200 tokens/s
  S-LoRA (100 adapters):     ~1150 tokens/s  (4% overhead)
  Naive per-adapter loading:  ~120 tokens/s  (10x slower)
```

---

## Rank Selection Strategies

Choosing the right LoRA rank is not just hyperparameter tuning. The optimal rank depends on the task, layer, and how much the layer needs to change from pretraining.

### SVD Analysis of Trained Adapters

After training a LoRA adapter, you can analyze the effective rank by looking at the singular values of the product BA:

```python
def analyze_adapter_rank(lora_A, lora_B, threshold=0.99):
    """SVD analysis of the trained weight update BA."""
    BA = (lora_B @ lora_A.T).float()
    U, S, Vh = torch.linalg.svd(BA, full_matrices=False)
    cumvar = (S ** 2).cumsum(0) / (S ** 2).sum()
    effective_rank = (cumvar < threshold).sum().item() + 1
    return effective_rank, S

# Typical findings per layer type:
#   Q/K projections: effective rank 2-4
#   V/O projections: effective rank 8-16
#   FFN up/down:     effective rank 16-64
```

### Layer-Wise Rank Allocation

Not all layers benefit equally from LoRA. A smarter approach allocates rank budget based on gradient sensitivity:

```python
def compute_layer_importance(model, dataloader, num_batches=10):
    """Gradient norm as proxy for how much each layer needs to change."""
    importance = {n: 0.0 for n, p in model.named_parameters()
                  if 'weight' in n and p.dim() == 2}
    for i, batch in enumerate(dataloader):
        if i >= num_batches: break
        model(**batch).loss.backward()
        for n, p in model.named_parameters():
            if n in importance and p.grad is not None:
                importance[n] += p.grad.norm().item()
        model.zero_grad()
    mx = max(importance.values())
    return {k: v/mx for k, v in importance.items()}
# High importance -> rank 32-64; Medium -> 8-16; Low -> 4 or skip
```

### Rules of Thumb

```
Task complexity vs. rank:
  Classification / sentiment:         rank 4-8
  Instruction following:              rank 16-32
  Domain-specific generation:         rank 32-64
  Code generation fine-tuning:        rank 64-128

Model size vs. rank:
  1-3B params:   rank 8-16 is usually sufficient
  7-13B params:  rank 16-32
  30-70B params: rank 32-64 (larger models are more over-parameterized)
```

---

## LoRA for Convolution and Embedding Layers

LoRA is not limited to linear layers. The same low-rank decomposition applies to any weight matrix.

### Convolutional Layers

A Conv2d weight is (out_channels, in_channels, kH, kW). Reshape to (out_channels, in_channels * kH * kW) and apply LoRA:

```python
class LoRAConv2d(torch.nn.Module):
    def __init__(self, base_conv, rank=8, alpha=16):
        super().__init__()
        self.base_conv = base_conv
        base_conv.weight.requires_grad_(False)
        flat_dim = base_conv.in_channels * base_conv.kernel_size[0] * base_conv.kernel_size[1]
        self.scaling = alpha / rank
        self.lora_A = torch.nn.Parameter(torch.randn(flat_dim, rank) * 0.01)
        self.lora_B = torch.nn.Parameter(torch.zeros(rank, base_conv.out_channels))

    def forward(self, x):
        base_out = self.base_conv(x)
        delta_w = (self.lora_A @ self.lora_B).T.reshape(self.base_conv.weight.shape)
        lora_out = torch.nn.functional.conv2d(
            x, delta_w * self.scaling,
            stride=self.base_conv.stride, padding=self.base_conv.padding)
        return base_out + lora_out
```

### Embedding Layers

For embedding tables (vocab_size, embed_dim), LoRA reduces the number of trainable parameters from vocab_size * embed_dim to (vocab_size + embed_dim) * rank:

```python
class LoRAEmbedding(torch.nn.Module):
    def __init__(self, base_embedding, rank=8, alpha=16):
        super().__init__()
        self.base = base_embedding
        base_embedding.weight.requires_grad_(False)
        self.scaling = alpha / rank
        self.lora_A = torch.nn.Embedding(base_embedding.num_embeddings, rank)
        self.lora_B = torch.nn.Linear(rank, base_embedding.embedding_dim, bias=False)
        torch.nn.init.zeros_(self.lora_B.weight)

    def forward(self, ids):
        return self.base(ids) + self.scaling * self.lora_B(self.lora_A(ids))
```

Useful for adapting models to new vocabularies without retraining the full embedding table.

---

## Merging Multiple Adapters: Task Arithmetic

When you have multiple LoRA adapters trained for different tasks, you can combine them through arithmetic operations on the weight updates.

### Task Vectors

A task vector is the difference between a fine-tuned model and its base: tau = W_ft - W_base. For LoRA, the task vector is simply the product BA (scaled by alpha/rank).

```python
def extract_task_vector(lora_A, lora_B, alpha, rank):
    """Task vector = scaled weight delta from LoRA."""
    return (alpha / rank) * (lora_B @ lora_A.T)

def merge_task_vectors(task_vectors, weights):
    return sum(w * tv for w, tv in zip(weights, task_vectors))
```

### Operations on Task Vectors

```python
# 1. Task addition: combine capabilities
#    e.g., merge a math adapter and a coding adapter
merged = 0.7 * tau_math + 0.5 * tau_code
W_merged = W_base + merged

# 2. Task negation: remove unwanted behavior
#    e.g., remove toxicity adapter from a model
W_detoxified = W_base - 0.5 * tau_toxic

# 3. Task analogies: transfer style
#    e.g., "formal English" minus "casual English" plus "casual French"
tau_formality = tau_formal_en - tau_casual_en
W_formal_fr = W_base + tau_casual_fr + tau_formality
```

### Resolving Interference Between Adapters

When merging adapters that modify the same parameters in conflicting ways, use TIES-Merging (Trim, Elect Sign, and Merge):

```python
def ties_merge(task_vectors, weights, density=0.2):
    """TIES: Trim low-magnitude, elect dominant sign, merge agreeing values."""
    trimmed = []
    for tv in task_vectors:
        thresh = tv.abs().quantile(1 - density)
        trimmed.append(tv * (tv.abs() >= thresh))
    stacked = torch.stack(trimmed)
    elected_sign = sum(w * t for w, t in zip(weights, trimmed)).sign()
    mask = (stacked.sign() == elected_sign.unsqueeze(0)) | (stacked == 0)
    return (stacked * mask).sum(0) / mask.float().sum(0).clamp(min=1)
```

### Practical Adapter Merging Workflow

```python
from peft import PeftModel

# Merge adapter into base weights permanently (no LoRA overhead at inference)
model = PeftModel.from_pretrained(base_model, "adapter_math")
merged = model.merge_and_unload()
merged.save_pretrained("merged_model")
```

---

## Training Stability Tips

LoRA training has its own failure modes distinct from full fine-tuning.

### Initialization Matters

```python
# Default: Kaiming for A, zeros for B (adapter starts as identity: BA = 0)
lora_A = torch.nn.Parameter(torch.randn(in_dim, rank) / rank**0.5)
lora_B = torch.nn.Parameter(torch.zeros(rank, out_dim))
# Alternative: small random B init (breaks zero-init but sometimes converges faster)
```

### Learning Rate and Alpha Coupling

The effective update magnitude is (alpha / rank) * lr. When changing rank, adjust alpha or lr to maintain the same effective scale:

```
rank=8,  alpha=16, lr=2e-4 -> effective scale: 2 * 2e-4 = 4e-4
rank=16, alpha=16, lr=2e-4 -> effective scale: 1 * 2e-4 = 2e-4  (halved!)
rank=16, alpha=32, lr=2e-4 -> effective scale: 2 * 2e-4 = 4e-4  (corrected)
```

Rule: keep alpha = 2 * rank as a starting point, then tune lr.

### Gradient Clipping for LoRA

LoRA parameters are much smaller than the base model, so gradients can be proportionally larger. Clip LoRA gradients aggressively (max_norm=0.3) separately from other trainable parameters (max_norm=1.0).

### Common Failure Modes

```
Problem: Loss spikes after many stable steps
Cause:   Learning rate too high for current rank
Fix:     Reduce lr by 2-4x, or increase rank

Problem: Training loss plateaus early, never reaches full FT loss
Cause:   Rank too low for the task complexity
Fix:     Increase rank or add LoRA to more layers (V, O, FFN)

Problem: Good train loss but poor eval (overfitting)
Cause:   Too many trainable params relative to dataset size
Fix:     Reduce rank, add dropout (0.05-0.1 on LoRA outputs)

Problem: NaN loss with QLoRA
Cause:   Dequantization instability with very small scales
Fix:     Use --bf16 instead of --fp16, ensure bitsandbytes >= 0.41
```

### Which Layers to Target

```
Minimum viable LoRA:  Q, V projections only
                      ~0.1% of params, decent for simple tasks

Recommended:          Q, K, V, O projections
                      ~0.2% of params, good balance

Full coverage:        Q, K, V, O, gate, up, down projections
                      ~0.5% of params, approaches full FT quality

Overkill:             All linear layers + embeddings + LM head
                      ~1-2% of params, rarely needed
```

### Training Checklist

1. Start with rank=16, alpha=32, lr=2e-4, target modules = [q_proj, v_proj]
2. Train for 1 epoch, evaluate
3. If underfitting: increase rank to 32, add k_proj and o_proj
4. If overfitting: add dropout=0.05, reduce rank, or reduce lr
5. For QLoRA: use bf16, paged_adamw_8bit, gradient_checkpointing=True
6. Always evaluate on held-out data every 100-200 steps
7. Save checkpoints: LoRA adapters are small (~20-100MB), save frequently

---

Get the video walkthrough: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
