# Adam Optimizer — Extended Notebook ($49)

> This document contains the extended content for the Adam optimizer lesson.
> Available at: https://www.skool.com/opensuperintelligencelab

---

## AdamW: Decoupled Weight Decay

### The Problem with L2 + Adam

Standard Adam with L2 regularization (weight decay) applies decay to all parameters equally via the loss:

```
loss = f(theta) + (lambda / 2) * ||theta||^2
```

But this interacts poorly with the adaptive LR. The gradient of the L2 term is `lambda * theta`, which gets scaled by the per-parameter adaptive LR — effectively making the weight decay *adaptive* when it should be *fixed*. Loshchilov & Hutter (2019) showed this leads to suboptimal regularization.

### Decoupled Weight Decay (AdamW)

AdamW separates the weight decay from the gradient update:

```python
# Standard Adam + L2 (incorrect):
grad += weight_decay * theta
theta -= lr * grad / sqrt(v)

# AdamW (correct — decoupled):
theta -= lr * (grad / sqrt(v) + weight_decay * theta)
```

The weight decay is applied directly to the parameters, not scaled by `1/sqrt(v)`. This gives a fixed decay regardless of the gradient geometry.

**Practical note:** AdamW is the default optimizer in most modern LLMs (Llama, Mistral, GPT-NeoX). Use `weight_decay=0.1` as a starting point for transformer models.

```python
class AdamW:
    def __init__(self, params, lr=1e-3, weight_decay=1e-2,
                 beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state = {
            id(p): {"m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": 0}
            for p in self.params
        }

    def step(self):
        for p in self.params:
            grad = p.grad
            if grad is None:
                continue
            s = self.state[id(p)]
            s["t"] += 1
            t = s["t"]

            # Decouple weight decay (apply before update)
            p.data.mul_(1 - self.lr * self.weight_decay)

            # Adam update
            s["m"].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            s["v"].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            m_hat = s["m"] / (1 - self.beta1 ** t)
            v_hat = s["v"] / (1 - self.beta2 ** t)
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad.zero_()
```

---

## LAMB: Layer-wise Adaptive Moments for Large Batch Training

### The Problem LAMB Solves

When training with large batch sizes (e.g., 4096+), the gradient becomes a sum of many independent per-sample gradients. The noise reduction allows for much larger learning rates — but naive scaling by batch size causes instability. LAMB adapts the LR at the layer level.

### LAMB Derivation

LAMB (You et al., 2020) modifies the Adam update:

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)

r_t = m_hat / (sqrt(v_hat) + eps)

# Layer normalization: use norm of parameter
theta_t = theta_{t-1} - lr * r_t / ||theta_{t-1}|| * ||r_t||
```

The key addition is the ratio `||theta|| / ||r_t||` which clamps the update norm to the parameter norm. This prevents gradient explosion in early layers with small parameters.

```python
class LAMB:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.state = {
            id(p): {"m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": 0}
            for p in self.params
        }

    def step(self):
        for p in self.params:
            grad = p.grad
            if grad is None:
                continue
            s = self.state[id(p)]
            s["t"] += 1
            t = s["t"]

            # Weight decay (decoupled, like AdamW)
            if self.weight_decay > 0:
                p.data.mul_(1 - self.lr * self.weight_decay)

            # Adam moments
            s["m"].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            s["v"].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            m_hat = s["m"] / (1 - self.beta1 ** t)
            v_hat = s["v"] / (1 - self.beta2 ** t)

            # LAMB: ratio of norms
            r_t = m_hat / (v_hat.sqrt() + self.eps)
            theta_norm = p.data.norm()
            r_norm = r_t.norm()
            # Clamp to avoid extreme ratios
            sigma = (theta_norm / (r_norm + self.eps)).clamp(max=10.0)
            p.data.add_(r_t, alpha=-self.lr * sigma.item())

    def zero_grad(self):
        for p in self.params:
            p.grad.zero_()
```

**When to use LAMB:** BERT training with batch sizes > 1024. LAMB enabled training with batch sizes up to 65536 without degradation.

---

## Sophia: A Smarter Second-Order-like Update

### The Core Idea

Sophia (Liu et al., 2023) replaces the `1/sqrt(v)` scaling in Adam with a Hessian-based preconditioner estimated via a simple running average of gradients. The update is:

```
h_t = beta2 * h_{t-1} + (1 - beta2) * g_t^2  # same as Adam's v
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t

update = -lr * m_t / (h_t + eps)
theta_{t+1} = theta_t + update * rho_t
```

where `rho_t = clip(update, delta)` prevents large steps. The key difference from Adam: `h_t` uses the **gradient itself** not the squared gradient — making Sophia a lightweight Hessian estimator.

**Why it works:** The Hessian diagonal tells us the local curvature. Using `g^2` as a proxy captures whether we're in a steep or flat region without expensive second-derivative computation.

```python
class Sophia:
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999,
                 eps=1e-8, delta=10.0):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.delta = delta
        self.state = {
            id(p): {"m": torch.zeros_like(p), "h": torch.zeros_like(p), "t": 0}
            for p in self.params
        }

    def step(self):
        for p in self.params:
            grad = p.grad
            if grad is None:
                continue
            s = self.state[id(p)]
            s["t"] += 1
            t = s["t"]

            # Update first moment (momentum)
            s["m"].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            # Update Hessian estimate (plain gradient, not squared)
            s["h"].mul_(self.beta2).add_(grad, alpha=1 - self.beta2)

            m_hat = s["m"] / (1 - self.beta1 ** t)
            h_hat = s["h"] / (1 - self.beta2 ** t)

            # Preconditioned update
            update = -self.lr * m_hat / (h_hat + self.eps)
            # Clamp step size (Sophia's key trick)
            update.clamp_(min=-self.delta, max=self.delta)
            p.data.add_(update)

    def zero_grad(self):
        for p in self.params:
            p.grad.zero_()
```

**Results:** Sophia achieves 2x better efficiency (samples to target loss) than Adam on GPT-2 training.

---

## Adaptive Optimizers for Large Batch Training

Large batch training (batch size > 1024) presents a challenge: gradient noise reduction lets you take bigger steps, but naive LR scaling causes instability. Several adaptations help:

### 1. Linear Warmup

Gradually increase the learning rate from 0 to the target LR over the first `k` steps (typically `k = 2000–10000`). This stabilizes early training when parameters are randomly initialized and gradients can be large.

```python
def get_lr(step, base_lr, warmup_steps, total_steps):
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    # Cosine decay after warmup
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return base_lr * 0.5 * (1 + np.cos(np.pi * progress))
```

### 2. Layer-wise Learning Rate Decay (LLRD)

In transformers, earlier layers need smaller learning rates than later layers. Apply a decay factor (e.g., 0.9) per layer from the top down:

```python
def apply_llrd(model, base_lr, decay_factor=0.9):
    lr = base_lr
    for layer in reversed(model.transformer.layers):
        for param in layer.parameters():
            param.optim = {"lr": lr}
        lr *= decay_factor
```

### 3. Gradient Clipping

Clip gradient norm to prevent explosion during large-batch training:

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

### 4. LAMB + Gradient Accumulation

For batches that don't fit in memory, accumulate gradients over smaller sub-batches then apply LAMB's layer-normalized update. This gives you the effective batch size without the memory overhead.

---

## Best Practices for LR Scheduling with Adam

### Schedule Types (in order of effectiveness for Adam-trained models)

**1. Cosine Annealing with Warm Restarts (CosineSchedule)**

Smooth decay that works well for most tasks. The learning rate decreases following a cosine curve from `lr_max` to `lr_min` over the schedule period.

```
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(t / T * pi))
```

**2. Inverse Square Root Schedule**

Used in original Transformer paper. The LR decays as `1/sqrt(max(step, warmup_steps))` after warmup.

```
lr_t = d_model^-0.5 * min(step^-0.5, step * warmup_steps^-1.5)
```

**3. Polynomial Decay**

Used in GPT-2. Decay following `(1 - step/total)^power` — often power = 2.

**4. Constant with Warmup (most common for LLMs)**

Warmup for 2000 steps, then hold at `lr_max` for most of training, optionally decay at the very end.

### Warmup is Non-Negotiable for Adam

Adam's bias correction depends on `t` being large. In early steps, `m` and `v` are biased. Warmup allows the moments to accumulate properly before large updates occur.

A good default for LLM training:
- Warmup steps: 2000 (or 0.5–1% of total steps, whichever is larger)
- Base LR: 1e-3 to 3e-4 (smaller for larger models)
- Minimum LR: 1e-5 to 1e-6 (to preserve fine-tuning)

### LR and Batch Size Scaling

When scaling batch size, scale the LR linearly initially, then optionally switch to LAMB or add gradient accumulation to maintain effective batch size:

```
effective_batch_size = batch_size * grad_accum_steps
lr = base_lr * (effective_batch_size / 256)^0.5  # sqrt scaling
```

This sqrt scaling (known as "linear + sqrt scaling") was popularized by Goyal et al. (2017) and works reasonably well for Adam up to batch sizes of ~4096.

---

## Quick Reference: Which Optimizer to Use?

| Scenario | Optimizer | Key Hyperparameter |
|---|---|---|
| General purpose (CNN, small transformers) | Adam | lr=1e-3 |
| Fine-tuning LLMs | AdamW | lr=1e-4 to 3e-4, wd=0.1 |
| Large batch training (BS > 4096) | LAMB | lr=1e-3 to 3e-3 |
| Very large models (trillion+ params) | Sophia | lr=1e-3, delta=10 |
| Simple SGD baseline | SGD + Momentum | lr=0.1, momentum=0.9 |
