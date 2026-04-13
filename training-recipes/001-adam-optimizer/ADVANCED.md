# Adam Optimizer: Advanced Topics

Beyond vanilla Adam — convergence bugs, memory-efficient variants, large-batch training, learning rate schedules, and the optimizers challenging Adam's dominance.

The core lesson covers Adam's mechanics: first moment (mean), second moment (variance), bias correction, and the update rule. This companion explores the failure modes, 8-bit compression, large-batch scaling laws, and practical scheduler recipes used in production LLM training.

---

## 1. The Adam convergence bug and AMSGrad

Adam has a proven convergence failure: in certain settings, the adaptive learning rate can become too large, causing the optimizer to diverge or converge to a bad solution.

### The problem

Adam divides by `sqrt(v_t)` where `v_t` is an exponential moving average of squared gradients. If a parameter gets a sequence of small gradients followed by a large gradient, `v_t` can decrease faster than the gradient grows, causing an enormous step.

```python
# Simplified example of Adam's convergence issue
# The second moment estimate can "forget" large past gradients

import torch

# Scenario: gradient is usually 0.001, occasionally 100
gradients = [0.001] * 100 + [100.0] + [0.001] * 100

v = 0.0
beta2 = 0.999
eps = 1e-8

for t, g in enumerate(gradients):
    v = beta2 * v + (1 - beta2) * g**2
    v_hat = v / (1 - beta2**(t+1))  # bias correction
    step = g / (v_hat**0.5 + eps)

    if abs(step) > 10:
        print(f"Step {t}: grad={g:.3f}, v_hat={v_hat:.6f}, "
              f"step={step:.1f}  <- HUGE STEP")
```

### AMSGrad: the fix

AMSGrad (Reddi et al., 2018) maintains the maximum of all past `v_t` values, preventing the denominator from shrinking.

```python
class AMSGrad:
    """Adam with AMSGrad fix for convergence guarantee."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0

        # State
        self.m = [torch.zeros_like(p) for p in self.params]  # first moment
        self.v = [torch.zeros_like(p) for p in self.params]  # second moment
        self.v_max = [torch.zeros_like(p) for p in self.params]  # AMSGrad max

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data

            # Update moments
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            # AMSGrad: take element-wise maximum
            self.v_max[i] = torch.maximum(self.v_max[i], self.v[i])

            # Bias correction
            m_hat = self.m[i] / (1 - self.beta1**self.t)
            # Use v_max instead of v for the denominator
            v_max_hat = self.v_max[i] / (1 - self.beta2**self.t)

            p.data -= self.lr * m_hat / (v_max_hat.sqrt() + self.eps)
```

In practice, AMSGrad rarely changes outcomes for LLM training because the convergence bug requires adversarial gradient sequences that rarely occur in practice. But it is theoretically important.

---

## 2. 8-bit Adam (bitsandbytes)

Adam stores two state tensors (m and v) per parameter, both in float32. For a 7B model, that is 7B * 2 * 4 bytes = 56 GB of optimizer state. 8-bit Adam compresses this to ~14 GB.

```
Optimizer state memory for a 7B model:

  Standard Adam:
    Parameters:  7B * 2 bytes (fp16)     = 14 GB
    Moments m:   7B * 4 bytes (fp32)     = 28 GB
    Moments v:   7B * 4 bytes (fp32)     = 28 GB
    Total:                                = 70 GB

  8-bit Adam:
    Parameters:  7B * 2 bytes (fp16)     = 14 GB
    Moments m:   7B * 1 byte  (int8)     =  7 GB
    Moments v:   7B * 1 byte  (int8)     =  7 GB
    Quantization maps:                    ≈ 0.1 GB
    Total:                                ≈ 28 GB  (2.5x reduction)
```

### How 8-bit Adam works

The key insight: optimizer states don't need full precision. The moments are running averages that change slowly, so they can be quantized with dynamic quantization maps.

```python
class Adam8bit:
    """Simplified 8-bit Adam (bitsandbytes approach)."""
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 block_size=2048):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.block_size = block_size

        # Quantized state
        self.m_quant = []  # int8 first moments
        self.v_quant = []  # int8 second moments
        self.m_absmax = []  # per-block scale factors for m
        self.v_absmax = []  # per-block scale factors for v

        for p in self.params:
            n = p.numel()
            n_blocks = (n + block_size - 1) // block_size
            self.m_quant.append(torch.zeros(n, dtype=torch.int8, device=p.device))
            self.v_quant.append(torch.zeros(n, dtype=torch.int8, device=p.device))
            self.m_absmax.append(torch.zeros(n_blocks, device=p.device))
            self.v_absmax.append(torch.zeros(n_blocks, device=p.device))

    def _quantize_blockwise(self, tensor, block_size):
        """Quantize a flat tensor to int8 with per-block scaling."""
        flat = tensor.reshape(-1)
        n = flat.numel()
        n_blocks = (n + block_size - 1) // block_size
        # Pad if needed
        padded = torch.zeros(n_blocks * block_size, device=flat.device)
        padded[:n] = flat

        blocks = padded.view(n_blocks, block_size)
        absmax = blocks.abs().amax(dim=1)  # (n_blocks,)
        scale = absmax / 127.0
        scale = scale.clamp(min=1e-12)

        quantized = (blocks / scale.unsqueeze(1)).round().clamp(-127, 127).to(torch.int8)
        return quantized.reshape(-1)[:n], absmax

    def _dequantize_blockwise(self, quant, absmax, block_size, n):
        """Dequantize int8 back to float."""
        n_blocks = (n + block_size - 1) // block_size
        padded = torch.zeros(n_blocks * block_size, device=quant.device)
        padded[:n] = quant[:n].float()

        blocks = padded.view(n_blocks, block_size)
        scale = absmax / 127.0
        dequant = blocks * scale.unsqueeze(1)
        return dequant.reshape(-1)[:n]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data.float().reshape(-1)
            n = g.numel()

            # Dequantize current state
            m = self._dequantize_blockwise(
                self.m_quant[i], self.m_absmax[i], self.block_size, n)
            v = self._dequantize_blockwise(
                self.v_quant[i], self.v_absmax[i], self.block_size, n)

            # Standard Adam update
            m = self.beta1 * m + (1 - self.beta1) * g
            v = self.beta2 * v + (1 - self.beta2) * g**2

            m_hat = m / (1 - self.beta1**self.t)
            v_hat = v / (1 - self.beta2**self.t)

            update = m_hat / (v_hat.sqrt() + self.eps)
            p.data -= self.lr * update.to(p.dtype).view(p.shape)

            # Re-quantize updated state
            self.m_quant[i], self.m_absmax[i] = self._quantize_blockwise(
                m, self.block_size)
            self.v_quant[i], self.v_absmax[i] = self._quantize_blockwise(
                v, self.block_size)
```

### Using bitsandbytes in practice

```python
import bitsandbytes as bnb

# Drop-in replacement for torch.optim.Adam
optimizer = bnb.optim.Adam8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

# Or AdamW variant
optimizer = bnb.optim.AdamW8bit(
    model.parameters(),
    lr=1e-4,
    betas=(0.9, 0.95),
    weight_decay=0.1,
)

# Works exactly like standard Adam — same API, ~2.5x less memory
```

---

## 3. LAMB and LARS for large-batch training

When scaling batch size from 256 to 32K+, Adam's learning rate needs careful per-layer tuning. LAMB (Layer-wise Adaptive Moments) and LARS (Layer-wise Adaptive Rate Scaling) solve this automatically.

### The large-batch problem

```
Small batch (256):    Adam with lr=1e-4 converges fine
Large batch (32K):    Same lr diverges. Lower lr converges but slower.

Why: gradient noise decreases with larger batches (√batch_size scaling).
Different layers have different optimal learning rates at large batch sizes.
The embedding layer wants a small lr, middle layers want medium, etc.
```

### LAMB: Layer-wise Adaptive Moments

```python
class LAMB:
    """LAMB optimizer for large-batch training (You et al., 2020).
    Used to train BERT in 76 minutes with batch size 64K.
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-6,
                 weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def step(self):
        self.t += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data

            # Standard Adam moment updates
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * g**2

            m_hat = self.m[i] / (1 - self.beta1**self.t)
            v_hat = self.v[i] / (1 - self.beta2**self.t)

            # Adam-style update direction (with weight decay)
            update = m_hat / (v_hat.sqrt() + self.eps) + self.weight_decay * p.data

            # LAMB trust ratio: scale update per-layer
            weight_norm = p.data.norm()
            update_norm = update.norm()

            if weight_norm > 0 and update_norm > 0:
                trust_ratio = weight_norm / update_norm
            else:
                trust_ratio = 1.0

            p.data -= self.lr * trust_ratio * update
```

The trust ratio is the key innovation: it scales each layer's update so that the relative change (update_norm / weight_norm) is consistent across layers.

---

## 4. Gradient accumulation

When your desired batch size does not fit in GPU memory, accumulate gradients across multiple mini-batches before taking an optimizer step.

```python
def train_with_gradient_accumulation(
    model, dataloader, optimizer, scheduler,
    accumulation_steps=8, max_grad_norm=1.0,
):
    """Effective batch size = micro_batch_size * accumulation_steps.
    
    If micro_batch = 4 and accumulation_steps = 8:
      effective_batch = 32
    """
    model.train()
    optimizer.zero_grad()

    for step, batch in enumerate(dataloader):
        # Forward pass
        outputs = model(**batch)
        # Scale loss by accumulation steps (so gradients average correctly)
        loss = outputs.loss / accumulation_steps
        loss.backward()

        if (step + 1) % accumulation_steps == 0:
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            # Take optimizer step
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            effective_step = (step + 1) // accumulation_steps
            if effective_step % 100 == 0:
                print(f"Step {effective_step}: loss={loss.item() * accumulation_steps:.4f}")
```

### Common gradient accumulation bugs

```python
# BUG 1: Forgetting to scale the loss
loss = outputs.loss
loss.backward()  # WRONG: gradients are accumulation_steps times too large

# FIX:
loss = outputs.loss / accumulation_steps
loss.backward()

# BUG 2: Using batch norm with gradient accumulation
# BatchNorm statistics are computed per micro-batch, not per effective batch
# Solution: use LayerNorm/RMSNorm (no batch statistics)

# BUG 3: Learning rate scheduler stepping per micro-batch instead of per step
scheduler.step()  # Called every micro-batch -> lr decays too fast!
# FIX: Only step scheduler when optimizer steps
if (step + 1) % accumulation_steps == 0:
    scheduler.step()
```

---

## 5. Learning rate finder

The learning rate finder (Smith, 2017) trains for a few hundred steps while exponentially increasing the learning rate. Plot loss vs lr to find the optimal range.

```python
import math

def lr_finder(model, dataloader, optimizer_class, min_lr=1e-7, max_lr=10,
              n_steps=300, smooth_factor=0.05):
    """Find optimal learning rate by scanning exponentially."""
    model_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Exponential lr schedule: min_lr -> max_lr over n_steps
    gamma = (max_lr / min_lr) ** (1 / n_steps)

    optimizer = optimizer_class(model.parameters(), lr=min_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    lrs = []
    losses = []
    smoothed_loss = None
    best_loss = float('inf')

    data_iter = iter(dataloader)
    for step in range(n_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        optimizer.zero_grad()
        loss = model(**batch).loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Exponential smoothing
        current_loss = loss.item()
        if smoothed_loss is None:
            smoothed_loss = current_loss
        else:
            smoothed_loss = smooth_factor * current_loss + (1 - smooth_factor) * smoothed_loss

        current_lr = scheduler.get_last_lr()[0]
        lrs.append(current_lr)
        losses.append(smoothed_loss)

        # Stop if loss explodes
        if smoothed_loss > best_loss * 4:
            break
        best_loss = min(best_loss, smoothed_loss)

    # Restore model state
    model.load_state_dict(model_state)

    # Optimal lr is typically 1/10 of the lr where loss starts increasing sharply
    # Find the steepest negative slope
    best_idx = 0
    best_slope = 0
    for i in range(1, len(losses)):
        slope = (losses[i-1] - losses[i]) / (math.log10(lrs[i]) - math.log10(lrs[i-1]))
        if slope > best_slope:
            best_slope = slope
            best_idx = i
    suggested_lr = lrs[best_idx] / 10

    print(f"Suggested learning rate: {suggested_lr:.2e}")
    return lrs, losses, suggested_lr
```

---

## 6. Scheduler deep dive

### Linear warmup + cosine decay (the LLM standard)

```python
import math

class WarmupCosineScheduler:
    """The standard LLM training scheduler.
    
    Phase 1 (warmup): lr increases linearly from 0 to max_lr
    Phase 2 (cosine): lr decreases following a cosine curve to min_lr
    """
    def __init__(self, optimizer, warmup_steps, total_steps,
                 max_lr, min_lr=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self.get_lr()
        for group in self.optimizer.param_groups:
            group['lr'] = lr

    def get_lr(self):
        if self.current_step < self.warmup_steps:
            # Linear warmup
            return self.max_lr * self.current_step / self.warmup_steps
        else:
            # Cosine decay
            progress = (self.current_step - self.warmup_steps) / \
                       (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            cosine = 0.5 * (1 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine
```

```
Learning rate over training:

  max_lr |    /\
         |   /  \
         |  /    \__
         | /        \___
  min_lr |/             \____
         +-----|------------>
         0   warmup    total_steps
```

### Cosine with warm restarts (SGDR)

Periodically reset the learning rate to escape local minima:

```python
class CosineWarmRestarts:
    """Cosine annealing with warm restarts (Loshchilov & Hutter, 2017).
    
    Each restart doubles the cycle length (T_mult=2).
    """
    def __init__(self, optimizer, T_0, T_mult=2, eta_min=0, warmup_steps=0):
        self.optimizer = optimizer
        self.T_0 = T_0          # first cycle length
        self.T_mult = T_mult    # cycle length multiplier
        self.eta_min = eta_min
        self.warmup_steps = warmup_steps
        self.current_step = 0
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self):
        self.current_step += 1

        if self.current_step < self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            step = self.current_step - self.warmup_steps
            # Find which cycle we're in
            cycle = 0
            cycle_start = 0
            cycle_len = self.T_0
            while step >= cycle_start + cycle_len:
                cycle_start += cycle_len
                cycle_len *= self.T_mult
                cycle += 1

            progress = (step - cycle_start) / cycle_len
            lr = self.eta_min + (self.base_lr - self.eta_min) * \
                 0.5 * (1 + math.cos(math.pi * progress))

        for group in self.optimizer.param_groups:
            group['lr'] = lr
```

```
Cosine with warm restarts (T_0=1000, T_mult=2):

  max_lr |/\/\    /\      /\          /\
         |    \  /  \    /  \        /  \
         |     \/    \  /    \      /    \
         |           \/      \    /      \
  min_lr |                    \/\/        \/
         +---|-----|----------|---------->
         0  1K    3K         7K       15K steps
              cycle 1   cycle 2     cycle 3
```

### Common scheduler configurations

```
Model          max_lr    min_lr     warmup     Schedule
──────────────────────────────────────────────────────────────
GPT-3 175B     6e-5      6e-6       375 steps  Cosine (300B tokens)
LLaMA 7B       3e-4      3e-5       2000 steps Cosine (1T tokens)
LLaMA 65B      1.5e-4    1.5e-5     2000 steps Cosine (1.4T tokens)
Chinchilla     1e-4      1e-5       400M tokens Cosine
Mistral 7B     3e-4      (unknown)  (unknown)  Cosine

Rules of thumb:
  - Warmup: 0.1-1% of total steps (rarely more than 2000 steps)
  - min_lr: 10% of max_lr (1/10 ratio)
  - max_lr: scale inversely with sqrt(model_size)
    7B -> 3e-4, 13B -> 2e-4, 70B -> 1.5e-4
```

---

## 7. Optimizer state checkpointing

Resuming training requires saving and loading optimizer state, not just model weights.

```python
def save_training_checkpoint(model, optimizer, scheduler, step, path):
    """Save everything needed to resume training."""
    checkpoint = {
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
        'rng_state': torch.random.get_rng_state(),
        'cuda_rng_state': torch.cuda.get_rng_state_all(),
    }
    torch.save(checkpoint, path)
    print(f"Saved checkpoint at step {step} to {path}")

def load_training_checkpoint(model, optimizer, scheduler, path):
    """Resume training from checkpoint."""
    checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if checkpoint['scheduler_state_dict'] and scheduler:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    torch.random.set_rng_state(checkpoint['rng_state'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state'])
    print(f"Resumed from step {checkpoint['step']}")
    return checkpoint['step']
```

### Checkpoint sizes

```
Model    Params   Weights (fp16)   Optimizer State   Total Checkpoint
──────────────────────────────────────────────────────────────────────
  1.3B    1.3B      2.6 GB          10.4 GB           13.0 GB
  7B      7B       14.0 GB          56.0 GB           70.0 GB
  13B    13B       26.0 GB         104.0 GB          130.0 GB
  70B    70B      140.0 GB         560.0 GB          700.0 GB

With 8-bit Adam:
  7B     7B       14.0 GB          14.0 GB           28.0 GB  (2.5x smaller)
```

---

## 8. Lion optimizer: the challenger

Lion (Chen et al., 2023) was discovered through evolutionary search over optimizer update rules. It uses only the sign of the moment, making it memory-efficient and surprisingly effective.

```python
class Lion:
    """Lion optimizer (EvoLved Sign Momentum).
    Uses sign(momentum) instead of adaptive learning rate.
    Only stores one moment (vs Adam's two) -> 50% less optimizer memory.
    """
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.1):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.weight_decay = weight_decay

        # Only ONE moment tensor (vs Adam's two)
        self.m = [torch.zeros_like(p) for p in self.params]

    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue
            g = p.grad.data

            # Interpolate between momentum and gradient for update direction
            update = self.beta1 * self.m[i] + (1 - self.beta1) * g

            # Take the SIGN — every parameter gets exactly +lr or -lr update
            p.data -= self.lr * (update.sign() + self.weight_decay * p.data)

            # Update momentum (using beta2, different from update beta1)
            self.m[i] = self.beta2 * self.m[i] + (1 - self.beta2) * g
```

### Lion vs Adam comparison

```
                   Adam            Lion
─────────────────────────────────────────────
Moments stored:    2 (m, v)        1 (m)
Memory per param:  8 bytes (fp32)  4 bytes (fp32)
Update rule:       m / sqrt(v)     sign(m)
Typical lr:        3e-4            1e-4 (3x smaller)
Weight decay:      0.1             0.3 (3x larger)
Best for:          General         Vision, moderate-scale LLMs

Memory savings for 7B model:
  Adam:  7B * 8 bytes = 56 GB optimizer state
  Lion:  7B * 4 bytes = 28 GB optimizer state (2x reduction)
```

Lion works well for vision models and moderate-scale language models. For the largest LLMs (70B+), Adam/AdamW remains the safe default because Lion's sign-based updates can be unstable with very aggressive learning rate schedules.

---

## 9. Practical optimizer selection

```
Choosing an optimizer for your project:

Standard LLM pre-training:
  -> AdamW with (beta1=0.9, beta2=0.95, eps=1e-8, wd=0.1)
  -> Warmup cosine schedule, min_lr = max_lr / 10
  -> This is the proven default

Memory-constrained training:
  -> AdamW 8-bit (bitsandbytes): 2.5x less optimizer memory
  -> Or Lion: 2x less optimizer memory, but needs lr/wd tuning

Large-batch training (batch > 8K):
  -> LAMB with layer-wise trust ratios
  -> Or AdamW with careful linear lr scaling: lr *= batch_size / 256

Fine-tuning:
  -> AdamW with lr 10-100x smaller than pre-training
  -> Short warmup (50-200 steps)
  -> Consider LoRA to reduce optimizer state entirely

Resume from checkpoint:
  -> MUST load optimizer state, not just model weights
  -> Verify lr schedule position matches saved step count
```

---

Get the video walkthrough: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
