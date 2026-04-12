# Mixed Precision Training From Scratch

Mixed precision cuts your memory footprint in half and speeds up training — but it silently destroys gradients if you do not understand what is happening inside. This is what `torch.cuda.amp` actually does.

---

## Why Bother With Precision?

Every tensor in PyTorch defaults to FP32 (32-bit floating point). That is 4 bytes per parameter. A 1 billion parameter model needs 4 GB just for weights — before gradients, optimizer states, or activations.

FP16 (half precision) cuts that to 2 GB. BF16 (brain float) also uses 2 bytes but with a smarter layout that is more numerically stable. On Ampere+ GPUs, TF32 (tensor float 32) runs at FP16 speed but with FP32 accuracy for matmuls.

The catch: FP16 gradients can round to zero. The fix: GradScaler. Understanding both is what this tutorial covers.

---

## FP16 vs BF16

Both formats use 16 bits. The difference is how those 16 bits are allocated.

```
FP16:  1 sign | 5 exponent | 10 mantissa
BF16:  1 sign | 8 exponent |  7 mantissa
```

BF16 has a larger exponent range. This means it rarely overflows or underflows — the big problem with FP16 on deep networks. BF16 handles the same range as FP32, just with less precision.

Use FP16 when:
- You have a newer GPU (Ampere+) with Tensor Cores optimized for FP16
- You need maximum speed and your model is numerically stable

Use BF16 when:
- You are unsure — it is the safer default
- Your model suffers from gradient underflow in FP16
- You are on older hardware that does not support TF32

---

## torch.cuda.amp.autocast

Autocast patches operations to run in lower precision when it is safe. Wrap a forward pass and PyTorch automatically uses FP16 or BF16 for supported ops.

Here is a minimal example:

```python
import torch
from torch.cuda.amp import autocast

model = model.cuda()
optimizer = torch.optim.Adam(model.parameters())

for inputs, targets in dataloader:
    inputs, targets = inputs.cuda(), targets.cuda()

    # autocast automatically chooses the right precision
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

What autocast actually does under the hood: it patches the dispatch table for specific ops. When you call `torch.matmul`, autocast intercepts it and calls `torch._amp_foreach_matmul` instead — a CUDA kernel that runs in FP16/BF16.

Not all ops are patched. Autocast falls back to FP32 for unsupported operations.

---

## The Problem: Small Gradients Round to Zero

FP16 has a narrower exponent range than FP32. Small gradient values can fall below the minimum representable FP16 value — they round to zero. This breaks training.

BF16 largely solves this with its larger exponent. But FP16 still needs a workaround: loss scaling.

---

## GradScaler: The Fix

GradScaler multiplies the loss by a scale factor before backpropagation. Gradients are proportionally larger — they stay representable in FP16. After the backward pass, GradScaler un-scales the gradients to recover the true values.

Step-by-step:

```
1. loss_scale = 128.0  (or auto-detected by GradScaler)

2. scaled_loss = loss * loss_scale

3. scaled_loss.backward()   # gradients are scaled UP

4. gradients = model.parameters().grad / loss_scale  # un-scale

5. optimizer.step()

6. loss_scale *= 1.01  # gradually increase if no overflow
   # or: loss_scale *= 0.9 if overflow detected
```

Using GradScaler:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in dataloader:
    inputs, targets = inputs.cuda(), targets.cuda()

    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

`scale(loss).backward()` scales the loss. `scaler.step(optimizer)` unscales gradients before the optimizer update. `scaler.update()` adjusts the scale factor for the next iteration.

---

## torch.set_float32_matmul_precision

Ampere+ GPUs (A100, RTX 30xx, RTX 40xx) support TF32 — a format that matmul operations use internally. TF32 has the range of FP32 with the precision of FP16. It runs at FP16 speed.

Enable TF32:

```python
torch.set_float32_matmul_precision("high")  # "highest" also available
```

This makes matmuls use TF32 internally while keeping your model weights in FP32. It is the fastest option on Ampere+ for most workloads.

---

## When Mixed Precision Helps vs Hurts

Mixed precision helps when:
- Your model is large and memory-bound
- You have an Ampere+ GPU with Tensor Cores
- You are running FP16 with GradScaler properly configured

Mixed precision hurts when:
- Your model is numerically unstable in FP16 — try BF16 instead
- You are on older hardware without good FP16 support
- You are I/O bound, not compute bound — memory savings do not help

---

## A Minimal End-to-End Example

### Piece 1: Model and data setup

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(512, 256)
        self.linear2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.linear2(x)
        return x

model = SimpleModel().cuda()
optimizer = torch.optim.Adam(model.parameters())
loss_fn = nn.CrossEntropyLoss()
```

### Piece 2: Enable TF32 on Ampere+ GPUs

```python
torch.set_float32_matmul_precision("high")
```

### Piece 3: GradScaler setup

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
```

### Piece 4: Training loop with autocast

```python
for inputs, targets in dataloader:
    inputs, targets = inputs.cuda(), targets.cuda()

    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## Recap

- **FP16 halves memory, BF16 is safer.** Both use 2 bytes vs 4 for FP32.
- **BF16 has a larger exponent.** It rarely underflows — better for deep networks.
- **Autocast patches ops.** Wrap `forward()` with `autocast()` and supported ops run in lower precision.
- **GradScaler fixes underflow.** Scale the loss up, unscale gradients before optimizer step.
- **TF32 on Ampere+ is free speed.** Set `torch.set_float32_matmul_precision("high")`.
- **Measure first.** Run the benchmark scripts to see if mixed precision helps your workload.

---

## Going Further

For convergence curves across precision modes, TF32 speed benchmarks, GradScaler recovery measurements, and numerical stability analysis — see [ADVANCED.md](./ADVANCED.md).

Sources:
- https://pytorch.org/docs/stable/amp.html
- https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
- https://arxiv.org/abs/1710.03740 (Mixed Precision Training)
