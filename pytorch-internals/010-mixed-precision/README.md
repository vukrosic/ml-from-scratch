# Mixed Precision Training From Scratch

> 🔴 YouTube Lesson: Coming soon | 🟡 Skool Advanced Video Lesson: [Join the advanced lesson](https://www.skool.com/become-ai-researcher-2669/about)

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

**Bit layout with actual bit patterns (example number: 1.5 in each format):**

```
FP16 layout:  [S][E][E][E][E][E][M][M][M][M][M][M][M][M][M][M]
              0   0  0  0  0  0  1  1  0  0  0  0  0  0  0  0
              |   \_____5 bits____/  \_____10 bits_____/
              |        exponent           mantissa (with implicit 1)
              sign

              Example: 1.5 in FP16
              - Sign: 0 (positive)
              - Exponent: 01111 (15) → bias 15 → actual exponent = 0
              - Mantissa: 1000000000 (binary) = 0.5 (implicit 1.0 + 0.5 = 1.5)

BF16 layout:  [S][E][E][E][E][E][E][E][E][M][M][M][M][M][M][M]
              0   0  0  0  0  0  0  0  0  1  1  0  0  0  0  0
              |   \_____8 bits____/  \_____7 bits_____/
              |        exponent           mantissa (with implicit 1)
              sign

              Example: 1.5 in BF16
              - Sign: 0 (positive)
              - Exponent: 01111111 (127) → bias 127 → actual exponent = 0
              - Mantissa: 1000000 (binary) = 0.5 (implicit 1.0 + 0.5 = 1.5)

Why the exponent difference matters:

  FP16 max exponent:  2^(2^5 - 1 - bias) = 2^(15 - 15 + 1) = 2^1 = ~65504
  BF16 max exponent:  2^(2^8 - 1 - bias) = 2^(127 - 127 + 1) = 2^1 = ~3.4e38

  The exponent range of BF16 matches FP32, just with less mantissa precision.
  FP16's small exponent range is why gradients underflow in deep networks.
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

**What ops get patched — before/after dispatch for matmul:**

```
BEFORE autocast (dispatch table):
  torch.matmul  →  torch.Tensor.matmul  →  CPU/CUDA kernel (FP32)

AFTER autocast() enters (dispatch table is patched):
  torch.matmul  →  torch._amp_foreach_matmul  →  Tensor Core kernel (FP16/BF16)

  Ops that get patched in autocast:
    - torch.matmul         (→ _amp_foreach_matmul)
    - torch.Linear.forward (→ _amp_foreach_linear)
    - torch.nn.functional.linear (→ _amp_foreach_linear)
    - torch.bmm, torch.baddbmm
    - torch.conv2d, torch.conv3d

  Ops that remain FP32 inside autocast:
    - torch.sum, torch.mean (reduction ops)
    - torch.batch_norm
    - torch.layer_norm
    - torch.softmax, torch.log_softmax
    - torch.relu, torch.sigmoid (pointwise — usually fast enough in FP32)
```

**Why only certain ops get patched:**

Matmuls and convolutions are the most memory-heavy and compute-intensive ops. They dominate runtime and memory usage. Patching them gives the biggest benefit. Reduction ops and pointwise ops have negligible memory impact compared to matmuls — leaving them in FP32 avoids numerical risk for minimal speed cost.

---

## The Problem: Small Gradients Round to Zero

FP16 has a narrower exponent range than FP32. Small gradient values can fall below the minimum representable FP16 value — they round to zero. This breaks training.

BF16 largely solves this with its larger exponent. But FP16 still needs a workaround: loss scaling.

---

## GradScaler: The Fix

GradScaler multiplies the loss by a scale factor before backpropagation. Gradients are proportionally larger — they stay representable in FP16. After the backward pass, GradScaler un-scales the gradients to recover the true values.

**Step-by-step with actual values (loss=0.5, scale=128.0):**

```
Starting values:
  loss = 0.5
  scale = 128.0

Step 1: scale the loss
  scaled_loss = loss * scale
  scaled_loss = 0.5 * 128.0
  scaled_loss = 64.0

Step 2: backward pass with scaled loss
  scaled_loss.backward()

  If we did NOT scale, gradients like 0.001 would underflow to 0 in FP16.
  By multiplying loss by 128, all gradients get multiplied by 128 too:
    gradient_before_scaling = 0.001
    gradient_in_backward_pass = 0.001 * 128 = 0.128  ← represents same
                                                        gradient, just scaled

Step 3: un-scale gradients before optimizer step
  Before: model.parameters().grad = 0.128 (scaled up)
  After:  model.parameters().grad / scale = 0.128 / 128 = 0.001 (true value)

Step 4: optimizer.step() uses true gradient values

Step 5: scaler.update() adjusts scale for next iteration
  - If no overflow detected:  scale *= 1.01  → 129.28
  - If overflow detected:     scale *= 0.9   → 115.2

The key insight: multiplying by scale does not change the SIGN of gradients.
The direction of optimization is preserved. Only the MAGNITUDE changes,
and it is restored before applying to weights.
```

Using GradScaler:

```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in dataloader:
    inputs, targets = inputs.cuda(), targets.cuda()

    # autocast: ops like matmul run in FP16/BF16 automatically
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)

    # scale(loss): multiply loss by scale factor before backward
    # this makes gradients larger, keeping them in representable FP16 range
    scaler.scale(loss).backward()

    # step unscales gradients before optimizer update (recovers true gradient values)
    scaler.step(optimizer)

    # update: adjust scale factor for next iteration
    # grows if no overflow occurred, shrinks if overflow was detected
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

**TF32 on Ampere: Three formats side by side with bit layouts:**

```
FP32 (full precision):  1 sign | 8 exponent | 23 mantissa
TF32 (Ampere native):    1 sign | 8 exponent | 10 mantissa  ← FP32 range, FP16 precision
FP16:                    1 sign | 5 exponent | 10 mantissa

TF32 bit layout (19 bits total, padded to 32 for storage):
  [S][E][E][E][E][E][E][E][E][M][M][M][M][M][M][M][M][M][M][?][?][?][?][?][?][?][?][?][?][?][?]
  0   0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0
  |   \_____8 bits_____/  \_____10 bits_____/  \____19 bits total, rest padding____/

FP16 bit layout (16 bits):
  [S][E][E][E][E][E][M][M][M][M][M][M][M][M][M][M]
  |   \___5 bits___/  \_____10 bits_____/

Why TF32 has FP32 range but FP16 precision:

  - FP32 range:  8 exponent bits → same numerical range as FP32 (exponent bias 127)
                can represent values from ~1e-38 to ~1e+38

  - FP16 precision: 10 mantissa bits (same as FP16)
                   only ~3 decimal digits of precision
                   but 10 bits is enough for matmul accumulation

  The matmul accumulation (the sum of many products) is what needs the range.
  The individual inputs (weights, activations) do not need more than FP16 precision.

  TF32 = take FP32 inputs, truncate mantissa to 10 bits, compute in TF32 accumulator,
         store result in FP32. The speedup comes from Tensor Cores running this
         19-bit-ish format at FP16 speeds with FP32 dynamic range.
```

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

**The AMP training loop broken into 6 numbered steps:**

```
Step 1: Move data to GPU
    inputs, targets = inputs.cuda(), targets.cuda()
    - tensors are FP32 by default on CPU
    - transferred to GPU as FP32
    - autocast will convert to FP16/BF16 during forward pass

Step 2: Forward pass with autocast (FP16/BF16 computation)
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    - model weights stay FP32 (optimizer needs FP32)
    - matmuls and convs run in FP16/BF16 via patched dispatch
    - activations stored in FP16/BF16 (saves memory)
    - loss computed in FP16/BF16

Step 3: Scale the loss before backward
    scaler.scale(loss).backward()
    - loss is multiplied by scale factor (e.g., 128.0)
    - scaled_loss.backward() computes gradients in FP16
    - scaled gradients are larger → no underflow

Step 4: Unscale gradients and optimizer step
    scaler.step(optimizer)
    - GradScaler divides gradients by scale factor
    - true gradient values are recovered
    - optimizer.step() updates FP32 weights using true gradients
    - weights stay FP32 (critical for optimizer stability)

Step 5: Update the scale factor
    scaler.update()
    - checks if overflow occurred during backward
    - if no overflow:  scale *= 1.01 (gradually increase)
    - if overflow:     scale *= 0.9  (back off)
    - prepares scale for next iteration

Step 6: Loop repeats
    - next iteration uses the updated scale factor
    - memory is freed/reused for next batch
    - activations from this step become eligible for garbage collection
```

A complete runnable version of this example is in `amp.py`. Run it with `python amp.py`.

---

## Recap

- **FP16 halves memory, BF16 is safer.** Both use 2 bytes vs 4 for FP32. A 1B parameter model: ~4 GB in FP32, ~1 GB per precision in FP16/BF16.
- **BF16 has a larger exponent.** It rarely underflows — better for deep networks.
- **Autocast patches ops.** Wrap `forward()` with `autocast()` and supported ops run in lower precision.
- **GradScaler fixes underflow.** Scale the loss up, unscale gradients before optimizer step.
- **TF32 on Ampere+ is free speed.** Set `torch.set_float32_matmul_precision("high")`. Matmuls typically run ~2-3x faster with TF32 vs FP32 on Ampere+ GPUs.
- **Measure first.** Run the benchmark scripts to see if mixed precision helps your workload.

---

## Common Pitfalls

- **Using `optimizer.step()` instead of `scaler.step(optimizer)` when using GradScaler.**
  GradScaler unscales gradients before the optimizer update. Calling `optimizer.step()` directly bypasses the unscale, leaving gradients at scaled values — this corrupts weight updates (effectively applies scaled gradient as if it were unscaled, causing severe weight decay or divergence). Always use `scaler.step(optimizer)`.

- **Using `autocast()` on CPU.**
  `autocast` only enables autocasting on CUDA. On CPU it is a no-op — all ops run in FP32 regardless of the `with autocast()` block. Mixed precision training requires CUDA.

- **Loss scaling factor too high, causing overflow in the forward pass.**
  If the scale factor is too large, activations overflow during the forward pass before gradients are even computed, producing NaN losses. GradScaler detects this and automatically reduces the scale factor, but a manually set high value (e.g., `init_scale=8192`) can cause this issue early in training before GradScaler adapts.

---

## Will This Break My Model? Decision Tree

Check these 3 things before enabling mixed precision:

```
1. IS YOUR MODEL NUMERICALLY STABLE IN FP16?
   │
   ├─ YES → FP16 with GradScaler works well.
   │         Monitor for overflow in first few steps.
   │         If loss goes NaN, reduce init_scale or switch to BF16.
   │
   └─ NO / UNSURE → Use BF16 instead.
                     BF16's larger exponent matches FP32 range.
                     GradScaler still recommended but less critical.

2. DO YOU HAVE AMPERE+ GPU (A100, RTX 3090/4090, H100)?
   │
   ├─ YES → Enable TF32: torch.set_float32_matmul_precision("high")
   │         Free 2-4x matmul speedup, no accuracy loss.
   │         Weights stay FP32, only matmul internals use TF32.
   │
   └─ NO → TF32 not available.
            FP16 or BF16 are still fine for memory savings.
            Speedup depends on your GPU's FP16 Tensor Core support.

3. IS YOUR MODEL LARGE ENOUGH TO BENEFIT?
   │
   ├─ YES (1B+ params) → Mixed precision cuts memory ~50%.
   │                     Train larger batches, fit models that
   │                     would not fit in FP32.
   │
   └─ NO (<100M params) → Memory savings may not matter.
                          Speedup is modest for small models.
                          Profile first to see if it is worth the complexity.

Summary:
  - BF16: safest choice (try this first if unsure)
  - FP16 + GradScaler: works well if model is numerically stable
  - TF32: free speed on Ampere+, no downsides
  - Small model + older GPU: profile first, may not help much
```

---

## Which File Demonstrates What

| File | What It Demonstrates |
|------|----------------------|
| `amp.py` | Full end-to-end training loop with autocast + GradScaler + TF32 |
| `tf32.py` | TF32 vs FP32 matmul speed benchmark |
| `grad_scaler.py` | GradScaler gradient recovery measurement |
| `convergence.py` | FP32 vs FP16 vs BF16 convergence curves |
| `verify.txt` | Expected numerical verification output for `convergence.py`. Run `python convergence.py` and compare output against `verify.txt` to check your implementation. |

---

## Going Further

For convergence curves across precision modes, TF32 speed benchmarks, GradScaler recovery measurements, and numerical stability analysis — see [ADVANCED.md](./ADVANCED.md).

Sources:
- https://pytorch.org/docs/stable/amp.html
- https://pytorch.org/docs/stable/generated/torch.set_float32_matmul_precision.html
- https://arxiv.org/abs/1710.03740 (Mixed Precision Training)

---

Get the video walkthrough of convergence curves across precision modes, TF32 benchmarks, and GradScaler recovery analysis: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
