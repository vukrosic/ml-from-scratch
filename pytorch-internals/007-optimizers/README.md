# Optimizer Internals From Scratch

Every optimizer is the same idea — update weights in the direction that reduces loss — but the path you take to get there changes everything. This is what SGD, Adam, and AdamW actually do under the hood.

---

## The Core Problem: Gradient Descent

Training a neural network means finding weights that minimize a loss function. The loss landscape is a surface in weight space. Gradient descent says: at your current position, find the steepest downhill direction and take a step.

The simplest update:

```
w = w - lr * gradient
```

One problem: the gradient is a local signal. It tells you the direction of steepest descent right now, but the landscape can be curved, ridged, or plateaud. A ball rolling down a hill using only the slope at its current contact point would oscillate wildly on a curved surface.

---

## SGD with Momentum: A Ball Rolling Downhill

The physics analogy: a ball rolling down a gradient has inertia. It does not stop and recalculate the slope at every point — it builds up velocity in the downhill direction.

Instead of resetting velocity each step, momentum accumulates a running average of past gradients:

```
v = beta * v + gradient           # velocity accumulation
w = w - lr * v                    # update with velocity instead of raw gradient
```

`beta` (typically 0.9) controls how much past velocity influences the current step. A high beta means the ball keeps rolling in the same direction even if the local gradient points elsewhere.

The effect: momentum dampens oscillations in one direction while reinforcing consistent movement in another. On a curved loss surface, SGD with momentum behaves like a ball that has mass — it rolls through valleys rather than bouncing across them.

### The momentum update in code

```python
def sgd_momentum(params, grads, velocities, lr=1e-3, momentum=0.9):
    for p, g, v in zip(params, grads, velocities):
        # Accumulate velocity: v = momentum * v + gradient
        # This builds up inertia in consistent directions
        v.data = momentum * v + g
        # Update: p = p - lr * velocity (not raw gradient)
        p.data = p - lr * v
```

Velocities must be initialized to zeros. They persist across steps — that is the memory of where you have been.

### Piece 1: Complete SGD with momentum example

```python
import torch

# Simple 2D problem: minimize (w - target)^2
target = 5.0
w = torch.tensor([0.0], requires_grad=True)

# Velocity persists across steps — initialize to zero
velocity = torch.zeros_like(w)

for step in range(100):
    loss = (w - target) ** 2
    loss.backward()

    # Manual momentum update: v = momentum * v + grad
    with torch.no_grad():
        velocity = 0.9 * velocity + w.grad
        w = w - 0.1 * velocity

    w.retain_grad()  # keep w.grad alive for next iteration
    print(f"Step {step}: w = {w.item():.4f}, loss = {loss.item():.4f}")
```

The momentum version reaches the target faster than plain SGD because it builds up velocity in the correct direction rather than taking independent steps.

---

## Adam: Adaptive Learning Rates via First and Second Moments

SGD with momentum uses one accumulator (velocity) to smooth the gradient direction. Adam extends this idea with two running averages — one for the direction (like momentum) and one for the scale of the gradient.

### Why two moments?

The first moment (mean of gradients) gives the direction, just like momentum.

The second moment (mean of squared gradients) tells you how large the gradient has been on average. A large second moment means gradients have consistently been large in that direction — the learning rate should shrink there. A small second moment means the signal is weak — the learning rate should compensate by being larger.

Adam divides the learning rate by the RMS of past gradients:

```
m = beta1 * m + (1 - beta1) * g          # first moment (direction)
v = beta2 * v + (1 - beta2) * g^2         # second moment (scale)
m_hat = m / (1 - beta1^t)                # bias correction for m
v_hat = v / (1 - beta2^t)                # bias correction for v
w = w - lr * m_hat / (sqrt(v_hat) + eps)
```

`beta1` is typically 0.9 (momentum-like), `beta2` is typically 0.999 (window for second moment), and `eps` (often 1e-8) prevents division by zero.

### Why bias correction?

Both `m` and `v` are initialized to zero. At the start of training, they are biased toward zero because the running average has not yet seen enough samples to be representative.

Consider `v` with `beta2 = 0.999`. After one step, `v = 0.999 * 0 + 0.001 * g^2 = 0.001 * g^2`. The true average over one step is `g^2`, but we get `0.001 * g^2` — off by a factor of 1000.

Bias correction divides by `(1 - beta^t)` where `t` is the step number. At step 1: `1 - 0.999^1 = 0.001`, so `v / 0.001 = g^2`. The correction disappears as `t` grows because the running average becomes accurate.

### The Adam update in code

```python
def adam(params, grads, m, v, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    for t, (p, g) in enumerate(zip(params, grads), 1):
        # Update biased first moment estimate: m = beta1 * m + (1-beta1) * g
        m.data = beta1 * m + (1 - beta1) * g
        # Update biased second moment estimate: v = beta2 * v + (1-beta2) * g^2
        v.data = beta2 * v + (1 - beta2) * g * g

        # Bias-correct both estimates: divide by (1 - beta^t)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Update parameters: p = p - lr * m_hat / (sqrt(v_hat) + eps)
        p.data = p - lr * m_hat / (v_hat.sqrt() + eps)
```

### Piece 1: Complete Adam example on one parameter

```python
import torch

target = 5.0
w = torch.tensor([0.0], requires_grad=True)

# Adam state: first and second moments (initialized to zero)
m = torch.zeros_like(w)
v = torch.zeros_like(w)

for step in range(100):
    loss = (w - target) ** 2
    loss.backward()

    g = w.grad

    # Bias-corrected first moment (direction)
    m = 0.9 * m + 0.1 * g
    m_hat = m / (1 - 0.9 ** (step + 1))
    # Bias-corrected second moment (scale)
    v = 0.999 * v + 0.001 * g * g
    v_hat = v / (1 - 0.999 ** (step + 1))

    # Update: divide lr by RMS of second moment
    w.data = w - 0.1 * m_hat / (v_hat.sqrt() + 1e-8)

    print(f"Step {step}: w = {w.item():.4f}, loss = {loss.item():.4f}")
```

---

## AdamW: Decoupled Weight Decay

Weight decay is a regularization technique that penalizes large weights by adding `lambda * ||w||^2` to the loss. The goal is to keep weights small, which improves generalization.

There are two ways to implement weight decay in gradient-based optimizers, and they are not equivalent in Adam.

### L2 regularization vs decoupled weight decay

**L2 regularization** (what most libraries do by default with `weight_decay`):

```
loss = original_loss + lambda * ||w||^2
grad = original_grad + 2 * lambda * w
w = w - lr * grad
```

The gradient of the L2 penalty is `2 * lambda * w`, which gets added to the gradient before the update. This couples the regularization with the gradient scaling from Adam's adaptive learning rates.

**Decoupled weight decay** (AdamW):

```
w = w - lr * (grad_from_adam + lambda * w)
```

The weight decay is applied directly to the weights, independent of the gradient computation. The update becomes:

```
w = w - lr * grad_from_adam - lr * lambda * w
w = (1 - lr * lambda) * w - lr * grad_from_adam
```

The key difference: with L2 regularization, the effective weight decay strength depends on the adaptive gradient scaling. With AdamW, it does not.

### Why does this matter?

Adaptive optimizers like Adam scale gradients by `1/sqrt(v)`. Large gradient components get shrunk, small ones get amplified. When you add L2 regularization as a gradient term, it also gets scaled by `1/sqrt(v)`. The result is that weight decay is stronger in directions where gradients are small and weaker where gradients are large — the opposite of what you want.

AdamW decouples the two, so the decay is uniform across all parameters.

### The AdamW update in code

```python
def adamw(params, grads, m, v, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
    for t, (p, g) in enumerate(zip(params, grads), 1):
        # Same bias-corrected moment estimates as Adam
        m.data = beta1 * m + (1 - beta1) * g
        v.data = beta2 * v + (1 - beta2) * g * g

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        # Adam update with adaptive lr
        p.data = p - lr * m_hat / (v_hat.sqrt() + eps)
        # Decoupled weight decay: applied directly to parameters, independent of gradient
        p.data = p - lr * weight_decay * p
```

### Piece 1: Complete AdamW example

```python
import torch

target = 5.0
w = torch.tensor([0.0], requires_grad=True)

m = torch.zeros_like(w)
v = torch.zeros_like(w)

for step in range(100):
    loss = (w - target) ** 2
    loss.backward()

    g = w.grad

    # Bias-corrected first and second moments
    m = 0.9 * m + 0.1 * g
    v = 0.999 * v + 0.001 * g * g
    m_hat = m / (1 - 0.9 ** (step + 1))
    v_hat = v / (1 - 0.999 ** (step + 1))

    # Adam step with decoupled weight decay
    w.data = w - 0.1 * m_hat / (v_hat.sqrt() + 1e-8)
    w.data = w - 0.1 * 0.01 * w  # separate decay step, not added to gradient

    print(f"Step {step}: w = {w.item():.4f}, loss = {loss.item():.4f}")
```

---

## Gradient Clipping: Why It Exists

During training, a single step can produce extremely large gradients — especially at the start or with certain architectures. A large gradient step can throw the model far across the loss landscape, destroying progress.

Gradient clipping caps the gradient norm to a maximum value:

```
if ||g|| > max_norm:
    g = g * (max_norm / ||g||)
```

This is applied before the optimizer update. `torch.nn.utils.clip_grad_norm_` computes the L2 norm of all parameters and scales them down if they exceed the threshold.

### When gradients explode

RNNs, LSTMs, and transformers early in training are particularly prone to large gradients. A single bad batch can produce gradients orders of magnitude larger than typical. Without clipping, one step can undo an entire epoch of progress.

### Gradient clipping in code

```python
import torch
import torch.nn.utils as utils

model = torch.nn.Linear(10, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch)
    loss.backward()

    # Clip gradients to max norm of 1.0
    utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()
```

The clipping is a separate operation from the update. It modifies the gradients in-place before they are consumed by the optimizer.

---

## Comparison: SGD vs Adam vs AdamW

Different optimizers make different trade-offs. The right choice depends on your problem.

| Optimizer | Best for | Strengths | Weaknesses |
|-----------|----------|-----------|------------|
| SGD + momentum | Vision, well-tuned training | Generalizes well, simple | Sensitive to learning rate, slow to converge |
| Adam | Fast initial progress, NLP | Adaptive rates, fast convergence | Can generalize worse, bias correction overhead |
| AdamW | Transformers, LLMs | Decoupled decay, better regularization | Still may need learning rate scheduling |

SGD with momentum typically generalizes better on image classification tasks and is the basis for most vision model training. Adam converges faster initially, which is useful in NLP where training is expensive. AdamW combines Adam's adaptive rates with proper weight decay, making it the standard for transformer training.

### The training loop comparison

Here is a minimal example comparing all three on the same problem:

```python
import torch

def train_with_optimizer(optimizer_cls, optimizer_kwargs, steps=100):
    target = 5.0
    w = torch.tensor([0.0], requires_grad=True)
    opt = optimizer_cls([w], **optimizer_kwargs)

    losses = []
    for _ in range(steps):
        loss = (w - target) ** 2
        losses.append(loss.item())

        loss.backward()
        opt.step()
        opt.zero_grad()

    return losses

# SGD with momentum
sgd_losses = train_with_optimizer(
    torch.optim.SGD,
    {"lr": 0.1, "momentum": 0.9}
)

# Adam
adam_losses = train_with_optimizer(
    torch.optim.Adam,
    {"lr": 0.1}
)

# AdamW
adamw_losses = train_with_optimizer(
    torch.optim.AdamW,
    {"lr": 0.1, "weight_decay": 0.01}
)
```

Run `python compare.py` to generate loss curves comparing all three optimizers. The script runs two experiments: a simple 1D problem (minimizing `(w - 5)^2`) and an MLP regression task on synthetic data. Two PNG files are saved: `optimizer_comparison.png` (1D problem) and `optimizer_comparison_mlp.png` (MLP regression).

#### What the PNG curves look like

**optimizer_comparison.png (1D problem):**

```
Loss
  ^
  |  SGD: slow start, steady descent
  |  Adam: fast initial drop
25|****
  |    ****  AdamW: slightly above Adam (weight decay opposes target=5)
  |        ****
  |             ****
  |                 ****
  |                      ****
  +------------------------------------> Step
  0    20    40    60    80   100
```

- SGD (orange dashed): starts at loss=25, descends slowly but steadily. Reaches ~0.01 by step 100.
- Adam (blue): drops to near-zero by step 20, fastest early convergence.
- AdamW (green): slightly above Adam throughout due to weight decay actively shrinking w toward 0 (opposite of target=5).

**optimizer_comparison_mlp.png (MLP regression):**

```
Loss
  ^
  |  SGD: lower final loss (better generalization)
  |  Adam/AdamW: lower initial loss
  |___________________
  |         ****      SGD
  |      ****         Adam
  |   ****            AdamW
  +------------------------------------> Step
```

- Adam/AdamW drop quickly initially but plateau higher.
- SGD with momentum takes longer to get going but settles to a lower final loss — the signature generalization advantage of SGD on structured tasks.

---

## Which File Demonstrates What

| File | What it demonstrates |
|------|----------------------|
| `sgd.py` | SGD with momentum implementation — manual velocity accumulation, step-by-step trace of weight, loss, and velocity values |
| `adam.py` | Adam implementation — manual first and second moment updates with bias correction, per-step trace of `m_hat` and `v_hat` |
| `adamw.py` | AdamW implementation — Adam update plus a separate decoupled weight decay step, showing how the two operations are kept independent |
| `compare.py` | Side-by-side comparison of all three optimizers — runs both a 1D toy problem and an MLP regression task, plots loss curves, saves PNG outputs |

Run any file directly with `python <filename>.py`. Each prints per-step diagnostics and returns the loss history.

---

## Recap

- **SGD with momentum** adds inertia — a velocity term that carries the update forward in consistent directions.
- **Adam** maintains two running averages: first moment (direction) and second moment (scale). It divides the learning rate by the RMS of past gradients.
- **Bias correction** in Adam compensates for the initialization bias of running averages when they start near zero.
- **AdamW** decouples weight decay from gradient scaling, making the regularization behave as intended.
- **Gradient clipping** prevents large gradient steps from destabilizing training.
- **No optimizer is universally best** — SGD generalizes better on vision, Adam/AdamW converge faster on NLP tasks.

---

## Going Further

For convergence benchmarks comparing SGD vs Adam vs AdamW on a 4-layer MLP across multiple learning rates, see [ADVANCED.md](./ADVANCED.md).

Sources:
- https://arxiv.org/abs/1609.04747 — Adam paper
- https://arxiv.org/abs/1711.05101 — Decoupled weight decay (AdamW)

---

Get the video walkthrough of SGD vs Adam vs AdamW convergence benchmarks and optimizer selection guide: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
