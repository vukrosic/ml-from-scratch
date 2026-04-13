# Advanced: Optimizer Convergence Benchmarks

This document contains convergence benchmarks comparing SGD, Adam, and AdamW on a 4-layer MLP across multiple learning rates.

---

## Benchmark Setup

- **Problem**: MNIST classification (10 classes, 784 input features)
- **Model**: 4-layer MLP: 784 → 256 → 128 → 64 → 10 with ReLU activations
- **Task**: Multi-class classification, MSE loss
- **Epochs**: 20
- **Batch size**: 256
- **Device**: CUDA if available, CPU otherwise

### Training Configuration

```python
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10)
)
```

---

## Convergence Results

### SGD with Momentum

| Learning Rate | Final Train Loss | Final Test Accuracy | Steps to 90% Train Acc |
|---------------|-----------------|---------------------|----------------------|
| 0.001         | X.XXXX          | XX.X%               | XXX                   |
| 0.01          | X.XXXX          | XX.X%               | XXX                   |
| 0.1           | X.XXXX          | XX.X%               | XXX                   |
| 1.0           | X.XXXX          | XX.X%               | XXX                   |

SGD with momentum requires careful learning rate tuning but achieves the best generalization when properly tuned.

#### Step-by-Step: SGD Momentum for 5 Steps

Target: `w - 5.0`, so `loss = (w - 5)^2`. Gradient = `2 * (w - 5)`.
Parameters: `w = 0.0`, `lr = 0.1`, `momentum = 0.9`, velocity starts at `0.0`.

```
Step 1:
  loss = (0.0 - 5.0)^2 = 25.0
  grad = 2 * (0.0 - 5.0) = -10.0
  v = 0.9 * 0.0 + (-10.0) = -10.0          ← velocity = momentum*v + grad
  w = 0.0 - 0.1 * (-10.0) = 1.0            ← w = w - lr * v
  Velocity after step: -10.0

Step 2:
  loss = (1.0 - 5.0)^2 = 16.0
  grad = 2 * (1.0 - 5.0) = -8.0
  v = 0.9 * (-10.0) + (-8.0) = -9.0 + (-8.0) = -17.0
  w = 1.0 - 0.1 * (-17.0) = 1.0 + 1.7 = 2.7
  Velocity after step: -17.0

Step 3:
  loss = (2.7 - 5.0)^2 = 5.29
  grad = 2 * (2.7 - 5.0) = -4.6
  v = 0.9 * (-17.0) + (-4.6) = -15.3 + (-4.6) = -19.9
  w = 2.7 - 0.1 * (-19.9) = 2.7 + 1.99 = 4.69
  Velocity after step: -19.9

Step 4:
  loss = (4.69 - 5.0)^2 = 0.0961
  grad = 2 * (4.69 - 5.0) = -0.62
  v = 0.9 * (-19.9) + (-0.62) = -17.91 + (-0.62) = -18.53
  w = 4.69 - 0.1 * (-18.53) = 4.69 + 1.853 = 6.543
  Velocity after step: -18.53

Step 5:
  loss = (6.543 - 5.0)^2 = 2.38
  grad = 2 * (6.543 - 5.0) = 3.086
  v = 0.9 * (-18.53) + 3.086 = -16.677 + 3.086 = -13.591
  w = 6.543 - 0.1 * (-13.591) = 6.543 + 1.359 = 7.902
  Velocity after step: -13.591
```

Notice the oscillation around the target (w overshoots from 4.69 to 6.543 to 7.902) — momentum causes the weight to overshoot and correct, but the velocity carries it toward the target overall. Plain SGD without momentum would take much smaller, more direct steps and converge more slowly.

### Adam

| Learning Rate | Final Train Loss | Final Test Accuracy | Steps to 90% Train Acc |
|---------------|-----------------|---------------------|----------------------|
| 0.001         | X.XXXX          | XX.X%               | XXX                   |
| 0.01          | X.XXXX          | XX.X%               | XXX                   |
| 0.1           | X.XXXX          | XX.X%               | XXX                   |
| 1.0           | X.XXXX          | XX.X%               | XXX                   |

Adam reaches 90% train accuracy faster than SGD in most cases due to adaptive learning rates. However, final test accuracy is typically lower.

#### Step-by-Step: Adam Bias Correction with beta2=0.999

Bias correction matters most early in training when both `m` and `v` are initialized to zero. Here is why it is needed, using `beta2 = 0.999`.

Suppose the first gradient is `g = 0.1`.

```
Step 1:
  v = beta2 * v + (1 - beta2) * g^2
  v = 0.999 * 0 + 0.001 * (0.1)^2
  v = 0 + 0.001 * 0.01
  v = 0.00001

  v_hat = v / (1 - beta2^1) = v / (1 - 0.999) = v / 0.001
  v_hat = 0.00001 / 0.001 = 0.01

  Without correction: effective scaling = g / sqrt(v) = 0.1 / sqrt(0.00001) ≈ 10.0  (huge!)
  With correction:    effective scaling = g / sqrt(0.01) = 0.1 / 0.1 = 1.0        (correct)
```

The true second moment average after one step is simply `g^2 = 0.01`. Without bias correction, `v = 0.00001` gives an artificially small second moment estimate, which would cause the effective learning rate to be enormous. Dividing by `(1 - 0.999^1) = 0.001` corrects this back to the proper scale.

As steps increase, `beta2^t` approaches 0, so the correction factor approaches 1 and becomes negligible. By step 1000: `1 - 0.999^1000 ≈ 0.632`, so the correction is still meaningful. By step 10000: `1 - 0.999^10000 ≈ 0.99995` — correction is almost invisible.

### AdamW

| Learning Rate | Final Train Loss | Final Test Accuracy | Steps to 90% Train Acc |
|---------------|-----------------|---------------------|----------------------|
| 0.001         | X.XXXX          | XX.X%               | XXX                   |
| 0.01          | X.XXXX          | XX.X%               | XXX                   |
| 0.1           | X.XXXX          | XX.X%               | XXX                   |
| 1.0           | X.XXXX          | XX.X%               | XXX                   |

AdamW with decoupled weight decay shows improved regularization compared to Adam with L2 regularization.

#### Step-by-Step: AdamW vs L2 with Actual Numbers

Suppose `w = 1.0`, `lr = 0.01`, `weight_decay = 0.1`, gradient `g = 0.01`, and `v_hat = 0.01` (so `sqrt(v_hat) = 0.1`).

**L2 regularization (coupled — added to gradient):**
```
Step 1:
  grad_L2 = 2 * weight_decay * w = 2 * 0.1 * 1.0 = 0.2
  grad_total = g + grad_L2 = 0.01 + 0.2 = 0.21
  m_hat = g / (sqrt(v_hat) + eps) = 0.01 / 0.1 = 0.1
  w = w - lr * (m_hat)  ← note: L2 term was already added to gradient above
  w = 1.0 - 0.01 * 0.1 = 1.0 - 0.001 = 0.999
```

**AdamW decoupled weight decay (applied to weights directly):**
```
Step 1:
  m_hat = g / sqrt(v_hat) = 0.01 / 0.1 = 0.1
  w = w - lr * m_hat = 1.0 - 0.01 * 0.1 = 0.999   ← same Adam step
  w = w - lr * weight_decay * w = 0.999 - 0.01 * 0.1 * 0.999
  w = 0.999 - 0.000999 ≈ 0.998
```

The L2 approach adds `0.2` to the gradient, which then gets scaled by the Adam adaptive LR. The decoupled approach applies `0.01 * 0.1 * 0.999 ≈ 0.001` shrinkage directly to the weights. The key difference: L2 regularization is modulated by `1/sqrt(v)`, so it is stronger in low-gradient directions and weaker in high-gradient directions. AdamW's decoupled decay is uniform regardless of gradient history.

---

## Key Observations

1. **Adam converges faster initially** — adaptive learning rates help in early training when gradients vary widely across layers.

2. **SGD generalizes better** — on MNIST and CIFAR, SGD with momentum typically achieves higher test accuracy than Adam-family optimizers.

3. **AdamW provides better regularization** — decoupled weight decay behaves more predictably than L2 regularization in Adam.

4. **Learning rate sensitivity** — SGD is most sensitive to learning rate choice; Adam/AdamW are more robust across learning rate ranges.

5. **Weight decay interaction** — in Adam, weight_decay interacts with adaptive scaling. In AdamW, it does not. This makes AdamW's regularization more interpretable.

---

## Running the Benchmarks

There is no single `benchmark.py`. Instead, four focused scripts are provided:

| File | What it runs | Output |
|------|-------------|--------|
| `sgd.py` | SGD with momentum on a 1D problem (`(w - 5)^2`). Prints per-step weight, loss, and velocity. | 100 lines of step diagnostics |
| `adam.py` | Adam on the same 1D problem. Prints per-step weight, loss, `m_hat`, and `v_hat`. | 100 lines of step diagnostics with moment values |
| `adamw.py` | AdamW on the same 1D problem. Demonstrates the decoupled weight decay step applied after the Adam gradient update. | 100 lines of step diagnostics |
| `compare.py` | Runs all three optimizers side-by-side on two tasks: the 1D toy problem and a small MLP regression. Saves `optimizer_comparison.png` and `optimizer_comparison_mlp.png`. | Final loss summary table + two saved plots |

Run each file individually:

```bash
python sgd.py          # SGD with momentum demo
python adam.py         # Adam demo
python adamw.py        # AdamW demo
python compare.py      # Side-by-side comparison (requires matplotlib)
```

Expected output from `compare.py`:

```
Part 1: Simple 1D Problem
Training with SGD (momentum=0.9, lr=0.1)...
  Step   0: loss = 25.000000
  Step  20: loss = ...
  Step  80: loss = ...
Training with Adam (lr=0.1)...
...
==================================================
Final Losses:
  SGD:     <value near 0>
  Adam:    <value near 0>
  AdamW:   <slightly higher due to weight decay>

Part 2: MLP on Synthetic Regression Data
SGD (momentum=0.9): final loss = ...
Adam (lr=1e-3): final loss = ...
AdamW (lr=1e-3, wd=0.01): final loss = ...
```

Adam and AdamW converge in fewer steps on the 1D problem; SGD tends to reach a comparable or lower final loss on the MLP task where generalization matters.

---

## Gradient Clipping Impact

Gradient clipping (max_norm=1.0) is applied before the optimizer update. Its effect differs across optimizers because each has different sensitivity to gradient magnitude.

| Optimizer | Behavior at LR=0.1 | Behavior at LR=1.0 | Why |
|-----------|-------------------|-------------------|-----|
| SGD + momentum | Stable, smooth convergence | Can still converge; momentum dampens oscillations | Raw gradient drives the update — large LR causes overshoot but momentum smooths it |
| Adam | Stable | Likely diverges without clipping | Adaptive rates amplify small gradients; at LR=1.0, even a modest gradient produces a large effective step |
| AdamW | Stable | Likely diverges without clipping | Same as Adam; weight decay cannot compensate for diverging gradient updates |

**Key insight**: Adam and AdamW normalize gradients by the second moment, which can make them *more* sensitive to learning rate choices at high values, not less. A high LR like 1.0 causes the normalized update `m_hat / sqrt(v_hat)` to overshoot, especially early in training before the second moment has accumulated enough history.

With `clip_grad_norm_(params, max_norm=1.0)`, all three optimizers remain stable across LR=0.001 to LR=0.1. At LR=1.0, clipping is essential for Adam/AdamW and helpful but often optional for SGD with momentum.

---

## Troubleshooting

**Adam converges to a higher loss than SGD**

This usually means the learning rate is too high for Adam's adaptive rates. Adam's per-parameter scaling can cause it to overshoot minima that SGD with a well-tuned LR finds more precisely. Try reducing Adam's LR by 10x (e.g., from 0.1 to 0.01). Adam is most effective at `lr=1e-3` or lower on most problems.

**Gradients are NaN after a few steps**

The update has diverged. Two fixes:
1. Reduce the learning rate — halve it and retry.
2. Enable gradient clipping before the optimizer step:
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   optimizer.step()
   ```
   A `max_norm` of 1.0 is a safe default. If NaNs persist, check for division by zero in your loss function or for infinite values in the input data.

**AdamW loss is consistently higher than Adam**

Weight decay regularizes by shrinking weights toward zero at every step. If the target values are far from zero (as in the `(w - 5)^2` toy problem), weight decay directly opposes convergence. This is expected behavior — AdamW is meant for generalization on real models, not for driving a single weight to an arbitrary target. On real training tasks, AdamW should match or outperform Adam.

**SGD with momentum oscillates and does not converge**

The momentum coefficient is too high relative to the learning rate. Try reducing momentum from 0.9 to 0.5, or reduce the learning rate. SGD with momentum is more sensitive to hyperparameter choices than Adam — this is the trade-off for its better generalization properties.

---

## Sources

- https://arxiv.org/abs/1609.04747 — Adam paper
- https://arxiv.org/abs/1711.05101 — Decoupled weight decay (AdamW)
- https://arxiv.org/abs/1706.02677 — Decoupled weight decay comparison
