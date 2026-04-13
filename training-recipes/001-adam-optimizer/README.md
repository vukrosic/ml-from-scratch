# Adam Optimizer From Scratch

**Adam is the most-used optimizer in deep learning.** It powers GPT, BERT, Stable Diffusion, AlphaFold — virtually every large model trained since 2015. Yet most practitioners just call `torch.optim.AdamW` and move on. How does it actually work? Where do `m` and `v` come from? Why is there a bias correction? What is the difference between Adam and AdamW, and why does it matter?

In this lesson we build Adam from the original paper equations, verify it matches PyTorch's implementation, then go beyond: AdamW, learning rate schedules, and the hyperparameter choices that actually matter in practice.

---

## The Hook: Why You Should Care

Imagine you are training a transformer. You have 50 million parameters. Some are embedding weights that see sparse gradients. Some are LayerNorm biases that converge in the first 100 steps. Some are attention projection matrices deep in the network that need thousands of steps to find the right direction.

**One learning rate cannot serve all of them.**

```
Parameter        Gradient magnitude     What it needs
─────────────────────────────────────────────────────
embed_weight     ~0.001 (sparse)        Larger steps
layernorm_bias   ~0.5   (nearly done)   Tiny steps
attn_proj        ~0.05  (mid-training)  Medium steps
ffn_weight       ~0.2   (noisy)         Smoothed steps
```

SGD applies the same learning rate to every single one. Adam gives each parameter its own effective learning rate — automatically, with no manual tuning per layer. That is why it "just works" and why it dominates modern deep learning.

Let's build it piece by piece.

---

## Piece 1: Why SGD Is Not Enough

Vanilla SGD updates every parameter the same way:

```
theta = theta - lr * gradient
```

This has two problems:

**Problem A: Noisy gradients cause oscillation.** Mini-batch gradients are noisy estimates. SGD follows them blindly, zig-zagging through narrow valleys instead of rolling smoothly to the bottom.

```
  Loss surface (2D slice)         SGD path (oscillates)

       ╱    ╲                        ╱→╲
      ╱      ╲                      ╱ ↗ ╲
     ╱  goal  ╲                    ╱↗  ↘ ╲
    ╱    ★     ╲                  ╱  ↘↗ ↘ ╲
   ╱            ╲                ╱  ↗ ↘  ↗ ╲
  ╱──────────────╲              start ~~~~~~→ ★
                                (many wasted steps)
```

**Problem B: No per-parameter scaling.** A parameter receiving gradients of magnitude 0.001 gets the same step size as one receiving gradients of magnitude 10.0. The first barely moves. The second explodes.

Adam solves both problems. Problem A is solved by **momentum** (first moment). Problem B is solved by **adaptive learning rates** (second moment).

---

## Piece 2: Momentum — The First Moment

Instead of following the raw gradient at each step, accumulate an exponential moving average (EMA) of past gradients. This smooths out noise and accelerates motion in consistent directions.

### The Formula

```
m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
```

- `g_t` is the gradient at step `t`
- `beta1 = 0.9` means "keep 90% of the old estimate, blend in 10% of the new gradient"
- `m_0 = 0` (initialized to zero)

### What This Looks Like

```
Step   Raw gradient g_t     Momentum m_t (beta1=0.9)
────   ────────────────     ────────────────────────
  1         +2.0             0.9(0)   + 0.1(2.0)  = 0.20
  2         -1.5             0.9(0.2) + 0.1(-1.5) = 0.03
  3         +1.8             0.9(0.03)+ 0.1(1.8)  = 0.207
  4         +2.1             0.9(.207)+ 0.1(2.1)  = 0.396
  5         +1.9             0.9(.396)+ 0.1(1.9)  = 0.546

The noisy gradients: +2.0, -1.5, +1.8, +2.1, +1.9  (wild swings)
The momentum signal:  0.20, 0.03, 0.21, 0.40, 0.55  (smooth ramp up)
```

The raw gradients fluctuate wildly, but momentum builds up a consistent signal: "this parameter should move in the positive direction." The outlier at step 2 (-1.5) barely dents the accumulated estimate.

### ASCII Visualization: Noisy vs. Smoothed

```
Gradient
  +2 |  *              *    *         Raw gradients (noisy)
     |       *    *              *
   0 |────────────────────────────→  step
     |  *         *
  -2 |       *              *

Momentum
  +2 |                               Smoothed (momentum)
     |                      ___----
   0 |────__------─────────────────→  step
     |
  -2 |
```

### The Code

```python
# Piece 1: accumulate biased first moment estimate
m = beta1 * m + (1 - beta1) * grad

# In-place PyTorch version (same thing, faster):
m.mul_(beta1).add_(grad, alpha=1 - beta1)
```

Momentum alone is already an improvement over SGD. But it does not solve the per-parameter scaling problem. For that we need the second moment.

---

## Piece 3: RMSProp — Adaptive Learning Rates via the Second Moment

RMSProp (Hinton, 2012 — introduced in a Coursera lecture, never formally published) tracks the magnitude of gradients per-parameter using an EMA of **squared** gradients:

### The Formula

```
v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
```

- `g_t^2` is the elementwise square of the gradient
- `beta2 = 0.999` means a longer memory window than momentum
- `v_0 = 0`

The update then divides by `sqrt(v_t)`:

```
theta = theta - lr * g_t / (sqrt(v_t) + eps)
```

### Why This Works

Think about what `sqrt(v_t)` represents: it is approximately the RMS (root mean square) of recent gradients for that parameter.

- **Parameter with large gradients** (e.g., RMS = 5.0): step size = `lr / 5.0` — small steps
- **Parameter with tiny gradients** (e.g., RMS = 0.01): step size = `lr / 0.01` — large steps

Each parameter automatically gets a learning rate inversely proportional to its typical gradient magnitude. Sparse parameters (embeddings) get amplified. Saturated parameters (biases near convergence) get dampened.

```
Parameter      avg |grad|    effective lr (lr=0.001)
───────────    ─────────     ───────────────────────
embed_weight     0.001        0.001 / 0.001  = 1.0
layernorm_bias   0.5          0.001 / 0.5    = 0.002
attn_proj        0.05         0.001 / 0.05   = 0.02
ffn_weight       0.2          0.001 / 0.2    = 0.005
```

### The Code

```python
# Piece 2: accumulate biased second moment estimate
v = beta2 * v + (1 - beta2) * grad ** 2

# In-place PyTorch version:
v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
```

---

## Piece 4: Adam = Momentum + RMSProp

Adam (Kingma & Ba, 2014) combines both ideas into one optimizer:

1. Use the **first moment** (momentum) for the direction
2. Use the **second moment** (RMSProp) for the scale
3. Add **bias correction** (see next section)

```
                        ┌───────────────┐
  gradient g_t ────────►│  First Moment │──► m_t  (direction)
                        │  EMA of g     │
                        └───────────────┘
                                                    ┌─────────┐
                        ┌───────────────┐           │  Adam   │
  gradient g_t ────────►│ Second Moment │──► v_t ──►│ Update  │──► new theta
                        │  EMA of g^2   │           │         │
                        └───────────────┘           └─────────┘

  Update:  theta = theta - lr * m_hat / (sqrt(v_hat) + eps)
```

The update is the momentum direction, scaled down by the RMS magnitude. Parameters with noisy/large gradients get smaller steps. Parameters with consistent/small gradients get larger steps. Everybody wins.

### Walking Through 5 Steps With Real Numbers

Let's trace a single parameter with `lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8`:

```
Step  g_t     m_t                    v_t                         m_hat    v_hat    update
────  ────    ────                   ────                        ─────    ─────    ──────
  1   0.50    0.9(0)+0.1(0.5)       0.999(0)+0.001(0.25)        0.500    0.250   -0.001*0.500/√0.250
              = 0.050                = 0.000250                                    = -0.001000

  2   0.40    0.9(0.05)+0.1(0.4)    0.999(0.00025)+0.001(0.16)  0.237    0.205   -0.001*0.237/√0.205
              = 0.085                = 0.000410                                    = -0.000523

  3   0.60    0.9(0.085)+0.1(0.6)   0.999(0.00041)+0.001(0.36)  0.126    0.213   -0.001*0.126/√0.213
              = 0.1365               = 0.000770                                    = -0.000273

  4   0.45    0.9(0.1365)+0.1(0.45) 0.999(0.00077)+0.001(0.2025) 0.148   0.222   -0.001*0.148/√0.222
              = 0.1679               = 0.000972                                    = -0.000314

  5   0.55    0.9(0.1679)+0.1(0.55) 0.999(0.000972)+0.001(0.3025) 0.172  0.232   -0.001*0.172/√0.232
              = 0.2061               = 0.001274                                    = -0.000357
```

Notice how the update magnitude is remarkably stable (~0.0003 to 0.001) even though the raw gradients bounce between 0.4 and 0.6. That is the self-normalizing property of Adam.

---

## Piece 5: Bias Correction — Why the First Steps Are Wrong Without It

Both `m` and `v` are initialized to zero. This introduces a systematic bias toward zero in the early steps.

### The Problem

At step 1 with `beta1 = 0.9`:

```
m_1 = 0.9 * 0 + 0.1 * g_1 = 0.1 * g_1
```

The momentum estimate is **only 10% of the true gradient**. We wanted an estimate of the gradient's moving average, but because we started at zero, the estimate is pulled down by 10x.

For the second moment with `beta2 = 0.999`, it is even worse:

```
v_1 = 0.999 * 0 + 0.001 * g_1^2 = 0.001 * g_1^2
```

The variance estimate is **1000x too small** at step 1.

### The Fix

The paper derives an unbiased estimator by dividing out the accumulated decay:

```
m_hat = m_t / (1 - beta1^t)
v_hat = v_t / (1 - beta2^t)
```

### Watching the Correction Fade Out

```
Step t    (1 - 0.9^t)    correction factor    (1 - 0.999^t)    correction factor
──────    ───────────     ─────────────────    ──────────────    ─────────────────
    1       0.100            10.000x              0.001           1000.000x
    2       0.190             5.263x              0.002            500.250x
    5       0.410             2.439x              0.005            200.400x
   10       0.651             1.536x              0.010            100.450x
   20       0.878             1.139x              0.020             50.949x
   50       0.995             1.005x              0.049             20.490x
  100       1.000             1.000x              0.095             10.516x
  500       1.000             1.000x              0.394              2.540x
 1000       1.000             1.000x              0.632              1.582x
```

Key insight: the first moment correction fades fast (~20 steps). The second moment correction lingers for hundreds of steps because `beta2` is so close to 1. **This is why warmup helps** — more on that later.

### The Code

```python
# Piece 3: bias correction
m_hat = m / (1 - beta1 ** t)
v_hat = v / (1 - beta2 ** t)
```

---

## Piece 6: The Full Adam Update

Combining everything (Algorithm 1 from the Kingma & Ba 2014 paper):

```
theta_t = theta_{t-1} - lr * m_hat / (sqrt(v_hat) + eps)
```

The `eps = 1e-8` prevents division by zero when a parameter has received near-zero gradients.

```python
# Piece 4: the parameter update
p.data.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
```

---

## Piece 5: Full Implementation

Here is the complete Adam class, matching Algorithm 1 of the [Kingma & Ba 2014 paper](https://arxiv.org/abs/1412.6980):

```python
class Adam:
    """Adam optimizer, from-scratch implementation matching the original paper."""

    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.state = {
            id(p): {"m": torch.zeros_like(p), "v": torch.zeros_like(p), "t": 0}
            for p in self.params
        }

    def step(self):
        for p in self.params:
            if p.grad is None:
                continue
            grad = p.grad
            s = self.state[id(p)]
            s["t"] += 1
            t = s["t"]

            # (1) Update biased first moment estimate
            s["m"].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
            # (2) Update biased second raw moment estimate
            s["v"].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
            # (3) Compute bias-corrected first moment estimate
            m_hat = s["m"] / (1 - self.beta1 ** t)
            # (4) Compute bias-corrected second raw moment estimate
            v_hat = s["v"] / (1 - self.beta2 ** t)
            # (5) Update parameters
            p.data.addcdiv_(m_hat, v_hat.sqrt().add_(self.eps), value=-self.lr)

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.grad.zero_()
```

---

## Piece 6: AdamW — Decoupled Weight Decay

This is one of the most misunderstood topics in deep learning optimization. **Adam and AdamW are not the same thing**, and using the wrong one can silently hurt your model.

### The Problem With L2 Regularization in Adam

The original Adam paper suggests adding L2 regularization by modifying the gradient:

```
g_t = gradient + lambda * theta     (L2 regularization)
```

This seems fine — you penalize large weights. But there is a subtle bug: the L2 penalty term gets divided by `sqrt(v_hat)` in the Adam update, just like the real gradient. For parameters with large `v_hat` (large gradients), the regularization effect is **weakened**. For parameters with small `v_hat`, it is **amplified**.

The regularization strength becomes inconsistent across parameters — the opposite of what you want.

### The Fix: Decoupled Weight Decay (Loshchilov & Hutter, 2019)

AdamW applies weight decay **directly to the parameters**, after the Adam step, bypassing the adaptive scaling:

```
Adam (L2 reg):    theta = theta - lr * (m_hat + lambda*theta) / (sqrt(v_hat) + eps)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^
                                       regularization is scaled by 1/sqrt(v_hat)

AdamW:            theta = theta - lr * m_hat / (sqrt(v_hat) + eps) - lr * lambda * theta
                                                                     ^^^^^^^^^^^^^^^^^^^
                                                                     regularization is NOT scaled
```

### The Code Difference

```python
# Adam with L2 regularization (WRONG way)
grad = grad + weight_decay * p.data          # modify gradient
# ... then run normal Adam update on modified grad

# AdamW (CORRECT way)
# ... run normal Adam update on original grad, THEN:
p.data.mul_(1 - lr * weight_decay)           # decay weights directly
```

### When It Matters

For small models and short training runs, the difference is negligible. For large models trained for many steps (LLMs, vision transformers), AdamW consistently outperforms Adam+L2. **PyTorch's default `torch.optim.AdamW` uses decoupled weight decay.** This is what you should use.

Typical weight decay values:
- **0.01**: safe default for most tasks
- **0.1**: common for large LLM pretraining (GPT-3, LLaMA)
- **0.0**: fine-tuning sometimes works better without decay

---

## Piece 7: Learning Rate Schedules — Warmup + Cosine Decay

The learning rate is not a constant in modern training. Nearly every large model uses a schedule.

### Why Warmup Helps (Especially With Adam)

Remember the bias correction table above? At step 1, the second moment estimate `v` is 1000x too small. Even with bias correction, the estimate is based on a single gradient — it is noisy and unreliable.

If you start with a full learning rate, those early updates can be wildly wrong. Warmup starts with a tiny learning rate and ramps up linearly, giving the second moment time to build a reliable estimate.

```
Learning rate during warmup (2000 steps, max_lr=3e-4):

lr
3e-4 |                    ╱─────────────
     |                  ╱
     |                ╱
     |              ╱
     |            ╱
     |          ╱
     |        ╱
     |      ╱
     |    ╱
     |  ╱
   0 |╱──────────────────────────────────→ step
     0        1000      2000
              warmup     training begins
```

### Cosine Decay

After warmup, decay the learning rate following a cosine curve to the minimum:

```
lr_t = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T))
```

```
Full schedule: linear warmup + cosine decay

lr
3e-4 |          ╱╲
     |        ╱    ╲
     |      ╱       ╲
     |    ╱           ╲
     |  ╱               ╲
     | ╱                  ╲
     |╱                     ╲
     |                        ╲
     |                          ╲
1e-5 |──────────────────────────────╲────→ step
     0   warmup               total steps
```

### The Code

```python
def get_lr(step, warmup_steps, max_lr, min_lr, total_steps):
    # Linear warmup
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps

    # Cosine decay
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))
```

### Typical Values

```
Setting              Small model       LLM pretraining
──────────           ───────────       ───────────────
max_lr               1e-3              3e-4
min_lr               1e-5              3e-5 (10% of max)
warmup_steps         500               2000
total_steps          10,000            100,000+
```

---

## Piece 8: Hyperparameter Guidance

### The Defaults (And When to Change Them)

```
Hyperparameter    Default     When to change
──────────────    ───────     ─────────────────────────────────────────
beta1             0.9         0.95 for very long training runs
beta2             0.999       0.95 for LLMs (Chinchilla, LLaMA recommend this)
eps               1e-8        1e-6 for mixed-precision training (fp16/bf16)
lr                1e-3        Almost always needs tuning per task
weight_decay      0.01        0.1 for large LLM pretraining
```

### Why beta2=0.95 for LLMs?

The default `beta2=0.999` gives the second moment a very long memory (~1000 steps). In LLM training, the loss landscape shifts significantly as the model learns. A long memory means `v` is tracking old gradient statistics that are no longer relevant.

`beta2=0.95` shortens the memory to ~20 steps, making the adaptive learning rate more responsive. This was identified in the Chinchilla paper (Hoffmann et al., 2022) and has become standard for LLM pretraining.

### Why eps=1e-6 for Mixed Precision?

In fp16/bf16 training, very small values get flushed to zero. If `sqrt(v_hat)` is small and `eps=1e-8`, the denominator might underflow. Bumping eps to `1e-6` adds a safety margin without meaningfully affecting training.

---

## Piece 9: Common Mistakes

### Mistake 1: Using Adam Instead of AdamW

```python
# WRONG: L2 regularization is scaled by adaptive term
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=0.01)

# CORRECT: weight decay is applied directly to parameters
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
```

In PyTorch, `torch.optim.Adam` with `weight_decay > 0` applies L2 regularization (the broken way). `torch.optim.AdamW` applies decoupled weight decay (the correct way). Always use `AdamW`.

### Mistake 2: No Learning Rate Warmup

Starting with the full learning rate from step 0 can cause training instability, especially with large batch sizes or large learning rates. The second moment needs time to calibrate.

```python
# WRONG: constant learning rate from the start
optimizer = AdamW(model.parameters(), lr=3e-4)
# ... training loop with no schedule

# CORRECT: warmup + cosine decay
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=3e-4, total_steps=total_steps,
    pct_start=0.05  # 5% of training is warmup
)
```

### Mistake 3: Wrong Weight Decay for Your Task

```
Task                    Recommended weight_decay
────                    ────────────────────────
Fine-tuning (small)     0.0 to 0.01
General training        0.01
LLM pretraining         0.1
```

Too much weight decay prevents the model from fitting. Too little leads to overfitting. The sweet spot depends on model size and dataset size.

### Mistake 4: Applying Weight Decay to Biases and LayerNorm

Weight decay should only apply to weight matrices, not to biases or normalization parameters. These are low-dimensional and do not benefit from regularization.

```python
# CORRECT: separate parameter groups
decay_params = [p for n, p in model.named_parameters() if "bias" not in n and "norm" not in n]
no_decay_params = [p for n, p in model.named_parameters() if "bias" in n or "norm" in n]

optimizer = AdamW([
    {"params": decay_params, "weight_decay": 0.01},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=3e-4)
```

### Mistake 5: Not Scaling LR With Batch Size

If you increase batch size, you often need to increase the learning rate proportionally (linear scaling rule). Doubling batch size? Try doubling the learning rate (and increase warmup steps).

---

## Verify Against PyTorch

The acid test: run the same training loop with our Adam and `torch.optim.Adam`. If our implementation is correct, the loss curves should be identical.

We trained a small MLP (20 -> 64 -> 64 -> 5) on a synthetic classification task for 30 epochs with identical seeds. Here is the result:

```
Max loss difference between implementations: 0.00e+00
Match: losses are essentially identical.
```

Run it yourself:
```
python compare.py
```

---

## Learning Rate Sweep

Adam is robust to learning rate choice — more so than SGD — but the LR still matters. We swept from `1e-5` to `1e-1` (log-spaced, 12 points) and ran 20 epochs per configuration.

```
python sweep.py
```

A well-tuned Adam typically sits in the `1e-4` to `1e-3` range. The loss landscape is flat around the optimum — unlike SGD which can be very sensitive.

---

## Recap

```
Component          What it does                          Key formula
─────────          ────────────                          ───────────
First moment (m)   EMA of gradients; smooths noise       m = b1*m + (1-b1)*g
Second moment (v)  EMA of squared grads; adaptive LR     v = b2*v + (1-b2)*g^2
Bias correction    Fixes zero-initialization bias        m_hat = m/(1-b1^t)
Adam update        Momentum direction, RMSProp scale     theta -= lr*m_hat/(sqrt(v_hat)+eps)
AdamW              Decoupled weight decay (use this)     theta *= (1 - lr*wd) after update
Warmup             Ramp LR up; let v stabilize           lr = max_lr * step/warmup_steps
Cosine decay       Anneal LR smoothly to minimum         lr = min + 0.5*(max-min)*(1+cos)
```

**The 80/20 of Adam in practice:**
- Use `torch.optim.AdamW`, not `Adam`
- `lr=1e-3` to `3e-4` for most tasks; sweep if in doubt
- `beta1=0.9, beta2=0.999` for small/medium models; `beta2=0.95` for LLMs
- Always use warmup (5-10% of total steps)
- Weight decay 0.01 default; 0.1 for large pretraining; exclude biases and norms
- For fp16/bf16: bump `eps` to `1e-6`

Our from-scratch implementation produces **identical results** to `torch.optim.Adam`. Now you know exactly what every line is doing.

---

Get the video walkthrough of AdamW, LAMB optimizer, Sophia optimizer, large batch training, and LR scheduling best practices: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
