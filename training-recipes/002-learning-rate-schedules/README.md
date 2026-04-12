# Learning Rate Schedules From Scratch

The learning rate is the most important hyperparameter in training deep learning models. Set it too high and your loss diverges; too low and you waste compute. But the learning rate also changes *during* training. A schedule that decays it over time can mean the difference between a model that converges and one that gets stuck.

Why does decaying the LR help? When you're far from the optimum, you want large updates to make fast progress. As you get closer, large updates overshoot — you need to shrink the step size. That's the intuition behind every learning rate schedule.

In this lesson we build four schedulers from scratch — Constant, Step Decay, Cosine Annealing, and Linear Warmup + Cosine Decay — and compare how they affect training.

---

## Hook

Most tutorials just call `torch.optim.lr_scheduler.StepLR` or `torch.optim.lr_scheduler.CosineAnnealingLR` and move on. But the formulas hide the geometry: cosine annealing smoothly interpolates between your initial LR and zero using the cosine function's shape. Linear warmup prevents the early gradient explosion that kills transformers. Understanding the math lets you choose and tune schedulers with intention, not guesswork.

---

## 1. The Baseline: Constant LR

The simplest approach: never change the learning rate. This works reasonably well with Adam (which adapts per-parameter), but it's rarely optimal. The LR that gives fast early progress will overshoot late in training.

```
lr(t) = initial_lr
```

```python
class ConstantLR:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.initial_lr = optimizer.param_groups[0]["lr"]

    def step(self):
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr
```

---

## 2. Step Decay

Drop the learning rate by a fixed factor `gamma` every `N` epochs. It's simple, interpretable, and was the standard before cosine schedules became popular.

```
lr(t) = initial_lr * gamma ** floor(t / step_size)
```

```python
class StepDecay:
    def __init__(self, optimizer, step_size=10, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.initial_lr = optimizer.param_groups[0]["lr"]

    def step(self):
        factor = self.gamma ** (self.last_epoch // self.step_size)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.initial_lr * factor
        self.last_epoch += 1
```

---

## 3. Cosine Annealing

Cosine annealing decays the LR along a half-cosine curve from `initial_lr` to near-zero. It was popularized by [Loshchilov & Hutter (2017)](https://arxiv.org/abs/1608.03983) as part of the SGDR method. The cosine shape gives slow decay in the middle and faster decay near the end — you stay near the peak LR longer than linear decay would allow.

```
lr(t) = initial_lr * (1 + cos(pi * t / T)) / 2
```

where `T` is the total number of steps.

```python
import math

class CosineAnnealing:
    def __init__(self, optimizer, total_steps, eta_min=0.0):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.initial_lr = optimizer.param_groups[0]["lr"]

    def get_lr(self):
        progress = self.last_epoch / self.total_steps
        cosine = (1 + math.cos(math.pi * progress)) / 2
        return self.eta_min + (self.initial_lr - self.eta_min) * cosine

    def step(self):
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group["lr"] = lr
        self.last_epoch += 1
```

---

## 4. Linear Warmup + Cosine Decay

This is the standard schedule for transformers (BERT, GPT, Stable Diffusion). Two phases:

1. **Warmup** (first `warmup_steps`): LR ramps from 0 to `initial_lr` linearly. This prevents gradient explosion in early training when parameters are random and gradients are large.
2. **Decay**: After warmup, LR follows a cosine decay to `eta_min`.

```
Phase 1 (t < warmup):     lr(t) = initial_lr * t / warmup_steps
Phase 2 (t >= warmup):     lr(t) = eta_min + (initial_lr - eta_min)
                           * (1 + cos(pi * (t - warmup_steps) / (T - warmup_steps))) / 2
```

```python
class LinearWarmupCosineDecay:
    def __init__(self, optimizer, warmup_steps, total_steps, eta_min=0.0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.eta_min = eta_min
        self.initial_lr = optimizer.param_groups[0]["lr"]

    def step(self):
        t = self.last_epoch
        if t < self.warmup_steps:
            lr = self.initial_lr * t / self.warmup_steps
        elif t < self.total_steps:
            progress = (t - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine = (1 + math.cos(math.pi * progress)) / 2
            lr = self.eta_min + (self.initial_lr - self.eta_min) * cosine
        else:
            lr = self.eta_min

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.last_epoch += 1
```

---

## 5. Compare Loss Curves

We trained identical MLPs with each scheduler for 80 epochs. Here is the result:

```
python compare.py
```

Constant LR converges slowest after the initial rapid descent. Step decay shows visible "jumps" when the LR drops. Cosine annealing and warmup+cosine produce the smoothest curves and typically reach the lowest final loss.

---

## 6. Benchmark: Throughput

The scheduler itself adds negligible overhead vs. the forward+backward pass.

```
python benchmark.py
```

All schedulers process the same number of samples per second because the overhead is O(1) per step. The real cost is the backward pass, not the LR update.

---

## Recap

- **Constant LR** is the baseline but is rarely optimal — early large steps can prevent late convergence.
- **Step Decay** drops the LR by a factor every N epochs — simple, interpretable, but the LR changes abruptly.
- **Cosine Annealing** smoothly decays along a cosine curve — popular because it stays near peak LR longer than linear decay.
- **Linear Warmup + Cosine Decay** is the standard for transformers — warmup prevents early gradient explosion, cosine decay gives smooth convergence.
- Scheduler overhead is negligible vs. the backward pass.

---

Get the video walkthrough of CosineAnnealingWarmRestarts, Layer-wise Learning Rate Decay (LLRD), and per-layer LR profiling: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
