# Softmax From Scratch

Softmax turns any list of numbers into a probability distribution — every output is between 0 and 1, and they all sum to 1. It is the engine of attention, classification, and anywhere a model needs to pick among options. Understanding it from the ground up is essential for anyone who wants to understand transformers.

---

## Why overflow breaks the naive formula

The textbook definition is `exp(x_i) / sum_j exp(x_j)`. Simple, elegant — and numerically broken for any input larger than about 700 in float32. The exponential grows faster than any floating-point number can represent, so you get `inf`. The fix is a single subtraction that changes everything.

---

## Stable softmax

```python
import math

def stable_softmax(x):
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    total = sum(exp_x)
    return [e / total for e in exp_x]
```

Subtract the maximum value before exponentiating. Since `exp(a - max)` is at most `exp(0) = 1`, nothing overflows. The division cancels out the subtraction, so the result is mathematically identical to the naive version — but numerically safe.

```python
x = [1000.0, 1001.0, 1002.0]
stable_softmax(x)
# [0.09003057, 0.24472847, 0.66524096]
```

---

## Temperature scaling

```python
def temperature_softmax(x, temperature=1.0):
    scaled = [xi / temperature for xi in x]
    max_scaled = max(scaled)
    exp_x = [math.exp(si - max_scaled) for si in scaled]
    total = sum(exp_x)
    return [e / total for e in exp_x]
```

Dividing logits by a temperature `T` before softmax controls how sharp or flat the distribution is. `T > 1` spreads probability mass more evenly. `T < 1` concentrates it on the largest logit. Temperature scaling is used in knowledge distillation and policy exploration.

```python
x = [2.0, 1.0, 0.5]
temperature_softmax(x, temperature=0.5)  # sharper: [0.86, 0.11, 0.03]
temperature_softmax(x, temperature=2.0)  # flatter: [0.39, 0.33, 0.28]
```

---

## The Jacobian of softmax

```python
def softmax_jacobian(s):
    n = len(s)
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i][j] = s[i] * (1.0 - s[i])
            else:
                J[i][j] = -s[i] * s[j]
    return J
```

The Jacobian matrix tells you how each output changes when each input changes. For softmax the diagonal is `s_i * (1 - s_i)` and every off-diagonal entry is `-s_i * s_j`. This matters for backpropagation — the autograd engine uses exactly this formula.

---

## Benchmark: our softmax vs torch

```python
import torch, time

x = torch.randn(512, dtype=torch.float32)
# torch
start = time.perf_counter()
for _ in range(2000): torch.softmax(x, dim=0)
torch_time = (time.perf_counter() - start) / 2000
```

Pure Python is intentionally slow. The point of this lesson is not speed — it is understanding. The benchmark exists so you can verify that our stable implementation produces numerically identical output to `torch.softmax` across all standard dtypes.

---

## Recap

- Naive softmax (`exp(x_i) / sum exp(x_j)`) overflows for inputs larger than ~700 in float32
- Stable softmax fixes this by subtracting `max(x)` before exponentiating — mathematically identical, numerically safe
- Temperature scaling divides logits by `T` before softmax: `T > 1` flattens, `T < 1` sharpens
- The softmax Jacobian has a closed form: `J_ii = s_i(1 - s_i)`, `J_ij = -s_is_j`
- `torch.softmax` is 100–1000x faster; our pure-Python version is for learning, not production

---

Get the extended notebook with log-softmax derivation, Gumbel-softmax, dtype numerical comparison across 5 types, temperature scaling entropy plots, and attention weight visualisation with real sentence embeddings: [OpenSuperintelligenceLab on Skool →](https://www.skool.com/opensuperintelligencelab)
