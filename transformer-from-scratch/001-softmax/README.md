# Softmax From Scratch

Every time a transformer decides which words to attend to, it runs softmax.
Every time a classifier picks a label, it runs softmax. Every time a language
model samples its next token, it runs softmax. It is arguably the single most
frequently executed nonlinearity in modern deep learning — and most tutorials
hand-wave past it with a one-line formula. That is a mistake, because softmax
has subtleties that cause real bugs in production: silent numerical overflow,
wrong-axis reductions that produce garbage probabilities, and memory blowups
in long-context attention. This lesson builds softmax from the ground up so
you understand every moving part.

What softmax actually does: it takes any vector of real numbers (called
**logits**) and maps them into a **probability distribution** — every output
is between 0 and 1, and they all sum to 1. The formula amplifies differences
exponentially: a logit that is 2 units larger than its neighbor gets roughly
7x more probability mass, not 2x. This exponential sharpening is what makes
attention patterns crisp and classification outputs decisive.

```
  logits:  [ 2.0,   1.0,   0.5 ]
              |       |       |
           exp(2)  exp(1)  exp(0.5)
              |       |       |
           [ 7.39,  2.72,   1.65 ]      raw exponentials
              |       |       |
           /  sum = 11.76         \
              |       |       |
           [ 0.63,  0.23,   0.14 ]      probabilities (sum to 1)
```

---

## Piece 1 — The naive formula and why it breaks

The textbook definition looks clean:

```
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

And the code is equally clean:

```python
import math

def naive_softmax(x):
    exp_x = [math.exp(xi) for xi in x]   # exponentiate each element
    total = sum(exp_x)                     # sum of all exponentials
    return [e / total for e in exp_x]      # normalize to get probabilities
```

This works for small inputs. But try it on numbers that a real model produces:

```python
>>> math.exp(710)
inf
>>> math.exp(709)
8.218407461554972e+307
```

In float32, `exp(x)` overflows to `inf` at roughly x = 88.7. In float64 it
survives until about x = 709.8. Real transformer logits routinely land in the
hundreds during early training before the model has learned to keep its outputs
tame. The moment a single `exp(x_i)` hits `inf`, the sum becomes `inf`, and
every output becomes `inf / inf = nan`. Your entire forward pass is garbage.

Here is what that looks like concretely:

```
logits = [1000.0, 1001.0, 1002.0]

naive_softmax:
  exp(1000) = inf
  exp(1001) = inf
  exp(1002) = inf
  sum       = inf
  output    = [nan, nan, nan]    <-- everything is destroyed
```

This is not a theoretical problem. It is the single most common numerical
bug in hand-rolled softmax implementations.

---

## Piece 2 — Stable softmax: one subtraction fixes everything

The fix is elegant. Since softmax is **shift-invariant** — subtracting the
same constant from every logit does not change the output probabilities — we
subtract the maximum value before exponentiating:

```python
import math

def stable_softmax(x):
    max_x = max(x)                                    # find the largest logit
    exp_x = [math.exp(xi - max_x) for xi in x]       # subtract max, then exp
    total = sum(exp_x)                                 # sum the safe exponentials
    return [e / total for e in exp_x]                  # normalize
```

Why does this work? After subtracting `max_x`, the largest element becomes
`exp(0) = 1`. Every other element becomes `exp(negative) < 1`. Nothing can
overflow. And the mathematical proof that the result is identical is one line:

```
exp(x_i - c) / sum_j exp(x_j - c)
= exp(x_i) * exp(-c) / (sum_j exp(x_j) * exp(-c))
= exp(x_i) / sum_j exp(x_j)                          # exp(-c) cancels
```

Now our earlier example works perfectly:

```python
x = [1000.0, 1001.0, 1002.0]
stable_softmax(x)
# [0.09003057, 0.24472847, 0.66524096]   <-- correct probabilities
```

The largest logit (1002) gets ~66% of the probability. The ratios are
exactly what the naive formula would give if it could handle the numbers.

---

## Piece 3 — Temperature scaling

Dividing every logit by a scalar `T` (temperature) before softmax controls
how **sharp** or **flat** the output distribution is:

```python
def temperature_softmax(x, temperature=1.0):
    scaled = [xi / temperature for xi in x]           # scale logits by T
    max_scaled = max(scaled)                           # stability trick
    exp_x = [math.exp(si - max_scaled) for si in scaled]
    total = sum(exp_x)
    return [e / total for e in exp_x]
```

The intuition: dividing by a large T shrinks all logits toward zero, making
their exponentials closer together. Dividing by a small T amplifies the gaps.

```
logits = [2.0, 1.0, 0.5]

T = 0.1 (very sharp):   [0.9998, 0.0002, 0.0000]   almost argmax
T = 0.5 (sharp):        [0.8360, 0.1142, 0.0498]
T = 1.0 (normal):       [0.6265, 0.2312, 0.1422]
T = 2.0 (flat):         [0.3936, 0.2978, 0.3086]
T = 5.0 (very flat):    [0.3548, 0.3226, 0.3226]   almost uniform
T -> inf:               [0.3333, 0.3333, 0.3333]   perfectly uniform
```

Here is what those distributions look like:

```
T=0.1   T=1.0   T=5.0
 |       |       |
 #       #       #  #  #
 #       #  #    #  #  #
 #       #  #  # #  #  #
---     ------  ------
 A       A B C   A  B  C

sharp   normal  flat
```

**Where temperature is used:**
- **Knowledge distillation** (T = 2-20): the teacher model's soft targets
  carry more information when flattened with high temperature
- **LLM sampling** (T = 0.0-2.0): low T for factual answers, high T for
  creative writing
- **Reinforcement learning** exploration: high T early, anneal down over time

---

## Piece 4 — Log-softmax: what cross-entropy actually needs

When you train a classifier, the loss function is almost always cross-entropy,
which PyTorch implements as `NLLLoss(log_softmax(logits))`. The key function
here is **log-softmax** — the logarithm of softmax:

```
log_softmax(x_i) = log(softmax(x_i))
                 = log(exp(x_i - max) / sum_j exp(x_j - max))
                 = (x_i - max) - log(sum_j exp(x_j - max))
```

Notice what happened: the `log` and `exp` canceled each other, leaving just
`x_i - max` minus the log of the sum. This avoids computing softmax
probabilities (which can underflow to zero for very negative logits) and
then taking `log(0) = -inf`. Log-softmax is more numerically stable than
computing softmax and log separately.

```python
def log_softmax(x):
    max_x = max(x)                                         # shift for stability
    shifted = [xi - max_x for xi in x]                     # x_i - max
    log_sum = math.log(sum(math.exp(s) for s in shifted))  # log(sum(exp(...)))
    return [s - log_sum for s in shifted]                   # final log-probs
```

**Why this matters in practice:** PyTorch's `F.cross_entropy` internally
computes log-softmax + NLLLoss in one fused operation. If you manually apply
`softmax` and then `log`, you lose numerical precision and can get `nan` in
your loss. This is one of the most common mistakes in PyTorch training loops.

```python
# WRONG — loses precision
probs = torch.softmax(logits, dim=-1)
loss = -torch.log(probs[target])          # log(0) = -inf if probs underflow

# RIGHT — numerically stable
loss = torch.nn.functional.cross_entropy(logits, target)
```

---

## Piece 5 — Online (streaming) softmax

Standard softmax requires **three passes** over the data: one to find the max,
one to compute the exponentials, and one to normalize. For long sequences in
attention, this means reading the full key vector from memory three times.
**Online softmax** (Milakov & Gimelshein, 2018) does it in a single pass. This
is the algorithm that Flash Attention uses internally.

The idea: maintain a running maximum and a running sum simultaneously, and
correct the sum every time a new maximum appears.

```python
def online_softmax(x):
    # Single-pass: track running max and running sum of exponentials
    m = float('-inf')       # running maximum
    d = 0.0                 # running sum of exp(x_i - m), will be corrected
    n = len(x)

    # Forward pass: compute m (global max) and d (sum of exp)
    for i in range(n):
        m_prev = m
        m = max(m, x[i])                 # update running max
        d = d * math.exp(m_prev - m) + math.exp(x[i] - m)
        #       ^^^^^^^^^^^^^^^^^^^^^^   correct old sum for new max
        #                                ^^^^^^^^^^^^^^^^^ add new term

    # Output pass: normalize
    return [math.exp(x[i] - m) / d for i in range(n)]
```

The critical line is `d = d * math.exp(m_prev - m) + math.exp(x[i] - m)`.
When `x[i]` becomes the new maximum, `m_prev - m` is negative, so the old
sum `d` gets scaled down by the ratio between old and new max. When the max
does not change, `exp(m_prev - m) = exp(0) = 1`, so `d` is untouched. This
rescaling trick is what lets us avoid storing all values and still get the
exact same result as three-pass softmax.

**Why Flash Attention cares:** In standard attention, you compute the full
`N x N` attention matrix, apply softmax row by row, then multiply by V.
This requires O(N^2) memory. Flash Attention tiles the computation and uses
online softmax to process each tile incrementally, reducing memory from
O(N^2) to O(N). Without online softmax, Flash Attention would not work.

---

## Piece 6 — Softmax in attention

In a transformer attention head, softmax creates the **attention weight
matrix** that decides how much each token attends to every other token:

```
scores = Q @ K^T / sqrt(d_k)      shape: [seq_len, seq_len]
weights = softmax(scores, dim=-1)  shape: [seq_len, seq_len]  (rows sum to 1)
output  = weights @ V              shape: [seq_len, d_v]
```

Each row of the weight matrix is a probability distribution over all keys.
Softmax along `dim=-1` means "for each query, distribute attention across
all keys." The `sqrt(d_k)` divisor is effectively temperature scaling — it
prevents the dot products from growing too large in high-dimensional spaces,
which would push softmax into a near-one-hot regime where gradients vanish.

```
Query "The"  attends to:  [The: 0.05, cat: 0.10, sat: 0.70, on: 0.10, mat: 0.05]
Query "cat"  attends to:  [The: 0.15, cat: 0.05, sat: 0.20, on: 0.10, mat: 0.50]
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                       each row sums to 1.0 (softmax output)
```

Without softmax, the raw dot-product scores are unbounded. Softmax
constrains them to valid probability weights, ensuring the weighted
combination of value vectors is a proper convex combination.

---

## Piece 7 — Gumbel-softmax: differentiable discrete sampling

Sometimes you need a model to make a **hard discrete choice** (pick one of N
options) but still be trainable with gradient descent. The problem: argmax is
not differentiable. Gumbel-softmax (Jang et al., 2017) solves this by adding
Gumbel noise before softmax, producing a sample that is approximately
one-hot but still differentiable:

```python
import random

def gumbel_softmax(logits, temperature=1.0):
    # Step 1: sample Gumbel noise for each logit
    gumbels = [-math.log(-math.log(random.random())) for _ in logits]

    # Step 2: add noise to logits (this is the "Gumbel trick")
    noisy = [l + g for l, g in zip(logits, gumbels)]

    # Step 3: apply temperature softmax to get a soft sample
    return temperature_softmax(noisy, temperature)
```

At low temperature, the output approaches a one-hot vector (hard sample).
At high temperature, it approaches a uniform distribution. During training,
you use a moderate temperature so gradients flow; at inference, you can
take the argmax of the logits directly.

**Where it is used:** variational autoencoders with discrete latent variables,
neural architecture search (choosing which operation to apply), and any model
that needs to route information through a discrete bottleneck.

---

## Piece 8 — The Jacobian of softmax

When backpropagation flows through softmax, it needs the **Jacobian** — the
matrix of all partial derivatives `ds_i / dx_j`. Softmax has a clean closed
form:

```
When i == j:  ds_i / dx_i = s_i * (1 - s_i)     (like sigmoid)
When i != j:  ds_i / dx_j = -s_i * s_j           (outputs are coupled)
```

In matrix notation: `J = diag(s) - s @ s^T`.

```python
def softmax_jacobian(s):
    n = len(s)
    J = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                J[i][j] = s[i] * (1.0 - s[i])    # diagonal: like sigmoid derivative
            else:
                J[i][j] = -s[i] * s[j]            # off-diagonal: coupling term
    return J
```

Two things to notice. First, the diagonal entries `s_i * (1 - s_i)` are
largest when `s_i = 0.5` and vanish when `s_i` approaches 0 or 1. This means
gradients are strongest for uncertain predictions and weakest for confident
ones — exactly the behavior you want for training. Second, the off-diagonal
entries are always negative: increasing one logit always decreases the
probability of every other class. The outputs compete with each other.

---

## Piece 9 — Benchmark: our softmax vs torch

```python
import torch, time

x = torch.randn(512, dtype=torch.float32)

# torch.softmax — backed by optimized C++/CUDA
start = time.perf_counter()
for _ in range(2000): torch.softmax(x, dim=0)
torch_time = (time.perf_counter() - start) / 2000

# our stable_softmax — pure Python, for understanding
start = time.perf_counter()
for _ in range(2000): stable_softmax(x.tolist())
our_time = (time.perf_counter() - start) / 2000
```

Pure Python is **100-1000x slower** than `torch.softmax`, which uses fused
C++/CUDA kernels. The point of this lesson is not speed — it is understanding.
The benchmark exists so you can verify that our stable implementation produces
**numerically identical output** to `torch.softmax` across float16, bfloat16,
float32, and float64. Run `benchmark.py` to see exact timings and error
margins for each dtype and dimension.

---

## Common mistakes

These are the bugs that actually show up in real codebases:

**1. Forgetting the max subtraction**

```python
# BUG: overflows for large logits
def bad_softmax(x):
    exp_x = [math.exp(xi) for xi in x]
    return [e / sum(exp_x) for e in exp_x]
```

Always subtract `max(x)` first. There is no reason not to.

**2. Wrong dim argument in PyTorch**

```python
scores = torch.randn(batch, seq_len, seq_len)

torch.softmax(scores, dim=-1)   # CORRECT: softmax over last axis (keys)
torch.softmax(scores, dim=0)    # WRONG: softmax over batch dimension
torch.softmax(scores, dim=1)    # WRONG: softmax over query dimension
```

For attention weights, you almost always want `dim=-1` so that each query's
attention distribution sums to 1 across keys. Getting this wrong produces
probabilities that sum to 1 across the wrong axis — the model trains but
performs terribly, and the bug is very hard to spot because the output shape
is the same.

**3. Applying softmax twice**

```python
# BUG: double softmax makes the distribution too flat
attn_weights = torch.softmax(scores, dim=-1)
attn_weights = torch.softmax(attn_weights, dim=-1)   # already probabilities!
```

This is surprisingly common when refactoring code. Since softmax inputs are
already between 0 and 1 after the first application, the second softmax
produces a nearly uniform distribution. Your attention becomes meaningless.

**4. Using softmax + log instead of log_softmax**

```python
# BAD: loses precision, can produce -inf
log_probs = torch.log(torch.softmax(logits, dim=-1))

# GOOD: numerically stable
log_probs = torch.log_softmax(logits, dim=-1)
```

As explained in Piece 4, the fused `log_softmax` avoids the intermediate
probability step where small values underflow to zero.

**5. Not handling the temperature=0 edge case**

```python
# Crashes with division by zero
def bad_temp_softmax(x, T):
    return softmax([xi / T for xi in x])    # T=0 -> inf logits -> nan
```

Temperature must be strictly positive. At T approaching 0, the output
approaches a one-hot vector on the largest logit (equivalent to argmax).
Handle this as a special case if your API allows T=0.

---

## Recap

Here is what we covered, and why each piece matters in practice:

- **Naive softmax** (`exp(x_i) / sum exp(x_j)`) overflows for inputs larger
  than ~88 in float32 or ~709 in float64. Never use it on raw model outputs.

- **Stable softmax** subtracts `max(x)` before exponentiating. Mathematically
  identical, numerically safe. This is what every production implementation
  does, including PyTorch and JAX.

- **Temperature scaling** divides logits by `T` before softmax. `T > 1`
  flattens the distribution (more exploration, softer targets). `T < 1`
  sharpens it (more exploitation, harder targets). Used in distillation,
  sampling, and RL.

- **Log-softmax** fuses `log` and `softmax` to avoid the `log(0)` trap. Use
  `F.cross_entropy` or `F.log_softmax` instead of `log(softmax(...))`.

- **Online softmax** computes the result in a single pass by maintaining a
  running max and correcting the running sum on the fly. This is the core
  algorithmic trick inside Flash Attention.

- **Softmax in attention** creates the weight matrix where each row sums to 1.
  The `sqrt(d_k)` scaling is temperature that prevents gradient vanishing.
  Always use `dim=-1`.

- **Gumbel-softmax** adds Gumbel noise for differentiable discrete sampling.
  Used in VAEs, NAS, and discrete bottleneck models.

- **The Jacobian** has a clean form: `diag(s) - s @ s^T`. Gradients are
  strongest for uncertain predictions (s near 0.5) and vanish for confident
  ones (s near 0 or 1).

- **`torch.softmax` is 100-1000x faster** than pure Python. Our version is
  for understanding the math, not for production.

---

Get the video walkthrough of log-softmax derivation, Gumbel-softmax, dtype numerical comparison, temperature scaling entropy plots, and attention weight visualisation: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
