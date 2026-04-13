# Softmax Advanced Topics

This is the deep-dive companion to the softmax tutorial. Everything here builds on the stable softmax foundation from the main lesson — if `exp(x_i - max(x))` does not feel automatic yet, go back and re-read that first.

---

## 1. Log-softmax derivation

Cross-entropy loss needs `log(softmax(x))`. Computing softmax first and then taking the log is wasteful and numerically dangerous — softmax outputs can be tiny, and `log(tiny)` amplifies floating-point error. Log-softmax fuses both operations into one stable formula.

### Starting point

```
softmax(x_i) = exp(x_i) / sum_j exp(x_j)
```

Take the log of both sides:

```
log(softmax(x_i)) = log(exp(x_i)) - log(sum_j exp(x_j))
                   = x_i - log(sum_j exp(x_j))
```

### Making it stable

The sum `sum_j exp(x_j)` overflows for large inputs. Apply the max-subtraction trick inside the log:

```
sum_j exp(x_j) = sum_j exp(x_j - m + m)        where m = max(x)
               = exp(m) * sum_j exp(x_j - m)
```

So:

```
log(sum_j exp(x_j)) = m + log(sum_j exp(x_j - m))
```

Substitute back:

```
log_softmax(x_i) = x_i - m - log(sum_j exp(x_j - m))
```

This is the formula PyTorch uses internally. Every term is safe: `x_j - m <= 0`, so `exp(x_j - m) <= 1`, so the sum never overflows. The log of a sum of values in `(0, 1]` is well-conditioned.

### Why not just log(softmax(x))?

```python
import math

def naive_log_softmax(x):
    s = stable_softmax(x)           # some outputs may be ~1e-38
    return [math.log(si) for si in s]  # log(1e-38) = -87.5 — loses precision

def fused_log_softmax(x):
    m = max(x)
    shifted = [xi - m for xi in x]
    log_sum = math.log(sum(math.exp(s) for s in shifted))
    return [s - log_sum for s in shifted]  # no tiny intermediate values
```

The fused version avoids the round-trip through tiny probabilities. This is why `torch.nn.functional.log_softmax` exists as a separate function and why `CrossEntropyLoss` expects raw logits — it calls log-softmax internally.

```python
x = [100.0, 101.0, 102.0]
fused_log_softmax(x)
# [-2.4076, -1.4076, -0.4076]
```

### Connection to CrossEntropyLoss

PyTorch's `CrossEntropyLoss(logits, target)` computes:

```
loss = -log_softmax(logits)[target]
```

It never materializes the softmax probabilities. This is both faster and more numerically stable than `NLLLoss(log(softmax(logits)), target)`.

---

## 2. Online softmax (the Flash Attention algorithm)

Standard softmax requires two passes over the data: one to find the max, one to compute `exp` and sum. When the sequence is millions of tokens long, this means loading from slow GPU memory twice. The online softmax algorithm does it in a single streaming pass.

### The core idea

Maintain two running values as you scan left to right:

- `m` — the running maximum seen so far
- `d` — the running sum of exponentials, always relative to the current `m`

When a new value `x_k` arrives:

1. If `x_k > m`, the old sum was computed relative to a smaller max. Rescale: `d = d * exp(m_old - m_new)`
2. Add the new term: `d = d + exp(x_k - m)`

After processing all values, the softmax denominator is `d` and the max is `m`.

### Implementation

```python
import math

def online_softmax(x):
    m = float('-inf')
    d = 0.0

    # single pass: compute max and denominator simultaneously
    for xi in x:
        m_old = m
        m = max(m, xi)
        d = d * math.exp(m_old - m) + math.exp(xi - m)

    # compute final probabilities
    return [math.exp(xi - m) / d for xi in x]
```

### Step-by-step walkthrough

Input: `x = [2.0, 5.0, 1.0, 8.0, 3.0, 4.0]`

```
Step 0: x=2.0  m_old=-inf  m=2.0   d = 0*exp(-inf-2) + exp(2-2)   = 0 + 1.0       = 1.0
Step 1: x=5.0  m_old=2.0   m=5.0   d = 1.0*exp(2-5) + exp(5-5)    = 0.0498 + 1.0   = 1.0498
Step 2: x=1.0  m_old=5.0   m=5.0   d = 1.0498*exp(0) + exp(1-5)   = 1.0498 + 0.0183 = 1.0681
Step 3: x=8.0  m_old=5.0   m=8.0   d = 1.0681*exp(5-8) + exp(8-8) = 0.0532 + 1.0   = 1.0532
Step 4: x=3.0  m_old=8.0   m=8.0   d = 1.0532*exp(0) + exp(3-8)   = 1.0532 + 0.0067 = 1.0599
Step 5: x=4.0  m_old=8.0   m=8.0   d = 1.0599*exp(0) + exp(4-8)   = 1.0599 + 0.0183 = 1.0783
```

Final: `m = 8.0`, `d = 1.0783`

```
softmax = [exp(xi - 8.0) / 1.0783 for xi in x]
        = [0.0025, 0.0499, 0.0009, 0.9274, 0.0062, 0.0170]
        sums to ≈ 1.0  ✓
```

### Why Flash Attention needs this

Flash Attention tiles the attention matrix into blocks that fit in SRAM. Each block sees only a slice of the key sequence. Online softmax lets each block update the running `(m, d)` state without ever needing the full row in memory. This is what makes O(N) memory attention possible.

---

## 3. Gumbel-softmax

Softmax converts logits into probabilities, but sampling from those probabilities (argmax or multinomial) is not differentiable — gradients cannot flow through discrete choices. Gumbel-softmax solves this with a continuous relaxation.

### The Gumbel-max trick

To sample a category from logits `z`, add Gumbel noise and take argmax:

```
g_i ~ Gumbel(0, 1)       — sample by: g = -log(-log(u)), u ~ Uniform(0,1)
category = argmax(z_i + g_i)
```

This produces exact samples from `softmax(z)` without ever computing the probabilities. But argmax is still not differentiable.

### The relaxation

Replace argmax with softmax at a low temperature:

```
y_i = softmax((z_i + g_i) / tau)
```

As `tau -> 0`, this approaches a one-hot vector (hard sample). As `tau -> inf`, it approaches uniform. The key insight: this is differentiable everywhere with respect to `z`.

### Full implementation

```python
import math
import random

def gumbel_sample():
    u = random.uniform(1e-10, 1.0 - 1e-10)
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature=1.0):
    noisy = [z + gumbel_sample() for z in logits]
    return temperature_softmax(noisy, temperature)

def temperature_softmax(x, temperature):
    scaled = [xi / temperature for xi in x]
    m = max(scaled)
    exp_x = [math.exp(s - m) for s in scaled]
    total = sum(exp_x)
    return [e / total for e in exp_x]
```

```python
logits = [2.0, 1.0, 0.5]

gumbel_softmax(logits, temperature=0.1)   # nearly one-hot: [0.99, 0.01, 0.00]
gumbel_softmax(logits, temperature=1.0)   # soft sample:    [0.58, 0.27, 0.15]
gumbel_softmax(logits, temperature=5.0)   # nearly uniform: [0.38, 0.33, 0.29]
```

### Straight-through estimator

In practice you often need a hard one-hot vector in the forward pass but smooth gradients in the backward pass. The straight-through estimator does exactly this:

```python
def straight_through_gumbel(logits, temperature=1.0):
    # forward: soft sample
    soft = gumbel_softmax(logits, temperature)

    # hard: one-hot from argmax
    idx = soft.index(max(soft))
    hard = [0.0] * len(logits)
    hard[idx] = 1.0

    # straight-through: use hard in forward, soft gradients in backward
    # in PyTorch: hard - soft.detach() + soft
    return hard, soft
```

In PyTorch the trick is `y_hard = (y_hard - y_soft).detach() + y_soft`. The forward value is `y_hard` (a one-hot), but autograd sees `y_soft` and computes gradients through it.

### Where it is used

- **VQ-VAE**: differentiable codebook selection without commitment loss hacks
- **Discrete latent variables**: any VAE that wants categorical latent codes
- **Architecture search**: DARTS uses relaxed categorical choices over operations
- **Reinforcement learning**: policy gradient with lower variance than REINFORCE

---

## 4. Softmax temperature and entropy

Entropy measures how "spread out" a probability distribution is. For softmax outputs, temperature directly controls entropy — and the relationship has a clean mathematical form.

### Entropy of a discrete distribution

```
H(p) = -sum_i p_i * log(p_i)
```

For softmax with temperature `T` over logits `z`:

```
p_i = exp(z_i / T) / sum_j exp(z_j / T)
```

### Boundary behavior

**T -> 0 (freezing):**
All probability mass concentrates on the largest logit. The distribution becomes a one-hot vector. Entropy goes to 0.

```
H -> 0    (pure argmax, no uncertainty)
```

**T -> inf (melting):**
Every `z_i / T -> 0`, so `exp(z_i / T) -> 1` for all `i`. The distribution becomes uniform: `p_i = 1/n`.

```
H -> log(n)    (maximum entropy, total uncertainty)
```

### Computing entropy vs temperature

```python
import math

def softmax_entropy(logits, temperature):
    scaled = [z / temperature for z in logits]
    m = max(scaled)
    exp_s = [math.exp(s - m) for s in scaled]
    total = sum(exp_s)
    probs = [e / total for e in exp_s]
    entropy = -sum(p * math.log(p + 1e-30) for p in probs)
    return entropy

logits = [3.0, 1.0, 0.5, -1.0]
max_entropy = math.log(len(logits))  # log(4) = 1.386

for T in [0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 50.0, 100.0]:
    H = softmax_entropy(logits, T)
    print(f"T={T:6.2f}  H={H:.4f}  H/Hmax={H/max_entropy:.3f}")
```

Output:

```
T=  0.10  H=0.0000  H/Hmax=0.000
T=  0.25  H=0.0003  H/Hmax=0.000
T=  0.50  H=0.0452  H/Hmax=0.033
T=  1.00  H=0.5765  H/Hmax=0.416
T=  2.00  H=1.0020  H/Hmax=0.723
T=  5.00  H=1.2627  H/Hmax=0.911
T= 10.00  H=1.3394  H/Hmax=0.966
T= 50.00  H=1.3824  H/Hmax=0.997
T=100.00  H=1.3844  H/Hmax=0.999
```

### ASCII entropy vs temperature plot

```
H/Hmax
1.00 |                                        ************************
     |                              **********
0.80 |                       *******
     |                   ****
0.60 |               ****
     |            ***
0.40 |          **
     |        **
0.20 |      **
     |    **
0.00 |****
     +----+----+----+----+----+----+----+----+----+----+----> T
     0    1    2    3    4    5    6    7    8    9    10
```

The curve rises steeply at first (small temperature changes near T=1 have big effects) and then flattens as it approaches the maximum entropy `log(n)`. This is why temperature tuning matters most in the range `[0.5, 2.0]`.

---

## 5. Numerical comparison across dtypes

Not all floating-point formats can handle the same input ranges. In transformers, mixed-precision training means softmax may execute in float16 or bfloat16 — and each has a different breaking point.

### Format properties

```
Format      Exponent bits   Mantissa bits   Max value       Min positive
float32     8               23              3.4028e+38      1.1755e-38
float16     5               10              65504           6.1035e-05
bfloat16    8               7               3.3895e+38      1.1755e-38
```

### Where each breaks for softmax

The critical operation is `exp(x_i - max(x))`. After subtracting the max, the shifted values are in `(-inf, 0]`. The danger is underflow — values so negative that `exp()` rounds to exactly zero.

```python
import torch

def test_softmax_dtype(dtype, label):
    # softmax of [0, -delta] — can we distinguish the two?
    for delta in [1.0, 10.0, 20.0, 50.0, 100.0]:
        x = torch.tensor([0.0, -delta], dtype=dtype)
        s = torch.softmax(x, dim=0)
        ratio = s[0].item() / s[1].item() if s[1].item() > 0 else float('inf')
        print(f"  {label}  delta={delta:5.1f}  softmax=[{s[0]:.6f}, {s[1]:.10f}]  ratio={ratio:.1f}")
```

### Results

```
float32:
  delta=  1.0  softmax=[0.731059, 0.2689414024]  ratio=2.7
  delta= 10.0  softmax=[0.999955, 0.0000453999]  ratio=22026.5
  delta= 50.0  softmax=[1.000000, 0.0000000000]  ratio=inf       # underflow at ~88
  delta=100.0  softmax=[1.000000, 0.0000000000]  ratio=inf

float16:
  delta=  1.0  softmax=[0.730957, 0.2690430000]  ratio=2.7
  delta= 10.0  softmax=[0.999512, 0.0000448227]  ratio=22304.0   # precision loss visible
  delta= 20.0  softmax=[1.000000, 0.0000000000]  ratio=inf       # underflow at ~11
  delta= 50.0  softmax=[1.000000, 0.0000000000]  ratio=inf

bfloat16:
  delta=  1.0  softmax=[0.734375, 0.2656250000]  ratio=2.8       # coarse mantissa
  delta= 10.0  softmax=[1.000000, 0.0000457764]  ratio=21845.3
  delta= 50.0  softmax=[1.000000, 0.0000000000]  ratio=inf       # underflow at ~88
  delta=100.0  softmax=[1.000000, 0.0000000000]  ratio=inf
```

### Key takeaways

| Property | float32 | float16 | bfloat16 |
|---|---|---|---|
| Underflow threshold (exp) | ~88 | ~11 | ~88 |
| Precision near 1.0 | 7 digits | 3 digits | 2 digits |
| Safe for attention logits | Yes | Marginal | Yes if range < 88 |
| Common usage | Master weights | Forward pass (AMP) | Forward + backward |

**float16** is the most fragile. Its small exponent range means logit differences above ~11 cause complete underflow — one token gets probability 1.0, all others get exactly 0.0. This is why attention implementations often upcast to float32 before softmax even in mixed-precision training.

**bfloat16** has the same exponent range as float32 (8 bits), so it survives the same input magnitudes. But its 7-bit mantissa means the probabilities themselves are coarse — fine distinctions between tokens are lost.

---

## 6. Softmax attention weight visualization

In a transformer, softmax converts raw attention scores into weights that sum to 1 across the key sequence. The sharpness of these weights determines whether the model "focuses" on one token or "spreads attention" across many.

### Attention score to weight conversion

```python
def attention_weights(scores, temperature=1.0):
    scaled = [s / temperature for s in scores]
    m = max(scaled)
    exp_s = [math.exp(s - m) for s in scaled]
    total = sum(exp_s)
    return [e / total for e in exp_s]
```

### Example: 8-token sequence

Raw scores (query-key dot products):

```
scores = [1.2, 0.8, 3.5, 0.1, 0.3, 3.7, 0.5, 1.0]
tokens = ["The", "cat", "sat", "on", "the", "mat", "and", "slept"]
```

#### T = 0.5 (sharp attention)

```
The   |##                              0.02
cat   |#                               0.01
sat   |############################    0.28
on    |                                0.00
the   |                                0.00
mat   |######################################  0.68
and   |                                0.00
slept |#                               0.01
```

Almost all weight on "mat" and "sat" — the two highest scores. The model is decisive.

#### T = 1.0 (standard attention)

```
The   |####                            0.04
cat   |###                             0.03
sat   |######################          0.22
on    |#                               0.01
the   |##                              0.02
mat   |##########################      0.27
and   |##                              0.02
slept |####                            0.04
```

Weight is concentrated on the top tokens but others still contribute. This is the default transformer behavior.

#### T = 5.0 (diffuse attention)

```
The   |###############                 0.13
cat   |#############                   0.12
sat   |##################              0.16
on    |###########                     0.10
the   |############                    0.11
mat   |##################              0.16
and   |############                    0.11
slept |##############                  0.12
```

Nearly uniform — every token gets roughly equal weight. The model cannot distinguish important from unimportant context.

### Why this matters

- **Too sharp** (low T): the model ignores context, attends to one token, brittle
- **Too diffuse** (high T): the model treats everything as equally relevant, cannot extract signal
- **The default `T = 1/sqrt(d_k)`**: scaled dot-product attention divides by `sqrt(d_k)` to keep the variance of scores around 1.0, which keeps softmax in the well-behaved regime

---

## 7. Blocked softmax for GPU

GPUs process data in parallel blocks (warps, thread blocks). A naive softmax requires a global max and global sum — two sequential reductions across the entire input. Blocked softmax splits the work so each block operates independently, then a lightweight global reduction combines the results.

### The algorithm

Given input `x` of length `N`, split into `B` blocks of size `K = N / B`:

**Phase 1: local reduction (parallel across blocks)**

Each block `b` computes:
```
m_b = max(x[b*K : (b+1)*K])               — local max
d_b = sum(exp(x_i - m_b)) for i in block   — local sum, relative to local max
```

**Phase 2: global reduction (single thread or small parallel reduction)**

```
M = max(m_0, m_1, ..., m_{B-1})            — global max
D = sum(d_b * exp(m_b - M)) for all b      — global sum, rescaled to global max
```

**Phase 3: final softmax (parallel across blocks)**

```
softmax(x_i) = exp(x_i - M) / D
```

### Implementation

```python
import math

def blocked_softmax(x, block_size=4):
    n = len(x)
    num_blocks = (n + block_size - 1) // block_size

    # phase 1: local max and sum per block
    local_m = []
    local_d = []
    for b in range(num_blocks):
        start = b * block_size
        end = min(start + block_size, n)
        block = x[start:end]

        m_b = max(block)
        d_b = sum(math.exp(xi - m_b) for xi in block)
        local_m.append(m_b)
        local_d.append(d_b)

    # phase 2: global reduction
    M = max(local_m)
    D = sum(d * math.exp(m - M) for m, d in zip(local_m, local_d))

    # phase 3: compute final probabilities
    return [math.exp(xi - M) / D for xi in x]
```

### Step-by-step example

Input: `x = [2.0, 5.0, 1.0, 8.0, 3.0, 4.0, 7.0, 6.0]`, block_size = 4

**Phase 1:**

```
Block 0: [2.0, 5.0, 1.0, 8.0]
  m_0 = 8.0
  d_0 = exp(-6) + exp(-3) + exp(-7) + exp(0) = 0.0025 + 0.0498 + 0.0009 + 1.0 = 1.0532

Block 1: [3.0, 4.0, 7.0, 6.0]
  m_1 = 7.0
  d_1 = exp(-4) + exp(-3) + exp(0) + exp(-1) = 0.0183 + 0.0498 + 1.0 + 0.3679 = 1.4360
```

**Phase 2:**

```
M = max(8.0, 7.0) = 8.0
D = 1.0532 * exp(8-8) + 1.4360 * exp(7-8)
  = 1.0532 * 1.0    + 1.4360 * 0.3679
  = 1.0532 + 0.5284
  = 1.5816
```

**Phase 3:**

```
softmax = [exp(x_i - 8.0) / 1.5816 for x_i in x]
        = [0.0016, 0.0315, 0.0006, 0.6323, 0.0039, 0.0107, 0.2327, 0.0856]
        sums to ≈ 1.0  ✓
```

### Why blocks matter on GPU

On a GPU, each CUDA thread block has fast shared memory (SRAM) but slow global memory (HBM). Blocked softmax:

1. **Phase 1** loads each chunk into SRAM once, computes local `(m, d)` — all fast local operations
2. **Phase 2** reduces `B` pairs instead of `N` values — tiny compared to the full vector
3. **Phase 3** loads each chunk again for the final divide — one more SRAM pass

Total HBM reads: `2N` (same as two-pass). But the reductions are local and fast, and each phase maps cleanly onto GPU thread blocks. This is the memory access pattern used in FlashAttention's tiled implementation and in Triton softmax kernels.

### Combining with online softmax

The blocked approach and the online approach are complementary. Within each block, you can use the online algorithm (single pass for local `m` and `d`). Across blocks, the global reduction uses the same rescaling trick. This is exactly what FlashAttention does: online softmax within each tile of the attention matrix, with inter-tile rescaling for the global result.

---

## Recap

- **Log-softmax** fuses `log` and `softmax` into `x_i - max(x) - log(sum(exp(x_j - max(x))))` — avoids the precision-destroying round-trip through tiny probabilities
- **Online softmax** computes the denominator in a single streaming pass by maintaining a running `(max, sum)` pair and rescaling when the max changes — the algorithm Flash Attention is built on
- **Gumbel-softmax** makes discrete sampling differentiable: add Gumbel noise, divide by temperature, apply softmax — with the straight-through estimator for hard samples with soft gradients
- **Temperature controls entropy**: `T -> 0` gives argmax (zero entropy), `T -> inf` gives uniform (max entropy `log(n)`), and the interesting range is `[0.5, 2.0]`
- **dtype matters**: float16 underflows at logit differences of ~11, bfloat16 survives to ~88 but with coarse precision, float32 is the safe default for softmax
- **Attention sharpness** is softmax temperature in disguise — scaled dot-product attention uses `1/sqrt(d_k)` to keep the distribution well-behaved
- **Blocked softmax** splits the reduction into local blocks and a global combine step, mapping cleanly onto GPU thread blocks and SRAM — the same pattern used in FlashAttention and Triton kernels

---

Get the video walkthrough: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
