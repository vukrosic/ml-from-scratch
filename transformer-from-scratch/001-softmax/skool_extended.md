# Softmax — Extended Notebook ($49)

This document supplements the free lesson with deeper derivations, numerical analysis, and advanced topics.

---

## 1. Log-softmax: derivation and why it matters

The log-softmax is `log(softmax_i(x)) = x_i - log(sum_j exp(x_j))`.

**Derivation:**

```
log(softmax_i(x))
= log(exp(x_i) / sum_j exp(x_j))
= log(exp(x_i)) - log(sum_j exp(x_j))
= x_i - log(sum_j exp(x_j))
```

The term `log(sum_j exp(x_j))` is the *log-sum-exp* (LSE). Computing it naively overflows the same way softmax does. The stable version:

```python
def log_sum_exp(x):
    max_x = max(x)
    return max_x + math.log(sum(math.exp(xi - max_x) for xi in x))
```

PyTorch exposes this as `torch.logsumexp`. The log-softmax is the foundation of cross-entropy loss — numerically stable CE is implemented as `F.nll_loss(log_softmax(x))`, not `cross_entropy(softmax(x), target)`.

---

## 2. Softmax across 5 dtypes — numerical behaviour

We compare float16, bfloat16, float32, float64, and a toy "int8 approximation" across three scenarios: small inputs, large inputs (stability test), and ill-conditioned inputs (near-equal values that need precision).

### 2.1 Small inputs — typical logits

```python
import torch, math

def stable_softmax(x):
    max_x = max(x)
    exp_x = [math.exp(xi - max_x) for xi in x]
    total = sum(exp_x)
    return [e / total for e in exp_x]

logits = torch.tensor([2.0, 1.0, 0.5], dtype=torch.float32)
for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
    t_out = torch.softmax(logits.to(dtype), dim=0)
    py_out = stable_softmax(logits.tolist())
    max_diff = (t_out.cpu().double() - torch.tensor(py_out)).abs().max()
    print(f"{str(dtype):<12}  max_diff={max_diff:.2e}")
```

Expected behaviour: float64 has the smallest error (reference), float32 is very close, bfloat16 is close but loses some precision in the tail, float16 accumulates more error but is still usable.

### 2.2 Large inputs — stability

```python
huge = [900.0, 901.0, 902.0]
for dtype in [torch.float16, torch.bfloat16, torch.float32, torch.float64]:
    t_out = torch.softmax(torch.tensor(huge, dtype=dtype), dim=0)
    print(f"{str(dtype):<12}  {t_out.tolist()}")
```

float16 will likely saturate to `inf` or all-ones. bfloat16 and float32 handle these values correctly. This is why attention scores in production models are often stored in bfloat16 rather than float16.

### 2.3 Int8 approximation — toy study

```python
def int8_softmax_approx(x, scale=127.0):
    """Clamp and rescale logits to [-127, 127] then apply softmax."""
    x_clamped = [max(-127, min(127, xi * scale)) for xi in x]
    return stable_softmax(x_clamped)
```

The int8 approximation introduces non-trivial error when the true logits span a large range. This is relevant for quantised inference: if you quantise activations to int8 without proper scaling, your softmax distribution can be significantly distorted.

---

## 3. Gumbel-softmax for differentiable sampling

Gumbel-softmax lets you sample from a categorical distribution and backpropagate through it. Instead of a one-hot sample (non-differentiable), you produce a soft one-hot vector that approximates it.

```python
import torch, math

def gumbel_softmax(logits, temperature=1.0, hard=False):
    """
    Gumbel-softmax sample.
    logits: unnormalised log probabilities (like softmax input).
    temperature: controls how close the sample is to one-hot.
    hard: if True, returns a true one-hot (straight-through estimator).
    """
    gumbels = [-math.log(-math.log(torch.rand(1).item() + 1e-10) + 1e-10)
               for _ in logits]
    gumbel_logits = [l + g for l, g in zip(logits, gumbels)]
    return gumbel_softmax
```

The Gumbel trick: adding i.i.d. Gumbel noise to logits and taking `argmax` is equivalent to sampling from the softmax distribution. The softmax version with temperature interpolates between the Gumbel-softmax (straight-through, differentiable) and the true categorical sample.

```python
def gumbel_softmax_sample(logits, temperature):
    # Step 1: add Gumbel noise
    noise = [torch.rand(1).item() for _ in logits]
    gumbel = [-math.log(-math.log(n + 1e-10) + 1e-10) for n in noise]
    perturbed = [l + g for l, g in zip(logits, gumbel)]
    # Step 2: softmax with temperature (soft argmax)
    max_p = max(perturbed)
    exp_p = [math.exp((p - max_p) / temperature) for p in perturbed]
    total = sum(exp_p)
    return [e / total for e in exp_p]
```

Use case: VAEs, GANs, reinforcement learning policy gradients, and anywhere you need a differentiable discrete choice.

---

## 4. Temperature scaling — entropy and confidence visualisation

```python
import math

def softmax_entropy(probs):
    """Entropy H = -sum_i p_i log p_i. Higher = more uncertain."""
    return -sum(p * math.log(p + 1e-10) for p in probs if p > 0)

def temperature_sweep(logits, temps):
    results = {}
    for T in temps:
        probs = temperature_softmax(logits, T)
        results[T] = {
            'probs': probs,
            'entropy': softmax_entropy(probs),
            'max_prob': max(probs)
        }
    return results
```

```python
logits = [2.0, 1.0, 0.5]
temps = [0.1, 0.25, 0.5, 1.0, 2.0, 4.0]
for T, data in temperature_sweep(logits, temps).items():
    print(f"T={T:4.2f}  entropy={data['entropy']:.4f}  max_prob={data['max_prob']:.4f}  {data['probs']}")
```

As T → 0, the distribution collapses to a Dirac (peak = 1.0, entropy → 0). As T → ∞, it becomes uniform (all ≈ 1/n, entropy → log n). Entropy is a useful diagnostic: if your attention maps have very low entropy, the model is extremely focused on a single token — sometimes good, sometimes a sign of mode collapse.

---

## 5. Attention weight visualisation with real sentence embeddings

We use a simple sentence embedding (average of word embeddings from a small vocabulary) to demonstrate how softmax produces attention-like weight distributions over words in a sentence.

```python
import torch

# Simple word embedding table (10 words, 8-dim embeddings)
vocab = {
    'the': 0, 'cat': 1, 'sat': 2, 'on': 3, 'mat': 4,
    'dog': 5, 'ran': 6, 'in': 7, 'yard': 8, 'sun': 9,
}
embeddings = torch.randn(10, 8)
embeddings = embeddings / embeddings.norm(dim=1, keepdim=True)  # unit length

def embed_sentence(sentence):
    words = sentence.lower().split()
    ids = [vocab[w] for w in words if w in vocab]
    return embeddings[ids].mean(dim=0)

queries = embed_sentence("the cat slept")
keys   = embed_sentence("the cat sat on the mat")

# Attention score: dot product between query and key
score = torch.dot(queries, keys).item()
attn_weights = stable_softmax([score], temperature=1.0)
print(f"Query: 'the cat slept'  Key: 'the cat sat on the mat'")
print(f"Attention weight: {attn_weights[0]:.4f}")
```

For a richer visualisation, compute pairwise dot products between every word in a sentence to build an attention-like heatmap — this is what self-attention does internally, but the softmax over dot products is the same operation.

```python
sentence = "the cat sat on the mat".split()
n = len(sentence)
# Build embedding matrix for sentence
ids = [vocab[w] for w in sentence if w in vocab]
sent_emb = embeddings[ids]  # (n, 8)

# Pairwise cosine similarity matrix (softmax over similarity scores)
sim = torch.mm(sent_emb, sent_emb.T)  # (n, n)
# Row-wise softmax (each word attends to all others)
attn = torch.softmax(sim, dim=1)
print("Attention heatmap (rows sum to 1):")
print(attn)
```

The diagonal entries are the self-attention weights — typically the largest values since a token is most similar to itself. Off-diagonal entries reveal semantic relationships. In a full transformer, Q, K, V projections would be learned to make these patterns meaningful. But the softmax is always doing the same job: turning similarity scores into a probability distribution over positions.
