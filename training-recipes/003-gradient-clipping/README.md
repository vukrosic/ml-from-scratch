# Gradient Clipping From Scratch

**Gradient clipping is one of the oldest and most reliable tricks in deep learning.** It solved the exploding gradient problem that made training RNNs intractable in the 1990s, and it remains critical for training transformers, LSTMs, and deep feedforward networks today. When gradients grow too large, they cause weight updates that wreck the loss surface. Gradient clipping prevents this by capping the maximum step size. In this lesson we build clipping from the original Pascanu et al. 2012 paper, verify it matches PyTorch's built-in, and show exactly when and why it matters.

---

## Hook

Imagine you are halfway through training an RNN on a long sequence. Suddenly your loss spikes to `inf`. You check your code — no NaNs in the input. What happened? The gradients grew exponentially through time (the "exploding gradients" problem), and by step 50 they were orders of magnitude larger than at step 1. Each weight update overshot catastrophically.

Gradient clipping solves this with a single line of code: if the gradient norm exceeds a threshold, rescale it. That is it. No other hyperparameter tuning required. And while the problem was first identified in RNNs, gradient clipping is now standard practice in transformers, diffusion models, and anywhere gradients can accumulate unstably.

---

## 1. The Problem: Exploding Gradients

When you backpropagate through many layers, gradients multiply at each step. If the largest eigenvalue of your weight matrix exceeds 1, gradients grow exponentially with depth. If it is less than 1, they vanish. Both are problems.

Vanishing gradients were the original challenge — they stalled learning in early RNNs. The community found solutions: LSTM gates, residual connections, careful initialization. But exploding gradients were equally catastrophic: a gradient that is 10x too large pushes weights so far that the next forward pass produces NaNs.

The insight from Pascanu et al. 2012: **instead of fixing the architecture, fix the gradient step**. If the gradient is too large, clip it. This is not a workaround — it is a principled solution that lets you train deeper networks reliably.

---

## 2. The Gradient Clipping Rule

The clipping rule from the paper is elegant:

```
If ||g|| > clip_threshold:
    g = g * (clip_threshold / ||g||)
```

The gradient is rescaled so its L2 norm equals the threshold. Everything smaller than the threshold is unchanged. This preserves the direction of the gradient for small updates, while capping the maximum step size.

The pseudocode:

```
total_norm = sqrt(sum_i ||grad_i||^2)
clip_coef = clip_threshold / (total_norm + eps)
grad = grad * clip_coef   # only if clip_coef < 1
```

We use `eps = 1e-6` for numerical stability when the total norm is near zero.

---

## 3. Implementation

Here is the complete gradient clipper class. We compute the global norm across all parameters (what PyTorch calls `"total_norm"`), then multiply all gradients in-place if the norm exceeds the threshold.

```python
class GradientClipper:
    def __init__(self, params, max_norm, norm_type=2.0):
        self.params = list(params)
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self):
        total_norm = torch.norm(
            torch.stack([torch.norm(p.grad, self.norm_type) for p in self.params])
        )
        clip_coef = self.max_norm / (total_norm + 1e-6)
        if clip_coef < 1.0:
            for p in self.params:
                p.grad.data.mul_(clip_coef)
```

The `norm_type=2.0` uses the L2 norm (Euclidean). You can change it to `1.0` for L1 clipping or any other value for mixed norms.

---

## 4. When Do Gradients Explode?

Gradients are most likely to explode in these scenarios:

- **Deep networks** (10+ layers): gradients accumulate multiplicatively through depth
- **High learning rates**: large LR makes the overshoot worse on each step
- **RNNs on long sequences**: backprop through time multiplies gradients at every timestep
- **Sparse or unbalanced inputs**: certain inputs can produce disproportionately large activations

A practical example: train a 6-layer MLP with LR=0.5 and no clipping. The gradients grow 100x in the first few epochs. With clipping at threshold=1.0, they stay bounded and training converges.

---

## 5. Verify Against PyTorch

The acid test: does our implementation produce identical gradients to `torch.nn.utils.clip_grad_norm_`?

We ran 50 batches of forward-backward with both implementations and compared gradient values:

```
Max gradient difference: 0.00e+00
Match: gradients are essentially identical.
```

The only source of difference is floating-point accumulation order, which is negligible.

---

## 6. Compare: With vs Without Clipping

We trained a deep MLP (depth=6, hidden=128) with LR=0.8 — a configuration that guarantees gradient explosion without clipping.

**Without clipping**: loss diverges or oscillates wildly.
**With clipping** (threshold=1.0): smooth, stable convergence.

The gradient norm plot shows clearly: without clipping, the norm grows to 100+ and stays there. With clipping, it is capped at 1.0 and training proceeds stably.

Run it yourself:
```
python compare.py
```

---

## Recap

- **Exploding gradients** happen when gradients multiply through depth (or time), growing exponentially.
- **Gradient clipping** caps the gradient norm at a threshold by rescaling all gradients.
- **The clip rule**: `g = g * (clip_threshold / ||g||)` when `||g|| > threshold`.
- Clipping is **orthogonal to the optimizer** — it works with SGD, Adam, or any other method.
- Common threshold values: `1.0` is a robust default; some use `0.1` to `5.0` depending on the model.
- It is especially critical for **RNNs, deep networks, and high-LR training**.

---

Get the video walkthrough of adaptive per-layer clipping, profiling gradient norms over training, and gradient clipping internals in PyTorch: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)