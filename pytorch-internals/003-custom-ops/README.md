# Custom Ops — PyTorch Internals

PyTorch's autograd engine computes gradients automatically, but sometimes you need to tell it exactly how. Custom ops let you define the forward and backward behavior yourself, and with `torch.library` you can register them so `torch.compile` sees them as first-class graph nodes.

In this lesson you will:
1. Build a custom ReLU as a `torch.autograd.Function`
2. See why you'd want to modify gradients (gradient amplification)
3. Register the op with `torch.library` so `torch.compile` can fuse it
4. Benchmark the overhead of a Python custom op vs built-in ReLU

---

## Why customize the backward pass?

Standard autograd gives you `loss.backward()` — gradients computed by the chain rule, nothing more. But some algorithms need to modify gradients during backpropagation:

- **Gradient clipping** — cap gradients to prevent exploding values
- **Gradient amplification** — boost signal in specific pathways (used in some training tricks)
- **Custom routing** — route gradients to different places than the forward pass went

To do any of these, you need to define your own `backward()` function. That's what `torch.autograd.Function` is for.

---

## Custom ReLU with torch.autograd.Function

> **Mental Model:** A `torch.autograd.Function` is a custom-defined backward pass. Normally PyTorch computes gradients automatically via the chain rule. This lets you say "when going backward through my operation, do THIS calculation instead of the standard one." The forward does computation, the backward defines the gradient formula.

**Before this section you should understand:** What a gradient is (signal flowing backward through a computation graph).

**After this section you will be able to:** Write a custom autograd Function with explicit forward and backward steps, and explain what `ctx.save_for_backward` does.

A `torch.autograd.Function` pairs a forward pass with a user-defined backward pass. PyTorch's autograd engine calls the backward pass automatically when you call `.backward()` on the output.

```python
import torch

class ReLUFunction(torch.autograd.Function):
    # Line 1: @staticmethod — this method belongs to the class, not instances.
    #           PyTorch calls ReLUFunction.forward(ctx, x), not instance.forward(x).
    #           There is no "self" because the class itself is the op definition.
    @staticmethod
    def forward(ctx, x):
        # ctx is a Context object — like a scratch pad for passing data
        #           from forward() to backward(). It is NOT the same as 'self'.
        #
        # ctx.save_for_backward(x) — stores x in a safe place that
        #           backward() can retrieve later. PyTorch holds onto these
        #           tensors until backpropagation finishes, then clears them.
        #           Why store? Because backward() runs LATER, after forward()
        #           has already finished. Without save_for_backward, x would
        #           be gone from scope.
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output — the gradient of (loss w.r.t. THIS op's output).
        #           Shape matches the forward output. Every element tells us
        #           "how much did the loss change if we nudge THIS element?"
        #
        # ctx.saved_tensors — tuple of tensors saved by save_for_backward().
        #           Unpacking with comma: x, = because it's always a tuple.
        x, = ctx.saved_tensors
        # x is the SAME tensor from forward — now we have it in backward.

        # mask = x > 0
        #   For each element: True (1) if positive, False (0) if negative.
        #   This IS the ReLU derivative: d/dx max(0,x) = 1 if x>0 else 0.
        mask = x > 0

        # grad_output * mask  — chain rule: grad_through_this_op = grad_output * local_grad
        #   Where x > 0:  grad_output * 1  = grad_output passes through unchanged
        #   Where x <= 0: grad_output * 0  = 0 (ReLU kills the gradient)
        return grad_output * mask
```

`forward()` does the computation. `backward()` receives the gradient of the loss with respect to the output, and returns the gradient with respect to the input.

The mask `x > 0` is exactly the derivative of ReLU: `1` where `x > 0`, `0` elsewhere. So `grad_output * mask` is the chain rule applied to ReLU.

---

## Gradient amplification

> **Mental Model:** Gradient amplification means "multiply the gradient by a factor before it flows further backward." You might do this to give extra training signal to certain pathways — in this case, we double the gradient for inputs that were already positive (active in the forward pass).

**Before this section you should understand:** What `ctx.save_for_backward` and `grad_output` mean in the backward pass.

**After this section you will be able to:** Modify gradients in a custom backward pass — doubling, clipping, or redirecting them.

Here's the "useful example" I promised. In `backward()`, we amplify gradients where `x > 0` by a factor of 2:

```python
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        mask = x > 0
        # Double the gradient for positive inputs.
        amplified = grad_output * mask * 2.0
        return amplified
```

Walk through with actual values — input `x = [1., -2., 3., -4.]`:

```
Step 1 — Forward pass (x.clamp(min=0)):
    x          = [ 1.,  -2.,   3.,  -4.]
    ReLU(x)    = [ 1.,   0.,   3.,   0.]   ← zeros out negatives

Step 2 — Suppose downstream gradient (grad_output) is the SAME as ReLU output
         (e.g., loss = sum(ReLU(x)), so d(loss)/d(ReLU) = 1 for each element):
    grad_output = [ 1.,   0.,   3.,   0.]

Step 3 — Compute mask (x > 0):
    mask = [ 1.,   0.,   1.,   0.]   ← True→1, False→0

Step 4 — Apply chain rule first (grad_output * mask):
    grad_after_chain = [ 1.,   0.,   3.,   0.]
    (element 0: 1*1=1, element 1: 0*0=0, element 2: 3*1=3, element 3: 0*0=0)

Step 5 — Amplify by factor of 2 for positive regions:
    amplified = grad_after_chain * 2.0
              = [ 2.,   0.,   6.,   0.]
```

Compare against standard (non-amplified) ReLU backward:

```
Standard ReLU backward:   [ 1.,   0.,   3.,   0.]
Amplified backward:        [ 2.,   0.,   6.,   0.]   ← 2x for active (positive) inputs
```

Run `custom_relu.py` and you will see:

```
Input:               tensor([ 1., -2.,  3., -4.])
ReLU output:         tensor([ 1.,  0.,  3.,  0.])
Gradient (amplified): tensor([ 2.,  0.,  6.,  0.])

Standard ReLU gradient: tensor([ 1.,  0.,  3.,  0.])
```

The custom op produces gradients that are 2x larger for the positive elements. This is impossible with a standard `nn.Module` because you cannot intercept the backward pass inside `forward()`.

To use it, call the Function's `.apply()` method:

```python
def custom_relu(x):
    return ReLUFunction.apply(x)
```

---

## Registering with torch.library for torch.compile

> **Mental Model:** `torch.compile` traces your code to build a graph of operations. When it sees a custom `autograd.Function`, it treats it as a "black box" — it can't see inside to fuse it with neighboring ops. Registering with `torch.library` puts your op in PyTorch's official operator registry, so `torch.compile` recognizes it and can optimize across it.

**Before this section you should understand:** How `torch.autograd.Function` defines a custom backward pass.

**After this section you will be able to:** Register a custom op with `torch.library` so that `torch.compile` treats it as a first-class graph node instead of a graph break.

`torch.compile` traces the autograd graph and generates optimized kernels. A custom `Function` appears in the trace as an opaque node — a "graph break". PyTorch falls back to running your op in Python, losing the performance benefits of compilation.

Registering with `torch.library` makes the op transparent to `torch.compile`:

```python
from torch.library import Library
import torch

lib = Library("custom_ops", "DEF")

def relu_impl(x):
    return x.clamp(min=0)

# "Autograd" role means PyTorch uses your autograd.Function for gradients.
lib.impl("relu", relu_impl, "Autograd")

def relu(x):
    return torch.ops.custom_ops.relu(x)
```

Now `torch.compile(relu)` can trace through the op, fuse it with adjacent operations, and emit a single optimized kernel.

Without registration:

```python
compiled_custom = torch.compile(custom_relu_unregistered)
```

PyTorch treats it as a fallback op. The results are correct, but compile cannot fuse it with neighbors.

---

## Benchmark: custom op vs nn.ReLU

> **Mental Model:** The benchmark answers: "Is writing a custom op faster?" The answer is usually no for Python-level ops — the speed benefit comes from writing GPU kernels (CUDA/Triton), not from the Python code itself.

**Before this section you should understand:** How `torch.autograd.Function` works and why registration exists.

**After this section you will be able to:** Measure op overhead and explain why custom Python ops are typically slower than built-ins.

The benchmark in `benchmark.py` runs forward+backward passes for both ops:

```
              Shape    nn.ReLU  CustomReLU    Ratio
-------------------------------------------------
      (128, 256)     0.0312     0.0389    1.247x
     (1024, 512)     0.0891     0.1023    1.148x
     (4096, 768)     0.2451     0.2782    1.135x
```

The custom op is slower due to Python call overhead. The real value of custom ops is not speed — it's **controlling the gradient flow**. Speedups come from kernel fusion when you write custom kernels in CUDA or Triton, not from Python-level custom autograd.

---

## Recap

- `torch.autograd.Function` defines a forward/backward pair. Call `.apply()` to use it.
- `ctx.save_for_backward(x)` stores tensors needed in `backward()`.
- `backward(grad_output)` receives the gradient from downstream and returns the gradient to upstream.
- Gradient amplification is one example where you genuinely need a custom backward.
- `torch.library` registers ops as first-class compile nodes, enabling graph fusion.
- Without registration, `torch.compile` treats the custom op as a graph break.

---

Get the video walkthrough of graph break diagnostics, custom CUDA kernel registration, and profiling across 5 architectures: [OpenSuperintelligenceLab on Skool](https://www.skool.com/opensuperintelligencelab)
