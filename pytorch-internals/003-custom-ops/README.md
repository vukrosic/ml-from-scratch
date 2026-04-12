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

A `torch.autograd.Function` pairs a forward pass with a user-defined backward pass. PyTorch's autograd engine calls the backward pass automatically when you call `.backward()` on the output.

```python
import torch

class ReLUFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # ctx.save_for_backward stores Tensors needed in backward().
        ctx.save_for_backward(x)
        return x.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # grad_output is the gradient flowing back from downstream ops.
        x, = ctx.saved_tensors
        mask = x > 0
        return grad_output * mask
```

`forward()` does the computation. `backward()` receives the gradient of the loss with respect to the output, and returns the gradient with respect to the input.

The mask `x > 0` is exactly the derivative of ReLU: `1` where `x > 0`, `0` elsewhere. So `grad_output * mask` is the chain rule applied to ReLU.

---

## Gradient amplification

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
