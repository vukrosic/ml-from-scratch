# Custom Ops — PyTorch Internals

> 🔴 YouTube Lesson: Coming soon | 🟡 Skool Advanced Video Lesson: [Join the advanced lesson](https://www.skool.com/become-ai-researcher-2669/about)

PyTorch's autograd engine computes gradients automatically, but sometimes you need to tell it exactly how. Custom ops let you define the forward and backward behavior yourself, and with `torch.library` you can register them so `torch.compile` sees them as first-class graph nodes.

In this lesson you will:
1. Build a custom ReLU as a `torch.autograd.Function`
2. See why you'd want to modify gradients (gradient amplification)
3. Register the op with `torch.library` so `torch.compile` can fuse it
4. Benchmark the overhead of a Python custom op vs built-in ReLU

---

## Files at a Glance

| File | What It Demonstrates | Key APIs |
|------|-----------------------|----------|
| `custom_relu.py` | A `torch.autograd.Function` with gradient amplification in `backward()`. Run standalone to see amplified vs standard gradients printed side-by-side. | `ReLUFunction.apply()`, `ctx.save_for_backward()` |
| `library_op.py` | Registers the same ReLU with `torch.library` so `torch.compile` can trace through it as a first-class node. Shows both registered (`torch.ops.custom_ops.relu`) and unregistered (`ReLUFunction.apply`) paths. | `Library("custom_ops", "DEF")`, `lib.define()`, `lib.impl(..., "Autograd")` |
| `benchmark.py` | Measures forward+backward latency (ms per iteration) for `nn.ReLU` vs the custom Python op across three tensor shapes. | `time.perf_counter`, warmup + timed-iterations pattern |

**Suggested reading order:** `custom_relu.py` first (mechanics of forward/backward), then `library_op.py` (how registration changes what `torch.compile` sees), then `benchmark.py` (what the overhead numbers actually mean).

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

### Why ctx.save_for_backward is necessary

The backward pass runs **after** the forward pass finishes. By the time `backward()` is called, the forward's local variables are out of scope. `ctx.save_for_backward` is PyTorch's mechanism for handing a tensor from forward to backward safely.

Consider what would happen without it:

```python
# WRONG — backward won't have access to x
@staticmethod
def forward(ctx, x):
    return x.clamp(min=0)   # x goes out of scope when forward returns

@staticmethod
def backward(ctx, grad_output):
    # x is gone — you cannot compute the mask x > 0
    mask = x > 0   # NameError: name 'x' is not defined
    return grad_output * mask
```

`ctx.save_for_backward` solves this by telling PyTorch: "Hold onto this tensor until backpropagation is done." PyTorch stores it in the autograd graph's node metadata, not in the Function instance itself. This is why `saved_tensors` is a tuple — PyTorch can store multiple tensors and they are cleared automatically after backprop to free memory.

**What you can and cannot save:**
- `ctx.save_for_backward(*tensors)` — only torch.Tensors, stored as a tuple
- `ctx.mark_dirty(x)` — tells autograd that you are mutating `x` in-place (advanced)
- You cannot save plain Python numbers or strings; only Tensors persist across the forward/backward boundary

A common confusion: `ctx.save_for_backward` is for tensors that come from **outside** the autograd graph (like the input). If you need to save a scalar computed inside forward (e.g., `keepdim=True` in a reduction), compute it in both forward and backward, or save it as a plain Python number before returning from forward.

```python
@staticmethod
def forward(ctx, x):
    ctx.save_for_backward(x)
    return x.clamp(min=0)

@staticmethod
def backward(ctx, grad_output):
    # saved_tensors is ALWAYS a tuple, even if you saved only one tensor.
    # The comma (x, =) unpacks it — not a typo.
    x, = ctx.saved_tensors
    return grad_output * (x > 0)
```

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

### From autograd.Function to torch.library: a continuum

Think of `torch.autograd.Function` and `torch.library` not as alternatives but as two points on a spectrum:

```
Level 1: Pure Python autograd.Function
        - .apply() to call
        - Correct gradients, but torch.compile sees a black box
        - Good for: gradient manipulation, custom routing

Level 2: torch.library registration (what we do here)
        - torch.ops.custom_ops.relu() to call
        - torch.compile can see through it, fuse with neighbors
        - Still runs your Python implementation
        - Good for: making custom ops visible to compile

Level 3: Custom C++/CUDA kernel registration
        - torch.library with a C++ or Triton implementation
        - torch.compile emits a single fused kernel for GPU
        - Real speedups come from here
        - Good for: production custom kernels
```

`library_op.py` shows Level 2. You keep your `ReLUFunction` (the Python autograd code) but register it so `torch.compile` knows about it. The compilation pipeline can now see the op as a node it can fuse with adjacent operations like `conv2d` or `linear`.

### How torch.library registration works

The `Library` API has two steps that always go together:

**Step 1 — Define the operator schema** (`lib.define`):
Declares to PyTorch's operator registry: "there exists an op called `custom_ops::relu` that takes a Tensor and returns a Tensor." This is the signature other PyTorch subsystems (dispatch, compilation) use to look up the op.

**Step 2 — Provide the implementation** (`lib.impl`):
Attaches your implementation to that schema. The third argument is the "role" — `"Autograd"` means "use this function for both forward AND backward in the autograd engine." PyTorch internally calls `ReLUFunction.apply` when the op is invoked during autograd.

```python
lib = Library("custom_ops", "DEF")          # namespace, "DEF"=define only
lib.define("relu(Tensor x) -> Tensor")      # schema declaration
lib.impl("relu", ReLUFunction.apply, "Autograd")  # attach implementation
```

After registration, you call the op through `torch.ops.custom_ops.relu(x)` — not through `ReLUFunction.apply` directly. This is the dispatcher path that `torch.compile` routes through.

```python
# Calling through the dispatcher (how registered ops are invoked)
def relu(x):
    return torch.ops.custom_ops.relu(x)   # goes through PyTorch's dispatcher

# NOT calling the Function.apply directly (unregistered path)
def custom_relu_unregistered(x):
    return ReLUFunction.apply(x)           # bypasses dispatcher
```

Both produce the same numerical results. The difference is that `torch.compile(relu)` (registered) can trace the op in the graph, while `torch.compile(custom_relu_unregistered)` treats it as a graph break and falls back to running it in Python.

### What graph breaks cost you

A graph break is not wrong — your results are still correct. The cost is opportunity:

```
Without registration (graph break):
    compiled_relu
    ├── trace: custom_relu is a "fallback" node
    └── emit: Python call to custom_relu for each invocation

With registration (fusable node):
    compiled_relu
    ├── trace: custom_ops::relu is a visible node
    ├── fuse:  merged with adjacent ops into one kernel
    └── emit:  single fused kernel (one CPU->GPU round-trip instead of many)
```

The real fusion benefit only materializes when your op is adjacent to other ops that can be merged with it — for example, `(conv2d -> custom_relu -> add)` might become a single fused kernel. For a standalone ReLU, the benefit is modest. The benefit is large when writing custom CUDA/Triton kernels (Level 3 above).

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

### Reading the numbers

Each row is the **average milliseconds per forward+backward iteration** over 500 timed runs (after 50 warmup runs to let the CPU cache settle).

- **nn.ReLU column** — PyTorch's built-in ReLU, implemented in C++. Very low overhead.
- **CustomReLU column** — your Python `ReLUFunction`, called via `.apply()`. Still in Python.
- **Ratio** — `CustomReLU / nn.ReLU`. A ratio of 1.25x means the custom op takes 25% longer.

The pattern across rows is informative:
- **(128, 256)** — smallest tensor. Ratio is highest (1.25x). Compute is fast, but Python call overhead is a larger fraction of total time.
- **(4096, 768)** — largest tensor. Ratio is lowest (1.13x). Compute dominates, so Python overhead is relatively smaller.

This illustrates a general principle: **Python-level custom ops penalize small-batch / small-tensor workloads the most**, because the overhead is fixed regardless of tensor size.

### Why is the custom op slower?

`nn.ReLU` is a single C++ call that does `tensor.clamp_min_(0)` in-place on the tensor's data buffer. Your `ReLUFunction.forward` is a Python function call that then calls `.clamp(min=0)` (which is also a Python call into C++). On backward, Python again calls into the autograd engine. The extra Python frames on the call stack add up.

The `ctx.save_for_backward(x)` also adds a small cost — PyTorch has to track the tensor in the autograd node.

### When custom ops ARE faster

Python-level custom ops are never faster than built-ins for simple element-wise operations. The speedup scenarios are:

1. **You skip unnecessary computation** — e.g., a custom op that detects and skips a no-op branch. If your forward pass is cheaper, the Python overhead may be worth it.
2. **You write a custom CUDA/Triton kernel** (Level 3 on the continuum) — fused kernels can be 2-10x faster by eliminating memory bandwidth waste (e.g., reading from HBM twice).
3. **torch.compile fusion** — when a registered custom op fuses with neighboring ops, the combined kernel can be faster than running each op separately.

For gradient manipulation use cases (the reason we're here), speed is not the goal — control over the backward pass is.

### Running the benchmark yourself

```bash
python benchmark.py
```

Expected output: `nn.ReLU` is consistently faster. If your results show `CustomReLU` faster, the warmup count is too low — the CPU hasn't settled into its final clock speed.

---

## Common Mistakes with Custom Ops

### Forgetting to save tensors for backward

```python
# WRONG
@staticmethod
def forward(ctx, x):
    return x.clamp(min=0)   # x goes out of scope after forward returns

@staticmethod
def backward(ctx, grad_output):
    mask = x > 0   # NameError — x is not in scope
    return grad_output * mask
```

```python
# CORRECT
@staticmethod
def forward(ctx, x):
    ctx.save_for_backward(x)   # hold x until backward is called
    return x.clamp(min=0)

@staticmethod
def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    return grad_output * (x > 0)
```

### Misunderstanding what grad_output represents

`grad_output` is NOT the gradient of the loss w.r.t. the input. It is the gradient of the loss w.r.t. the **output** of this op. You must multiply it by the local gradient (the derivative of the op itself) before passing it upstream. This is the chain rule.

```python
# WRONG — treating grad_output as if it were already the input gradient
@staticmethod
def backward(ctx, grad_output):
    x, = ctx.saved_tensors
    return grad_output   # missing the mask / local gradient factor
```

### Using .apply() instead of the registered torch.ops path for compile

If you register an op with `torch.library` but then call `MyFunction.apply(x)` instead of `torch.ops.my_library.my_op(x)`, you bypass the registration and `torch.compile` still sees a graph break. Make sure your call site matches the registration style.

### Assuming saved tensors persist across multiple backward calls

`ctx.saved_tensors` are cleared automatically after each `.backward()` call. If you call `loss.backward()` twice (without `retain_graph=True`), the second call will crash if it tries to access `saved_tensors`. In normal training loops this never happens because each forward pass saves fresh tensors, but it trips people up when debugging.

### Returning the wrong number of gradients

If your forward takes multiple inputs, your backward must return a gradient for each one (or `None` for inputs that don't need gradients). Mismatching the number of return values causes cryptic autograd errors.

```python
@staticmethod
def forward(ctx, x, y):
    ctx.save_for_backward(x, y)
    return x + y

@staticmethod
def backward(ctx, grad_output):
    x, y = ctx.saved_tensors
    # Must return TWO things: grad w.r.t. x and grad w.r.t. y
    return grad_output, grad_output   # both get the same gradient here
```

### Confusing torch.library "Autograd" role with a regular function

When you call `lib.impl("relu", ReLUFunction.apply, "Autograd")`, you pass `ReLUFunction.apply` — the bound method object — not `ReLUFunction` (the class) and not `ReLUFunction.forward` (the forward static method). Passing the wrong thing causes a TypeError at implementation time.

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
