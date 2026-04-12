# Custom Ops — Extended Notes

This document covers advanced topics NOT in the free lesson. For the full notebook with runnable code, see the Skool $49 tier.

---

## When to write a custom op vs using builtins

The decision tree is simple:

1. **Is your forward pass something PyTorch already has?** Use the builtin. `torch.relu`, `nn.functional.softmax`, `F.layer_norm` — these are battle-tested, fused, and faster than anything you write in Python.

2. **Do you need to modify gradients?** You must use `torch.autograd.Function`. Builtin `nn.Module`s give you no hook into the backward pass.

3. **Do you need kernel-level fusion with other ops?** Register with `torch.library` or write a custom kernel (CUDA/Triton). Pure Python autograd Functions cannot be fused at the kernel level.

4. **Are you targeting quantization or low-precision kernels?** Custom ops with `torch.library` + a lowered kernel (quantized CPU, CUDA, or Triton) are the path.

**Rule of thumb**: reach for a builtin first. Write a custom op when the builtin cannot express what you need. Premature optimization at the op level rarely pays off.

---

## torch._dynamo.allow_in_graph

`torch._dynamo.allow_in_graph` is a decorator that tells Dynamo: "treat this function as a graph node even though you can't trace through it."

```python
import torch._dynamo as dynamo

@dynamo.allow_in_graph
def my_custom_op(x):
    # This might have side effects, use Python control flow, etc.
    return x.clamp(min=0)
```

Without this decorator, `torch.compile` would graph-break at this call and fall back to Python execution. With it, Dynamo accepts the function as an opaque node in the traced graph.

Compare to `torch.library`:
- `torch.library`: You provide a backend implementation (could be a Triton kernel, a quantized kernel). Dynamo traces the abstract op and dispatches to the registered kernel.
- `allow_in_graph`: You say "don't try to trace inside this function, just call it as-is". No kernel registration, just graph compatibility.

**When to use `allow_in_graph`**: Your op is pure Python with no autograd, and you don't want to register a full library entry. It's a lighter-weight escape hatch.

**When to use `torch.library`**: You want Dynamo to see the op symbolically, enable fusion, and provide a specific kernel implementation for one or more backends.

---

## Custom ops for quantization

Quantization replaces float32 weights/activations with int8 or float16 to reduce memory bandwidth and compute. PyTorch's `torch.quantization` APIs use custom ops internally to perform the quantized matmul, ReLU, convolutions, etc.

A quantized custom op typically has two implementations:

```python
# Registered for CPU in fp32
lib.impl("my_relu", my_relu_fp32, "CPU")

# Registered for CPU in int8
lib.impl("my_relu", my_relu_int8, "QuantizedCPU")
```

The dispatcher inside `torch.ops` picks the right kernel based on the input dtype and the current backend. You don't write the dispatcher — you just register kernels with `torch.library` using role tags like `"QuantizedCPU"`.

**The pattern**:
1. Define the op schema in `lib.define(...)` (the abstract signature).
2. Register one or more kernels with `lib.impl(...)` for different roles/backends.
3. Call via `torch.ops.my_namespace.my_op(...)`.

PyTorch's `torch.ops.quantized` namespace is built entirely on this mechanism.

---

## Custom ops for memory-efficient kernels

Memory-efficient kernels matter when you are write-sensitive: gradient checkpointing, online distillation, streaming models, or any model where you cannot hold the full activation trace in memory.

**Persistent buffers**: Some kernels (e.g., causal attention masks) are constant but depend on sequence length. Instead of materializing them during forward, you can register a custom op with a persistent CUDA allocation that is initialized once and reused across calls:

```python
lib.define("my_op(Tensor seq_len) -> Tensor")

@staticmethod
def forward(ctx, seq_len):
    # Allocate on first call, reuse on subsequent calls.
    if not my_op_cache[seq_len]:
        my_op_cache[seq_len] = torch.empty(seq_len, seq_len, device="cuda")
        torch.cuda.synchronize()
    return my_op_cache[seq_len]
```

**In-place gradient accumulation**: Instead of storing activations for the full backward trace, you can accumulate gradients in a pre-allocated buffer using a custom op with a persistent state:

```python
# Forward accumulates into a buffer; backward reads from it.
# This trades compute for memory by avoiding intermediate activation storage.
```

**Triton kernels for fused attention**: The FlashAttention paper showed that fusing `S = Q @ K^T`, `S = S * mask`, `P = softmax(S)` into a single Triton kernel cuts memory by ~50% vs storing the full attention matrix. This is implemented as a custom op in the `flash_attn` package, registered with `torch.library` and backed by a Triton kernel.

**Key insight**: The memory savings come from not materializing full intermediate tensors. A custom op can compute `softmax` and `mask` in a single fused kernel that only ever holds one row of the attention matrix at a time, never the full `N x N` matrix.

---

## Further reading

- [PyTorch Custom ops documentation](https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)
- [torch.library API reference](https://pytorch.org/docs/stable/library.html)
- [Dynamo graph break troubleshooting](https://pytorch.org/docs/stable/torch.compiler_troubleshooting.html)
- FlashAttention paper: `arXiv:2205.14135` — the canonical example of a memory-efficient custom op
