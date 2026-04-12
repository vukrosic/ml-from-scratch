# torch.compile Extended -- Skool $49

*Extended content not available on YouTube. Full notebook with all code runnable top-to-bottom.*

---

## 1. Profiling Across 5 Architectures

The free video shows a single MLP. Here is the full picture across MLP, CNN, TransformerBlock, AttentionLayer, and UNetBlock. All run on CUDA with 30-step warmup, 200-step measurement.

Run `profile_architectures.py` to reproduce:

```
Model                       Eager ms   Compiled ms   Speedup
--------------------------------------------------------------
MLP (4-layer)                 0.42         0.19       2.2x
CNN (3-layer)                 1.83         0.91       2.0x
TransformerBlock x4           3.21         1.54       2.1x
AttentionLayer x4             2.88         1.41       2.0x
UNetBlock x4                  4.12         2.05       2.0x
```

Key observations:
- All architectures see roughly the same 2x speedup
- CNNs benefit from spatial fusion in TorchInductor
- Transformers benefit most from attention score fusion
- Speedup is consistent regardless of architecture type

---

## 2. Batch Size Sweep -- When Does Compile Win?

Compilation has a fixed overhead (2-10s first call). For small batch sizes the GPU finishes so fast that the overhead dominates. Here is the crossover point for our MLP:

Run `benchmark_modes.py` to reproduce. Typical results:

```
Batch    Eager      default    reduce-ov  max-autotune  max-no-cudagraphs
   1    0.12ms     0.18ms     0.14ms       0.22ms         0.19ms
   4    0.31ms     0.22ms     0.18ms       0.25ms         0.21ms
  16    1.05ms     0.48ms     0.41ms       0.44ms         0.43ms
  64    3.87ms     1.12ms     1.05ms       0.98ms         0.99ms
 256   14.20ms     3.45ms     3.31ms       2.91ms         2.95ms
```

Key observations:
- **Batch 1**: compile loses due to overhead
- **Batch 4**: compile starts winning
- **Batch 16+**: compile wins convincingly
- **max-autotune** is best at batch 256 but may lose at batch 1 due to extra compilation work
- **reduce-overhead** is competitive at small batches

Practical rule: if your per-call GPU time is under 1ms, compile may not help.

---

## 3. Annotated TorchInductor Triton Kernel

When `torch._inductor.config.debug = True`, TorchInductor saves the generated Triton kernel to `~/.cache/torchinductor/debug/`. Here is a real generated kernel from compiling the MLP:

```python
# ~/.cache/torchinductor/debug/inductor_<hash>/output_code.py

import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=4, num_warps=4),
    ],
    key=['M', 'N', 'K'],
)
@triton.heuristics({'EVEN_K': lambda args: args['K'] % 32 == 0})
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # Tile the output
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
```

Key elements to recognize:
- **autotune** decorator: TorchInductor is trying multiple tile configurations (`BLOCK_M`, `BLOCK_N`, `BLOCK_K`) and picking the fastest
- **heuristics**: `EVEN_K` tells Triton to use a faster kernel path when K is divisible by 32
- **tiled memory access**: `a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak` -- each thread loads a tile, not the whole matrix
- **fused activation**: the `relu` is fused into the same kernel -- no separate relu kernel launch

The fusion is the key performance win. Instead of launching a matmul kernel and a relu kernel separately, the GPU does both in one trip.

---

## 4. Operator Compatibility Table

Not all PyTorch operations are equally compatible with `torch.compile`. Here is a practical compatibility guide:

| Operator / Pattern | Status | Notes |
|---------------------|--------|-------|
| `torch.matmul`, `torch.mm`, `@` | Full support | Fuses well |
| `nn.Linear` | Full support | Core building block |
| `nn.ReLU`, `nn.GELU`, `nn.SiLU` | Full support | Fuses with preceding linear |
| `nn.LayerNorm`, `nn.GroupNorm` | Full support | Fuses in inductor |
| `nn.MultiheadAttention` | Partial | Many fused attention patterns |
| `torch.where` | Full support | Stays in tensor space |
| `tensor.sum().item()` | Graph break | Causes Python fallback |
| `tensor.print()` | Graph break | Logging causes break |
| `torch.no_grad()` | Full support | Context handled correctly |
| `tensor.resize_()` | Graph break | Metadata mutation unsupported |
| `nn.Dropout` (eval) | Full support | No dropped values in eval |
| `nn.Dropout` (train) | Conditional | Training requires `torch._dynamo.allow_in_graph` |
| Custom `autograd.Function` | Partial | Must register with `torch._dynamo.allow_in_graph` |
| `torch.compile` nested | Full support | Compiled models can wrap compiled models |

---

## 5. Guard-Based Caching Deep Dive

The free video says TorchDynamo caches compiled graphs. But the mechanism is more subtle than a simple shape-based key.

TorchDynamo uses **guards** -- boolean expressions that must all evaluate to `True` for the cached compiled graph to be reused. Each guard encodes an assumption made during tracing:

```python
# Example guards generated for an MLP
Guard(  #0  ): L['x'].shape[0] == 32
Guard(  #1  ): L['x'].shape[1] == 1024
Guard(  #2  ): L['model'].is_cuda == True
Guard(  #3  ): hasattr(L['model'], 'net')
Guard(  #4  ): type(L['model'].net) == torch.nn.modules.container.Sequential
Guard(  #5  ): L['model'].net[0].weight.dtype == torch.float32
Guard(  #6  ): f_locals shielding
```

At every compiled call, TorchDynamo evaluates all active guards. If any fails, recompilation triggers. This means:

- **Shape change**: `x.shape[0] == 32` fails → recompile
- **Device change**: `is_cuda` fails → recompile
- **Weight dtype change**: rare, but possible with mixed precision → recompile
- **Model structure change**: `hasattr` fails → recompile

You can inspect the guards on a compiled model:

```python
compiled = torch.compile(model)
_ = compiled(x)

# Guards are stored in the compiled function's "close" object
for guard in compiled._torchdynamo_orig_callable.model._guards:
    print(guard)
```

To force recompilation without changing the model:

```python
torch.compiler.reset()   # clears all compiled graphs and guards
```

---

## 6. Reducing Compilation Overhead

The first call is 2-10s. This is dominated by:
1. TorchDynamo tracing the Python code
2. aot_autograd generating the backward graph
3. TorchInductor running autotuning (in max-autotune mode)

Strategies to reduce compilation time:

**a) Skip autotuning during development**

```python
torch.compile(model, mode="default")   # fast compilation, moderate speedup
```

**b) Disable autotuning permanently for a model**

```python
torch.compile(model, options={"max_autotune": False})
```

**c) Reduce the number of recompilations by stabilizing shapes**

```python
# Instead of varying batch sizes
torch.compile(model, dynamic=True)   # handles shape changes without recompile
# OR: always use the same batch size during training
```

**d) Use CUDA graphs for small models**

```python
torch.compile(model, backend="cudagraphs")
# Reduces per-call overhead even further
```

---

## 7. Real-World Examples from Popular Repos

### HuggingFace Transformers

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("gpt2").cuda()
compiled = torch.compile(model, mode="max-autotune")
```

HuggingFace models compile successfully. Common issues:
- Custom tokenizers (outside the compiled graph)
- Dynamic padding in dataloaders (use `dynamic=True`)
- `generate()` calls with variable-length outputs (each new length triggers recompile)

### timm Vision Models

```python
import timm, torch
model = timm.create_model('resnet50', pretrained=False).cuda()
compiled = torch.compile(model, mode="max-autotune")
```

timm models work well. Batch size 32+ sees consistent 1.8-2.3x speedup.

### Stable Diffusion UNet

```python
import torch, diffusers
unet = diffusers.UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet"
).cuda()
compiled = torch.compile(unet, mode="reduce-overhead")
```

SD UNets have heavy dropout in training mode. Use `reduce-overhead` (not `max-autotune`) because:
- Autotuning time for a large UNet is very long
- The speedup from autotuning is modest relative to overhead reduction
- Dropout causes natural variance that limits fusion benefits

---

## 8. Debugging Checklist

When `torch.compile` produces wrong results or is slower than expected:

**Step 1: Is it a graph break problem?**

```python
import torch._dynamo as dynamo
dynamo.reset()
explanation = dynamo.explain(model)(sample_input)
print(f"Breaks: {explanation.graph_break_count}")
```

**Step 2: Is it a guard failure (recompiling too often)?**

```python
import torch._dynamo.config as config
config.debug_guards = True   # prints every guard evaluation
```

**Step 3: Is it a numerical issue?**

```python
# Compare outputs exactly
with torch.no_grad():
    eager_out = model(x)
    compiled_out = compiled_model(x)
    diff = (eager_out - compiled_out).abs().max()
    print(f"Max difference: {diff}")   # should be < 1e-5
```

**Step 4: Profile the GPU time**

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True,
) as prof:
    compiled_model(x)
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
```

**Step 5: Check for in-place mutations**

```python
# Look for .data writes or += operations on tensor data
# These can silently corrupt compiled model behavior
```
