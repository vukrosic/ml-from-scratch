# Tutorial Plan: PyTorch Internals

## Existing Tutorials

| # | Title | Status |
|---|-------|--------|
| 001 | torch.compile From Scratch | Done — README + ADVANCED + critique |
| 002 | Autograd From Scratch | Done — README + toy engine |
| 003 | Custom Ops | Done — README + code |
| 004 | GPU Memory Management | Done — README + code |

---

## The Gap: 001 torch.compile Needs

Before treating this as done, 001 needs actual GPU benchmark runs to replace placeholders in ADVANCED.md. The files are ready, the structure is right, but no one has measured on a real GPU yet.

---

## Future Tutorial Ideas

Sorted by priority — core PyTorch internals every ML engineer should know.

### 005 · Distributed Training From Scratch

**What it covers:**
- `torch.distributed` init methods (env://, tcp://, file://)
- `DistributedDataParallel` — how it shards gradients across ranks
- `NCCL` backend — GPU-to-GPU collective operations
- Collective ops: `all_reduce`, `all_gather`, `broadcast`, `reduce`
- Gradient synchronization — how DDP syncs before the optimizer step
- `torchrun` launcher vs `torch.distributed.launch`

**Python files needed:**
- `init.py` — benchmark init methods
- `all_reduce.py` — measure bandwidth
- `ddp.py` — compare DDP vs single-GPU
- `collective_ops.py` — time each NCCL op

**Why:** Multi-GPU training is standard for anything above 1B params. Understanding the synchronization model is the difference between a model that scales and one that doesn't.

---

### 006 · Attention From Scratch

**What it covers:**
- Scaled dot-product attention: `softmax(q @ k.T / sqrt(d)) @ v`
- Masking — causal, padding, arbitrary
- Flash Attention — how tiling over the attention matrix avoids O(n²) memory
- Multi-head attention — split heads, project, concat, project
- KV caching — how autoregressive models store key/value tensors

**Python files needed:**
- `attention.py` — naive attention vs torch.nn.functional.scaled_dot_product_attention
- `flash_attention.py` — simulate the tiling that Flash uses
- `multihead.py` — MHA module from scratch
- `kv_cache.py` — benchmark with and without caching

**Why:** Attention is the core primitive of every modern LLM. Understanding the memory dynamics (the O(n²) problem, KV caching) is prerequisite to understanding why Flash Attention and paged attention exist.

---

### 007 · Optimizer Internals From Scratch

**What it covers:**
- SGD with momentum — the physics analogy (ball rolling down a gradient)
- Adam — how adaptive learning rates work (first moment / second moment)
- Weight decay — L2 regularization vs explicit decay
- Gradient clipping — `torch.nn.utils.clip_grad_norm_`
- Comparison: SGD, Adam, AdamW on a simple problem

**Python files needed:**
- `sgd.py` — SGD from scratch with momentum
- `adam.py` — Adam from scratch
- `compare.py` — all three optimizers on a simple training loop

**Why:** Most engineers treat optimizers as black boxes. The math is not hard — SGD+momentum is one equation. Once you see it, Adam's bias correction makes sense too.

---

### 008 · DataLoader From Scratch

**What it covers:**
- `torch.utils.data.Dataset`, `Sampler`, `collate_fn`
- Why `num_workers=0` is sometimes faster (the GIL, multiprocessing overhead)
- `pin_memory` — why it speeds up host-to-device transfers
- `prefetch_factor` — controlling the pipeline depth
- Memory: what a DataLoader worker actually holds

**Python files needed:**
- `dataset.py` — map-style and iterable-style datasets
- `benchmark_workers.py` — measure num_workers=0 vs 4 vs 8
- `pin_memory.py` — benchmark transfer speeds
- `custom_collate.py` — handle variable-length sequences

**Why:** Data loading is the hidden bottleneck in most training loops. The GIL and multiprocessing overhead are non-obvious. Understanding the pipeline helps you tune batch size and num_workers correctly.

---

### 009 · CUDA Streams and Async Execution From Scratch

**What it covers:**
- Default stream vs custom streams
- Stream priorities and how they interact with CUDA work queues
- `torch.cuda.stream()` context manager
- Synchronization primitives: `wait()`, `synchronize()`, events
- How async execution interacts with profiling (`torch.profiler`)

**Python files needed:**
- `streams.py` — default stream vs custom stream benchmark
- `event_sync.py` — measure event timing
- `profiler.py` — profile async overhead

**Why:** Most code runs on the default stream and never thinks about it. But when you have overlapping data transfers and computation, or multiple models on one GPU, streams are how you control the schedule.

---

### 010 · Mixed Precision Training From Scratch

**What it covers:**
- FP16 vs BF16 — why BF16 has a larger exponent
- `torch.cuda.amp.autocast` — how it patches ops to their low-precision versions
- `GradScaler` — how it prevents underflow in FP16 gradients
- Loss scaling — the problem: small gradients round to zero in FP16
- `torch.set_float32_matmul_precision` — TF32 on Ampere+

**Python files needed:**
- `amp.py` — compare FP32 vs FP16 vs BF16 training curves
- `grad_scaler.py` — measure what GradScaler recovers
- `tf32.py` — measure TF32 speedup on Ampere+

**Why:** Every large model trains in FP16 or BF16. Understanding how GradScaler works is the difference between a training run that diverges and one that converges.

---

## Lower Priority

These are worth considering but less foundational than the above:

| # | Title | Why |
|---|-------|-----|
| 011 | Activation Functions From Scratch | ReLU, GELU, SiLU, Mish — the derivatives and why GELU has no hard cutoff |
| 012 | Normalization Layers From Scratch | BatchNorm tracks running stats; LayerNorm doesn't. Why that matters for RL / LLM training |
| 013 | Weight Initialization From Scratch | Xavier, Kaiming — the math and why it matters for deep networks |
| 014 | LR Schedulers From Scratch | Cosine annealing, warmup, step decay — how the learning rate evolves |

---

## Suggested Order

```
001 torch.compile        ← done
002 autograd            ← done
003 custom ops          ← done
004 memory management   ← done
-------- gap --------   ← run 001 benchmarks on GPU
005 distributed         ← most impactful for large models
006 attention           ← core LLM primitive
007 optimizers         ← foundational
008 dataloader         ← practical performance tuning
009 cuda streams       ← advanced performance
010 mixed precision     ← standard for large model training
011 activations        ← if time allows
012 normalization      ← if time allows
```

---

## Principles for Writing New Tutorials

1. **Hook first.** One sentence that answers "why do I care?" before any code.
2. **Build the toy.** Like 002 (Value engine) and 004 (ManualCheckpoint), show the concept from scratch before showing the library call.
3. **Placeholders for unmeasured numbers.** No invented benchmarks. `~X ms` until someone runs it.
4. **Text leads before every code block.** No bare code.
5. **Recap as checklist.** 5-8 bullets, one sentence each.
6. **Python files are the hands-on.** Every major concept gets a runnable file.
7. **ADVANCED.md optional.** Only if there is genuinely advanced content (benchmarks, kernel output, profiling) that would clutter the main README.
