# Distributed Training From Scratch

Multi-GPU training speeds up your job proportionally to the number of GPUs — until communication overhead eats your gains. Understanding how PyTorch actually moves data between processes is what lets you push beyond toy benchmarks and scale to real workloads. This is how `torch.distributed` actually works.

---

## What Problem Distributed Training Solves

### The Problem: One GPU Is Not Enough

A large model and a large dataset mean training takes weeks on a single GPU. You need multiple GPUs working together. But every GPU needs to see the full model and the full dataset — and they need to agree on gradients after each step.

Naive approaches fail fast. If GPU 0 holds the model and computes gradients, then sends them to GPU 1, GPU 1 sits idle during the entire compute phase. Parallelism without synchronization is just sequential execution with extra steps.

### The Fix: Run the Same Model on Every GPU, Split the Data

Each GPU holds a full copy of the model. Each GPU processes a different slice of the batch. After the backward pass, all GPUs communicate their gradients — they all compute the same averaged gradient, then all take the same optimizer step. Every GPU stays busy computing while the others are communicating.

```
GPU 0: forward(batch_slice_0) → backward → all_reduce gradients
GPU 1: forward(batch_slice_1) → backward → all_reduce gradients
...
```

This is DistributedDataParallel. Every GPU computes, then they sync.

---

## How torch.distributed Init Works

Before any communication can happen, every process needs to find its peers. PyTorch supports three init methods:

### env:// — The Standard Launcher Method

`torchrun` (the recommended launcher) sets environment variables that tell each process how to connect. With `env://`, PyTorch reads those variables automatically:

```python
dist.init_process_group(backend="nccl", init_method="env://")
```

`torchrun` sets `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and `RANK` for every process. You do not configure anything manually.

### tcp:// — Manual Address Specification

Specify an IP address and port that all processes can reach. Useful when `torchrun` is not available:

```python
dist.init_process_group(backend="nccl", init_method="tcp://192.168.1.100:29500")
```

Every process needs the same address. One machine acts as the coordinator.

### file:// — Shared Filesystem Required

All processes communicate through a shared filesystem. Simple but slow — do not use this for multi-node training:

```python
dist.init_process_group(backend="nccl", init_method="file:///tmp/shared_store")
```

---

## The Backend: NCCL vs Gloo

**NCCL** (NVIDIA Collective Communications Library) is the fast path for NVIDIA GPUs. It uses NVLink and GPUDirect RDMA for peer-to-peer communication without touching CPU memory. Always use NCCL when you have NVIDIA GPUs.

**Gloo** is the open-source fallback. It works on CPUs and any GPU via CUDA but is significantly slower than NCCL. Only use Gloo for debugging or when NCCL is not available.

Rule: use `backend="nccl"` for GPU training, `backend="gloo"` for CPU.

---

## Collective Operations

Distributed training is built from five primitive operations. Each has specific semantics and cost.

### all_reduce — Sum Across All Ranks

Every rank has a tensor. After `all_reduce`, every rank has the sum of all tensors:

```
Before:        rank0=[1,2]   rank1=[3,4]
After all_reduce:  rank0=[4,6]   rank1=[4,6]
```

This is how gradient synchronization works. Each rank computes its gradient contribution, then all_reduce sums them so every rank has the identical averaged gradient.

### all_gather — Collect Tensors From All Ranks

Every rank contributes a tensor. After `all_gather`, every rank has the concatenation of all tensors:

```
Before:        rank0=[1,2]   rank1=[3,4]
After all_gather:  rank0=[1,2,3,4]   rank1=[1,2,3,4]
```

This appears in pipeline parallelism when a later stage needs the full input.

### broadcast — One-to-All Distribution

One rank has a tensor. After `broadcast`, all ranks have a copy:

```
Before:        rank0=[1,2]   rank1=[?,?]  (rank1 has garbage)
After broadcast from rank0:  rank0=[1,2]   rank1=[1,2]
```

Used to distribute model parameters or configuration from rank 0 to everyone else.

### reduce — All-to-One Sum

Every rank contributes a tensor. After `reduce`, only the root rank has the sum:

```
Before:        rank0=[1,2]   rank1=[3,4]
After reduce to rank0:  rank0=[4,6]   rank1=[?,?] (unchanged)
```

Useful when only one rank needs the aggregated result — saves communication for non-essential ranks.

### barrier — Synchronization Point

Every rank waits until all ranks have arrived. Not a data movement operation — it only synchronizes. Useful for timing sections of code across ranks.

---

## DistributedDataParallel: How It Shards Gradients

DDP wraps a model and handles gradient synchronization automatically. Here is what actually happens during a training step:

### What DDP Actually Does

When you wrap a model with `DDP`, every rank has a full copy of the model. During the backward pass, each rank computes gradients for its portion of the batch. These gradients are local — they only reflect that rank's data, not the full batch.

Before the optimizer step, DDP runs `all_reduce` on every parameter's gradient. Every rank contributes its local gradient; every rank receives the averaged gradient. Then all ranks take the same optimizer step on identical gradients.

```
Step 1: rank0 forward(batch0), backward → grad0 (local gradient)
        rank1 forward(batch1), backward → grad1 (local gradient)

Step 2: DDP all_reduce(grad0) + all_reduce(grad1) → averaged_grad (same on every rank)

Step 3: optimizer.step() with identical averaged_grad on every rank
```

The key insight: no data is transferred during forward or backward. Only gradients are synchronized — and only once per training step, after all compute is done.

### Gradient Synchronization in Detail

Gradient averaging is the core of DDP's correctness guarantee. The algorithm:

1. Each rank computes `loss.backward()` which fills `.grad` with local gradients
2. DDP hooks detect when `.grad` is ready and trigger `all_reduce` asynchronously
3. `all_reduce` sums all local gradients and divides by world size — yielding the correct average
4. The optimizer reads the averaged gradient and updates parameters

```
rank0 grad:  [2.0, 4.0]
rank1 grad:  [4.0, 6.0]

all_reduce sum:  [6.0, 10.0]
divide by 2:     [3.0, 5.0]  ← same on every rank
```

Without DDP, each rank would apply its own local gradient and the models would diverge.

---

## torchrun: The Standard Launcher

`torchrun` replaces manual process spawning. It launches N processes on your machine (or cluster), sets environment variables correctly, and handles graceful shutdown.

Basic usage:

```bash
torchrun --nproc_per_node=2 train.py
```

This launches 2 processes on the local node. `torchrun` sets `RANK=0` and `RANK=1`, `WORLD_SIZE=2`, and a shared `MASTER_ADDR` and `MASTER_PORT`.

For multi-node training:

```bash
# On node 0:
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr=192.168.1.100 train.py

# On node 1:
torchrun --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr=192.168.1.100 train.py
```

The `--master_addr` must be reachable from all nodes. Usually the first node's IP address.

---

## torchrun Sets Everything Up Automatically

When you use `torchrun`, you do not configure rank or world size manually. The launcher injects the correct values via environment variables. Your script reads them:

```python
import torch.distributed as dist

# torchrun sets these environment variables automatically
dist.init_process_group(backend="nccl", init_method="env://")

rank = dist.get_rank()      # 0, 1, 2, ...  N-1
world_size = dist.get_world_size()  # total number of processes
```

Your training code stays the same regardless of how many GPUs you use.

---

## Minimal End-to-End Example

### Piece 1: The model

```python
import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.relu(x)
        return self.fc2(x)
```

### Piece 2: Init distributed

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# torchrun sets rank, world_size, and master address automatically
dist.init_process_group(backend="nccl", init_method="env://")
rank = dist.get_rank()
world_size = dist.get_world_size()
```

### Piece 3: Wrap with DDP

```python
model = SimpleModel().cuda(rank)
model = DDP(model, device_ids=[rank])
```

### Piece 4: Training loop

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for inputs, targets in dataloader:
    # DDP handles gradient synchronization
    inputs = inputs.cuda(rank)
    targets = targets.cuda(rank)

    optimizer.zero_grad()
    output = model(inputs)
    loss = nn.functional.cross_entropy(output, targets)
    loss.backward()
    optimizer.step()
```

### Piece 5: Launch

```bash
torchrun --nproc_per_node=2 train.py
```

Run with 4 GPUs:

```bash
torchrun --nproc_per_node=4 train.py
```

The same script scales to any number of GPUs.

---

## Why DDP Beats DataParallel

PyTorch also has `DataParallel` (DP). It is simpler but slower:

| | DDP | DataParallel |
|--|-----|--------------|
| Communication | Once per step (all_reduce) | Every batch split (gather + broadcast) |
| Memory | One copy per GPU | Replicated on master GPU |
| Multi-GPU | All GPUs compute | One GPU coordinates |
| Multi-node | Works | Does not work |

DDP communicates less, uses memory more efficiently, and scales to multiple nodes. Always prefer DDP.

---

## Recap

- **Init methods:** `env://` with `torchrun` is the standard setup. `tcp://` for manual multi-node. `file://` for single-node shared filesystem.
- **Backend:** Use NCCL for NVIDIA GPUs, Gloo as the CPU fallback.
- **Collective ops:** `all_reduce` sums across ranks (gradient sync), `all_gather` collects tensors, `broadcast` distributes from root, `reduce` aggregates to root, `barrier` synchronizes.
- **DDP:** Every rank has a full model copy. After backward, gradients are all_reduced so all ranks have identical averaged gradients before the optimizer step.
- **torchrun:** Launches processes, sets environment variables, handles cleanup. Use `--nproc_per_node` to control GPU count.
- **Scaling:** DDP scales near-linearly for compute-bound workloads. Small models or slow interconnects bottleneck on communication.

---

## Going Further

For real benchmark numbers across GPU counts, NCCL vs Gloo bandwidth tables, common error messages and fixes, multi-node setup walkthrough, and profiling distributed training with PyTorch Profiler — see [ADVANCED.md](./ADVANCED.md).

Sources:
- https://pytorch.org/tutorials/intermediate/dist_tuto.html
- https://pytorch.org/docs/stable/distributed.html
- https://pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html
