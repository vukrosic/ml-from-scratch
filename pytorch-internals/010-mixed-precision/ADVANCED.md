# Advanced: Convergence and Speed Benchmarks

This file contains numerical results and convergence analysis for mixed precision training. All measurements use the same architecture and hyperparameters unless noted.

---

## FP16 vs BF16 Convergence

**Hardware: GTX 1070 Ti** (Turing, no Tensor Cores for FP16 — uses FP32 paths with FP16 emulation)

**Note: All numbers are placeholders. Run `python convergence.py` on your GPU to get real measurements.**

Run `convergence.py` to generate loss curves. The script trains a 4-layer MLP on a synthetic dataset in three modes: FP32 (baseline), FP16 with GradScaler, and BF16 with GradScaler.

Expected behavior:

- FP32 converges cleanly — the reference curve.
- BF16 closely tracks FP32 — the larger exponent prevents gradient underflow.
- FP16 may diverge from FP32 if loss scaling is too aggressive or too conservative. The gap depends on model depth and batch size.

The key metric is the gap between FP16/BF16 and FP32 at the end of training. A large gap indicates numerical instability, not just speed differences.

---

## GradScaler Recovery

**Hardware: GTX 1070 Ti**

**Note: All numbers are placeholders. Run `python grad_scaler.py` on your GPU to get real measurements.**

Run `grad_scaler.py` to measure what GradScaler actually recovers. The script:

1. Computes gradients in FP32 (ground truth).
2. Computes gradients in FP16 (with underflow).
3. Computes gradients in FP16 with GradScaler.
4. Reports recovery rate: how much of the FP32 gradient magnitude GradScaler restores.

Expected result: GradScaler recovers roughly 80-95% of gradient magnitude depending on scale factor choice. Too low a scale factor and gradients underflow. Too high and overflow occurs during the forward pass.

The optimal scale factor is dynamic — GradScaler adjusts it automatically based on overflow detection.

---

## TF32 Speed vs FP32 on Ampere+

**Hardware: GTX 1070 Ti**

**Note: All numbers are placeholders. Run `python tf32.py` on your GPU to get real measurements.**

**GTX 1070 Ti vs A100 on TF32:** The GTX 1070 Ti is a Turing-generation GPU. Turing supports FP16 Tensor Cores but does **not** have the dedicated TF32 format that Ampere+ GPUs expose via `torch.set_float32_matmul_precision("high")`. On Ampere+ (A100, H100, RTX 3090/4090), TF32 is a native format that Tensor Cores accelerate — yielding 2-4x matmul speedup over FP32. On GTX 1070 Ti, the precision setting has no effect because TF32 is not supported; matmuls run in true FP32 or fall back to FP16 paths. Benchmark numbers on this hardware will differ significantly from published A100 results.

Run `tf32.py` to measure TF32 speedup. The script benchmarks matrix multiplication throughput with and without TF32 enabled.

```python
torch.set_float32_matmul_precision("high")   # TF32 enabled
torch.set_float32_matmul_precision("highest") # use FP32 matmul precision
```

Expected results on Ampere+ (A100, RTX 3090, RTX 4090):

- TF32 matmuls are roughly 2-4x faster than FP32 matmuls.
- The speedup comes from Tensor Cores operating on TF32 inputs.
- Memory bandwidth savings are minimal — weights stay FP32.

Note: TF32 speedup depends on batch size and matrix dimensions. Small matrices may see no speedup. Run the benchmark with your actual workload dimensions.

---

## Numerical Stability Analysis

### Gradient Magnitude Over Time

Track gradient magnitude per layer across training. FP16 gradients will show more variance than FP32 — small values round to zero, large values may overflow.

BF16 shows gradient magnitude closer to FP32 because its larger exponent handles the same numerical range.

### Loss Scaling Impact

- Scale factor 1.0 (no scaling): expect significant underflow for deep networks.
- Scale factor 128.0: good starting point for most models.
- Scale factor 512.0+: may cause overflow in forward pass.

GradScaler finds the right factor dynamically by doubling until it sees overflow, then backing off.

---

## Benchmark Configuration

Each script (`amp.py`, `tf32.py`, `grad_scaler.py`, `convergence.py`) accepts command-line arguments to configure:

- `--hidden-size`: width of hidden layers
- `--batch-size`: training batch size
- `--steps`: number of training steps to run
- `--warmup`: warmup steps before timing

Run with `--help` to see all options.

---

## Hardware Support

| Format | Hardware | Tensor Cores |
|--------|----------|--------------|
| FP16   | All NVIDIA GPUs | Yes (Ampere+), emulation (older) |
| BF16   | Ampere+ (A100, RTX 30xx+) | Yes |
| TF32   | Ampere+ only | Yes |

BF16 Tensor Core support arrived with Ampere. On older hardware (V100, Turing), BF16 runs through FP32 paths — no speedup over FP32.

Check your GPU:

```python
import torch
print(torch.cuda.get_device_name())
print(torch.cuda.get_device_capability())
```
