# Custom Ops

Register your own operations in PyTorch so they work with autograd, compile, and vmap.

## What we build
- Custom forward+backward op with `torch.autograd.Function`
- Register with `torch.library` for compile compatibility
- Benchmark custom vs built-in

## Files (planned)
- `custom_relu.py` — custom op with autograd
- `library_op.py` — torch.library registration
- `benchmark.py`
