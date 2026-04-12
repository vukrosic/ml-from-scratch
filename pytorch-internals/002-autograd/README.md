# Autograd From Scratch

Build a tiny autograd engine. Forward pass records a tape, backward pass replays it.

## What we build
- `Value` class with `+`, `*`, `relu`, and `backward()`
- Computation graph visualization
- Compare against `torch.autograd`

## Files (planned)
- `autograd.py` — the engine
- `visualize.py` — draw the graph
- `compare.py` — verify against PyTorch
