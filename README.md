# ML From Scratch

> 🔴 YouTube Channel: [Open Super Intelligence Lab](https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g)
> 🟡 Skool Advanced Lesson: [Become AI Researcher](https://www.skool.com/become-ai-researcher-2669/about)

Learn how deep learning actually works by building everything from scratch in Python. Every tutorial starts from zero — no frameworks, no wrappers — then compares against PyTorch to prove it works.

All code is free and open source. Video walkthroughs on YouTube, with deeper follow-up lessons on Skool.

## Tutorials

### PyTorch Internals

How PyTorch works under the hood. Each lesson builds a toy version of a real PyTorch subsystem, then shows you the real thing.

| # | Tutorial | What you build | What you learn |
|---|----------|---------------|----------------|
| 001 | [torch.compile](pytorch-internals/001-torch-compile/) | A graph tracer, fusion pass, and code generator | How TorchDynamo traces Python, how TorchInductor fuses operations into Triton kernels, what graph breaks are and how to fix them |
| 002 | [Autograd](pytorch-internals/002-autograd/) | A scalar autograd engine with `backward()` | How `loss.backward()` works — computation graphs, the chain rule, topological sort, gradient accumulation |

More tutorials coming soon.

## How Each Tutorial Works

Every tutorial folder contains:

- **README.md** -- the core article explaining the concept step by step, with small code blocks and concrete examples
- **Python files** -- runnable implementations you can execute immediately (`python autograd.py`)
- **ADVANCED.md** -- deeper coverage: benchmarks, profiling, PyTorch internals, edge cases, and debugging techniques

The README teaches the concept. The Python files let you run it. ADVANCED goes deeper for those who want the full picture.

## Running the Code

```bash
# Clone the repo
git clone https://github.com/vukrosic/ml-from-scratch.git
cd ml-from-scratch

# Pick a tutorial and run it
cd pytorch-internals/002-autograd
python autograd.py          # run the from-scratch implementation
python compare.py           # verify against PyTorch
python visualize.py         # see the computation graph
```

Each tutorial is self-contained. No cross-dependencies between lessons. You need Python 3.8+ and PyTorch for the comparison scripts.

```bash
pip install torch
```

## Structure

```
ml-from-scratch/
└── pytorch-internals/
    ├── 001-torch-compile/     # graph tracing, fusion, code generation
    │   ├── README.md
    │   ├── ADVANCED.md
    │   ├── tracer.py
    │   ├── fusion.py
    │   ├── codegen.py
    │   ├── benchmark.py
    │   ├── graph_breaks.py
    │   ├── graph_break_scanner.py
    │   ├── benchmark_modes.py
    │   └── profile_architectures.py
    │
    └── 002-autograd/          # computation graphs, chain rule, backward pass
        ├── README.md
        ├── ADVANCED.md
        ├── autograd.py
        ├── compare.py
        └── visualize.py
```

## Links

- [YouTube](https://www.youtube.com/channel/UC7XJj9pv_11a11FUxCMz15g) -- main channel
- [Skool](https://www.skool.com/become-ai-researcher-2669/about) -- advanced lesson and deeper follow-up material
