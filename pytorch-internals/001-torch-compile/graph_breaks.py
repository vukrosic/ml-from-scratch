"""Graph break examples and fixes.

What it does:
  - Defines two modules: BadModule (uses .item() causing a graph break)
    and FixedModule (uses torch.where with no break).
  - Runs dynamo.explain on both to show the difference in break counts.

Run:
  python graph_breaks.py

Expected output:
  === Bad module (has graph breaks) ===
  Graph breaks: 1
  Graphs: 1

  === Fixed module (no graph breaks) ===
  Graph breaks: 0
  Graphs: 1

Note:
  The dynamo.explain API may vary by PyTorch version.
  If break_count causes an AttributeError, try graph_break_count instead.
"""

import torch
import torch.nn as nn
import torch._dynamo as dynamo


# BAD — .item() causes a graph break
class BadModule(nn.Module):
    def forward(self, x):
        if x.sum().item() > 0:   # .item() forces Python fallback — graph break
            return x * 2
        return x - 2


# FIXED — torch.where keeps everything as tensors
class FixedModule(nn.Module):
    def forward(self, x):
        cond = (x.sum() > 0)         # stays as a tensor — no break
        return torch.where(cond, x * 2, x - 2)


if __name__ == "__main__":
    x = torch.randn(10)

    print("=== Bad module (has graph breaks) ===")
    bad_fn = BadModule()
    explanation = dynamo.explain(bad_fn)(x)
    print(f"Graph breaks: {explanation.graph_break_count}")
    print(f"Graphs: {explanation.graph_count}")

    print("\n=== Fixed module (no graph breaks) ===")
    dynamo.reset()
    fixed_fn = FixedModule()
    explanation = dynamo.explain(fixed_fn)(x)
    print(f"Graph breaks: {explanation.graph_break_count}")
    print(f"Graphs: {explanation.graph_count}")
