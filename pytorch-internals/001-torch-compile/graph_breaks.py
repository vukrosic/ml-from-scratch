"""Graph break examples and fixes."""

import torch
import torch.nn as nn
import torch._dynamo as dynamo


# BAD — .item() causes a graph break
class BadModule(nn.Module):
    def forward(self, x):
        if x.sum().item() > 0:
            return x * 2
        return x - 2


# FIXED — torch.where keeps everything as tensors
class FixedModule(nn.Module):
    def forward(self, x):
        cond = (x.sum() > 0)
        return torch.where(cond, x * 2, x - 2)


if __name__ == "__main__":
    x = torch.randn(10)

    print("=== Bad module (has graph breaks) ===")
    bad_fn = BadModule()
    explanation = dynamo.explain(bad_fn)(x)
    print(f"Graph breaks: {explanation.break_count}")
    print(f"Graphs: {explanation.graph_count}")

    print("\n=== Fixed module (no graph breaks) ===")
    dynamo.reset()
    fixed_fn = FixedModule()
    explanation = dynamo.explain(fixed_fn)(x)
    print(f"Graph breaks: {explanation.break_count}")
    print(f"Graphs: {explanation.graph_count}")
