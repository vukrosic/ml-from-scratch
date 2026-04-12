"""Automated graph break scanner — reports every graph break in a model with suggested fixes.

What it does:
  - Wraps dynamo.explain to get a structured break report.
  - Prints total breaks, total compiled graphs, and the reason for each break.
  - Includes a fix lookup table for the most common causes (.item(), print, resize_, etc.).

Run:
  python graph_break_scanner.py

Expected output:
  ============================================================
  Graph Break Report
  Total breaks: 2
  Compiled graphs: 2
  ============================================================

    Break 1
    Reason : ...
    Handler: ...

    Break 2
    Reason : ...
    Handler: ...

Note:
  dynamo.explain returns an object whose fields vary by PyTorch version.
  Key fields used here: graph_break_count, graph_count, graph_breaks.
  If this errors on your version, run dynamo.explain directly on your model
  to see what fields are available.
"""

import torch
import torch.nn as nn
import torch._dynamo as dynamo


class GraphBreakScanner:
    def __init__(self, model):
        self.model = model
        dynamo.reset()

    def scan(self, sample_input):
        """Run dynamo.explain and return a structured break report."""
        explanation = dynamo.explain(self.model)(sample_input)

        breaks = []
        if hasattr(explanation, 'graph_breaks'):
            for i, detail in enumerate(getattr(explanation, 'graph_breaks', [])):
                breaks.append({
                    "index": i,
                    "reason": getattr(detail, 'reason', str(detail)),
                    "handler": getattr(detail, 'handler', 'unknown'),
                })

        return {
            "total_breaks": getattr(explanation, 'graph_break_count', 0),
            "total_graphs": getattr(explanation, 'graph_count', 0),
            "breaks": breaks,
        }

    def report(self, sample_input):
        report = self.scan(sample_input)
        print("=" * 60)
        print("Graph Break Report")
        print("=" * 60)
        print(f"Total breaks : {report['total_breaks']}")
        print(f"Compiled graphs: {report['total_graphs']}")
        print()

        if not report['breaks']:
            print("No graph breaks found.")
            return

        for b in report['breaks']:
            print(f"  Break {b['index']+1}")
            print(f"  Reason : {b['reason']}")
            print(f"  Handler: {b['handler']}")
            print()


# Common graph break causes and their fixes
GRAPH_BREAK_FIXES = {
    ".item()": "Replace tensor.item() with tensor.tolist() or keep the value in tensor space.",
    "tensor.sum() > 0": "Replace `if tensor.sum() > 0` with torch.where or torch.any().",
    "print(tensor)": "Remove print(tensor) from the forward pass, or use torch._logging instead.",
    "resize_": "resize_() mutates tensor metadata. Use reshape() instead, which returns a new tensor.",
    "share_memory_": "share_memory_() causes breaks. Avoid it in compiled forward passes.",
    "data_ptr()": "data_ptr() returns a raw pointer. Avoid in compiled forward passes.",
}


def suggest_fix(reason_str):
    """Given a graph break reason string, return a suggested fix."""
    for cause, fix in GRAPH_BREAK_FIXES.items():
        if cause in reason_str:
            return fix
    return "Review the specific line causing the break. Consider restructuring or using torch.where."


if __name__ == "__main__":
    class BuggyModel(nn.Module):
        def forward(self, x):
            if x.sum().item() > 0:     # graph break
                x = x * 2
            print(x.sum().item())       # graph break
            return x

    model = BuggyModel()
    scanner = GraphBreakScanner(model)
    scanner.report(torch.randn(8))
