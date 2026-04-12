"""Automated graph break scanner -- reports every graph break in a model with fixes."""

import torch
import torch.nn as nn
import torch._dynamo as dynamo


class GraphBreakScanner:
    def __init__(self, model):
        self.model = model
        dynamo.reset()

    def scan(self, sample_input):
        """Run dynamo.explain and return structured break report."""
        explanation = dynamo.explain(self.model)(sample_input)

        breaks = []
        if hasattr(explanation, 'graph_break_count'):
            for i, detail in enumerate(getattr(explanation, 'graph_breaks', [])):
                breaks.append({
                    "index": i,
                    "reason": getattr(detail, 'reason', str(detail)),
                    "handler": getattr(detail, 'handler', 'unknown'),
                })

        return {
            "total_breaks": explanation.graph_break_count if hasattr(explanation, 'graph_break_count') else 0,
            "total_graphs": explanation.graph_count if hasattr(explanation, 'graph_count') else 0,
            "breaks": breaks,
        }

    def report(self, sample_input):
        report = self.scan(sample_input)
        print("=" * 60)
        print(f"Graph Break Report")
        print(f="=" * 60)
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
    ".item()": "Replace `tensor.item()` with `tensor.tolist()` or keep in tensor space.",
    "tensor.sum() > 0": "Replace `if tensor.sum() > 0` with `torch.where` or `torch.any()`.",
    "print(tensor)": "Remove print(tensor) from forward pass, or use `torch._dynamo.log_state`.",
    "resize_": "resize_() mutates tensor metadata and breaks tracing. Use reshape() instead.",
    "share_memory_": "share_memory_() causes breaks. Avoid in compiled forward passes.",
    "data_ptr()": "data_ptr() returns a raw pointer. Avoid in compiled forward.",
}


def suggest_fix(reason_str):
    """Given a graph break reason string, suggest a fix."""
    for cause, fix in GRAPH_BREAK_FIXES.items():
        if cause in reason_str:
            return fix
    return "Review the specific line causing the break. Consider restructuring or using torch.where."


# Example usage
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
