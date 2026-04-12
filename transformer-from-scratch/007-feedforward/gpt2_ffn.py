"""
Compare FFN choices across GPT-2 model sizes.

This script shows how different FFN variants perform across the GPT-2 model sizes:
- GPT-2 Small:  12 layers, 768 dim, 3072 d_ff
- GPT-2 Medium: 24 layers, 1024 dim, 4096 d_ff
- GPT-2 Large:  36 layers, 1280 dim, 5120 d_ff

We compare:
1. Standard FFN (Linear -> GELU -> Linear)
2. FFN-SwiGLU (Linear -> SiLU*gate -> Linear) [used in Llama, PaLM]
3. FFN-ReGLU (Linear -> ReLU*gate -> Linear) [from GLU paper]
4. FFN-GeGLU (Linear -> GELU*gate -> Linear) [from GLU paper]
"""

import torch
import torch.nn as nn
from swiglu import FFNSwiGLU, FFNReGLU, FFNGeGLU
from feedforward import FeedForward


# GPT-2 model size configurations
GPT2_CONFIGS = {
    "gpt2_small":  {"n_layers": 12,  "d_model": 768,  "d_ff": 3072},
    "gpt2_medium": {"n_layers": 24,  "d_model": 1024, "d_ff": 4096},
    "gpt2_large":  {"n_layers": 36,  "d_model": 1280, "d_ff": 5120},
}


def build_transformer_block(d_model, d_ff, ffn_type="standard"):
    """
    A single transformer block with configurable FFN type.

    ffn_type options:
    - "standard": Linear -> GELU -> Linear
    - "swiglu":    Linear -> SiLU*gate -> Linear
    - "reglu":     Linear -> ReLU*gate -> Linear
    - "geglu":     Linear -> GELU*gate -> Linear
    """
    class TransformerBlock(nn.Module):
        def __init__(self):
            super().__init__()
            # Self-attention (simplified — just for FFN comparison)
            self.attn = nn.MultiheadAttention(d_model, num_heads=12, batch_first=True)
            self.ln1 = nn.LayerNorm(d_model)
            self.ln2 = nn.LayerNorm(d_model)
            # FFN
            if ffn_type == "standard":
                self.ffn = FeedForward(d_model, d_ff)
            elif ffn_type == "swiglu":
                self.ffn = FFNSwiGLU(d_model, d_ff)
            elif ffn_type == "reglu":
                self.ffn = FFNReGLU(d_model, d_ff)
            elif ffn_type == "geglu":
                self.ffn = FFNGeGLU(d_model, d_ff)
            else:
                raise ValueError(f"Unknown FFN type: {ffn_type}")

        def forward(self, x):
            # Pre-norm transformer architecture
            x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))[0]
            x = x + self.ffn(self.ln2(x))
            return x

    return TransformerBlock()


def count_ffn_params(n_layers, d_model, d_ff, ffn_type="standard"):
    """Count parameters in the FFN portion of the model."""
    if ffn_type == "standard":
        # Two linear layers: d_model*d_ff + d_ff*d_model + biases
        return n_layers * (d_model * d_ff + d_ff * d_model + d_ff + d_model)
    else:
        # Three linear layers: d_model*d_ff + d_model*d_ff + d_ff*d_model + biases
        return n_layers * (2 * d_model * d_ff + d_ff * d_model + 2 * d_ff + d_model)


def flops_per_token(d_model, d_ff, ffn_type="standard"):
    """
    Approximate FLOPs for one FFN forward pass on one token.
    Each matmul is 2 * N_in * N_out FLOPs.
    """
    if ffn_type == "standard":
        # Linear1: 2 * d_model * d_ff
        # GELU: ~5 FLOPs per element (approximation)
        # Linear2: 2 * d_ff * d_model
        return 2 * d_model * d_ff + 2 * d_ff * d_model
    else:
        # Three matmuls instead of two
        # Gate + Up: 2 * d_model * d_ff
        # Down: 2 * d_ff * d_model
        return 2 * d_model * d_ff + 2 * d_model * d_ff + 2 * d_ff * d_model


if __name__ == "__main__":
    print("=" * 70)
    print("FFN COMPARISON ACROSS GPT-2 MODEL SIZES")
    print("=" * 70)

    ffn_types = ["standard", "swiglu", "reglu", "geglu"]

    print(f"\n{'Model':>12} {'d_model':>8} {'d_ff':>8}  {'FFN Type':>10}  "
          f"{'FFN Params':>14}  {'FLOPs/token':>12}")
    print("-" * 75)

    for model_name, config in GPT2_CONFIGS.items():
        n_layers = config["n_layers"]
        d_model = config["d_model"]
        d_ff = config["d_ff"]

        for ffn_type in ffn_types:
            params = count_ffn_params(n_layers, d_model, d_ff, ffn_type)
            flops = flops_per_token(d_model, d_ff, ffn_type)
            print(f"{model_name:>12} {d_model:>8} {d_ff:>8}  "
                  f"{ffn_type:>10}  {params:>14,}  {flops:>12,}")

        print()

    print("\nKEY OBSERVATIONS:")
    print("  1. SwiGLU, ReGLU, GeGLU all have ~50% more FFN parameters than standard")
    print("     (three matmuls instead of two)")
    print("  2. SwiGLU is the most common in production LLMs (Llama, PaLM, Gemma)")
    print("  3. ReGLU was one of the first GLU variants shown to improve transformers")
    print("  4. GeGLU uses GELU for the gate signal, intermediate in complexity")

    # Verify all FFN variants produce the same output shape
    print("\n" + "=" * 70)
    print("SHAPE VERIFICATION")
    print("=" * 70)
    batch, seq_len = 2, 16
    x = torch.randn(batch, seq_len, GPT2_CONFIGS["gpt2_small"]["d_model"])

    for ffn_type in ffn_types:
        block = build_transformer_block(
            GPT2_CONFIGS["gpt2_small"]["d_model"],
            GPT2_CONFIGS["gpt2_small"]["d_ff"],
            ffn_type=ffn_type
        )
        out = block(x)
        print(f"  {ffn_type:>10}: output shape = {out.shape}")
