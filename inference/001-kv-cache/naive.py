"""
Naive autoregressive generation: no KV-cache.

At each step we recompute ALL previous key/value tensors from scratch.
This is O(n^2) — slow for long sequences.
"""

import torch


class NaiveAttention(torch.nn.Module):
    """Single-layer attention without any KV-cache."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        Returns: (batch, seq_len, d_model)
        """
        B, T, _ = x.shape

        q = self.W_q(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Full attention — O(T^2) per step
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.W_o(out)


class NaiveTransformerBlock(torch.nn.Module):
    """Transformer block using naive attention."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.attn = NaiveAttention(d_model, n_heads)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.GELU(),
            torch.nn.Linear(d_model * 4, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class NaiveGenerator(torch.nn.Module):
    """Minimal language model — recomputes everything at every step."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.blocks = torch.nn.ModuleList(
            [NaiveTransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f.bias = None  # Tie with embedding

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """
        token_ids: (batch, seq_len)
        Returns: (batch, seq_len, vocab_size)
        """
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x)
        x = self.ln_f(x)
        return self.lm_head(x)


@torch.no_grad()
def generate_naive(model: torch.nn.Module, prompt: torch.Tensor, max_new: int) -> list[int]:
    """
    Autoregressive generation WITHOUT a KV-cache.

    At every step we feed the entire sequence so far into the model.
    The model recomputes K and V for ALL tokens seen up to that point.
    """
    model.eval()
    generated = prompt.tolist()

    for _ in range(max_new):
        # Always pass the full sequence — model recomputes K/V from scratch
        logits = model(prompt)
        next_token_logits = logits[:, -1, :]  # (batch, vocab)
        next_token = next_token_logits.argmax(dim=-1)

        generated.append(next_token.item())
        prompt = torch.cat([prompt, next_token.unsqueeze(0)], dim=1)

    return generated


if __name__ == "__main__":
    torch.manual_seed(42)

    # Tiny model for demonstration
    model = NaiveGenerator(vocab_size=512, d_model=128, n_heads=4, n_layers=2)

    # Token IDs for the prompt
    prompt = torch.tensor([[1, 2, 3, 4, 5]])

    # Generate 10 new tokens
    result = generate_naive(model, prompt, max_new=10)
    print(f"Generated {len(result) - len(prompt[0])} tokens")
    print(f"Full sequence length: {len(result)}")
