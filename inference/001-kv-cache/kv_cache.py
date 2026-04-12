"""
KV-cache autoregressive generation.

We store key/value tensors in a dict keyed by (layer, position).
At each step we only compute K and V for the new token, then
append them to the cache. Attention attends over cached K/V.
"""

import torch


class CachedAttention(torch.nn.Module):
    """Single-layer attention WITH explicit KV-cache."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.W_q = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_k = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_v = torch.nn.Linear(d_model, d_model, bias=False)
        self.W_o = torch.nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        cache_k: dict[int, torch.Tensor],
        cache_v: dict[int, torch.Tensor],
        layer_idx: int,
        start_pos: int,
    ) -> torch.Tensor:
        """
        x: (batch, 1, d_model) — single new token
        cache_k, cache_v: dict mapping layer_idx -> (batch, n_heads, seq_sofar, head_dim)
        layer_idx: which layer this attention instance belongs to
        start_pos: the position index where this layer's cached sequence starts
        Returns: (batch, 1, d_model)
        """
        B, _, _ = x.shape

        q = self.W_q(x).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.W_k(x).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.W_v(x).view(B, 1, self.n_heads, self.head_dim).transpose(1, 2)

        # Append new K/V to cached tensors
        key = torch.cat([cache_k[layer_idx], k], dim=2)
        val = torch.cat([cache_v[layer_idx], v], dim=2)

        # Update the cache for next step
        cache_k[layer_idx] = key
        cache_v[layer_idx] = val

        # Attention over full cached sequence — O(1) for us, but the cache IS the full history
        attn = (q @ key.transpose(-2, -1)) * self.scale
        attn = torch.softmax(attn, dim=-1)
        out = attn @ val

        out = out.transpose(1, 2).contiguous().view(B, 1, -1)
        return self.W_o(out)


class CachedAttentionLayer(torch.nn.Module):
    """Wrapper to give each layer its own cache index."""

    def __init__(self, d_model: int, n_heads: int, layer_idx: int):
        super().__init__()
        self._layer_idx = layer_idx
        self.attn = CachedAttention(d_model, n_heads)
        self.ln1 = torch.nn.LayerNorm(d_model)
        self.ln2 = torch.nn.LayerNorm(d_model)
        self.ff = torch.nn.Sequential(
            torch.nn.Linear(d_model, d_model * 4),
            torch.nn.GELU(),
            torch.nn.Linear(d_model * 4, d_model),
        )

    def forward(
        self,
        x: torch.Tensor,
        cache_k: dict[int, torch.Tensor],
        cache_v: dict[int, torch.Tensor],
        start_pos: int,
    ) -> torch.Tensor:
        h = self.ln1(x)
        h = self.attn(h, cache_k, cache_v, self._layer_idx, start_pos)
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


class CachedGenerator(torch.nn.Module):
    """Minimal language model with cached generation support."""

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d_model)
        self.blocks = torch.nn.ModuleList(
            [
                CachedAttentionLayer(d_model, n_heads, layer_idx=i)
                for i in range(n_layers)
            ]
        )
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.ln_f.bias = None

    def forward(
        self,
        token_ids: torch.Tensor,
        cache_k: dict[int, torch.Tensor],
        cache_v: dict[int, torch.Tensor],
        start_pos: int,
    ) -> torch.Tensor:
        """
        token_ids: (batch, seq_len) — seq_len is 1 during generation after warmup
        cache_k, cache_v: shared dict for all layers
        start_pos: position of first token in this batch
        """
        x = self.embed(token_ids)
        for block in self.blocks:
            x = block(x, cache_k, cache_v, start_pos)
        x = self.ln_f(x)
        return self.lm_head(x)


@torch.no_grad()
def generate_cached(model: torch.nn.Module, prompt: torch.Tensor, max_new: int) -> list[int]:
    """
    Autoregressive generation WITH a KV-cache.

    cache_k and cache_v are dicts: layer_idx -> (batch, n_heads, seq_sofar, head_dim)
    At step 0 we pass the full prompt. At subsequent steps we only pass
    the single new token, and the cache stores all K/V history.
    """
    model.eval()

    # Warmup: process the full prompt and fill the cache
    # After W_k/v and transpose(1,2), K/V tensors have shape (batch, n_heads, seq, head_dim)
    cache_k = {i: torch.empty(1, 4, 0, 32) for i in range(2)}
    cache_v = {i: torch.empty(1, 4, 0, 32) for i in range(2)}

    # Process prompt
    seq_len = prompt.shape[1]
    for pos in range(seq_len):
        start_pos = pos
        token = prompt[:, pos:pos+1]
        logits = model(token, cache_k, cache_v, start_pos)

    # Generate new tokens — one at a time, cache grows
    generated = prompt.tolist()[0]
    for _ in range(max_new):
        start_pos = seq_len
        token = prompt[:, -1:]  # last token only
        logits = model(token, cache_k, cache_v, start_pos)
        next_token = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_token.item())
        prompt = next_token.unsqueeze(0)
        seq_len += 1

    return generated


if __name__ == "__main__":
    torch.manual_seed(42)

    model = CachedGenerator(vocab_size=512, d_model=128, n_heads=4, n_layers=2)

    prompt = torch.tensor([[1, 2, 3, 4, 5]])

    result = generate_cached(model, prompt, max_new=10)
    print(f"Generated {len(result) - len(prompt[0])} tokens")
    print(f"Full sequence length: {len(result)}")
