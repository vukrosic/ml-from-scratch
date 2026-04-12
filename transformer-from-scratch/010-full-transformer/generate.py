"""
generate.py — Greedy decoding from a trained Transformer.

Given a source sequence, the model generates one token at a time.
At each step:
  1. Encode the full source
  2. Start with an initial token (SOS/eos token)
  3. Decode one step, take the argmax, append to the target
  4. Stop when EOS is produced or max_len is reached

This is called "greedy" because we always pick the most likely next token.
It is fast but can get stuck in locally optimal sequences.
See beam_search.py for a simple alternative that considers multiple candidates.
"""

import torch


# ---------------------------------------------------------------------------
# Mask helpers (must match full_transformer.py)
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0)
    return mask.unsqueeze(0).unsqueeze(0)


def make_src_mask(seq_len, device):
    return torch.ones(1, 1, seq_len, seq_len, device=device)


def make_cross_mask(src_seq_len, tgt_seq_len, device):
    return torch.ones(1, 1, tgt_seq_len, src_seq_len, device=device)


# ---------------------------------------------------------------------------
# Greedy decode
# ---------------------------------------------------------------------------

def greedy_decode(model, src, max_len, sos_token, eos_token, device):
    """
    Greedy decoding for a single source sequence.

    Args:
        model:       trained Transformer
        src:         (1, src_len) token indices
        max_len:     maximum generated sequence length
        sos_token:   token index that starts the target (same as vocab_size from train)
        eos_token:   token index that ends the target (0, used as PAD/EOS)
        device:      cpu or cuda

    Returns:
        Generated token indices (list, excluding SOS)
    """
    model.eval()

    src = src.to(device)
    src_mask = make_src_mask(src.shape[1], device).to(device)

    # Encode source once
    enc_output = model.encode(src, src_mask)

    # Start with SOS token
    tgt = torch.tensor([[sos_token]], device=device)
    seq = []

    for _ in range(max_len):
        tgt_mask = make_causal_mask(tgt.shape[1], device).to(device)
        cross_mask = make_cross_mask(src.shape[1], tgt.shape[1], device).to(device)

        logits = model.decode(tgt, enc_output, tgt_mask, cross_mask)
        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)   # (1, 1)

        token = next_token.item()
        seq.append(token)

        if token == eos_token:
            break

        tgt = torch.cat([tgt, next_token], dim=1)

    return seq


# ---------------------------------------------------------------------------
# Run on a sample from the dataset
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    from full_transformer import Transformer

    checkpoint = torch.load("transformer_cipher.pt", map_location=device)
    config = checkpoint["config"]

    model = Transformer(
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        d_ff=config["d_ff"],
        n_layers=config["n_layers"],
        src_vocab_size=config["vocab_size"],
        tgt_vocab_size=config["vocab_size"],
    )
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()

    vocab_size = config["vocab_size"]
    sos_token = vocab_size          # SOS = vocab_size (as in train.py CipherDataset)
    eos_token = 0                   # EOS = 0 (as in train.py)

    # Generate a few examples
    print("\n=== Greedy Decoding Examples ===")
    torch.manual_seed(42)

    src_len = 12
    for i in range(5):
        # Random source
        src = torch.randint(1, vocab_size, (1, src_len))
        # Compute what the correct target should be (shift=7 as in train.py)
        correct = ((src + 7 - 1) % (vocab_size - 1)) + 1

        gen = greedy_decode(model, src, max_len=20, sos_token=sos_token, eos_token=eos_token, device=device)

        print(f"\nSource:     {src[0].tolist()}")
        print(f"Expected:  {correct[0].tolist()}")
        print(f"Generated: {gen}")
        match = (gen == correct[0].tolist()[:len(gen)])
        print(f"Match: {match}")
