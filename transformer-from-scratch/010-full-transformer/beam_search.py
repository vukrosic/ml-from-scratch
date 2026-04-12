"""
beam_search.py — Beam search decoding for the Transformer.

Greedy decoding always picks the single most likely next token.
Beam search keeps the top-k candidates at each step and scores
the accumulated sequence, returning the best-scoring complete
hypothesis at the end.

Why beam search over greedy?
  Greedy can get stuck in locally optimal sequences where an early
  high-probability token leads to a poor overall sentence.
  Beam search explores multiple paths simultaneously, trading compute
  for quality.

How it works:
  1. Initialise with k sequences, each starting with SOS.
  2. For each step, expand every active beam by every possible next token.
  3. Score each new beam as: sum(log P(token_i | prefix_i-1)) / len(prefix)
     (length-normalised to avoid bias toward short sequences)
  4. Keep the top-k beams by score.
  5. Repeat until all beams produce EOS or max_len is reached.
"""

import torch
import torch.nn.functional as F


def make_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0)
    return mask.unsqueeze(0).unsqueeze(0)


def make_src_mask(seq_len, device):
    return torch.ones(1, 1, seq_len, seq_len, device=device)


def make_cross_mask(src_seq_len, tgt_seq_len, device):
    return torch.ones(1, 1, tgt_seq_len, src_seq_len, device=device)


def beam_search(model, src, beam_size, max_len, sos_token, eos_token, device,
                length_penalty=0.6):
    """
    Beam search decoding.

    Args:
        model:         trained Transformer
        src:           (1, src_len) source token indices
        beam_size:     number of candidate beams to keep
        max_len:       maximum decode length
        sos_token:     start-of-sequence token index
        eos_token:     end-of-sequence token index
        device:        cpu or cuda
        length_penalty: exponent for length normalisation (>0, higher = prefer longer)

    Returns:
        Best generated sequence (list of token indices, no SOS/EOS)
    """
    model.eval()

    src = src.to(device)
    src_mask = make_src_mask(src.shape[1], device).to(device)
    enc_output = model.encode(src, src_mask)

    # Initialise: each beam starts with SOS
    # beams: list of (log_prob_sum, token_sequence)
    beams = [(0.0, [sos_token])]
    completed = []

    for _ in range(max_len):
        all_candidates = []

        for score, seq in beams:
            if seq[-1] == eos_token:
                # Already finished — keep but don't expand
                completed.append((score, seq))
                continue

            tgt = torch.tensor([[seq]], device=device)
            tgt_mask = make_causal_mask(len(seq), device).to(device)
            cross_mask = make_cross_mask(src.shape[1], len(seq), device).to(device)

            logits = model.decode(tgt, enc_output, tgt_mask, cross_mask)
            log_probs = F.log_softmax(logits[:, -1, :], dim=-1)   # (1, vocab)

            # Top beam_size tokens
            top_log_probs, top_tokens = log_probs.topk(beam_size, dim=-1)

            for i in range(beam_size):
                token = top_tokens[0, i].item()
                log_p = top_log_probs[0, i].item()
                new_score = score + log_p
                new_seq = seq + [token]
                all_candidates.append((new_score, new_seq))

        if not all_candidates:
            break

        # Length-normalise and rank
        ranked = sorted(all_candidates, key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
        beams = ranked[:beam_size]

        # Check for completion
        if all(seq[-1] == eos_token for _, seq in beams):
            completed.extend(beams)
            break

    # Combine completed and in-progress, pick the best
    all_seqs = completed + beams
    all_seqs = sorted(all_seqs, key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
    best_seq = all_seqs[0][1]

    # Strip SOS/EOS
    return [t for t in best_seq if t not in (sos_token, eos_token)]


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    sos_token = vocab_size
    eos_token = 0

    torch.manual_seed(42)

    print("\n=== Beam Search (k=3) ===")
    src_len = 12
    for i in range(5):
        src = torch.randint(1, vocab_size, (1, src_len))
        correct = ((src + 7 - 1) % (vocab_size - 1)) + 1

        gen = beam_search(model, src, beam_size=3, max_len=20,
                          sos_token=sos_token, eos_token=eos_token,
                          device=device)

        print(f"\nSource:    {src[0].tolist()}")
        print(f"Expected:  {correct[0].tolist()}")
        print(f"Generated: {gen}")
