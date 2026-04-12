"""
train.py — Simple training loop for the full Transformer.

We train on a tiny synthetic translation task:
  Source: a sequence of N tokens drawn from a small vocabulary
  Target: the same sequence but each token incremented by a fixed offset
         (a deterministic "cipher" that the model must learn to reverse)

This is not a real language but it is real sequence-to-sequence learning:
the model must attend over the entire source to produce each target token.

Training uses teacher forcing — the decoder receives the full target
sequence at once.  The loss is cross-entropy on next-token prediction.
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class CipherDataset(Dataset):
    """
    Source: random token sequences of length src_len from [1, vocab_size].
    Target: each token shifted by +shift, wrapped modulo vocab_size.

    The model must learn: output[i] = (input[i] + shift) % vocab_size.
    Since shift is small (< vocab_size), attending to the source alone
    is sufficient — no copying or rare tokens needed.
    """

    def __init__(self, n_samples, src_len, tgt_len, vocab_size, shift=7):
        super().__init__()
        self.src = torch.randint(1, vocab_size, (n_samples, src_len))
        self.tgt = ((self.src + shift - 1) % (vocab_size - 1)) + 1
        # Add SOS token at the start of the target for teacher forcing
        self.sos = torch.full((n_samples, 1), vocab_size)      # vocab_size = EOS/SOS
        self.tgt_input = torch.cat([self.sos, self.tgt], dim=1)
        self.tgt_labels = torch.cat([self.tgt, torch.full((n_samples, 1), 0)], dim=1)  # 0 = PAD

    def __len__(self):
        return self.src.shape[0]

    def __getitem__(self, idx):
        return self.src[idx], self.tgt_input[idx], self.tgt_labels[idx]


# ---------------------------------------------------------------------------
# Mask helpers (also used in full_transformer.py)
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len, device):
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device), diagonal=0)
    return mask.unsqueeze(0).unsqueeze(0)


def make_src_mask(seq_len, device):
    return torch.ones(1, 1, seq_len, seq_len, device=device)


def make_cross_mask(src_seq_len, tgt_seq_len, device):
    return torch.ones(1, 1, tgt_seq_len, src_seq_len, device=device)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------

def train_step(model, src, tgt_input, tgt_labels, optimizer, criterion, device):
    model.train()
    optimizer.zero_grad()

    src_mask = make_src_mask(src.shape[1], device)
    tgt_mask = make_causal_mask(tgt_input.shape[1], device)
    cross_mask = make_cross_mask(src.shape[1], tgt_input.shape[1], device)

    src = src.to(device)
    tgt_input = tgt_input.to(device)
    tgt_labels = tgt_labels.to(device)
    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)
    cross_mask = cross_mask.to(device)

    logits = model(src, tgt_input, src_mask, tgt_mask, cross_mask)
    loss = criterion(
        logits.reshape(-1, logits.shape[-1]),
        tgt_labels.reshape(-1)
    )
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(model, train_loader, n_epochs, lr, device, print_every=200):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model = model.to(device)

    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        n_batches = 0

        for src, tgt_input, tgt_labels in train_loader:
            loss = train_step(model, src, tgt_input, tgt_labels, optimizer, criterion, device)
            total_loss += loss
            n_batches += 1

            if n_batches % print_every == 0:
                avg = total_loss / n_batches
                print(f"  Epoch {epoch} | Batch {n_batches} | Loss {avg:.4f}")

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch:>3} complete | Avg loss {avg_loss:.4f}")

    return model


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Hyperparameters (small for quick CPU training)
    d_model = 128
    n_heads = 4
    d_ff = 512
    n_layers = 2
    vocab_size = 50          # small vocabulary
    src_len = 12
    tgt_len = 12             # same length as source (target is shifted version)
    batch_size = 64
    n_epochs = 30
    lr = 1e-3

    # Build model
    from full_transformer import Transformer

    model = Transformer(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        src_vocab_size=vocab_size,
        tgt_vocab_size=vocab_size,
    )

    # Build dataloaders
    train_ds = CipherDataset(n_samples=10000, src_len=src_len, tgt_len=tgt_len,
                              vocab_size=vocab_size, shift=7)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    trained = train(model, train_loader, n_epochs=n_epochs, lr=lr, device=device)

    # Save
    torch.save({
        "model_state": trained.state_dict(),
        "config": {
            "d_model": d_model, "n_heads": n_heads, "d_ff": d_ff,
            "n_layers": n_layers, "vocab_size": vocab_size,
        },
    }, "transformer_cipher.pt")
    print("Saved to transformer_cipher.pt")
