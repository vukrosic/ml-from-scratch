"""
train.py — Training loop for the character-level English -> French Transformer.

We use label smoothing (a key regularisation technique from the paper) and
backpropagation through the full encoder-decoder graph.  The model is trained
to minimise cross-entropy on the next-token prediction task.

Key design decisions:
  - Label smoothing: reduces overconfidence, improves generalisation (Section 5.4)
  - Gradient clipping: prevents gradient explosion in deep networks (Section 5.3)
  - Teacher forcing: decoder receives correct prefix at each step (standard)
  - Warm-up learning rate: Section 5.3 describes this schedule
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import Transformer, make_causal_mask, make_src_mask
from data import build_dataloaders


# ---------------------------------------------------------------------------
# 1.  Learning Rate Schedule  (Section 5.3, warm-up)
# ---------------------------------------------------------------------------
# lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
#
# The model uses sinusoidal positional encoding so it works with any sequence
# length up to max_seq_len.  We set warmup_steps=400 as a reasonable default.

class WarmupScheduler:
    """Linear warm-up followed by inverse-square-root decay."""

    def __init__(self, optimizer, d_model, warmup_steps=400):
        self.optimizer = optimizer
        self.d_model   = d_model
        self.warmup    = warmup_steps
        self._step_cnt = 0
        self.base_lr   = optimizer.param_groups[0]["lr"]

    def step(self):
        self._step_cnt += 1
        scale = self.d_model ** -0.5
        warmup = min(self._step_cnt ** -0.5,
                     self._step_cnt * self.warmup ** -1.5)
        lr = scale * warmup * self.base_lr
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        return lr


# ---------------------------------------------------------------------------
# 2.  Training step
# ---------------------------------------------------------------------------
def train_step(model, src, tgt, optimizer, scheduler, criterion,
               clip_grad=1.0, device="cpu"):
    """
    One forward + backward pass over a batch.

    src: (batch, src_len)
    tgt: (batch, tgt_len)
    """
    model.train()
    optimizer.zero_grad()

    # Decoder input: all tokens except the last (predict next token)
    tgt_input  = tgt[:, :-1]
    # Labels:    all tokens except the first  (SOS is not a valid label)
    tgt_labels = tgt[:, 1:]

    # Masks
    src_mask = make_src_mask(src.shape[1], device)
    tgt_mask = make_causal_mask(tgt_input.shape[1], device)

    src = src.to(device)
    tgt_input = tgt_input.to(device)
    tgt_labels = tgt_labels.to(device)
    src_mask = src_mask.to(device)
    tgt_mask = tgt_mask.to(device)

    # Forward pass
    logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
    # logits: (batch, tgt_len-1, vocab_size)

    # Cross-entropy loss (ignore padding positions)
    loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_labels.reshape(-1))

    # Backward pass
    loss.backward()

    # Gradient clipping (Section 5.3: "gradient clipping was performed")
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

    optimizer.step()
    scheduler.step()

    return loss.item()


# ---------------------------------------------------------------------------
# 3.  Validation perplexity
# ---------------------------------------------------------------------------
def evaluate(model, val_loader, criterion, device="cpu"):
    """Compute average cross-entropy loss on the validation set."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt, *_ in val_loader:
            tgt_input  = tgt[:, :-1]
            tgt_labels = tgt[:, 1:]

            src_mask = make_src_mask(src.shape[1], device)
            tgt_mask = make_causal_mask(tgt_input.shape[1], device)

            src = src.to(device)
            tgt_input = tgt_input.to(device)
            tgt_labels = tgt_labels.to(device)
            src_mask = src_mask.to(device)
            tgt_mask = tgt_mask.to(device)

            logits = model(src, tgt_input, src_mask=src_mask, tgt_mask=tgt_mask)
            loss = criterion(logits.reshape(-1, logits.shape[-1]), tgt_labels.reshape(-1))

            # Count non-padding tokens
            total_loss   += loss.item() * tgt_labels.numel()
            total_tokens += tgt_labels.numel()

    avg_loss = total_loss / total_tokens
    return math.exp(avg_loss)   # perplexity = exp(loss)


# ---------------------------------------------------------------------------
# 4.  Main training loop
# ---------------------------------------------------------------------------
def train(model, train_loader, val_loader, n_epochs=30,
         lr=0.0001, device="cpu", print_every=5):
    """
    Train the transformer and report training loss + validation perplexity.
    """
    model = model.to(device)

    # Label smoothing: a regularisation technique from Section 5.4
    # "Label smoothing of epsilon_ls = 0.1 was used"
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9)
    scheduler = WarmupScheduler(optimizer, model.d_model, warmup_steps=400)

    print(f"{'Epoch':>6} | {'Step':>5} | {'Train Loss':>11} | {'Val PPL':>8} | {'LR':>12}")
    print("-" * 60)

    global_step = 0
    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0
        n_batches  = 0

        for src, tgt, *_ in train_loader:
            loss = train_step(model, src, tgt, optimizer, scheduler, criterion,
                              device=device)
            epoch_loss += loss
            n_batches  += 1
            global_step += 1

            # Periodic logging
            if global_step % print_every == 0:
                val_ppl = evaluate(model, val_loader, criterion, device=device)
                current_lr = optimizer.param_groups[0]["lr"]
                avg_train_loss = epoch_loss / n_batches
                print(f"{epoch:>6} | {global_step:>5} | {avg_train_loss:>11.4f} "
                      f"| {val_ppl:>8.2f} | {current_lr:>12.6f}")

        # End-of-epoch validation
        val_ppl = evaluate(model, val_loader, criterion, device=device)
        avg_train_loss = epoch_loss / n_batches
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"{epoch:>6} | {global_step:>5} | {avg_train_loss:>11.4f} "
              f"| {val_ppl:>8.2f} | {current_lr:>12.6f}")
        print("-" * 60)

    return model


# ---------------------------------------------------------------------------
# 5.  Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Hyperparameters (paper defaults)
    d_model  = 512
    n_heads  = 8
    d_ff     = 2048
    n_layers = 6
    batch_size = 32
    n_epochs   = 60      # ~60 epochs is enough to see convergence on this tiny corpus

    train_loader, val_loader, src_vocab, tgt_vocab = build_dataloaders(
        batch_size=batch_size
    )

    model = Transformer(
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        n_layers=n_layers,
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    trained_model = train(
        model,
        train_loader,
        val_loader,
        n_epochs=n_epochs,
        device=device,
        print_every=10,
    )

    # Save the trained model
    torch.save({
        "model_state": trained_model.state_dict(),
        "src_vocab":   src_vocab,
        "tgt_vocab":   tgt_vocab,
        "config":      {"d_model": d_model, "n_heads": n_heads,
                        "d_ff": d_ff, "n_layers": n_layers},
    }, "transformer_checkpoint.pt")
    print("\nModel saved to transformer_checkpoint.pt")
