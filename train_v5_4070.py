"""
XORZO v5 — Circumpunct Brain Training Script
Tuned for RTX 4070 (12GB VRAM)

Usage:
    python train_v5_4070.py --test    # ~1 min smoke test
    python train_v5_4070.py --mini    # ~10 min real test
    python train_v5_4070.py           # full training
"""

import sys
import os
import time
import math
import json
import glob

# ── Mode selection ──
TEST_MODE = "--test" in sys.argv
MINI_MODE = "--mini" in sys.argv

if TEST_MODE:
    N_EPOCHS = 10
    MAX_CORPUS_CHARS = 200_000
    CHECKPOINT_EVERY = 5
    MODE_NAME = "TEST"
    MODE_DESC = "1 run, 10 epochs, small corpus — ~1 min smoke test"
elif MINI_MODE:
    N_EPOCHS = 30
    MAX_CORPUS_CHARS = 500_000
    CHECKPOINT_EVERY = 10
    MODE_NAME = "MINI"
    MODE_DESC = "1 run, 30 epochs, 500K corpus — ~10 min"
else:
    N_EPOCHS = 80
    MAX_CORPUS_CHARS = 4_000_000
    CHECKPOINT_EVERY = 20
    MODE_NAME = "FULL"
    MODE_DESC = "1 run, 80 epochs, 4M corpus — ~2 hours"

import torch
import torch.nn as nn
import torch.nn.functional as F

# Import the v5 architecture
from v5_architecture import CircumpunctBrain

PHI = (1 + math.sqrt(5)) / 2


def banner():
    print("=" * 70)
    print(f"  ⊙ XORZO v5 — CIRCUMPUNCT BRAIN — {MODE_NAME} MODE")
    print(f"  {MODE_DESC}")
    print(f"  A network of networks: ⊙ = Φ(•, ○)")
    print(f"  Binary aperture (÷t) + Shared field (Φ) + Boundary (○)")
    print(f"  Tuned for RTX 4070 (12GB VRAM)")
    print("=" * 70)


def gather_corpus(max_chars):
    """Gather training text from available sources."""
    corpus = ""

    # Try local training data first
    training_dir = os.path.join(os.path.dirname(__file__), "training")
    if os.path.isdir(training_dir):
        for fp in sorted(glob.glob(os.path.join(training_dir, "*.txt"))):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    corpus += f.read() + "\n"
            except Exception:
                pass

    # Also try HTML files
    if os.path.isdir(training_dir):
        for fp in sorted(glob.glob(os.path.join(training_dir, "*.html"))):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    corpus += f.read() + "\n"
            except Exception:
                pass

    # Fallback: any .py files in the project
    if len(corpus) < 10000:
        for fp in sorted(glob.glob(os.path.join(os.path.dirname(__file__), "*.py"))):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    corpus += f.read() + "\n"
            except Exception:
                pass

    corpus = corpus[:max_chars]
    return corpus


def main():
    banner()

    # ── GPU ──
    device = "cpu"
    vram_total = 0.0
    if torch.cuda.is_available():
        device = "cuda"
        gpu = torch.cuda.get_device_properties(0)
        vram_total = gpu.total_memory / 1024**3
        print(f"  GPU: {gpu.name} ({vram_total:.1f} GB)")
    else:
        print("  ⚠ No GPU — running on CPU (will be slow)")

    # ── Corpus ──
    print(f"  ── Gathering Corpus ──")
    corpus = gather_corpus(MAX_CORPUS_CHARS)
    print(f"    Total corpus: {len(corpus):,} chars")

    chars = sorted(set(corpus))
    vocab_size = len(chars)
    print(f"    Unique chars: {vocab_size}")

    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    data = torch.tensor([char_to_idx.get(c, 0) for c in corpus], dtype=torch.long)

    # ── Model ──
    SEQ_LEN = 256
    BATCH_SIZE = 8 if TEST_MODE else 12

    brain = CircumpunctBrain(
        vocab_size=vocab_size,
        d_model=192,
        n_nodes_l0=6,
        n_nodes_l1=2,
        n_heads=4,
        max_len=SEQ_LEN,
        dropout=0.1,
    )

    print()
    print(brain.status())
    print()

    brain = brain.to(device)
    n_params = sum(p.numel() for p in brain.parameters())
    print(f"  Parameters: {n_params:,} total")

    # ── VRAM check ──
    if device == "cuda":
        vram_used = torch.cuda.memory_allocated() / 1024**3
        vram_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM: {vram_used:.1f}G / {vram_total:.1f}G ({vram_used/vram_total*100:.0f}%)")

    # ── Optimizer ──
    LR = 3e-4
    optimizer = torch.optim.AdamW(brain.parameters(), lr=LR, weight_decay=0.01)

    # ── LR schedule (fractal: ⊛→i→☀) ──
    def fractal_lr(epoch):
        progress = epoch / max(N_EPOCHS - 1, 1)
        if progress < 1/3:
            p = progress * 3
            return 0.1 + 0.9 * (1 - math.cos(math.pi * p)) / 2
        elif progress < 2/3:
            p = (progress - 1/3) * 3
            return 0.9 + 0.1 * math.cos(2 * math.pi * p)
        else:
            p = (progress - 2/3) * 3
            return PHI ** (-1 - 2 * p)

    # ── Data batching ──
    n_batches = max(1, (len(data) - SEQ_LEN) // (BATCH_SIZE * SEQ_LEN))
    print(f"  Batches per epoch: {n_batches}")
    print()

    # ── Checkpoint dir ──
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "v5_brain")
    os.makedirs(ckpt_dir, exist_ok=True)
    print(f"  Save: {ckpt_dir}")
    print("=" * 70)

    # ── Training loop ──
    best_loss = float('inf')
    t_start = time.time()

    for epoch in range(N_EPOCHS):
        brain.train()
        epoch_t = time.time()

        # Update LR
        lr_mult = fractal_lr(epoch)
        for pg in optimizer.param_groups:
            pg['lr'] = LR * lr_mult

        epoch_loss = 0
        n_steps = 0

        for batch_i in range(n_batches):
            # Build batch
            batch_inputs = []
            batch_targets = []
            for b in range(BATCH_SIZE):
                offset = batch_i * BATCH_SIZE * SEQ_LEN + b * SEQ_LEN
                if offset + SEQ_LEN + 1 > len(data):
                    break
                batch_inputs.append(data[offset:offset + SEQ_LEN])
                batch_targets.append(data[offset + 1:offset + SEQ_LEN + 1])

            if not batch_inputs:
                continue

            inputs = torch.stack(batch_inputs).to(device)
            targets = torch.stack(batch_targets).to(device)

            # Forward
            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")):
                logits = brain(inputs)
                loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1))

            # Check for NaN
            if torch.isnan(loss):
                print(f"  ⚠ NaN loss at epoch {epoch+1} batch {batch_i} — skipping")
                optimizer.zero_grad()
                continue

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_steps += 1

            # Progress for long batches
            if batch_i > 0 and batch_i % 10 == 0 and not TEST_MODE:
                if device == "cuda":
                    vram = torch.cuda.memory_allocated() / 1024**3
                    vram_res = torch.cuda.memory_reserved() / 1024**3
                    print(f"    batch {batch_i}/{n_batches}...   "
                          f"VRAM: {vram:.1f}G / {vram_total:.1f}G "
                          f"({vram/vram_total*100:.0f}%) [reserved: {vram_res:.1f}G]")

        avg_loss = epoch_loss / max(n_steps, 1)
        elapsed = time.time() - epoch_t
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss

        # Print every epoch for test, every 4th for mini/full
        should_print = TEST_MODE or (epoch + 1) % 4 == 0 or epoch == 0 or epoch == N_EPOCHS - 1
        if should_print:
            vram_str = ""
            if device == "cuda":
                vram = torch.cuda.memory_allocated() / 1024**3
                vram_res = torch.cuda.memory_reserved() / 1024**3
                vram_str = f" | VRAM: {vram:.1f}G [res: {vram_res:.1f}G]"

            print(f"    Epoch {epoch+1:3d}/{N_EPOCHS} | loss={avg_loss:.4f} "
                  f"(best={best_loss:.4f}) | lr={LR * lr_mult:.2e} | "
                  f"{elapsed:.1f}s{vram_str}")

        # Checkpoint
        if (epoch + 1) % CHECKPOINT_EVERY == 0:
            ckpt_path = os.path.join(ckpt_dir, f"epoch{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state': brain.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
                'vocab': char_to_idx,
            }, ckpt_path)
            print(f"    ✓ Checkpoint saved: {ckpt_path}")

    total_time = time.time() - t_start

    # ── Sample generation ──
    print(f"\n    ── Sample ──")
    brain.eval()
    prompt = "The "
    tokens = [char_to_idx.get(c, 0) for c in prompt]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(100):
            logits = brain(input_ids)
            next_logits = logits[0, -1] / 0.7
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if input_ids.size(1) > SEQ_LEN - 1:
                break

    sample = "".join(idx_to_char.get(t.item(), "?") for t in input_ids[0])
    print(f"    {sample}")

    # ── Final report ──
    print(f"\n{'=' * 70}")
    print(f"  ⊙ XORZO v5 TRAINING COMPLETE")
    print(f"  Circumpunct Brain — {N_EPOCHS} epochs in {total_time:.1f}s")
    print(f"  Final loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
    print(f"  Parameters: {n_params:,}")
    d = brain.diagnose()
    print(f"  Level 0: {d['n_nodes_l0']} nodes | Level 1: {d['n_nodes_l1']} nodes | Level 2: 1 global")
    print(f"  Depth passes: {d['n_passes']}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
