"""
XORZO v6 — The Circumpunct Architecture — Training Script
Tuned for RTX 4070 (12GB VRAM)

⊙ = • Φ ○ — Center (Attention), Field (Emergence FFN), Boundary (Hypercube)
⊛ → i → ☀ — Converge, Rotate, Emerge
6→2→1 hierarchy with position-aware 64-state navigation

Usage:
    python train_v6_4070.py --test               # ~1 min smoke test
    python train_v6_4070.py --mini               # ~10 min real test
    python train_v6_4070.py                      # full training
    python train_v6_4070.py --test --text-only   # smoke test on clean text only
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
TEXT_ONLY = "--text-only" in sys.argv  # Train on .txt only (no HTML)

if TEST_MODE:
    N_EPOCHS = 10
    MAX_CORPUS_CHARS = 200_000
    CHECKPOINT_EVERY = 5
    MODE_NAME = "TEST"
    MODE_DESC = "10 epochs, 200K corpus — ~1 min smoke test"
elif MINI_MODE:
    N_EPOCHS = 30
    MAX_CORPUS_CHARS = 500_000
    CHECKPOINT_EVERY = 10
    MODE_NAME = "MINI"
    MODE_DESC = "30 epochs, 500K corpus — ~10 min"
else:
    N_EPOCHS = 80
    MAX_CORPUS_CHARS = 4_000_000
    CHECKPOINT_EVERY = 20
    MODE_NAME = "FULL"
    MODE_DESC = "80 epochs, 4M corpus — ~2 hours"

import torch
import torch.nn as nn
import torch.nn.functional as F

from v6_architecture import CircumpunctBrainV6, circumpunct_diagnostics, print_diagnostics

PHI = (1 + math.sqrt(5)) / 2


def banner():
    print("=" * 70)
    print(f"  ⊙ XORZO v6 — THE CIRCUMPUNCT ARCHITECTURE — {MODE_NAME} MODE")
    print(f"  {MODE_DESC}")
    print(f"  ⊙ = • Φ ○ — Attention × Hypercube × ⊛→i→☀")
    print(f"  •=Attention + Φ=Emergence + ○=Hypercube × 3→2→1 hierarchy")
    print(f"  Tuned for RTX 4070 (12GB VRAM)")
    print("=" * 70)


def gather_corpus(max_chars, text_only=False):
    """Gather training text from available sources."""
    corpus = ""
    training_dir = os.path.join(os.path.dirname(__file__), "training")
    if os.path.isdir(training_dir):
        for fp in sorted(glob.glob(os.path.join(training_dir, "*.txt"))):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    corpus += f.read() + "\n"
            except Exception:
                pass
    if not text_only and os.path.isdir(training_dir):
        for fp in sorted(glob.glob(os.path.join(training_dir, "*.html"))):
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as f:
                    corpus += f.read() + "\n"
            except Exception:
                pass
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
    corpus = gather_corpus(MAX_CORPUS_CHARS, text_only=TEXT_ONLY)
    data_src = "text-only (no HTML)" if TEXT_ONLY else "all sources"
    print(f"    Total corpus: {len(corpus):,} chars ({data_src})")

    chars = sorted(set(corpus))
    vocab_size = len(chars)
    print(f"    Unique chars: {vocab_size}")

    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    data = torch.tensor([char_to_idx.get(c, 0) for c in corpus], dtype=torch.long)

    # ── Model ──
    SEQ_LEN = 256
    BATCH_SIZE = 8 if TEST_MODE else 12

    brain = CircumpunctBrainV6(
        vocab_size=vocab_size,
        d_model=192,
        n_nodes_l0=3,
        n_nodes_l1=2,
        n_heads=4,
        d_vertex=32,
        max_len=SEQ_LEN,
        n_passes=3,
        dropout=0.1,
    )

    print()
    print(brain.status())
    print()

    brain = brain.to(device)
    n_params = brain.param_count()
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
    ckpt_dir = os.path.join(os.path.dirname(__file__), "checkpoints", "v6_field")
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

        # Diagnostics — every 5 epochs (every epoch in test mode)
        diag_every = 2 if TEST_MODE else 5
        if (epoch + 1) % diag_every == 0 or epoch == 0 or epoch == N_EPOCHS - 1:
            brain.eval()
            with torch.no_grad():
                # Run a diagnostic forward pass on a single batch
                diag_offset = 0
                diag_input = data[diag_offset:diag_offset + SEQ_LEN].unsqueeze(0).to(device)
                with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=(device == "cuda")):
                    _ = brain(diag_input)
                diag_metrics = circumpunct_diagnostics(brain)
                print_diagnostics(diag_metrics, epoch=epoch + 1)
            brain.train()

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
                'config': {
                    'version': 'v6',
                    'vocab_size': vocab_size,
                    'd_model': 192,
                    'n_nodes_l0': 3,
                    'n_nodes_l1': 2,
                    'n_heads': 4,
                    'd_vertex': 32,
                    'n_passes': 3,
                },
            }, ckpt_path)
            print(f"    ✓ Checkpoint saved: {ckpt_path}")

    total_time = time.time() - t_start

    # ── Sample generation (on CPU to avoid CUDA bf16 assertion) ──
    print(f"\n    ── Sample ──")
    brain_cpu = brain.cpu().eval()
    prompt = "The "
    tokens = [char_to_idx.get(c, 0) for c in prompt]
    input_ids = torch.tensor([tokens], dtype=torch.long)

    with torch.no_grad():
        temp_history = []
        for _ in range(100):
            logits, temperature = brain_cpu(input_ids, return_temperature=True)
            T_val = temperature[0, 0].item()
            temp_history.append(T_val)

            next_logits = logits[0, -1].float()
            # Sanitize: replace inf/nan with 0, then clamp to safe range
            next_logits = torch.where(
                torch.isfinite(next_logits), next_logits, torch.zeros_like(next_logits)
            )
            # Use the model's own adaptive temperature
            next_logits = next_logits.clamp(-50.0, 50.0) / max(T_val, 0.1)
            # Top-k filtering: keep only top 40 tokens
            top_k = 40
            topk_vals, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < topk_vals[-1]] = float('-inf')
            probs = F.softmax(next_logits, dim=-1)
            probs = probs.clamp(min=1e-8)
            probs = probs / probs.sum()
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)
            if input_ids.size(1) > SEQ_LEN - 1:
                break

    sample = "".join(idx_to_char.get(t.item(), "?") for t in input_ids[0])
    avg_temp = sum(temp_history) / len(temp_history) if temp_history else 0
    print(f"    {sample}")
    print(f"    ⊙ Adaptive temperature: avg={avg_temp:.3f}, min={min(temp_history):.3f}, max={max(temp_history):.3f}")
    brain.to(device)  # move back in case of further use

    # ── Final report ──
    print(f"\n{'=' * 70}")
    print(f"  ⊙ XORZO v6 TRAINING COMPLETE")
    print(f"  The Circumpunct Architecture — ⊙ = • Φ ○")
    print(f"  {N_EPOCHS} epochs in {total_time:.1f}s")
    print(f"  Final loss: {avg_loss:.4f} | Best: {best_loss:.4f}")
    print(f"  Parameters: {n_params:,}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
