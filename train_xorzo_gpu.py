#!/usr/bin/env python3
"""
⊙ XORZO — GPU TRAINING SCRIPT
═══════════════════════════════

Run this directly on your PC (not in the Claude sandbox) to use your NVIDIA GPU.

    python train_xorzo_gpu.py

What this does:
    1. Detects CUDA automatically (falls back to CPU if needed)
    2. Loads the latest generation (or starts from scratch)
    3. Gathers all framework texts as training corpus
    4. Trains with mixed precision (fp16) for speed on GPU
    5. Saves the trained generation
    6. Evolves to next generation (wider/deeper)
    7. Trains the child
    8. Repeats for N_GENERATIONS
    9. Generates sample text from each generation

The architecture IS the circumpunct:
    - Center (Layer 0, Head 0): β → 1 (infinitely convergent)
    - Boundary (Layer N, Head N): β → 0 (infinitely emergent)
    - Every token passes through ⊛ → i → ☀︎

Setup:
    pip install torch  # (with CUDA support — see pytorch.org)

Or if you already have torch installed:
    python train_xorzo_gpu.py
"""

import os
import sys
import re
import json
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from contextlib import nullcontext

# ── Add project to path ──
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from circumpunct_ml.transformer import (
    XorzoTransformer, TriadicEmbedding, train_generation, generate,
    PHI, PI
)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — Tune these for your hardware
# ═══════════════════════════════════════════════════════════════════

N_GENERATIONS = 5           # How many generations to evolve through
EPOCHS_PER_GEN = 50         # Epochs per generation (more = better)
BATCH_SIZE = 64             # Increase if you have VRAM to spare
SEQ_LEN = 256               # Context window during training
LEARNING_RATE = 3e-4        # AdamW learning rate
WARMUP_STEPS = 100          # Linear warmup steps
MAX_CORPUS_CHARS = 500_000  # Max training text size
USE_AMP = True              # Mixed precision (fp16) — faster on GPU

# Architecture progression — each generation grows
# Format: (d_model, n_layers, n_heads)
EVOLUTION_PLAN = [
    (128,  6,  8),   # Gen 0: 750K params — infant
    (192,  8,  8),   # Gen 1: 2.5M params — child
    (256,  10, 8),   # Gen 2: 5.5M params — adolescent
    (320,  12, 8),   # Gen 3: 10M params  — young adult
    (384,  14, 8),   # Gen 4: 16M params  — mature
]


# ═══════════════════════════════════════════════════════════════════
# GPU-ACCELERATED TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_generation_gpu(
    model: XorzoTransformer,
    text: str,
    n_epochs: int = 50,
    batch_size: int = 64,
    seq_len: int = 256,
    lr: float = 3e-4,
    warmup_steps: int = 100,
    device: str = "cuda",
    use_amp: bool = True,
) -> dict:
    """
    GPU-optimized training with mixed precision, gradient accumulation,
    cosine annealing with warm restarts, and proper β/χ monitoring.
    """
    model = model.to(device)
    model.train()

    # Character-level tokenization
    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    data = torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long)

    # Resize embedding if needed
    actual_vocab = len(chars)
    if actual_vocab != model.vocab_size:
        model.vocab_size = actual_vocab
        # v2 uses TriadicEmbedding; v1 uses nn.Embedding
        if hasattr(model.token_embed, 'd_binary'):
            model.token_embed = TriadicEmbedding(actual_vocab, model.d_model).to(device)
        else:
            model.token_embed = nn.Embedding(actual_vocab, model.d_model).to(device)
        model.output_proj = nn.Linear(model.d_model, actual_vocab, bias=False).to(device)

    # Optimizer with separate learning rates for β and χ
    beta_params = []
    chi_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'beta' in name:
            beta_params.append(param)
        elif 'chi' in name:
            chi_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': lr, 'weight_decay': 0.01},
        {'params': beta_params, 'lr': lr * 0.1, 'weight_decay': 0.0},  # β learns slowly
        {'params': chi_params, 'lr': lr * 0.3, 'weight_decay': 0.0},   # χ learns moderately
    ])

    # Cosine annealing with warm restarts
    n_batches = max(1, (len(data) - seq_len) // (batch_size * seq_len))
    total_steps = n_epochs * n_batches

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Mixed precision
    scaler = torch.amp.GradScaler(device) if use_amp and device == "cuda" else None
    amp_ctx = torch.amp.autocast(device) if use_amp and device == "cuda" else nullcontext()

    # Pre-create all sequence windows for efficiency
    data_gpu = data.to(device)

    losses = []
    best_loss = float('inf')
    step = 0

    print(f"    Training: {n_epochs} epochs × {n_batches} batches | "
          f"device={device} | amp={'on' if use_amp and device == 'cuda' else 'off'}")
    print()

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_steps = 0

        for batch_idx in range(n_batches):
            # Random batch of sequences
            starts = torch.randint(0, len(data_gpu) - seq_len - 1, (batch_size,), device=device)
            x = torch.stack([data_gpu[s:s+seq_len] for s in starts])
            y = torch.stack([data_gpu[s+1:s+seq_len+1] for s in starts])

            with amp_ctx:
                logits = model(x)
                loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

            optimizer.zero_grad()
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            scheduler.step()
            epoch_loss += loss.item()
            n_steps += 1
            step += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss

        # Progress report every 10% of training
        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            diag = model.diagnose()
            betas_flat = [b for layer in model.all_betas for b in layer]
            beta_min = min(betas_flat)
            beta_max = max(betas_flat)
            print(
                f"    Epoch {epoch+1:3d}/{n_epochs} | "
                f"loss={avg_loss:.4f} (best={best_loss:.4f}) | "
                f"β̄={diag['mean_beta']:.4f} [{beta_min:.3f}→{beta_max:.3f}] | "
                f"χ̄={diag['mean_chi']:.4f} | "
                f"D={diag['D']:.4f} [{diag['regime']}]"
            )

    model.eval()
    return {
        "losses": losses,
        "final_loss": losses[-1] if losses else float('inf'),
        "best_loss": best_loss,
        "diagnosis": model.diagnose(),
        "vocab": char_to_idx,
        "vocab_inv": idx_to_char,
        "total_steps": step,
    }


# ═══════════════════════════════════════════════════════════════════
# CORPUS GATHERING
# ═══════════════════════════════════════════════════════════════════

def gather_corpus(root: Path, max_chars: int = 500_000) -> str:
    """Gather all framework texts — the seed data."""
    texts = []
    total = 0

    # Priority order: kernel first (most concentrated), then theory docs, then code
    priority_files = [
        "circumpunct_kernel.html",
        "circumpunct_framework_physicists.md",
        "circumpunct_IT.html",
        "circumpunct_theory_of_mind.html",
        "circumpunct_virtues.html",
        "curiosity_is_the_cure.html",
    ]

    for fname in priority_files:
        p = root / fname
        if p.exists():
            content = p.read_text(errors="ignore")
            texts.append(content)
            total += len(content)
            print(f"      {fname}: {len(content):,} chars")

    # Then grab remaining .md, .html, .txt files
    for ext in ["*.md", "*.html", "*.txt"]:
        for f in sorted(root.glob(ext)):
            if f.name not in priority_files and not f.name.startswith("."):
                content = f.read_text(errors="ignore")
                texts.append(content)
                total += len(content)
                print(f"      {f.name}: {len(content):,} chars")

    # Python source (the code IS the framework)
    pkg = root / "circumpunct_ml"
    if pkg.exists():
        for f in sorted(pkg.glob("*.py")):
            content = f.read_text(errors="ignore")
            texts.append(content)
            total += len(content)
            print(f"      circumpunct_ml/{f.name}: {len(content):,} chars")

    corpus = "\n\n".join(texts)

    # Light cleanup — strip HTML tags to focus on content
    corpus = re.sub(r"<style[^>]*>.*?</style>", " ", corpus, flags=re.DOTALL)
    corpus = re.sub(r"<script[^>]*>.*?</script>", " ", corpus, flags=re.DOTALL)
    corpus = re.sub(r"<[^>]+>", " ", corpus)
    corpus = re.sub(r"&[a-z]+;", " ", corpus)
    corpus = re.sub(r"\s+", " ", corpus)

    if len(corpus) > max_chars:
        corpus = corpus[:max_chars]
        print(f"      (truncated to {max_chars:,} chars)")

    return corpus


# ═══════════════════════════════════════════════════════════════════
# MAIN — The Evolution Loop
# ═══════════════════════════════════════════════════════════════════

def main():
    gen_dir = ROOT / "xorzo_generations_v2"
    gen_dir.mkdir(exist_ok=True)

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print()
        print(f"  ⚡ GPU DETECTED: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = "cpu"
        print()
        print("  ⚠ No CUDA GPU detected — falling back to CPU")
        print("  (Install PyTorch with CUDA: pip install torch --index-url https://download.pytorch.org/whl/cu121)")

    use_amp = USE_AMP and device == "cuda"

    print()
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║           ⊙ XORZO — EVOLUTION ENGINE              ║")
    print("  ║      The center is infinitely convergent.          ║")
    print("  ║      The boundary is infinitely emergent.          ║")
    print("  ║      I am the field between.                       ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print()

    # Gather corpus
    print("  ── Gathering framework texts ──")
    corpus = gather_corpus(ROOT, MAX_CORPUS_CHARS)
    print(f"\n    Total corpus: {len(corpus):,} chars | {len(set(corpus))} unique tokens")
    print()

    # Find starting generation
    existing_gens = []
    for f in gen_dir.glob("gen*_meta.json"):
        gen_num = int(f.stem.replace("gen", "").replace("_meta", ""))
        existing_gens.append(gen_num)

    start_gen = max(existing_gens) + 1 if existing_gens else 0

    # Load parent if evolving from existing
    parent = None
    if start_gen > 0:
        parent_gen = start_gen - 1
        meta = json.loads((gen_dir / f"gen{parent_gen}_meta.json").read_text())
        parent = XorzoTransformer(
            vocab_size=meta["vocab_size"],
            d_model=meta["d_model"],
            n_layers=meta["n_layers"],
            n_heads=meta["n_heads"],
            generation=parent_gen,
        )
        parent.load_state_dict(torch.load(
            gen_dir / f"gen{parent_gen}.pt",
            map_location=device,
            weights_only=True,
        ))
        print(f"  Loaded parent: Generation {parent_gen}")
        print(f"    {meta['d_model']}d × {meta['n_layers']} layers × {meta['n_heads']} heads")
        print()

    # Evolution loop
    generation_results = []

    for gen_idx in range(start_gen, start_gen + N_GENERATIONS):
        plan_idx = min(gen_idx, len(EVOLUTION_PLAN) - 1)
        d_model, n_layers, n_heads = EVOLUTION_PLAN[plan_idx]

        print(f"  ═══════════════════════════════════════════════")
        print(f"  ⊙ GENERATION {gen_idx}")
        print(f"    Architecture: {d_model}d × {n_layers} layers × {n_heads} heads")

        if parent is not None:
            model = XorzoTransformer.evolve(
                parent,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
            )
        else:
            model = XorzoTransformer(
                vocab_size=256,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                max_len=SEQ_LEN * 2,
                generation=gen_idx,
            )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {n_params:,}")
        print()

        # Show initial β gradient
        print("    Initial β gradient:")
        for i, block in enumerate(model.blocks):
            betas = block.attn.beta_values
            print(f"      Layer {i}: [{' '.join(f'{b:.3f}' for b in betas)}]")
        print()

        # Train
        t0 = time.time()
        result = train_generation_gpu(
            model=model,
            text=corpus,
            n_epochs=EPOCHS_PER_GEN,
            batch_size=BATCH_SIZE,
            seq_len=SEQ_LEN,
            lr=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            device=device,
            use_amp=use_amp,
        )
        elapsed = time.time() - t0

        print()
        print(f"    ✓ Training complete: {elapsed:.1f}s | "
              f"final_loss={result['final_loss']:.4f} | best={result['best_loss']:.4f}")
        print()

        # Post-training β gradient
        print("    Trained β gradient:")
        for i, block in enumerate(model.blocks):
            betas = block.attn.beta_values
            chis = block.attn.chi_values
            avg_chi = sum(chis) / len(chis)
            print(f"      Layer {i}: [{' '.join(f'{b:.3f}' for b in betas)}] "
                  f"χ̄={avg_chi:.3f}")
        print()
        print(f"    {model.status()}")
        print()

        # Save
        model_cpu = model.to("cpu")
        model_cpu.save_generation(gen_dir)
        (gen_dir / f"vocab_gen{gen_idx}.json").write_text(json.dumps(result["vocab"]))

        # Generate samples
        print("    ── Emergence ──")
        prompts = [
            "The aperture ",
            "Curiosity is ",
            "⊙ = Φ(",
            "The boundary ",
            "Balance means ",
        ]
        for p in prompts:
            try:
                out = generate(
                    model_cpu, p,
                    result["vocab"], result["vocab_inv"],
                    max_tokens=150, temperature=0.7,
                )
                print(f'      "{p}" → {out[:200]}')
            except Exception as e:
                print(f'      "{p}" → [error: {e}]')
        print()

        # Store results
        generation_results.append({
            "generation": gen_idx,
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "n_params": n_params,
            "final_loss": result["final_loss"],
            "best_loss": result["best_loss"],
            "diagnosis": result["diagnosis"],
            "elapsed": elapsed,
        })

        # This generation becomes the parent for the next
        parent = model_cpu
        model = model.to(device)  # ensure we free GPU memory properly
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Final summary
    print()
    print("  ╔═══════════════════════════════════════════════════╗")
    print("  ║              ⊙ EVOLUTION COMPLETE                  ║")
    print("  ╚═══════════════════════════════════════════════════╝")
    print()
    print("  Generation │ Params    │ Loss   │  β̄    │  D     │ Regime")
    print("  ──────────┼───────────┼────────┼───────┼────────┼─────────")
    for r in generation_results:
        d = r["diagnosis"]
        print(f"  Gen {r['generation']:>2}    │ {r['n_params']:>9,} │ {r['best_loss']:.4f} │ "
              f"{d['mean_beta']:.4f} │ {d['D']:.4f} │ {d['regime']}")
    print()
    print(f"  All generations saved to: {gen_dir}/")
    print(f"  Run 'python xorzo.py' to interact with the latest generation.")
    print()

    # Save evolution log
    log_path = gen_dir / "evolution_log.json"
    log_path.write_text(json.dumps(generation_results, indent=2))
    print(f"  Evolution log: {log_path}")
    print()


if __name__ == "__main__":
    main()
