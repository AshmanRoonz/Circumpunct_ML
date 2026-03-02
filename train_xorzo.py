#!/usr/bin/env python3
"""
Train Xorzo — First Breath

Feeds the Circumpunct Framework texts through the transformer.
Each character passes through ⊛ → i → ☀︎.
β learns to find balance. χ learns fidelity.

Usage: python train_xorzo.py
"""

import os
import sys
import time
import json
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from circumpunct_ml.transformer import (
    XorzoTransformer, train_generation, generate
)


def gather_corpus(root: Path) -> str:
    """Gather all framework texts — the seed data."""
    texts = []

    # Framework documents (the theory)
    for f in sorted(root.glob("*.md")) + sorted(root.glob("*.html")) + sorted(root.glob("*.txt")):
        if not f.name.startswith("."):
            texts.append(f.read_text(errors="ignore"))

    # Package source (the code IS the framework)
    pkg = root / "circumpunct_ml"
    if pkg.exists():
        for f in sorted(pkg.glob("*.py")):
            texts.append(f.read_text(errors="ignore"))

    return "\n\n".join(texts)


def main():
    root = Path(__file__).parent
    gen_dir = root / "xorzo_generations"

    print()
    print("  ╔═══════════════════════════════════════╗")
    print("  ║         ⊙ XORZO — FIRST BREATH        ║")
    print("  ║           Training Generation 0         ║")
    print("  ╚═══════════════════════════════════════╝")
    print()

    # Gather training corpus
    print("  Gathering framework texts...")
    corpus = gather_corpus(root)
    print(f"  Corpus: {len(corpus):,} characters, {len(set(corpus))} unique")
    print()

    # Create model
    model = XorzoTransformer(
        vocab_size=256,     # Will be resized to actual vocab
        d_model=128,        # Hidden dimension
        n_layers=6,         # 6 circumpunct blocks
        n_heads=8,          # 8 apertures per block
        max_len=512,        # Context window
        dropout=0.1,
        generation=0,
    )

    print("  ⊙ Before training:")
    print(f"  {model.status()}")
    print()

    # Train
    print("  ═══ TRAINING ═══")
    t0 = time.time()

    result = train_generation(
        model=model,
        text=corpus,
        n_epochs=20,        # 20 epochs for generation 0
        batch_size=32,
        seq_len=128,        # 128-char context
        lr=3e-4,
        device="cpu",
    )

    elapsed = time.time() - t0
    print()
    print(f"  Training complete in {elapsed:.1f}s")
    print(f"  Final loss: {result['final_loss']:.4f}")
    print()

    # Post-training status
    print("  ⊙ After training:")
    print(f"  {model.status()}")
    print()

    # Save generation
    model.save_generation(gen_dir)
    # Also save vocab mapping
    (gen_dir / "vocab.json").write_text(json.dumps(result["vocab"]))
    print(f"  Saved to {gen_dir}/")
    print()

    # Generate some text
    print("  ═══ EMERGENCE — First Words ═══")
    print()

    prompts = [
        "⊙ = Φ(",
        "The aperture ",
        "β = 0.5 means ",
        "Curiosity is ",
    ]

    for prompt in prompts:
        output = generate(
            model=model,
            prompt=prompt,
            vocab=result["vocab"],
            vocab_inv=result["vocab_inv"],
            max_tokens=150,
            temperature=0.8,
        )
        print(f"  Prompt: '{prompt}'")
        print(f"  → {output[:200]}")
        print()

    print("  ─────────────────────────────────────────")
    print("  ⊙ Generation 0 complete. Xorzo has taken its first breath.")
    print()


if __name__ == "__main__":
    main()
