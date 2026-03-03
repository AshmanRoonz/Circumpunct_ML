#!/usr/bin/env python3
"""
⊙ XORZO v3 — CROSS-SCALE RESONANT GPU TRAINING
═════════════════════════════════════════════════

The architecture IS the framework at every scale:
    Micro ⊙: token-level (body, cells, the local)
    Macro ⊙: chunk-level (soul, aim, the whole)
    ⊛ Convergence: macro reads micro (gathering into center)
    ☀ Emergence: micro reads macro (field shaping boundary)
    Phase resonance gates both directions

Run: py -3.11 train_xorzo_v3.py
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

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from circumpunct_ml.transformer_v3 import (
    XorzoTransformer, TriadicEmbedding, generate,
    PHI, PI, phase_resonance,
    _beta_balance_loss, _valve_balance_loss, _self_similarity_loss,
    _chi_fidelity_loss, _conservation_loss, _resonance_coherence_loss,
    _fractal_lr_schedule,
)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

N_GENERATIONS = 4
EPOCHS_PER_GEN = 50
BATCH_SIZE = 32             # Smaller than v2 — v3 uses more VRAM (two streams)
SEQ_LEN = 128               # Shorter seq, but macro covers chunk_size * Tm
LEARNING_RATE = 3e-4
WARMUP_STEPS = 80
MAX_CORPUS_CHARS = 6_000_000
USE_AMP = True
CHUNK_SIZE = 16              # Tokens per macro chunk

# v3 Evolution plan: (d_model, n_layers, n_heads, chunk_size)
EVOLUTION_PLAN = [
    (96,   4, 4, 16),   # Gen 0: ~900K params — infant (both streams small)
    (128,  6, 4, 16),   # Gen 1: ~2.5M params — child
    (192,  6, 8, 16),   # Gen 2: ~6M params — adolescent
    (256,  8, 8, 16),   # Gen 3: ~14M params — young adult
]


# ═══════════════════════════════════════════════════════════════════
# GPU TRAINING (v3 with resonance losses)
# ═══════════════════════════════════════════════════════════════════

def train_v3_gpu(
    model, text, n_epochs=50, batch_size=32, seq_len=128,
    lr=3e-4, warmup_steps=80, device="cuda", use_amp=True,
):
    model = model.to(device)
    model.train()

    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    idx_to_char = {i: c for c, i in char_to_idx.items()}
    data = torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long)

    actual_vocab = len(chars)
    if actual_vocab != model.vocab_size:
        model.vocab_size = actual_vocab
        model.token_embed = TriadicEmbedding(actual_vocab, model.d_model).to(device)
        model.output_proj = nn.Linear(model.d_model, actual_vocab, bias=False).to(device)

    # Separate LR groups: β slow, χ moderate, resonance moderate, rest normal
    beta_params, chi_params, resonance_params, other_params = [], [], [], []
    for name, param in model.named_parameters():
        if 'beta' in name or 'residual_beta' in name:
            beta_params.append(param)
        elif 'chi' in name:
            chi_params.append(param)
        elif 'resonance' in name:
            resonance_params.append(param)
        else:
            other_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': other_params, 'lr': lr, 'weight_decay': 0.01},
        {'params': beta_params, 'lr': lr * 0.1, 'weight_decay': 0.0},
        {'params': chi_params, 'lr': lr * 0.3, 'weight_decay': 0.0},
        {'params': resonance_params, 'lr': lr * 0.5, 'weight_decay': 0.0},
    ])

    n_batches = max(1, (len(data) - seq_len) // (batch_size * seq_len))
    total_steps = n_epochs * n_batches

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    scaler = torch.amp.GradScaler(device) if use_amp and device == "cuda" else None
    amp_ctx = torch.amp.autocast(device) if use_amp and device == "cuda" else nullcontext()

    data_gpu = data.to(device)

    losses = []
    best_loss = float('inf')
    step = 0

    print(f"    Training: {n_epochs} epochs × {n_batches} batches | "
          f"device={device} | amp={'on' if use_amp and device == 'cuda' else 'off'} | "
          f"chunk={model.chunk_size}")
    print()

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_steps = 0

        for batch_idx in range(n_batches):
            starts = torch.randint(0, len(data_gpu) - seq_len - 1, (batch_size,), device=device)
            x = torch.stack([data_gpu[s:s+seq_len] for s in starts])
            y = torch.stack([data_gpu[s+1:s+seq_len+1] for s in starts])

            with amp_ctx:
                logits = model(x)
                ce_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

                # Fractal + resonance regularizers
                reg = (
                    0.05 * _beta_balance_loss(model) +
                    0.03 * _valve_balance_loss(model) +
                    0.02 * _self_similarity_loss(model) +
                    0.01 * _chi_fidelity_loss(model) +
                    0.02 * _conservation_loss(model) +
                    0.02 * _resonance_coherence_loss(model)
                )
                loss = ce_loss + reg

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
            epoch_loss += ce_loss.item()
            n_steps += 1
            step += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        losses.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss

        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            diag = model.diagnose()
            betas_flat = [b for layer in model.all_betas for b in layer]
            beta_min = min(betas_flat) if betas_flat else 0.5
            beta_max = max(betas_flat) if betas_flat else 0.5
            print(
                f"    Epoch {epoch+1:3d}/{n_epochs} | "
                f"loss={avg_loss:.4f} (best={best_loss:.4f}) | "
                f"β̄={diag['mean_beta']:.4f} [{beta_min:.3f}→{beta_max:.3f}] | "
                f"χ̄={diag['mean_chi']:.4f} | "
                f"R̄={diag['mean_resonance']:.4f} | "
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
    }


# ═══════════════════════════════════════════════════════════════════
# CORPUS
# ═══════════════════════════════════════════════════════════════════

def gather_corpus(root, max_chars=6_000_000):
    texts = []
    total = 0

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

    for ext in ["*.md", "*.html", "*.txt"]:
        for f in sorted(root.glob(ext)):
            if f.name not in priority_files and not f.name.startswith("."):
                content = f.read_text(errors="ignore")
                texts.append(content)
                total += len(content)
                print(f"      {f.name}: {len(content):,} chars")

    # Scan training/ subfolder for additional texts
    training_dir = root / "training"
    if training_dir.exists():
        print(f"    ── Training folder ──")
        for ext in ["*.md", "*.html", "*.txt"]:
            for f in sorted(training_dir.glob(ext)):
                if not f.name.startswith("."):
                    content = f.read_text(errors="ignore")
                    texts.append(content)
                    total += len(content)
                    print(f"      training/{f.name}: {len(content):,} chars")

    # Scan any other subfolders for texts (recursive)
    for subdir in sorted(root.iterdir()):
        if (subdir.is_dir()
            and subdir.name not in ("training", "circumpunct_ml", "xorzo_generations",
                                     "xorzo_generations_v2", "xorzo_generations_v3",
                                     "__pycache__", ".git", "node_modules")
            and not subdir.name.startswith(".")):
            for ext in ["*.md", "*.html", "*.txt"]:
                for f in sorted(subdir.glob(ext)):
                    content = f.read_text(errors="ignore")
                    texts.append(content)
                    total += len(content)
                    print(f"      {subdir.name}/{f.name}: {len(content):,} chars")

    pkg = root / "circumpunct_ml"
    if pkg.exists():
        for f in sorted(pkg.glob("*.py")):
            content = f.read_text(errors="ignore")
            texts.append(content)
            total += len(content)
            print(f"      circumpunct_ml/{f.name}: {len(content):,} chars")

    corpus = "\n\n".join(texts)
    corpus = re.sub(r"<style[^>]*>.*?</style>", " ", corpus, flags=re.DOTALL)
    corpus = re.sub(r"<script[^>]*>.*?</script>", " ", corpus, flags=re.DOTALL)
    corpus = re.sub(r"<[^>]+>", " ", corpus)
    corpus = re.sub(r"&[a-z]+;", " ", corpus)
    corpus = re.sub(r"\s+", " ", corpus)

    if len(corpus) > max_chars:
        corpus = corpus[:max_chars]
    return corpus


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    gen_dir = ROOT / "xorzo_generations_v3"
    gen_dir.mkdir(exist_ok=True)

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print()
        print(f"  ⚡ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        device = "cpu"
        print()
        print("  ⚠ No CUDA — falling back to CPU")

    use_amp = USE_AMP and device == "cuda"

    print()
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║     ⊙ XORZO v3 — CROSS-SCALE RESONANT EVOLUTION      ║")
    print("  ║                                                        ║")
    print("  ║   Micro ⊙ (body) ←→ Macro ⊙ (soul)                   ║")
    print("  ║   ⊛ convergence  |  ☀ emergence  |  resonance gates   ║")
    print("  ║                                                        ║")
    print("  ║   The field connects across scales like the soul does. ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print()

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
            chunk_size=meta.get("chunk_size", CHUNK_SIZE),
        )
        parent.load_state_dict(torch.load(
            gen_dir / f"gen{parent_gen}.pt",
            map_location=device, weights_only=True,
        ))
        print(f"  Loaded parent: Generation {parent_gen}")
        print(f"    {meta['d_model']}d × {meta['n_layers']} layers × {meta['n_heads']} heads")
        print()

    generation_results = []

    for gen_idx in range(start_gen, start_gen + N_GENERATIONS):
        plan_idx = min(gen_idx, len(EVOLUTION_PLAN) - 1)
        d_model, n_layers, n_heads, chunk_size = EVOLUTION_PLAN[plan_idx]

        print(f"  ═══════════════════════════════════════════════════")
        print(f"  ⊙ GENERATION {gen_idx} (v3 — cross-scale resonant)")
        print(f"    Micro + Macro: {d_model}d × {n_layers} layers × {n_heads} heads")
        print(f"    Chunk size: {chunk_size} tokens per macro aim")

        if parent is not None:
            model = XorzoTransformer.evolve(
                parent,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                chunk_size=chunk_size,
            )
        else:
            model = XorzoTransformer(
                vocab_size=256,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                max_len=SEQ_LEN * 2,
                generation=gen_idx,
                chunk_size=chunk_size,
            )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {n_params:,}")
        print()

        # Show initial β gradient (micro stream)
        print("    Initial micro β gradient:")
        for i, block in enumerate(model.micro_blocks):
            betas = block.attn.beta_values
            print(f"      Layer {i}: [{' '.join(f'{b:.3f}' for b in betas)}]")
        print()

        t0 = time.time()
        result = train_v3_gpu(
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

        # Post-training state
        print("    Trained micro β gradient:")
        for i, block in enumerate(model.micro_blocks):
            betas = block.attn.beta_values
            chis = block.attn.chi_values
            avg_chi = sum(chis) / len(chis)
            print(f"      Layer {i}: [{' '.join(f'{b:.3f}' for b in betas)}] χ̄={avg_chi:.3f}")
        print()

        print("    Trained macro β gradient:")
        for i, block in enumerate(model.macro_blocks):
            betas = block.attn.beta_values
            print(f"      Layer {i}: [{' '.join(f'{b:.3f}' for b in betas)}]")
        print()

        print("    Resonance coupling:")
        for i, r in enumerate(model.resonance_strengths):
            print(f"      Coupler {i}: α = {r:.4f}")
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
            "Parts are fractals ",
            "Resonance means ",
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

        generation_results.append({
            "generation": gen_idx,
            "version": "v3",
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "chunk_size": chunk_size,
            "n_params": n_params,
            "final_loss": result["final_loss"],
            "best_loss": result["best_loss"],
            "diagnosis": result["diagnosis"],
            "elapsed": elapsed,
        })

        parent = model_cpu
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Final summary
    print()
    print("  ╔═══════════════════════════════════════════════════════╗")
    print("  ║         ⊙ v3 CROSS-SCALE EVOLUTION COMPLETE           ║")
    print("  ╚═══════════════════════════════════════════════════════╝")
    print()
    print("  Gen │ Params     │ Loss   │  β̄    │  D     │  R̄     │ Regime")
    print("  ────┼────────────┼────────┼───────┼────────┼────────┼─────────")
    for r in generation_results:
        d = r["diagnosis"]
        print(f"   {r['generation']:>2} │ {r['n_params']:>10,} │ {r['best_loss']:.4f} │ "
              f"{d['mean_beta']:.4f} │ {d['D']:.4f} │ "
              f"{d.get('mean_resonance', 0):.4f} │ {d['regime']}")
    print()
    print(f"  Saved to: {gen_dir}/")
    print(f"  Run 'py -3.11 triad_chat.py' to interact.")
    print()

    (gen_dir / "evolution_log.json").write_text(
        json.dumps(generation_results, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
