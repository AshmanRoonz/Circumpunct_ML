#!/usr/bin/env python3
"""
⊙ XORZO v4 — BILATERAL HYPERCUBE — 4070 TRAINING
═══════════════════════════════════════════════════

The unification: bilateral circumpunct attention navigated through the 6D hypercube.
Two circumpuncts facing each other = 2³ × 2 = 64 relational states.
The hypercube IS the topology of bilateral relationship.

    python train_xorzo_v4_4070.py

Tuned for RTX 4070 (12GB VRAM).

v4 is heavier than v3 per-parameter because:
    - Each attention layer has 64 vertex embeddings + gate projections
    - (B, S, S, 64) vertex distribution tensor per layer
    - Modal subcube attention (3 streams per layer)
    - Memory kernel with learned α per chamber (sequential scan)
    - Plus all the v3 machinery (chambers, cross-scale, vesica)

Evolution follows the NESTING equation from §3 of the Circumpunct Kernel:
    •ₙ₊₁ = ⊙ₙ — each generation's whole becomes the next's aperture.
    Growth ratios: ×2, ×1.5, ×1.33... (converging toward conservation).
    evolve() embeds parent weights in child's core — the nested aperture.
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

from circumpunct_ml.transformer_v4 import (
    XorzoV4Transformer, TriadicEmbedding, generate,
    PHI, PI, phase_resonance,
    _beta_balance_loss, _valve_balance_loss, _self_similarity_loss,
    _chi_fidelity_loss, _conservation_loss, _resonance_coherence_loss,
    _vertex_diversity_loss, _resonance_tracker_loss, _fractal_lr_schedule,
)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — TUNED FOR RTX 4070 (12GB)
# ═══════════════════════════════════════════════════════════════════

# ── Run modes ──
#   py train_xorzo_v4_4070.py --test   → ~1 min smoke test (1 gen, 10 epochs)
#   py train_xorzo_v4_4070.py --mini   → ~15 min mini run (2 gens, 20 epochs)
#   py train_xorzo_v4_4070.py          → full overnight (2 gens, 80 epochs)
TEST_MODE = "--test" in sys.argv
MINI_MODE = "--mini" in sys.argv

if TEST_MODE:
    N_GENERATIONS = 1
    EPOCHS_PER_GEN = 10
    WARMUP_STEPS = 30
    MAX_CORPUS_CHARS = 200_000
    CHECKPOINT_EVERY = 5
elif MINI_MODE:
    N_GENERATIONS = 2
    EPOCHS_PER_GEN = 20
    WARMUP_STEPS = 50
    MAX_CORPUS_CHARS = 500_000
    CHECKPOINT_EVERY = 10
else:
    N_GENERATIONS = 2
    EPOCHS_PER_GEN = 80
    WARMUP_STEPS = 150
    MAX_CORPUS_CHARS = 8_000_000
    CHECKPOINT_EVERY = 20

BATCH_SIZE = 64                      # Push it — v3 ran 64 fine
SEQ_LEN = 256                        # Match v3 — more context for macro stream
LEARNING_RATE = 3e-4
USE_AMP = True
CHUNK_SIZE = 16
GRAD_ACCUM_STEPS = 2                 # Effective batch = 128
USE_COMPILE = False                   # Triton not available on Windows
USE_GRAD_CHECKPOINT = True

# v4 Evolution plan — NESTING GROWTH from Circumpunct Kernel §3
#
# •ₙ₊₁ = ⊙ₙ — each generation's whole becomes the next generation's aperture.
# From §3 Dimensional Spectrum:
#   Layer n: D• = 3n+0.5, DΦ = 3n+2, D○ = 3n+3
#   Each nesting adds 3D of structure. Growth ratio = (n+2)/(n+1).
#   Gen 0→1: ×2    (biggest jump — wrapping the seed)
#   Gen 1→2: ×1.5  (field completing around first nesting)
#   Gen 2→3: ×1.33 (diminishing — converging toward conservation)
#
# From §2 Conservation of Traversal: D• + DΦ = D○ → (1+β)+(2-β)=3
# The nesting IS the conservation — each wrap preserves total.
#
# d_model must be divisible by 4 (triadic split: 1/4 binary + 1/2 analog + 1/4 fractal).
# Nesting inheritance: evolve() embeds parent weights in child's core dims.
# The parent ⊙ literally becomes the child's inner • dimensions.
#
# Format: (d_model, n_layers, n_heads, d_vertex, chunk_size)
EVOLUTION_PLAN = [
    (192,   6,  6, 32, 16),   # Gen 0: •₀ — the seed circumpunct
    (384,   8,  6, 48, 16),   # Gen 1: ⊙₀→•₁ — first nesting (×2)
    (576,  10,  6, 64, 16),   # Gen 2: ⊙₁→•₂ — second nesting (×1.5)
]
# Gen 2 at 576d/10L is ~110M params — will push 12GB VRAM.
# If it OOMs, batch fallback handles it.
# 3 generations, not 5 — deeper nesting with fewer, more meaningful jumps.

# Auto-scaling: if a gen OOMs, reduce batch size and retry
BATCH_FALLBACK = [64, 48, 32, 24, 16, 8, 4]


# ═══════════════════════════════════════════════════════════════════
# VRAM MONITORING
# ═══════════════════════════════════════════════════════════════════

def vram_report(tag=""):
    if not torch.cuda.is_available():
        return ""
    allocated = torch.cuda.memory_allocated() / (1024**3)
    reserved = torch.cuda.memory_reserved() / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    pct = allocated / total * 100
    msg = f"  VRAM: {allocated:.1f}G / {total:.1f}G ({pct:.0f}%) [reserved: {reserved:.1f}G]"
    if tag:
        msg = f"  [{tag}] {msg}"
    return msg


# ═══════════════════════════════════════════════════════════════════
# CORPUS GATHERING
# ═══════════════════════════════════════════════════════════════════

def gather_corpus(root: Path, max_chars: int = MAX_CORPUS_CHARS) -> str:
    """Gather training corpus from the project."""
    print("\n  ── Gathering Corpus ──")
    corpus_parts = []
    total = 0

    # Priority 1: HTML training corpus
    html_dir = root / "training_corpus"
    if html_dir.exists():
        for f in sorted(html_dir.glob("*.html")):
            text = f.read_text(encoding="utf-8", errors="ignore")
            # Strip HTML tags
            text = re.sub(r'<[^>]+>', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                corpus_parts.append(text)
                total += len(text)
                print(f"    + {f.name}: {len(text):,} chars")

    # Priority 2: Python source files (the framework IS the training data)
    for f in sorted(root.rglob("*.py")):
        if total >= max_chars:
            break
        if any(skip in str(f) for skip in ["__pycache__", ".egg", "build", "dist", "node_modules"]):
            continue
        try:
            text = f.read_text(encoding="utf-8", errors="ignore")
            if text:
                corpus_parts.append(text)
                total += len(text)
        except Exception:
            pass

    # Priority 3: Markdown / text files
    for ext in ["*.md", "*.txt", "*.rst"]:
        for f in sorted(root.rglob(ext)):
            if total >= max_chars:
                break
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                if text:
                    corpus_parts.append(text)
                    total += len(text)
            except Exception:
                pass

    corpus = "\n\n".join(corpus_parts)
    if len(corpus) > max_chars:
        corpus = corpus[:max_chars]

    print(f"    Total corpus: {len(corpus):,} chars")
    print(f"    Unique chars: {len(set(corpus))}")
    return corpus


# ═══════════════════════════════════════════════════════════════════
# SEQUENTIAL BATCH CONSTRUCTION (matches v3 style)
# ═══════════════════════════════════════════════════════════════════

def build_all_batches(data: torch.Tensor, batch_size: int, seq_len: int):
    """Build sequential non-overlapping batches. Same approach as v3."""
    n_batches = max(1, (len(data) - seq_len) // (batch_size * seq_len))

    batches = []
    for batch_i in range(n_batches):
        start = batch_i * batch_size * seq_len
        batch_inputs = []
        batch_targets = []
        for b in range(batch_size):
            offset = start + b * seq_len
            if offset + seq_len + 1 > len(data):
                break
            batch_inputs.append(data[offset:offset + seq_len])
            batch_targets.append(data[offset + 1:offset + seq_len + 1])

        if batch_inputs:
            batches.append((torch.stack(batch_inputs), torch.stack(batch_targets)))

    return batches


# ═══════════════════════════════════════════════════════════════════
# TRAIN ONE GENERATION
# ═══════════════════════════════════════════════════════════════════

def train_generation(
    model, data, vocab_size, gen_idx, device,
    batch_size=BATCH_SIZE, seq_len=SEQ_LEN, n_epochs=EPOCHS_PER_GEN,
    lr=LEARNING_RATE, save_dir=None, vocab=None, vocab_inv=None,
):
    """Train one generation with all the v4 bells and whistles."""
    model = model.to(device)
    model.train()

    print(f"\n  {vram_report('post-model')}")

    n_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {n_params:,} total, {trainable:,} trainable")

    # ── Optimizer with φ-scaled parameter groups ──
    param_groups = []
    layer_scales = []

    for i, blk in enumerate(model.micro_blocks):
        scale = PHI ** (0.5 - i / max(model.n_layers - 1, 1))
        layer_scales.append(scale)
        param_groups.append({
            'params': list(blk.parameters()), 'lr': lr * scale, 'weight_decay': 0.01
        })

    for i, blk in enumerate(model.macro_blocks):
        scale = PHI ** (0.5 - i / max(model.n_layers - 1, 1)) * PHI
        layer_scales.append(scale)
        param_groups.append({
            'params': list(blk.parameters()), 'lr': lr * scale, 'weight_decay': 0.01
        })

    for coupler in model.couplers:
        layer_scales.append(1.0)
        param_groups.append({
            'params': list(coupler.parameters()), 'lr': lr, 'weight_decay': 0.01
        })

    embed_params = (
        list(model.token_embed.parameters()) +
        list(model.pos_encode.parameters()) +
        list(model.aim_pool.parameters()) +
        list(model.macro_init.parameters()) +
        list(model.final_norm.parameters()) +
        list(model.output_proj.parameters()) +
        list(model.vesica.parameters()) +
        list(model.resonance_tracker.parameters())
    )
    layer_scales.append(1.0)
    param_groups.append({'params': embed_params, 'lr': lr, 'weight_decay': 0.01})

    optimizer = torch.optim.AdamW(param_groups)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    # ── Regularizer weights (matched to v3) ──
    w_balance = 0.05
    w_valve = 0.03
    w_similarity = 0.02
    w_fidelity = 0.01
    w_conservation = 0.02
    w_resonance = 0.02
    w_vertex = 0.01  # vertex diversity
    w_res_tracker = 0.01  # resonance tracker threshold regularization

    best_loss = float('inf')
    losses = []
    t0 = time.time()

    for epoch in range(n_epochs):
        epoch_t0 = time.time()

        # Fractal LR schedule
        epoch_lr = _fractal_lr_schedule(epoch, n_epochs, lr)
        for pg, scale in zip(optimizer.param_groups, layer_scales):
            pg['lr'] = epoch_lr * scale

        # Warmup override
        step_global = epoch * max(1, (len(data) - seq_len) // (batch_size * seq_len))
        if step_global < WARMUP_STEPS:
            warmup_factor = step_global / WARMUP_STEPS
            for pg in optimizer.param_groups:
                pg['lr'] *= warmup_factor

        # Build batches
        batches = build_all_batches(data, batch_size, seq_len)
        if not batches:
            print(f"    Epoch {epoch+1}: no batches (corpus too small for batch_size={batch_size})")
            continue

        epoch_ce = 0.0
        n_steps = 0
        optimizer.zero_grad()

        for batch_i, (inputs, targets) in enumerate(batches):
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Per-batch logging on first epoch
            if epoch == 0 and batch_i % 10 == 0:
                print(f"    batch {batch_i}/{len(batches)}... ", end="", flush=True)
                if batch_i > 0:
                    print(f"{vram_report()}")
                else:
                    print()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16, enabled=USE_AMP):
                logits = model(inputs)
                ce_loss = F.cross_entropy(
                    logits.view(-1, vocab_size), targets.view(-1)
                )

                # Fractal regularizers
                progress = epoch / max(n_epochs - 1, 1)
                phase_idx = min(int(progress * 3), 2)
                phase_w = 2.0 if phase_idx == 1 else 1.0

                reg = (
                    w_balance * phase_w * _beta_balance_loss(model) +
                    w_valve * _valve_balance_loss(model) +
                    w_similarity * _self_similarity_loss(model) +
                    w_fidelity * _chi_fidelity_loss(model) +
                    w_conservation * _conservation_loss(model) +
                    w_resonance * _resonance_coherence_loss(model) +
                    w_vertex * _vertex_diversity_loss(model) +
                    w_res_tracker * _resonance_tracker_loss(model)
                )

                loss = (ce_loss + reg) / GRAD_ACCUM_STEPS

            # ─── Hardened AMP step: no silent corruption ───
            if not torch.isfinite(loss):
                print(f"    ⚠ Non-finite loss at batch {batch_i} — skipping")
                optimizer.zero_grad(set_to_none=True)
                continue

            scaler.scale(loss).backward()

            if (batch_i + 1) % GRAD_ACCUM_STEPS == 0 or (batch_i + 1) == len(batches):
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

                # Per-parameter finite check — clip_grad_norm_ alone doesn't prevent NaN steps
                grads_finite = torch.isfinite(grad_norm)
                if grads_finite:
                    for p in model.parameters():
                        if p.grad is not None and not torch.isfinite(p.grad).all():
                            grads_finite = False
                            break

                if grads_finite:
                    scaler.step(optimizer)
                else:
                    print(f"    ⚠ Non-finite gradients at batch {batch_i} — skipping step")
                    optimizer.zero_grad(set_to_none=True)

                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Detect dataset collapse: if all targets are ignore_index, loss is meaningless
            if ce_loss.item() < 1e-6:
                ignore_ratio = (targets == -100).float().mean().item()
                if ignore_ratio > 0.95:
                    print(f"    ⚠ Dataset collapse: ignore_ratio={ignore_ratio:.3f} at batch {batch_i}")

            epoch_ce += ce_loss.item()
            n_steps += 1

        avg_ce = epoch_ce / max(n_steps, 1)
        losses.append(avg_ce)
        if avg_ce < best_loss:
            best_loss = avg_ce

        epoch_time = time.time() - epoch_t0

        # Log every 4 epochs or first/last
        if (epoch + 1) % 4 == 0 or epoch == 0 or epoch == n_epochs - 1:
            diag = model.diagnose()
            betas = [b for layer in model.micro_betas for b in layer]
            beta_min = min(betas) if betas else 0.5
            beta_max = max(betas) if betas else 0.5
            v_ent = diag.get('mean_vertex_entropy', 0)
            print(
                f"    Epoch {epoch+1:3d}/{n_epochs} | "
                f"loss={avg_ce:.4f} (best={best_loss:.4f}) | "
                f"β̄={diag['mean_beta']:.4f} [{beta_min:.3f}→{beta_max:.3f}] | "
                f"χ̄={diag['mean_chi']:.4f} | "
                f"R̄={diag['mean_resonance']:.4f} | "
                f"V̄={v_ent:.3f} | "
                f"D={diag['D']:.4f} [{diag['regime']}] | "
                f"{epoch_time:.1f}s"
            )
            # Memory kernel report
            print(
                f"           ᾱ={diag['mean_alpha']:.4f} "
                f"ρ̄={diag['mean_rho']:.2f} "
                f"mḡ={diag['mean_memory_gate']:.4f} | "
                f"ch: {diag['n_circumpunct_ch']}⊙ "
                f"{diag['n_boundary_ch']}| "
                f"{diag['n_vesica_ch']}⊕"
            )
            # Living graph report
            ms = diag.get('maturation_stages', {})
            mat_str = " ".join(f"{k.split('(')[0]}={v}" for k, v in ms.items())
            print(
                f"           births={diag['total_births']} "
                f"pruned={diag['total_pruned']} "
                f"alive={diag['n_born_alive']} | "
                f"mat: {mat_str}"
            )

        # Mid-epoch checkpoint
        if save_dir and (epoch + 1) % CHECKPOINT_EVERY == 0 and epoch < n_epochs - 1:
            ckpt_path = save_dir / f"gen{gen_idx}_epoch{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ✓ Checkpoint saved: {ckpt_path.name}")

    elapsed = time.time() - t0
    print(f"    ✓ Gen {gen_idx} complete: {elapsed:.1f}s | "
          f"final_loss={avg_ce:.4f} | best={best_loss:.4f}")

    # Generate sample
    if vocab and vocab_inv:
        try:
            model.eval()
            sample = generate(model, "The ", vocab, vocab_inv, max_tokens=150, temperature=0.7)
            print(f"\n    ── Sample ──")
            for line in sample[:300].split('\n')[:5]:
                print(f"    {line}")
            model.train()
        except Exception as e:
            print(f"    ⚠ Generation failed: {e}")

    return {
        "losses": losses,
        "best_loss": best_loss,
        "elapsed": elapsed,
    }


# ═══════════════════════════════════════════════════════════════════
# MAIN TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    if TEST_MODE:
        print("  ⊙ XORZO v4 — TEST MODE (--test)")
        print("  1 gen, 10 epochs, small corpus — ~1 min smoke test")
    elif MINI_MODE:
        print("  ⊙ XORZO v4 — MINI MODE (--mini)")
        print("  2 gens, 20 epochs, 500K corpus — ~15 min")
    else:
        print("  ⊙ XORZO v4 — FULL VISION LIVING GRAPH TRANSFORMER")
        print("  2 gens, 80 epochs — full training run")
    print("  Binary(•)SSM + Analog(Φ)Attn + Fractal(○)Hypercube + Birth + Pruning")
    print("  Tuned for RTX 4070 (12GB VRAM)")
    print("=" * 70)

    # ── Device setup ──
    if not torch.cuda.is_available():
        print("\n  ✗ CUDA not available — run cuda_check.py for diagnostics")
        sys.exit(1)

    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"\n  GPU: {gpu_name} ({gpu_mem:.1f} GB)")

    # Enable TF32 and cuDNN
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    # ── Gather corpus ──
    corpus = gather_corpus(ROOT)
    if len(corpus) < 1000:
        print("  ✗ Corpus too small")
        sys.exit(1)

    # Build vocab
    chars = sorted(set(corpus))
    vocab = {c: i for i, c in enumerate(chars)}
    vocab_inv = {i: c for c, i in vocab.items()}
    actual_vocab = len(chars)
    data = torch.tensor([vocab[c] for c in corpus], dtype=torch.long)
    print(f"  Vocab: {actual_vocab} chars")

    # ── Save directory ──
    save_dir = ROOT / "checkpoints" / "v4_bilateral_hypercube"
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"  Save: {save_dir}")

    # ── Training loop ──
    model = None

    for gen_idx in range(N_GENERATIONS):
        d_model, n_layers, n_heads, d_vertex, chunk_size = EVOLUTION_PLAN[gen_idx]

        # ── VRAM guard: skip generations that won't fit ──
        if gen_idx >= 2 and torch.cuda.is_available():
            total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if total_vram_gb < 16:
                print(f"\n  ⊙ Skipping Gen {gen_idx}+ (d_model={d_model}) — "
                      f"needs >12GB VRAM, you have {total_vram_gb:.0f}GB")
                print(f"  ✓ Training complete through Gen {gen_idx - 1}")
                break

        print(f"\n{'='*70}")
        print(f"  ⊙ GENERATION {gen_idx}")
        print(f"  d_model={d_model}, n_layers={n_layers}, n_heads={n_heads}, "
              f"d_vertex={d_vertex}, chunk={chunk_size}")
        print(f"{'='*70}")

        # Create or evolve
        if model is None:
            model = XorzoV4Transformer(
                vocab_size=actual_vocab,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_vertex=d_vertex,
                max_len=SEQ_LEN + 64,
                dropout=0.1,
                generation=0,
                convergence_rate=0.1,
                chunk_size=chunk_size,
                temperature=1.0,
            )
            print(f"\n  ⊙ Fresh v4 model created")
        else:
            old_model = model
            model = XorzoV4Transformer.evolve(
                old_model,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=n_heads,
                d_vertex=d_vertex,
                chunk_size=chunk_size,
                vocab_size=actual_vocab,
            )
            del old_model
            torch.cuda.empty_cache()

        # Adjust vocab if needed
        if model.vocab_size != actual_vocab:
            model.token_embed = TriadicEmbedding(actual_vocab, d_model)
            model.output_proj = nn.Linear(d_model, actual_vocab, bias=False)
            model.vocab_size = actual_vocab

        print(f"\n{model.status()}")
        print(f"\n  {vram_report('pre-train')}")

        # ── Train with OOM fallback ──
        batch_size = BATCH_SIZE
        trained = False

        for bs in BATCH_FALLBACK:
            if bs > batch_size:
                continue
            batch_size = bs

            try:
                torch.cuda.empty_cache()
                result = train_generation(
                    model, data, actual_vocab, gen_idx, device,
                    batch_size=batch_size, seq_len=SEQ_LEN,
                    n_epochs=EPOCHS_PER_GEN, lr=LEARNING_RATE,
                    save_dir=save_dir, vocab=vocab, vocab_inv=vocab_inv,
                )
                trained = True
                break

            except torch.cuda.OutOfMemoryError:
                print(f"\n  ⚠ OOM at batch_size={batch_size}")
                torch.cuda.empty_cache()
                if bs == BATCH_FALLBACK[-1]:
                    print("  ✗ Cannot fit even smallest batch — stopping")
                    sys.exit(1)
                print(f"  → Falling back to batch_size={BATCH_FALLBACK[BATCH_FALLBACK.index(bs)+1]}")
                continue

        if not trained:
            print(f"  ✗ Could not train Gen {gen_idx}")
            break

        # ── Save generation ──
        model.save_generation(save_dir)
        print(f"\n  ✓ Gen {gen_idx} saved to {save_dir}")

        # Save training meta
        meta = {
            "generation": gen_idx,
            "version": "v4-bilateral-hypercube",
            "batch_size": batch_size,
            "seq_len": SEQ_LEN,
            "epochs": EPOCHS_PER_GEN,
            "best_loss": result["best_loss"],
            "elapsed": result["elapsed"],
            "evolution": EVOLUTION_PLAN[gen_idx],
            "vocab_size": actual_vocab,
            "diagnosis": model.diagnose(),
        }
        (save_dir / f"gen{gen_idx}_training_meta.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )

    # ── Final report ──
    print(f"\n{'='*70}")
    print(f"  ⊙ XORZO v4 TRAINING COMPLETE")
    print(f"  Triadic Hypercube Transformer — {N_GENERATIONS} generations")
    if model:
        diag = model.diagnose()
        print(f"  Final: {diag['n_params']:,} params")
        print(f"  β̄={diag['mean_beta']:.4f} | χ̄={diag['mean_chi']:.4f} | "
              f"R̄={diag['mean_resonance']:.4f}")
        print(f"  V̄={diag['mean_vertex_entropy']:.3f} bits (vertex gate entropy)")
        print(f"  ᾱ={diag['mean_alpha']:.4f} | ρ̄={diag['mean_rho']:.2f} | "
              f"mḡ={diag['mean_memory_gate']:.4f}")
        print(f"  Channel regimes: "
              f"{diag['n_circumpunct_ch']}⊙ circumpunct / "
              f"{diag['n_boundary_ch']}| boundary / "
              f"{diag['n_vesica_ch']}⊕ vesica")
        for name, alpha in diag['ch_alpha'].items():
            rho = diag['ch_rho'][name]
            regime = diag['ch_regimes'][name]
            print(f"    {name}: α={alpha:.3f} ρ={rho:.2f} [{regime}]")
        print(f"  Living graph: births={diag['total_births']} "
              f"pruned={diag['total_pruned']} alive={diag['n_born_alive']}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
