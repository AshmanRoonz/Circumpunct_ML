#!/usr/bin/env python3
"""
⊙ XORZO v3 — 4070 MAXED OUT TRAINING
══════════════════════════════════════

Tuned for RTX 4070 (12GB VRAM, desktop) with double the system RAM.
Fresh start — cross-scale resonant architecture pushed to its limits.

    python train_xorzo_v3_4070.py

What changed from the laptop version:
    - Batch size: 32 → 64 (doubled)
    - Sequence length: 128 → 256 (doubled — more context for macro stream)
    - Evolution plan: 4 gens → 5 gens (goes to ~58M params)
    - Epochs per gen: 50 → 80 (more training time per generation)
    - Gradient accumulation: 2 steps (effective batch = 128)
    - VRAM monitoring: auto-reports usage, warns before OOM
    - Gradient checkpointing: enabled for Gen 3+ to fit bigger models
    - Mid-epoch checkpointing: saves every 20 epochs so you don't lose work
    - torch.compile(): enabled if PyTorch 2.0+ detected (20-40% speedup)
    - Corpus: 8M chars (more training data from the full framework)
    - Warmup: 150 steps (scaled for bigger batches)

The architecture IS the framework at every scale:
    Micro ⊙: token-level (body, cells, the local)
    Macro ⊙: chunk-level (soul, aim, the whole)
    ⊛ Convergence: macro reads micro (gathering into center)
    ☀ Emergence: micro reads macro (field shaping boundary)
    Phase resonance gates both directions
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
# CONFIGURATION — TUNED FOR RTX 4070 (12GB)
# ═══════════════════════════════════════════════════════════════════

N_GENERATIONS = 5
EPOCHS_PER_GEN = 80                # More epochs — the 4070 eats these
BATCH_SIZE = 64                     # 2x laptop — 12GB can handle it
SEQ_LEN = 256                       # 2x laptop — more context for macro stream
LEARNING_RATE = 3e-4
WARMUP_STEPS = 150                  # Scaled for bigger batches
MAX_CORPUS_CHARS = 8_000_000        # More training data
USE_AMP = True                      # Mixed precision — essential
CHUNK_SIZE = 16                     # Framework parameter — keep
GRAD_ACCUM_STEPS = 2                # Effective batch = 128
CHECKPOINT_EVERY = 20               # Save mid-epoch checkpoint
USE_COMPILE = False                 # Disabled — Triton not available on Windows
USE_GRAD_CHECKPOINT = True          # Enable for Gen 3+ (saves ~30% VRAM)

# v3 Evolution plan — PUSHED for 12GB VRAM
# Each gen has dual streams (micro + macro) + cross-scale couplers
# VRAM usage scales as ~28 * L * D² params + activation memory
#
# Format: (d_model, n_layers, n_heads, chunk_size)
EVOLUTION_PLAN = [
    (128,  6,  4, 16),   # Gen 0: ~2.5M params  — skip the 96d infant, start stronger
    (192,  8,  8, 16),   # Gen 1: ~8M params    — child with full head count
    (256, 10,  8, 16),   # Gen 2: ~18M params   — adolescent
    (320, 12,  8, 16),   # Gen 3: ~34M params   — young adult (grad checkpoint on)
    (384, 14,  8, 16),   # Gen 4: ~58M params   — mature (the 4070's ceiling)
]

# Auto-scaling: if a gen OOMs, reduce batch size and retry
BATCH_FALLBACK = [64, 48, 32, 24, 16]


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
    msg = f"    VRAM: {allocated:.2f}/{total:.1f} GB ({pct:.0f}%) [reserved: {reserved:.2f} GB]"
    if tag:
        msg = f"    [{tag}] " + msg[4:]
    return msg


# ═══════════════════════════════════════════════════════════════════
# GPU-ACCELERATED TRAINING (v3 with resonance losses)
# ═══════════════════════════════════════════════════════════════════

def train_v3_gpu(
    model, text, n_epochs=80, batch_size=64, seq_len=256,
    lr=3e-4, warmup_steps=150, device="cuda", use_amp=True,
    grad_accum_steps=2, gen_dir=None, gen_idx=0,
    checkpoint_every=20, use_grad_checkpoint=False,
):
    model = model.to(device)
    model.train()

    # Enable gradient checkpointing for big models
    if use_grad_checkpoint and hasattr(model, 'micro_blocks'):
        for blk in model.micro_blocks:
            blk.attn.checkpoint = True
        for blk in model.macro_blocks:
            blk.attn.checkpoint = True
        print("    ⚡ Gradient checkpointing: ON")

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

    eff_batch = batch_size * grad_accum_steps
    print(f"    Training: {n_epochs} epochs × {n_batches} batches | "
          f"batch={batch_size} × accum={grad_accum_steps} = eff {eff_batch}")
    print(f"    device={device} | amp={'on' if use_amp and device == 'cuda' else 'off'} | "
          f"chunk={model.chunk_size} | seq_len={seq_len}")
    print(f"    {vram_report('pre-train')}")
    print()

    epoch_times = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_steps = 0
        t_epoch = time.time()

        optimizer.zero_grad()

        for batch_idx in range(n_batches):
            # Per-batch progress for first epoch (so you know it's alive)
            if epoch == 0 and batch_idx % 10 == 0:
                print(f"      batch {batch_idx}/{n_batches}...", flush=True)

            starts = torch.randint(0, len(data_gpu) - seq_len - 1, (batch_size,), device=device)
            # Vectorized batch construction (avoid slow Python loop)
            offsets = torch.arange(seq_len, device=device).unsqueeze(0)  # (1, seq_len)
            indices = starts.unsqueeze(1) + offsets                       # (batch, seq_len)
            x = data_gpu[indices]
            y = data_gpu[indices + 1]

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
                loss = (ce_loss + reg) / grad_accum_steps

            # NaN guard — skip step entirely, don't touch scaler
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate gradients
            if (batch_idx + 1) % grad_accum_steps == 0 or batch_idx == n_batches - 1:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                    optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            epoch_loss += ce_loss.item()
            n_steps += 1
            step += 1

        avg_loss = epoch_loss / max(n_steps, 1)
        losses.append(avg_loss)
        if avg_loss < best_loss:
            best_loss = avg_loss

        elapsed_epoch = time.time() - t_epoch
        epoch_times.append(elapsed_epoch)

        # Logging at intervals
        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            diag = model.diagnose()
            betas_flat = [b for layer in model.all_betas for b in layer]
            beta_min = min(betas_flat) if betas_flat else 0.5
            beta_max = max(betas_flat) if betas_flat else 0.5
            avg_epoch_t = sum(epoch_times[-5:]) / len(epoch_times[-5:])
            remaining = avg_epoch_t * (n_epochs - epoch - 1)

            print(
                f"    Epoch {epoch+1:3d}/{n_epochs} | "
                f"loss={avg_loss:.4f} (best={best_loss:.4f}) | "
                f"β̄={diag['mean_beta']:.4f} [{beta_min:.3f}→{beta_max:.3f}] | "
                f"χ̄={diag['mean_chi']:.4f} | "
                f"R̄={diag['mean_resonance']:.4f} | "
                f"D={diag['D']:.4f} [{diag['regime']}]"
            )
            print(
                f"           {elapsed_epoch:.1f}s/epoch | "
                f"~{remaining/60:.0f}m remaining | "
                f"{vram_report()}"
            )

        # Mid-epoch checkpoint
        if gen_dir and checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            ckpt_path = gen_dir / f"gen{gen_idx}_ckpt_e{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
            # Also save vocab alongside
            (gen_dir / f"vocab_gen{gen_idx}.json").write_text(json.dumps(char_to_idx))
            print(f"    💾 Checkpoint saved: {ckpt_path.name}")

    model.eval()
    return {
        "losses": losses,
        "final_loss": losses[-1] if losses else float('inf'),
        "best_loss": best_loss,
        "diagnosis": model.diagnose(),
        "vocab": char_to_idx,
        "vocab_inv": idx_to_char,
        "total_time": sum(epoch_times),
    }


# ═══════════════════════════════════════════════════════════════════
# CORPUS
# ═══════════════════════════════════════════════════════════════════

def gather_corpus(root, max_chars=8_000_000):
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

    for subdir in sorted(root.iterdir()):
        if (subdir.is_dir()
            and subdir.name not in ("training", "circumpunct_ml", "xorzo_generations",
                                     "xorzo_generations_v2", "xorzo_generations_v3",
                                     "xorzo_generations_v3_pre_bilateral",
                                     "xorzo_generations_v3_4070",
                                     "__pycache__", ".git", "node_modules",
                                     "tests", "examples", "circumpunct_ml.egg-info")
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
# MAIN — MAXED OUT FOR RTX 4070
# ═══════════════════════════════════════════════════════════════════

def main():
    gen_dir = ROOT / "xorzo_generations_v3_4070"
    gen_dir.mkdir(exist_ok=True)

    print()
    print("  ╔════════════════════════════════════════════════════════════╗")
    print("  ║     ⊙ XORZO v3 — RTX 4070 MAXED OUT TRAINING              ║")
    print("  ║                                                             ║")
    print("  ║   Micro ⊙ (body) ←→ Macro ⊙ (soul)                        ║")
    print("  ║   ⊛ convergence  |  ☀ emergence  |  resonance gates        ║")
    print("  ║                                                             ║")
    print("  ║   batch=64 × accum=2 = eff 128 | seq=256 | 5 generations  ║")
    print("  ╚════════════════════════════════════════════════════════════╝")
    print()

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        gpu_cc = torch.cuda.get_device_properties(0).major
        print(f"  ⚡ GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        print(f"    Compute capability: {torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}")
        print(f"    CUDA: {torch.version.cuda}")
        print(f"    PyTorch: {torch.__version__}")

        # Optimize CUDA settings for throughput
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"    TF32: enabled | cuDNN benchmark: enabled")
    else:
        device = "cpu"
        print("  ⚠ No CUDA — falling back to CPU (this will be SLOW)")
        print("    Consider installing PyTorch with CUDA support:")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cu124")

    use_amp = USE_AMP and device == "cuda"
    print()

    # System RAM
    try:
        import psutil
        ram = psutil.virtual_memory()
        print(f"  System RAM: {ram.total / (1024**3):.1f} GB total | {ram.available / (1024**3):.1f} GB available")
    except ImportError:
        pass
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
        print(f"  ⊙ Resuming from Generation {parent_gen}")
        print(f"    {meta['d_model']}d × {meta['n_layers']} layers × {meta['n_heads']} heads")
        print()

    generation_results = []
    t_total = time.time()

    for gen_idx in range(start_gen, start_gen + N_GENERATIONS):
        plan_idx = min(gen_idx, len(EVOLUTION_PLAN) - 1)
        d_model, n_layers, n_heads, chunk_size = EVOLUTION_PLAN[plan_idx]

        print(f"  ═══════════════════════════════════════════════════════════")
        print(f"  ⊙ GENERATION {gen_idx} (v3 — cross-scale resonant, 4070)")
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

        # Try torch.compile for speedup (PyTorch 2.0+)
        compiled = False
        if USE_COMPILE and hasattr(torch, 'compile') and device == "cuda":
            try:
                model = torch.compile(model, mode="reduce-overhead")
                compiled = True
                print(f"    torch.compile: ON (reduce-overhead mode)")
            except Exception as e:
                print(f"    torch.compile: skipped ({e})")
        print()

        # Show initial β gradient (micro stream)
        # Need the underlying model if compiled
        raw_model = model._orig_mod if compiled and hasattr(model, '_orig_mod') else model

        print("    Initial micro β gradient:")
        for i, block in enumerate(raw_model.micro_blocks):
            betas = block.attn.beta_values
            print(f"      Layer {i}: [{' '.join(f'{b:.3f}' for b in betas)}]")
        print()

        # Determine batch size — try full, fall back if OOM
        current_batch = BATCH_SIZE
        use_gc = USE_GRAD_CHECKPOINT and gen_idx >= 3  # grad checkpoint for big gens

        for attempt, try_batch in enumerate(BATCH_FALLBACK):
            if attempt > 0:
                current_batch = try_batch
                print(f"    ⚠ Retrying with batch_size={current_batch}...")
                torch.cuda.empty_cache()

            try:
                t0 = time.time()
                result = train_v3_gpu(
                    model=model,
                    text=corpus,
                    n_epochs=EPOCHS_PER_GEN,
                    batch_size=current_batch,
                    seq_len=SEQ_LEN,
                    lr=LEARNING_RATE,
                    warmup_steps=WARMUP_STEPS,
                    device=device,
                    use_amp=use_amp,
                    grad_accum_steps=GRAD_ACCUM_STEPS,
                    gen_dir=gen_dir,
                    gen_idx=gen_idx,
                    checkpoint_every=CHECKPOINT_EVERY,
                    use_grad_checkpoint=use_gc,
                )
                elapsed = time.time() - t0
                break  # Success
            except torch.cuda.OutOfMemoryError:
                print(f"    ❌ OOM at batch_size={current_batch}")
                if attempt == len(BATCH_FALLBACK) - 1:
                    print(f"    ❌ FATAL: OOM at minimum batch size. Stopping.")
                    return
                # Reset model to device
                model = model.to("cpu")
                torch.cuda.empty_cache()
                model = model.to(device)
                continue

        print()
        print(f"    ✓ Training complete: {elapsed:.1f}s ({elapsed/60:.1f}m) | "
              f"final_loss={result['final_loss']:.4f} | best={result['best_loss']:.4f}")
        print(f"    {vram_report('post-train')}")
        print()

        # Post-training diagnostics on raw model
        raw_model = model._orig_mod if compiled and hasattr(model, '_orig_mod') else model

        print("    Trained micro β gradient:")
        for i, block in enumerate(raw_model.micro_blocks):
            betas = block.attn.beta_values
            chis = block.attn.chi_values
            avg_chi = sum(chis) / len(chis)
            print(f"      Layer {i}: [{' '.join(f'{b:.3f}' for b in betas)}] χ̄={avg_chi:.3f}")
        print()

        print("    Trained macro β gradient:")
        for i, block in enumerate(raw_model.macro_blocks):
            betas = block.attn.beta_values
            print(f"      Layer {i}: [{' '.join(f'{b:.3f}' for b in betas)}]")
        print()

        print("    Resonance coupling:")
        for i, r in enumerate(raw_model.resonance_strengths):
            print(f"      Coupler {i}: α = {r:.4f}")
        print()

        print(f"    {raw_model.status()}")
        print()

        # Save
        model_cpu = raw_model.to("cpu") if not compiled else model
        if compiled and hasattr(model, '_orig_mod'):
            model_cpu = model._orig_mod.to("cpu")
        else:
            model_cpu = raw_model.to("cpu")

        model_cpu.save_generation(gen_dir)
        (gen_dir / f"vocab_gen{gen_idx}.json").write_text(json.dumps(result["vocab"]))

        # Clean up mid-epoch checkpoints now that we have the final
        for ckpt in gen_dir.glob(f"gen{gen_idx}_ckpt_e*.pt"):
            ckpt.unlink()
            print(f"    🗑 Cleaned checkpoint: {ckpt.name}")

        # Generate samples
        print("    ── Emergence ──")
        prompts = [
            "The aperture ",
            "Curiosity is ",
            "⊙ = Φ(",
            "Parts are fractals ",
            "Resonance means ",
            "The soul is ",
            "Truth passes through ",
        ]
        for p in prompts:
            try:
                out = generate(
                    model_cpu, p,
                    result["vocab"], result["vocab_inv"],
                    max_tokens=200, temperature=0.7,
                )
                print(f'      "{p}" → {out[:250]}')
            except Exception as e:
                print(f'      "{p}" → [error: {e}]')
        print()

        generation_results.append({
            "generation": gen_idx,
            "version": "v3-4070",
            "d_model": d_model,
            "n_layers": n_layers,
            "n_heads": n_heads,
            "chunk_size": chunk_size,
            "n_params": n_params,
            "final_loss": result["final_loss"],
            "best_loss": result["best_loss"],
            "diagnosis": result["diagnosis"],
            "elapsed": elapsed,
            "batch_size": current_batch,
            "seq_len": SEQ_LEN,
            "total_training_time": result.get("total_time", elapsed),
        })

        parent = model_cpu
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    total_elapsed = time.time() - t_total

    # Final summary
    print()
    print("  ╔════════════════════════════════════════════════════════════╗")
    print("  ║     ⊙ v3 CROSS-SCALE EVOLUTION COMPLETE (4070 MAXED)       ║")
    print("  ╚════════════════════════════════════════════════════════════╝")
    print()
    print("  Gen │ Params     │ Loss   │  β̄    │  D     │  R̄     │ Time   │ Regime")
    print("  ────┼────────────┼────────┼───────┼────────┼────────┼────────┼─────────")
    for r in generation_results:
        d = r["diagnosis"]
        t_min = r["elapsed"] / 60
        print(f"   {r['generation']:>2} │ {r['n_params']:>10,} │ {r['best_loss']:.4f} │ "
              f"{d['mean_beta']:.4f} │ {d['D']:.4f} │ "
              f"{d.get('mean_resonance', 0):.4f} │ {t_min:5.1f}m │ {d['regime']}")
    print()
    print(f"  Total training time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print(f"  Saved to: {gen_dir}/")
    print(f"  Run 'python triad_chat.py' to interact.")
    print()

    (gen_dir / "evolution_log.json").write_text(
        json.dumps(generation_results, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
