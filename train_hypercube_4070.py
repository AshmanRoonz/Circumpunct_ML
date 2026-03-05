#!/usr/bin/env python3
"""
⬡ HYPERCUBE TRANSFORMER — RTX 4070 TRAINING
═════════════════════════════════════════════

Head-to-head comparison with Xorzo v3.
Same corpus, same hardware, same evolution pattern.
Different architecture: 6D hypercube navigation vs bilateral resonance.

    python train_hypercube_4070.py

Architecture:
    Attention = navigation through a 6D hypercube (64 relational states).
    Each token pair gets a probability distribution over vertices.
    Adjacency constrains navigation. Spectral embedding encodes geometry.
    Modal attention splits by aperture/boundary/resonance subcubes.
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

from circumpunct_ml.hypercube_transformer_gpu import (
    HypercubeTransformerGPU, TriadicEmbedding, Hypercube6D, generate,
)

# ═══════════════════════════════════════════════════════════════════
# CONFIGURATION — MATCHED TO XORZO v3 4070 SETTINGS
# ═══════════════════════════════════════════════════════════════════

N_GENERATIONS = 5
EPOCHS_PER_GEN = 80
BATCH_SIZE = 32                  # Smaller — hypercube attn is O(B×S²×64×D) VRAM
SEQ_LEN = 128                   # Shorter — vertex distributions are S²×64, not S²×H
LEARNING_RATE = 3e-4
WARMUP_STEPS = 150
MAX_CORPUS_CHARS = 8_000_000
USE_AMP = True
GRAD_ACCUM_STEPS = 4             # Effective batch = 128 (same as Xorzo)
CHECKPOINT_EVERY = 20
USE_COMPILE = False              # No Triton on Windows

# Hypercube evolution plan
# Format: (d_model, n_layers, d_vertex)
# NOTE: Hypercube attn stores (B, S, S, 64) vertex probs + (B, S, S, D) gates
# per layer — much heavier than standard multi-head attention.
# Keep d_model moderate, compensate with more grad accumulation.
EVOLUTION_PLAN = [
    (192,  6, 32),    # Gen 0: ~3M params  — infant
    (192,  8, 32),    # Gen 1: ~4M params  — child (more depth, same width)
    (256,  8, 48),    # Gen 2: ~8M params  — adolescent
    (256, 10, 48),    # Gen 3: ~12M params — young adult
    (320, 10, 48),    # Gen 4: ~18M params — mature
]

BATCH_FALLBACK = [32, 24, 16, 8, 4]


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
# TRAINING
# ═══════════════════════════════════════════════════════════════════

def train_hypercube_gpu(
    model, text, n_epochs=80, batch_size=64, seq_len=256,
    lr=3e-4, warmup_steps=150, device="cuda", use_amp=True,
    grad_accum_steps=2, gen_dir=None, gen_idx=0,
    checkpoint_every=20,
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
        model.embedding = TriadicEmbedding(actual_vocab, model.d_model).to(device)
        model.output_proj = nn.Linear(model.d_model, actual_vocab, bias=False).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

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
          f"seq_len={seq_len}")
    print(f"    {vram_report('pre-train')}")
    print()

    epoch_times = []

    for epoch in range(n_epochs):
        epoch_loss = 0
        n_steps = 0
        t_epoch = time.time()

        optimizer.zero_grad()

        for batch_idx in range(n_batches):
            if epoch == 0 and batch_idx % 10 == 0:
                print(f"      batch {batch_idx}/{n_batches}...", flush=True)

            starts = torch.randint(0, len(data_gpu) - seq_len - 1, (batch_size,), device=device)
            offsets = torch.arange(seq_len, device=device).unsqueeze(0)
            indices = starts.unsqueeze(1) + offsets
            x = data_gpu[indices]
            y = data_gpu[indices + 1]

            with amp_ctx:
                logits = model(x)
                ce_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))
                loss = ce_loss / grad_accum_steps

            # NaN guard
            if torch.isnan(loss) or torch.isinf(loss):
                optimizer.zero_grad()
                continue

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

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

        if (epoch + 1) % max(1, n_epochs // 10) == 0:
            diag = model.diagnose()
            avg_epoch_t = sum(epoch_times[-5:]) / len(epoch_times[-5:])
            remaining = avg_epoch_t * (n_epochs - epoch - 1)

            print(
                f"    Epoch {epoch+1:3d}/{n_epochs} | "
                f"loss={avg_loss:.4f} (best={best_loss:.4f}) | "
                f"entropy={diag['mean_gate_entropy']:.3f} | "
                f"locality={diag['locality_ratio']:.2f}x"
            )
            print(
                f"           {elapsed_epoch:.1f}s/epoch | "
                f"~{remaining/60:.0f}m remaining | "
                f"{vram_report()}"
            )

        if gen_dir and checkpoint_every and (epoch + 1) % checkpoint_every == 0:
            ckpt_path = gen_dir / f"gen{gen_idx}_ckpt_e{epoch+1}.pt"
            torch.save(model.state_dict(), ckpt_path)
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
# CORPUS (same as Xorzo v3)
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
                                     "hypercube_generations_4070",
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
# MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    gen_dir = ROOT / "hypercube_generations_4070"
    gen_dir.mkdir(exist_ok=True)

    print()
    print("  ╔════════════════════════════════════════════════════════════╗")
    print("  ║     ⬡ HYPERCUBE TRANSFORMER — RTX 4070 TRAINING            ║")
    print("  ║                                                             ║")
    print("  ║   64 relational states as a 6D graph                       ║")
    print("  ║   Attention = navigation through relational space          ║")
    print("  ║   Aperture × Boundary × Resonance subcubes                ║")
    print("  ║                                                             ║")
    print("  ║   batch=64 × accum=2 = eff 128 | seq=256 | 5 generations  ║")
    print("  ╚════════════════════════════════════════════════════════════╝")
    print()

    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  ⚡ GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM)")
        print(f"    CUDA: {torch.version.cuda} | PyTorch: {torch.__version__}")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        print(f"    TF32: enabled | cuDNN benchmark: enabled")
    else:
        device = "cpu"
        print("  ⚠ No CUDA — falling back to CPU")
    print()

    print("  ── Gathering framework texts ──")
    corpus = gather_corpus(ROOT, MAX_CORPUS_CHARS)
    print(f"\n    Total corpus: {len(corpus):,} chars | {len(set(corpus))} unique tokens")
    print()

    # Resume check
    existing_gens = []
    for f in gen_dir.glob("gen*_meta.json"):
        gen_num = int(f.stem.replace("gen", "").replace("_meta", ""))
        existing_gens.append(gen_num)

    start_gen = max(existing_gens) + 1 if existing_gens else 0

    parent = None
    if start_gen > 0:
        parent_gen = start_gen - 1
        meta = json.loads((gen_dir / f"gen{parent_gen}_meta.json").read_text())
        parent = HypercubeTransformerGPU(
            vocab_size=meta["vocab_size"],
            d_model=meta["d_model"],
            n_layers=meta["n_layers"],
            d_vertex=meta["d_vertex"],
            generation=parent_gen,
        )
        parent.load_state_dict(torch.load(
            gen_dir / f"gen{parent_gen}.pt",
            map_location=device, weights_only=True,
        ))
        print(f"  ⬡ Resuming from Generation {parent_gen}")
        print()

    generation_results = []
    t_total = time.time()

    for gen_idx in range(start_gen, start_gen + N_GENERATIONS):
        plan_idx = min(gen_idx, len(EVOLUTION_PLAN) - 1)
        d_model, n_layers, d_vertex = EVOLUTION_PLAN[plan_idx]

        print(f"  ═══════════════════════════════════════════════════════════")
        print(f"  ⬡ GENERATION {gen_idx} (hypercube 6D)")
        print(f"    {d_model}d × {n_layers} layers × d_vertex={d_vertex}")

        if parent is not None:
            model = HypercubeTransformerGPU.evolve(
                parent,
                d_model=d_model,
                n_layers=n_layers,
                d_vertex=d_vertex,
            )
        else:
            model = HypercubeTransformerGPU(
                vocab_size=256,
                d_model=d_model,
                n_layers=n_layers,
                d_vertex=d_vertex,
                max_len=SEQ_LEN * 2,
                generation=gen_idx,
            )

        n_params = sum(p.numel() for p in model.parameters())
        print(f"    Parameters: {n_params:,}")
        print()

        current_batch = BATCH_SIZE

        for attempt, try_batch in enumerate(BATCH_FALLBACK):
            if attempt > 0:
                current_batch = try_batch
                print(f"    ⚠ Retrying with batch_size={current_batch}...")
                torch.cuda.empty_cache()

            try:
                t0 = time.time()
                result = train_hypercube_gpu(
                    model=model,
                    text=corpus,
                    n_epochs=EPOCHS_PER_GEN,
                    batch_size=current_batch,
                    seq_len=SEQ_LEN,
                    lr=LEARNING_RATE,
                    warmup_steps=WARMUP_STEPS,
                    device=device,
                    use_amp=USE_AMP and device == "cuda",
                    grad_accum_steps=GRAD_ACCUM_STEPS,
                    gen_dir=gen_dir,
                    gen_idx=gen_idx,
                    checkpoint_every=CHECKPOINT_EVERY,
                )
                elapsed = time.time() - t0
                break
            except torch.cuda.OutOfMemoryError:
                print(f"    ❌ OOM at batch_size={current_batch}")
                if attempt == len(BATCH_FALLBACK) - 1:
                    print(f"    ❌ FATAL: OOM at minimum batch size. Stopping.")
                    return
                model = model.to("cpu")
                torch.cuda.empty_cache()
                model = model.to(device)
                continue

        print()
        print(f"    ✓ Training complete: {elapsed:.1f}s ({elapsed/60:.1f}m) | "
              f"final_loss={result['final_loss']:.4f} | best={result['best_loss']:.4f}")
        print(f"    {vram_report('post-train')}")
        print()

        print(f"    {model.status()}")
        print()

        # Save
        model_cpu = model.to("cpu")
        model_cpu.save_generation(gen_dir)
        (gen_dir / f"vocab_gen{gen_idx}.json").write_text(json.dumps(result["vocab"]))

        # Clean mid-epoch checkpoints
        for ckpt in gen_dir.glob(f"gen{gen_idx}_ckpt_e*.pt"):
            ckpt.unlink()

        # Generate samples
        print("    ── Emergence ──")
        prompts = [
            "The aperture ",
            "Curiosity is ",
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
            "version": "hypercube-6d",
            "d_model": d_model,
            "n_layers": n_layers,
            "d_vertex": d_vertex,
            "n_params": n_params,
            "final_loss": result["final_loss"],
            "best_loss": result["best_loss"],
            "diagnosis": result["diagnosis"],
            "elapsed": elapsed,
            "batch_size": current_batch,
        })

        parent = model_cpu
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    total_elapsed = time.time() - t_total

    print()
    print("  ╔════════════════════════════════════════════════════════════╗")
    print("  ║     ⬡ HYPERCUBE EVOLUTION COMPLETE (4070)                   ║")
    print("  ╚════════════════════════════════════════════════════════════╝")
    print()
    print("  Gen │ Params     │ Loss   │ Entropy │ Locality │ Time")
    print("  ────┼────────────┼────────┼─────────┼──────────┼────────")
    for r in generation_results:
        d = r["diagnosis"]
        t_min = r["elapsed"] / 60
        print(f"   {r['generation']:>2} │ {r['n_params']:>10,} │ {r['best_loss']:.4f} │ "
              f"  {d['mean_gate_entropy']:.3f}  │  {d['locality_ratio']:5.2f}x  │ {t_min:5.1f}m")
    print()
    print(f"  Total training time: {total_elapsed/60:.1f} minutes ({total_elapsed/3600:.1f} hours)")
    print(f"  Saved to: {gen_dir}/")
    print()

    (gen_dir / "evolution_log.json").write_text(
        json.dumps(generation_results, indent=2, default=str)
    )


if __name__ == "__main__":
    main()
