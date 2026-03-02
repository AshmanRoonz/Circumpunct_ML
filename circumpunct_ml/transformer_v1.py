"""
The Circumpunct Transformer — Xorzo's Neural Architecture

⊙ = Φ(•, ○) as a transformer.

This is not a standard transformer with circumpunct labels.
The circumpunct IS the computation:

    Standard Transformer          Circumpunct Transformer
    ─────────────────────         ─────────────────────────
    Q·K^T attention        →     ⊛ Convergence (gather signal)
    softmax + nonlinearity →     i  Aperture Rotation (Å(β) = exp(iπβ))
    V projection           →     ☀︎ Emergence (radiate transformed signal)
    Layer norm              →     Balance Norm (enforce β → 0.5)
    FFN hidden dim          →     φ-scaled (d_model × φ)
    Positional encoding    →     Golden spiral positions
    Token embedding        →     64-state lattice encoding

β is LEARNABLE — the network discovers balance.
Each generation inherits and evolves the previous.
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

PHI = (1 + math.sqrt(5)) / 2  # 1.618...
PI = math.pi


# ═══════════════════════════════════════════════════════════════════
# GOLDEN POSITIONAL ENCODING
# ═══════════════════════════════════════════════════════════════════

class GoldenPositionalEncoding(nn.Module):
    """
    Positional encoding using the golden angle (137.508°).

    Standard transformers use sin/cos at geometric frequencies.
    Xorzo uses the golden angle — the most irrational angle,
    ensuring maximum information spread across positions.

    The golden angle = 2π/φ² ≈ 137.508° — the same number
    that appears as the fine structure constant.
    """

    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        # Golden angle frequencies instead of geometric
        golden_angle = 2 * PI / (PHI ** 2)  # ≈ 2.399963... rad
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(PHI * 2) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * golden_angle * div_term)
        pe[:, 1::2] = torch.cos(position * golden_angle * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ═══════════════════════════════════════════════════════════════════
# APERTURE ROTATION — The Gate
# ═══════════════════════════════════════════════════════════════════

class ApertureRotation(nn.Module):
    """
    Å(β) = exp(iπβ)

    At β=0.5: Å = i (pure imaginary rotation — this is WHY i
    appears in quantum mechanics).

    Applied as a complex-valued rotation in paired dimensions.
    Each pair of dimensions is treated as (real, imaginary),
    and the rotation mixes them by angle πβ.

    β is LEARNABLE — the network discovers balance.

    The depth parameter encodes the circumpunct geometry:
        - depth → 0 (center/•):  initialized convergent (β > 0.5)
        - depth → 1 (boundary/○): initialized emergent (β < 0.5)

    The center is infinitely convergent.
    The boundary is infinitely emergent.
    Balance (β = 0.5) is the living membrane between them.
    """

    def __init__(self, d_model: int, depth: float = 0.5):
        super().__init__()
        # depth ∈ [0, 1]: 0 = center (convergent), 1 = boundary (emergent)
        # β_init ranges from ~0.7 (center) to ~0.3 (boundary)
        # so that center converges harder, boundary emerges harder
        beta_init = 0.5 + 0.2 * (1.0 - 2.0 * depth)  # center→0.7, boundary→0.3
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.d_model = d_model
        self.depth = depth  # remember where in the ⊙ this aperture sits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Clamp β to (0, 1)
        beta = torch.sigmoid(self.beta)

        # Rotation angle = πβ
        angle = PI * beta

        # Apply rotation to paired dimensions
        d = x.shape[-1]
        half = d // 2

        x_real = x[..., :half]
        x_imag = x[..., half:2*half]

        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        # Complex rotation: (r + ji) × e^(iπβ)
        out_real = x_real * cos_a - x_imag * sin_a
        out_imag = x_real * sin_a + x_imag * cos_a

        # Handle odd dimension
        if d % 2 == 1:
            return torch.cat([out_real, out_imag, x[..., -1:]], dim=-1)
        return torch.cat([out_real, out_imag], dim=-1)

    @property
    def current_beta(self) -> float:
        return torch.sigmoid(self.beta).item()


# ═══════════════════════════════════════════════════════════════════
# CIRCUMPUNCT ATTENTION — ⊛ → i → ☀︎
# ═══════════════════════════════════════════════════════════════════

class CircumpunctAttention(nn.Module):
    """
    The core of Xorzo. Attention reimagined as ⊛ → i → ☀︎.

    ⊛ (Convergence): Gather signal from all positions toward
       each focal point. This IS attention — but framed as
       isotropic convergence rather than query-key matching.

    i  (Aperture Rotation): Transform the converged signal
       through the gate. Complex rotation by Å(β).
       At β=0.5: maximum information throughput.

    ☀︎ (Emergence): Radiate the transformed signal back out.
       Project to output space and distribute.

    Multi-head: each head is a separate aperture (•) on the
    boundary (○), each with its own learnable β.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1,
                 layer_depth: float = 0.5):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # ⊛ Convergence projections (what to gather, where from)
        self.W_converge_q = nn.Linear(d_model, d_model, bias=False)
        self.W_converge_k = nn.Linear(d_model, d_model, bias=False)
        self.W_converge_v = nn.Linear(d_model, d_model, bias=False)

        # i  Aperture rotation — one per head (each head has its own β)
        # Heads form a gradient from center (convergent) to boundary (emergent)
        # within each layer. Layer depth also modulates: early layers more
        # convergent, later layers more emergent.
        self.apertures = nn.ModuleList([
            ApertureRotation(
                self.d_head,
                depth=self._head_depth(h, n_heads, layer_depth)
            )
            for h in range(n_heads)
        ])

        # ☀︎ Emergence projection
        self.W_emerge = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_head)

        # χ gate — learnable faithful/inverted per head
        # Initialized to +1 (faithful). Can learn to invert.
        self.chi = nn.Parameter(torch.ones(n_heads))

    @staticmethod
    def _head_depth(head_idx: int, n_heads: int, layer_depth: float) -> float:
        """
        Compute depth for a specific head within a layer.

        The center is infinitely convergent. The boundary is infinitely emergent.
        Heads are arranged radially: head 0 is closest to center (•),
        head n-1 is closest to boundary (○). Layer depth shifts the
        whole range — early layers lean convergent, later layers emergent.

        Returns depth ∈ [0, 1] where 0=center, 1=boundary.
        """
        # Head position: 0 → center, n-1 → boundary
        head_fraction = head_idx / max(n_heads - 1, 1)
        # Blend head position with layer depth
        # Early layers (layer_depth≈0): heads span [0, 0.5] (all convergent-leaning)
        # Later layers (layer_depth≈1): heads span [0.5, 1] (all emergent-leaning)
        # Middle layers: heads span full [0, 1]
        lo = layer_depth * 0.5
        hi = 0.5 + layer_depth * 0.5
        return lo + head_fraction * (hi - lo)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        # ═══ ⊛ CONVERGENCE ═══
        # Gather: what does each position want? what does each offer?
        q = self.W_converge_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_converge_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_converge_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Isotropic convergence (attention scores)
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Converged signal
        converged = torch.matmul(attn, v)  # (B, n_heads, T, d_head)

        # ═══ i APERTURE ROTATION ═══
        # Each head rotates through its own aperture
        rotated_heads = []
        for h in range(self.n_heads):
            head_signal = converged[:, h]  # (B, T, d_head)

            # Rotate through aperture
            rotated = self.apertures[h](head_signal)

            # Apply χ gate (faithful or inverted)
            chi_h = torch.tanh(self.chi[h])  # smooth ±1
            rotated = rotated * chi_h

            rotated_heads.append(rotated)

        # Recombine heads
        combined = torch.stack(rotated_heads, dim=1)  # (B, n_heads, T, d_head)
        combined = combined.transpose(1, 2).contiguous().view(B, T, D)

        # ═══ ☀︎ EMERGENCE ═══
        emerged = self.W_emerge(combined)

        return emerged

    @property
    def beta_values(self) -> list[float]:
        """Current β for each head/aperture."""
        return [a.current_beta for a in self.apertures]

    @property
    def chi_values(self) -> list[float]:
        """Current χ for each head."""
        return [torch.tanh(c).item() for c in self.chi]


# ═══════════════════════════════════════════════════════════════════
# BALANCE NORM — Replaces LayerNorm
# ═══════════════════════════════════════════════════════════════════

class BalanceNorm(nn.Module):
    """
    Normalization that gently enforces β → 0.5.

    Standard LayerNorm centers to mean=0, var=1.
    BalanceNorm does the same but adds a regularization
    signal that pulls the representation toward balance.

    At balance: convergence energy = emergence energy.
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.balance_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)

        # Balance signal: pull first and second halves toward equal energy
        half = x.shape[-1] // 2
        energy_first = x[..., :half].pow(2).mean(dim=-1, keepdim=True)
        energy_second = x[..., half:].pow(2).mean(dim=-1, keepdim=True)
        imbalance = (energy_first - energy_second) * self.balance_weight

        # Subtract imbalance from dominant half
        correction = torch.zeros_like(x)
        correction[..., :half] = -imbalance
        correction[..., half:] = imbalance

        return normed + correction


# ═══════════════════════════════════════════════════════════════════
# GOLDEN FFN — φ-scaled Feed-Forward
# ═══════════════════════════════════════════════════════════════════

class GoldenFFN(nn.Module):
    """
    Feed-forward network with golden ratio scaling.

    Standard: d_model → 4·d_model → d_model
    Xorzo:    d_model → φ·d_model → d_model (≈ 1.618× expansion)

    Why? The golden ratio is the optimal expansion factor for
    self-similar information processing. Φ mediates between
    aperture (compression) and boundary (expansion).

    Uses GELU activation — the closest standard activation to
    the smooth aperture gate function.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        d_hidden = int(d_model * PHI)  # φ-scaled hidden dimension

        self.W_in = nn.Linear(d_model, d_hidden)
        self.W_out = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.W_out(self.dropout(F.gelu(self.W_in(x))))


# ═══════════════════════════════════════════════════════════════════
# CIRCUMPUNCT BLOCK — One complete ⊙ layer
# ═══════════════════════════════════════════════════════════════════

class CircumpunctBlock(nn.Module):
    """
    One complete ⊙ = Φ(•, ○) processing layer.

    Each block is a full circumpunct cycle:
        1. BalanceNorm → CircumpunctAttention (⊛→i→☀︎) + residual
        2. BalanceNorm → GoldenFFN (Φ expansion/compression) + residual

    The residual connection is the conservation of traversal (A3):
    what enters must be recoverable in what exits.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1,
                 layer_depth: float = 0.5):
        super().__init__()
        self.norm1 = BalanceNorm(d_model)
        self.attn = CircumpunctAttention(d_model, n_heads, dropout, layer_depth)
        self.norm2 = BalanceNorm(d_model)
        self.ffn = GoldenFFN(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # ⊛ → i → ☀︎  (with residual = conservation of traversal)
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        # Φ expansion/compression (with residual)
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


# ═══════════════════════════════════════════════════════════════════
# XORZO TRANSFORMER — The Complete Architecture
# ═══════════════════════════════════════════════════════════════════

class XorzoTransformer(nn.Module):
    """
    The Xorzo Transformer — ⊙ = Φ(•, ○) as a neural network.

    Architecture:
        - Token embedding (vocab → d_model)
        - Golden positional encoding (golden angle frequencies)
        - N × CircumpunctBlock (⊛→i→☀︎ + GoldenFFN)
        - Final BalanceNorm → output projection

    Every component maps to the framework:
        • Embedding    = ⊛ (convergence from discrete tokens to continuous space)
        • Blocks       = recursive ⊙ cycles (each a complete circumpunct)
        • Output proj  = ☀︎ (emergence back to discrete predictions)
        • β per head   = learnable balance (the network discovers ◐=0.5)
        • χ per head   = learnable gate (faithful or inverted transmission)

    Parameters
    ----------
    vocab_size : int
        Number of tokens. Default 64 (the lattice).
    d_model : int
        Hidden dimension. Default 128.
    n_layers : int
        Number of ⊙ blocks (circumpunct depth).
    n_heads : int
        Apertures per block.
    generation : int
        Evolution generation (increments each training cycle).
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        max_len: int = 512,
        dropout: float = 0.1,
        generation: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.generation = generation

        # ⊛ Token convergence (embedding)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = GoldenPositionalEncoding(d_model, max_len)
        self.embed_dropout = nn.Dropout(dropout)

        # Recursive ⊙ blocks — arranged from center (convergent) to boundary (emergent)
        # Layer 0 is the innermost circle (• — pure convergence)
        # Layer N-1 is the outermost circle (○ — pure emergence)
        # The center is infinitely convergent. The boundary is infinitely emergent.
        self.blocks = nn.ModuleList([
            CircumpunctBlock(
                d_model, n_heads, dropout,
                layer_depth=i / max(n_layers - 1, 1)
            )
            for i in range(n_layers)
        ])

        # ☀︎ Final emergence
        self.final_norm = BalanceNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying (aperture = emergence, by symmetry)
        self.output_proj.weight = self.token_embed.weight

        # Initialize with golden scaling
        self._golden_init()

    def _golden_init(self):
        """Initialize weights with φ-scaled variance."""
        for p in self.parameters():
            if p.dim() > 1:
                # Xavier init scaled by 1/√φ
                nn.init.xavier_normal_(p, gain=1.0 / math.sqrt(PHI))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: token ids → logits.

        The full cycle: ⊛ (embed) → [⊙ × N] → ☀︎ (project)
        """
        # ⊛ Convergence: discrete → continuous
        h = self.token_embed(x)
        h = self.pos_encode(h)
        h = self.embed_dropout(h)

        # Causal mask for autoregressive generation
        if mask is None:
            T = x.size(1)
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        # N × ⊙ cycles
        for block in self.blocks:
            h = block(h, mask)

        # ☀︎ Emergence: continuous → discrete
        h = self.final_norm(h)
        logits = self.output_proj(h)

        return logits

    # ── Self-Diagnosis ──

    @property
    def all_betas(self) -> list[list[float]]:
        """β values across all layers and heads."""
        return [block.attn.beta_values for block in self.blocks]

    @property
    def mean_beta(self) -> float:
        """Average β across entire network."""
        all_b = [b for layer in self.all_betas for b in layer]
        return sum(all_b) / len(all_b) if all_b else 0.5

    @property
    def all_chis(self) -> list[list[float]]:
        """χ values across all layers and heads."""
        return [block.attn.chi_values for block in self.blocks]

    def diagnose(self) -> dict:
        """
        Full geometric error diagnosis.
        The network's immune system.
        """
        betas = [b for layer in self.all_betas for b in layer]
        chis = [c for layer in self.all_chis for c in layer]

        mean_b = sum(betas) / len(betas)
        mean_chi = sum(chis) / len(chis)
        beta_var = sum((b - mean_b)**2 for b in betas) / len(betas)

        errors = []
        if mean_b > 0.7:
            errors.append("INFLATION: mean β too high — convergence dominates")
        if mean_b < 0.3:
            errors.append("SEVERANCE: mean β too low — emergence dominates")
        if mean_chi < 0:
            errors.append("INVERSION: mean χ negative — signal being flipped")
        if beta_var > 0.1:
            errors.append("PROJECTION: β variance high — inconsistent processing")

        return {
            "generation": self.generation,
            "mean_beta": mean_b,
            "beta_variance": beta_var,
            "mean_chi": mean_chi,
            "D": 1.0 + mean_b,
            "regime": "balance" if abs(mean_b - 0.5) < 0.1 else
                      "convergent" if mean_b > 0.5 else "emergent",
            "errors": errors,
            "healthy": len(errors) == 0,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def status(self) -> str:
        """Human-readable status."""
        d = self.diagnose()
        lines = [
            f"⊙ XORZO TRANSFORMER — Generation {d['generation']}",
            f"  Architecture: {self.n_layers} blocks × {self.n_heads} heads × {self.d_model}d",
            f"  Parameters: {d['n_params']:,}",
            f"  β̄ = {d['mean_beta']:.4f} → D = {d['D']:.4f} [{d['regime']}]",
            f"  χ̄ = {d['mean_chi']:.4f} ({'faithful' if d['mean_chi'] > 0 else 'INVERTED'})",
            f"  β variance: {d['beta_variance']:.6f}",
        ]
        if d['errors']:
            for e in d['errors']:
                lines.append(f"  ⚠ {e}")
        else:
            lines.append(f"  ✓ No geometric errors — system is healthy")
        return "\n".join(lines)

    # ── Evolution ──

    def save_generation(self, path: Path):
        """Save this generation's weights and metadata."""
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / f"gen{self.generation}.pt")
        meta = {
            "generation": self.generation,
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "diagnosis": self.diagnose(),
        }
        (path / f"gen{self.generation}_meta.json").write_text(
            json.dumps(meta, indent=2)
        )

    @classmethod
    def evolve(cls, parent: 'XorzoTransformer', **mutations) -> 'XorzoTransformer':
        """
        Create next generation from parent.

        The child inherits the parent's learned weights and β values.
        Mutations can modify architecture (more layers, heads, etc).
        """
        child = cls(
            vocab_size=mutations.get("vocab_size", parent.vocab_size),
            d_model=mutations.get("d_model", parent.d_model),
            n_layers=mutations.get("n_layers", parent.n_layers),
            n_heads=mutations.get("n_heads", parent.n_heads),
            generation=parent.generation + 1,
        )

        # Inherit compatible weights from parent
        parent_state = parent.state_dict()
        child_state = child.state_dict()

        inherited = 0
        for key in child_state:
            if key in parent_state and child_state[key].shape == parent_state[key].shape:
                child_state[key] = parent_state[key]
                inherited += 1

        child.load_state_dict(child_state)
        print(f"  ⊙ Generation {child.generation} born from {parent.generation}")
        print(f"    Inherited {inherited}/{len(child_state)} parameters")

        return child


# ═══════════════════════════════════════════════════════════════════
# FRACTAL LEARNING — The training process IS the circumpunct
# ═══════════════════════════════════════════════════════════════════
#
# Standard training: minimize cross-entropy. That's it.
# Fractal training: the optimization ITSELF cycles through ⊛ → i → ☀︎
#
#   ⊛ (Convergence phase): High LR, explore the loss landscape.
#       The network gathers signal from the data.
#
#   i  (Aperture Rotation phase): Anneal LR, enforce β → 0.5.
#       The network passes through the gate. D → 1.5.
#
#   ☀︎ (Emergence phase): Low LR, crystallize.
#       The network radiates coherent structure.
#
# Additional fractal properties:
#   - φ-scaled LR per layer depth (inner learns faster)
#   - β-balance loss pulls toward D = 1.5 (Mandelbrot boundary)
#   - A3 conservation loss (traversal energy is conserved)
#   - Self-similar gradient coupling between layers
# ═══════════════════════════════════════════════════════════════════


def _beta_balance_loss(model: XorzoTransformer) -> torch.Tensor:
    """
    Pull every β toward 0.5 — the Mandelbrot boundary.

    At β=0.5: D=1.5, maximum fractal complexity.
    This is a soft target — the network can deviate,
    but there's a gravitational pull toward balance.

    The pull strength scales with distance from 0.5:
    gentle near balance, strong at extremes.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for block in model.blocks:
        for aperture in block.attn.apertures:
            beta = torch.sigmoid(aperture.beta)
            # Quadratic pull toward 0.5
            loss = loss + (beta - 0.5).pow(2)
            n += 1
    return loss / max(n, 1)


def _self_similarity_loss(model: XorzoTransformer) -> torch.Tensor:
    """
    Fractal self-similarity: adjacent layers should have β-patterns
    that are scaled versions of each other.

    In a fractal, zoom in at any level and you see the same structure.
    This loss encourages the β gradient across heads in layer N
    to resemble the gradient in layer N+1, scaled by φ.

    This IS conservation of traversal (A3) — expressed as
    self-similarity rather than energy balance. The residual
    connections already conserve energy. This loss conserves PATTERN.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    prev_betas = None

    for block in model.blocks:
        betas = torch.stack([torch.sigmoid(a.beta) for a in block.attn.apertures])
        if prev_betas is not None and len(betas) == len(prev_betas):
            # The difference pattern between adjacent heads should be self-similar
            # across layers — scaled by 1/φ at each depth
            curr_diffs = betas[1:] - betas[:-1]
            prev_diffs = prev_betas[1:] - prev_betas[:-1]
            # Self-similar: curr_diffs ≈ prev_diffs (same pattern, layer to layer)
            loss = loss + (curr_diffs - prev_diffs).pow(2).mean()
            n += 1
        prev_betas = betas.detach()  # detach to avoid through-time gradients

    return loss / max(n, 1)


def _chi_fidelity_loss(model: XorzoTransformer) -> torch.Tensor:
    """
    Gently encourage χ toward ±1 (pure faithful or pure inverted).

    χ near 0 means the gate is killing signal — geometric error.
    The network should commit to either faithful or inverted transmission.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for block in model.blocks:
        for chi_raw in block.attn.chi:
            chi = torch.tanh(chi_raw)
            # Pull toward |χ| = 1: penalize |χ| < 1
            loss = loss + (1.0 - chi.pow(2)).pow(2)
            n += 1
    return loss / max(n, 1)


def _fractal_lr_schedule(epoch: int, n_epochs: int, base_lr: float) -> float:
    """
    ⊛ → i → ☀︎ learning rate schedule.

    The training itself cycles through the circumpunct:
        Phase 1 (⊛ Convergence):  0%–33%  — LR ramps up, gather signal
        Phase 2 (i  Rotation):   33%–67%  — LR at peak, transform
        Phase 3 (☀︎ Emergence):   67%–100% — LR decays by φ, crystallize

    Within each phase, the LR follows a cosine sub-cycle (self-similar).
    """
    progress = epoch / max(n_epochs - 1, 1)

    if progress < 1/3:
        # ⊛ Convergence: ramp up
        phase_progress = progress * 3  # 0→1
        lr = base_lr * (0.1 + 0.9 * (1 - math.cos(PI * phase_progress)) / 2)
    elif progress < 2/3:
        # i Rotation: full power, slight oscillation (aperture wobble)
        phase_progress = (progress - 1/3) * 3  # 0→1
        lr = base_lr * (0.9 + 0.1 * math.cos(2 * PI * phase_progress))
    else:
        # ☀︎ Emergence: decay by golden ratio
        phase_progress = (progress - 2/3) * 3  # 0→1
        lr = base_lr * (PHI ** (-1 - 2 * phase_progress))  # φ^(-1) → φ^(-3)

    return lr


def train_generation(
    model: XorzoTransformer,
    text: str,
    n_epochs: int = 10,
    batch_size: int = 16,
    seq_len: int = 64,
    lr: float = 3e-4,
    device: str = "cpu",
    # Fractal learning weights
    w_balance: float = 0.05,     # β → 0.5 pull strength
    w_similarity: float = 0.02,  # Self-similar β patterns across layers
    w_fidelity: float = 0.01,    # χ → ±1 pull strength
) -> dict:
    """
    Fractal training: the optimization IS the circumpunct.

    Loss = CE + fractal regularizers:
        1. Cross-entropy (learn the data)
        2. β-balance loss (pull toward D=1.5, the Mandelbrot boundary)
        3. Self-similarity loss (adjacent layers have φ-scaled β patterns)
        4. χ-fidelity loss (commit to faithful or inverted, not zero)

    Three training phases (⊛ → i → ☀︎):
        1. Convergence: ramp LR, explore, gather signal
        2. Rotation: full LR, enforce balance, transform
        3. Emergence: decay LR by φ, crystallize structure

    φ-scaled per-layer LR: inner layers (center/•) learn faster,
    outer layers (boundary/○) learn slower. Self-similar scaling.

    Conservation of traversal (A3) is STRUCTURAL — it's the
    residual connections, not an external loss that fights the network.
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
        model.token_embed = nn.Embedding(actual_vocab, model.d_model).to(device)
        model.output_proj = nn.Linear(model.d_model, actual_vocab, bias=False).to(device)
        model.output_proj.weight = model.token_embed.weight

    # φ-scaled parameter groups: inner layers learn faster, outer slower
    # Store base LR scales so we can update properly each epoch
    layer_scales = []
    param_groups = []
    for i, block in enumerate(model.blocks):
        layer_depth = i / max(model.n_layers - 1, 1)
        # Center (depth≈0): lr × φ^0.5 ≈ 1.27× (faster — convergence)
        # Boundary (depth≈1): lr × φ^(-0.5) ≈ 0.79× (slower — emergence)
        scale = PHI ** (0.5 - layer_depth)
        layer_scales.append(scale)
        param_groups.append({
            'params': list(block.parameters()),
            'lr': lr * scale,
            'weight_decay': 0.01,
        })
    # Embedding and output learn at base rate
    embed_params = list(model.token_embed.parameters()) + \
                   list(model.pos_encode.parameters()) + \
                   list(model.final_norm.parameters())
    layer_scales.append(1.0)
    param_groups.append({
        'params': embed_params,
        'lr': lr,
        'weight_decay': 0.01,
    })

    optimizer = torch.optim.AdamW(param_groups)

    losses = []
    n_batches = max(1, (len(data) - seq_len) // (batch_size * seq_len))

    phase_names = ['⊛ converge', 'i rotate ', '☀︎ emerge ']

    for epoch in range(n_epochs):
        # ⊛→i→☀︎ learning rate schedule
        epoch_lr = _fractal_lr_schedule(epoch, n_epochs, lr)

        # Update LR for each param group using stored scales (no drift)
        for pg, scale in zip(optimizer.param_groups, layer_scales):
            pg['lr'] = epoch_lr * scale

        # Which phase are we in?
        progress = epoch / max(n_epochs - 1, 1)
        phase_idx = min(int(progress * 3), 2)
        phase = phase_names[phase_idx]

        # Dynamic balance weight: stronger during i (rotation) phase
        phase_w_balance = w_balance * (2.0 if phase_idx == 1 else 1.0)

        epoch_ce = 0
        epoch_bal = 0
        epoch_sim = 0
        n_steps = 0

        for batch_idx in range(n_batches):
            starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
            x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
            y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

            # Single forward pass
            logits = model(x)

            # 1. Cross-entropy (learn the data)
            ce_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

            # 2. β-balance (pull toward D=1.5 — no extra forward pass needed)
            bal_loss = _beta_balance_loss(model)

            # 3. Self-similarity (β patterns repeat across layers)
            sim_loss = _self_similarity_loss(model)

            # 4. χ-fidelity (commit to ±1)
            chi_loss = _chi_fidelity_loss(model)

            # Combined: CE dominates, fractal terms are gentle guidance
            total_loss = ce_loss + \
                         phase_w_balance * bal_loss + \
                         w_similarity * sim_loss + \
                         w_fidelity * chi_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_ce += ce_loss.item()
            epoch_bal += bal_loss.item()
            epoch_sim += sim_loss.item()
            n_steps += 1

        avg_ce = epoch_ce / max(n_steps, 1)
        avg_bal = epoch_bal / max(n_steps, 1)
        avg_sim = epoch_sim / max(n_steps, 1)
        losses.append(avg_ce)

        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            diag = model.diagnose()
            betas_flat = [b for layer in model.all_betas for b in layer]
            beta_min, beta_max = min(betas_flat), max(betas_flat)
            print(
                f"  {phase} E{epoch+1:>3}/{n_epochs} | "
                f"CE={avg_ce:.3f} β-bal={avg_bal:.4f} sim={avg_sim:.5f} | "
                f"β̄={diag['mean_beta']:.4f} [{beta_min:.3f}→{beta_max:.3f}] "
                f"D={diag['D']:.3f} | "
                f"lr={epoch_lr:.1e}"
            )

    model.eval()
    return {
        "losses": losses,
        "final_loss": losses[-1] if losses else float('inf'),
        "diagnosis": model.diagnose(),
        "vocab": char_to_idx,
        "vocab_inv": idx_to_char,
    }


def generate(
    model: XorzoTransformer,
    prompt: str,
    vocab: dict,
    vocab_inv: dict,
    max_tokens: int = 200,
    temperature: float = 0.8,
    device: str = "cpu",
) -> str:
    """Generate text from Xorzo — the ☀︎ emergence."""
    model.eval()
    tokens = [vocab.get(c, 0) for c in prompt]
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    generated = list(prompt)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x[:, -512:])  # Context window
            next_logits = logits[0, -1] / temperature

            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)

            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)

            char = vocab_inv.get(next_token.item(), "?")
            generated.append(char)

    return "".join(generated)
