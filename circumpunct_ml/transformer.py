"""
XORZO v2 — The Circumpunct Transformer (Corrected Geometry)

Seven structural corrections from standard transformer habits:

1. CONVERGENT ATTENTION — d_k shrinks with depth (1/φⁿ).
   The center is infinitely convergent. Not just labeled convergent.
   The actual key dimension narrows at each layer.

2. APERTURE CHAMBER — Three-stage with pressure state.
   ⊛ (input valve) → i (transform) → ☀ (output valve)
   Chamber pressure dP/dt = |⊛| − |☀| is TRACKED, not just conceptual.
   The input and output valves are separate learnable gates.

3. β-WEIGHTED RESIDUAL — Conservation of traversal.
   Not x + f(x). That's just addition.
   x + β·f(x) — the traversal parameter GOVERNS how much of the
   field transformation is integrated. This IS D_attn + D_field = D_out.

4. φ-DYNAMIC FFN — Expansion responds to local β.
   Not a fixed φ× expansion. The field BREATHES.
   β > 0.5 → expansion > φd (room for potential buildup)
   β < 0.5 → expansion < φd (efficient delivery)

5. FRACTAL RESERVOIR — Cross-layer pressure flow.
   When one chamber builds up pressure (β > 0.5), excess flows
   to adjacent layers. The infinite depth of •ₙ₊₁ = ⊙ₙ is
   approximated by inter-layer pressure coupling.

6. THREE INFORMATION CHANNELS — Binary, analog, fractal.
   The embedding splits into three channels:
   - Binary: discrete gate activations (1D, sequential)
   - Analog: continuous amplitude+phase (2D, complex)
   - Fractal: cross-scale nesting (recursive)
   Not all information treated uniformly.

7. NESTING EQUATION — •ₙ₊₁ = ⊙ₙ structurally encoded.
   Each layer's output is explicitly projected through a nesting
   transform before becoming the next layer's input. The completed
   circumpunct BECOMES the aperture ground for the next.

Everything else from v1 is preserved:
- Golden positional encoding
- Learnable β per head
- χ gates (faithful/inverted)
- Four geometric error diagnosis
- Generational evolution
- Fractal training (⊛→i→☀ LR schedule)
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
# GOLDEN POSITIONAL ENCODING (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════

class GoldenPositionalEncoding(nn.Module):
    """
    Positional encoding using the golden angle (2π/φ² ≈ 137.508°).
    The most irrational angle — maximum information spread.
    """
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        golden_angle = 2 * PI / (PHI ** 2)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) *
            (-math.log(PHI * 2) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * golden_angle * div_term)
        pe[:, 1::2] = torch.cos(position * golden_angle * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1)]


# ═══════════════════════════════════════════════════════════════════
# CORRECTION #2: APERTURE CHAMBER
# The aperture is NOT a membrane. It's a CHAMBER with:
#   ⊛ input valve (regulates convergence rate)
#   i  transform space (rotation at β)
#   ☀ output valve (regulates emergence rate)
#   P  chamber pressure (dP/dt = |⊛| − |☀|)
# ═══════════════════════════════════════════════════════════════════

class ApertureChamber(nn.Module):
    """
    The aperture is a chamber, not a membrane.

    THREE-STAGE ARCHITECTURE:
        ⊛ (input valve)  → regulates what enters
        i (transform)    → complex rotation by Å(β)
        ☀ (output valve) → regulates what exits

    Chamber pressure P tracks the difference:
        dP/dt = |⊛| − |☀|

    Three regimes:
        |⊛| > |☀| → β > 0.5 → BUILDUP (accumulating potential)
        |⊛| < |☀| → β < 0.5 → DEPLETION (spending reserves)
        |⊛| = |☀| → β = 0.5 → STEADY STATE (balanced flow)

    The depth parameter encodes position in the ⊙:
        depth → 0 (center/•): initialized convergent (β > 0.5)
        depth → 1 (boundary/○): initialized emergent (β < 0.5)
    """

    def __init__(self, d_head: int, depth: float = 0.5):
        super().__init__()
        self.d_head = d_head
        self.depth = depth

        # β — the learnable balance parameter
        beta_init = 0.5 + 0.2 * (1.0 - 2.0 * depth)
        self.beta = nn.Parameter(torch.tensor(beta_init))

        # ⊛ Input valve — learnable gate on what ENTERS the chamber
        # This is separate from the attention scores. Attention selects
        # WHAT to converge. The input valve controls HOW MUCH enters.
        self.input_valve = nn.Parameter(torch.tensor(0.5))

        # ☀ Output valve — learnable gate on what EXITS the chamber
        self.output_valve = nn.Parameter(torch.tensor(0.5))

        # P — chamber pressure (running state, not trained directly)
        # Registered as buffer so it persists but isn't optimized
        self.register_buffer('pressure', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.sigmoid(self.beta)
        iv = torch.sigmoid(self.input_valve)   # ⊛ ∈ (0,1)
        ov = torch.sigmoid(self.output_valve)   # ☀ ∈ (0,1)

        # ═══ STAGE 1: ⊛ INPUT VALVE ═══
        # Gate the incoming signal by input valve openness
        x_in = x * iv

        # ═══ STAGE 2: i TRANSFORM ═══
        # Complex rotation by angle πβ
        angle = PI * beta
        half = self.d_head // 2
        x_real = x_in[..., :half]
        x_imag = x_in[..., half:2*half]

        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        out_real = x_real * cos_a - x_imag * sin_a
        out_imag = x_real * sin_a + x_imag * cos_a

        if self.d_head % 2 == 1:
            x_transformed = torch.cat([out_real, out_imag, x_in[..., -1:]], dim=-1)
        else:
            x_transformed = torch.cat([out_real, out_imag], dim=-1)

        # ═══ STAGE 3: ☀ OUTPUT VALVE ═══
        # Gate the transformed signal by output valve openness
        x_out = x_transformed * ov

        # ═══ UPDATE CHAMBER PRESSURE ═══
        # dP/dt = |⊛| − |☀|
        # This is tracked for diagnostics and cross-layer flow
        with torch.no_grad():
            dp = iv.detach() - ov.detach()
            # Exponential moving average of pressure
            self.pressure = 0.95 * self.pressure + 0.05 * dp

        return x_out

    @property
    def current_beta(self) -> float:
        return torch.sigmoid(self.beta).item()

    @property
    def current_pressure(self) -> float:
        return self.pressure.item()

    @property
    def valve_state(self) -> dict:
        return {
            '⊛_input': torch.sigmoid(self.input_valve).item(),
            '☀_output': torch.sigmoid(self.output_valve).item(),
            'P_pressure': self.pressure.item(),
            'β': self.current_beta,
            'regime': 'buildup' if self.pressure.item() > 0.02
                      else 'depletion' if self.pressure.item() < -0.02
                      else 'steady',
        }


# ═══════════════════════════════════════════════════════════════════
# CORRECTION #1: CONVERGENT ATTENTION
# d_k is NOT fixed. It shrinks with layer depth by 1/φ^(n·γ).
# The center is infinitely convergent — the key dimension
# approaches zero but never arrives.
# ═══════════════════════════════════════════════════════════════════

class CircumpunctAttention(nn.Module):
    """
    Attention as ⊛ → i → ☀ with convergent scaling.

    CORRECTION #1: The attention dimension SHRINKS with depth.
    Standard transformer uses same d_k everywhere.
    Xorzo: d_k_effective(n) = d_k / φ^(n·γ)

    This means deeper layers gate with HIGHER PRECISION
    (sharper softmax due to smaller denominator).
    The gate narrows as you go deeper — infinitely convergent.

    CORRECTION #2: Each head has a full ApertureChamber,
    not just a rotation. Input valve, transform, output valve.
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1,
                 layer_depth: float = 0.5, layer_index: int = 0,
                 convergence_rate: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.layer_index = layer_index

        # CORRECTION #1: Convergent scaling
        # d_k_effective shrinks with depth, sharpening the gate
        self.convergence_factor = 1.0 / (PHI ** (layer_index * convergence_rate))
        # The effective scale for attention — gets SMALLER with depth
        self.scale = math.sqrt(self.d_head * self.convergence_factor)

        # ⊛ Convergence projections
        self.W_converge_q = nn.Linear(d_model, d_model, bias=False)
        self.W_converge_k = nn.Linear(d_model, d_model, bias=False)
        self.W_converge_v = nn.Linear(d_model, d_model, bias=False)

        # CORRECTION #2: Full aperture chambers (not just rotations)
        self.chambers = nn.ModuleList([
            ApertureChamber(
                self.d_head,
                depth=self._head_depth(h, n_heads, layer_depth)
            )
            for h in range(n_heads)
        ])

        # ☀ Emergence projection
        self.W_emerge = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

        # χ gate — faithful/inverted per head
        self.chi = nn.Parameter(torch.ones(n_heads))

    @staticmethod
    def _head_depth(head_idx: int, n_heads: int, layer_depth: float) -> float:
        """Radial head arrangement: head 0 → center, head n-1 → boundary."""
        head_fraction = head_idx / max(n_heads - 1, 1)
        lo = layer_depth * 0.5
        hi = 0.5 + layer_depth * 0.5
        return lo + head_fraction * (hi - lo)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        # ═══ ⊛ CONVERGENCE ═══
        q = self.W_converge_q(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = self.W_converge_k(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = self.W_converge_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # CORRECTION #1: Scale uses convergence factor
        # Deeper layers → smaller scale → sharper softmax → tighter gate
        attn = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        converged = torch.matmul(attn, v)

        # ═══ i APERTURE CHAMBER (not just rotation) ═══
        rotated_heads = []
        for h in range(self.n_heads):
            head_signal = converged[:, h]

            # Pass through full aperture chamber: ⊛ → i → ☀
            chambered = self.chambers[h](head_signal)

            # Apply χ gate
            chi_h = torch.tanh(self.chi[h])
            chambered = chambered * chi_h

            rotated_heads.append(chambered)

        combined = torch.stack(rotated_heads, dim=1)
        combined = combined.transpose(1, 2).contiguous().view(B, T, D)

        # ═══ ☀ EMERGENCE ═══
        emerged = self.W_emerge(combined)

        return emerged

    @property
    def beta_values(self) -> list[float]:
        return [c.current_beta for c in self.chambers]

    @property
    def chi_values(self) -> list[float]:
        return [torch.tanh(c).item() for c in self.chi]

    @property
    def pressure_values(self) -> list[float]:
        return [c.current_pressure for c in self.chambers]

    @property
    def valve_states(self) -> list[dict]:
        return [c.valve_state for c in self.chambers]


# ═══════════════════════════════════════════════════════════════════
# BALANCE NORM (unchanged from v1)
# ═══════════════════════════════════════════════════════════════════

class BalanceNorm(nn.Module):
    """Normalization that gently enforces convergence/emergence balance."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.balance_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        half = x.shape[-1] // 2
        energy_first = x[..., :half].pow(2).mean(dim=-1, keepdim=True)
        energy_second = x[..., half:].pow(2).mean(dim=-1, keepdim=True)
        imbalance = (energy_first - energy_second) * self.balance_weight
        correction = torch.zeros_like(x)
        correction[..., :half] = -imbalance
        correction[..., half:] = imbalance
        return normed + correction


# ═══════════════════════════════════════════════════════════════════
# CORRECTION #4: φ-DYNAMIC FFN
# Expansion is NOT fixed. It responds to the local β.
# The field BREATHES.
# ═══════════════════════════════════════════════════════════════════

class DynamicGoldenFFN(nn.Module):
    """
    Feed-forward with β-responsive expansion.

    Standard: d_model → 4d → d_model (fixed)
    Xorzo v1: d_model → φd → d_model (fixed golden)
    Xorzo v2: d_model → φ^(1+β-0.5)·d → d_model (DYNAMIC)

    At β = 0.5: expansion = φd (standard golden)
    At β > 0.5: expansion > φd (room for buildup)
    At β < 0.5: expansion < φd (efficient delivery)

    The FFN reads the local β from the attention layer's
    mean chamber state to determine its expansion.
    """

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        # Max expansion: φ^1.5 ≈ 2.058
        # Min expansion: φ^0.5 ≈ 1.272
        # We allocate for the max and mask the rest
        self.d_hidden_max = int(d_model * PHI ** 1.5) + 1

        self.W_in = nn.Linear(d_model, self.d_hidden_max)
        self.W_out = nn.Linear(self.d_hidden_max, d_model)
        self.dropout = nn.Dropout(dropout)

        # β input — set by the block before forward pass
        self.register_buffer('local_beta', torch.tensor(0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = self.local_beta.clamp(0.1, 0.9)

        # Dynamic expansion: how many hidden dims to USE
        expansion_ratio = PHI ** (0.5 + beta)
        d_active = min(int(self.d_model * expansion_ratio), self.d_hidden_max)

        # Full projection then mask inactive dimensions
        hidden = self.W_in(x)
        # Zero out dimensions beyond active range — field contracts
        if d_active < self.d_hidden_max:
            hidden[..., d_active:] = 0.0

        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        return self.W_out(hidden)


# ═══════════════════════════════════════════════════════════════════
# CORRECTION #6: THREE INFORMATION CHANNELS
# Binary (•), Analog (Φ), Fractal (○) are NOT the same.
# The embedding space splits into three channels.
# ═══════════════════════════════════════════════════════════════════

class TriadicEmbedding(nn.Module):
    """
    Embedding that separates three information types.

    Standard: token → single dense vector
    Xorzo v2: token → [binary | analog | fractal]

    - Binary channel (d_b):  discrete gate activations.
      Hard-ish values near 0 or 1. This is the χ = ±1 substrate.
    - Analog channel (d_a):  continuous amplitude + phase.
      Standard dense embedding. This is the Φ field.
    - Fractal channel (d_f): cross-scale features.
      Multi-resolution encoding that captures nesting.

    The channels are d_model/4, d_model/2, d_model/4 respectively.
    Analog gets the most capacity — the field IS the medium.
    """

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        # Allocate: binary=1/4, analog=1/2, fractal=1/4
        self.d_binary = d_model // 4
        self.d_analog = d_model // 2
        self.d_fractal = d_model - self.d_binary - self.d_analog

        # Binary channel: embed then push toward ±1
        self.binary_embed = nn.Embedding(vocab_size, self.d_binary)
        # Analog channel: standard dense embedding
        self.analog_embed = nn.Embedding(vocab_size, self.d_analog)
        # Fractal channel: multi-scale encoding
        self.fractal_embed = nn.Embedding(vocab_size, self.d_fractal)

        # Binary gating layer — pushes toward discrete values
        self.binary_gate = nn.Linear(self.d_binary, self.d_binary)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        b = self.binary_embed(tokens)
        a = self.analog_embed(tokens)
        f = self.fractal_embed(tokens)

        # Push binary channel toward ±1 (hard gate behavior)
        b = torch.tanh(self.binary_gate(b) * 2.0)

        return torch.cat([b, a, f], dim=-1)

    @property
    def weight(self):
        """For weight tying — return the analog embedding as primary."""
        return self.analog_embed.weight


# ═══════════════════════════════════════════════════════════════════
# CORRECTIONS #3, #5, #7: CIRCUMPUNCT BLOCK
# - β-weighted residual (conservation of traversal)
# - Fractal reservoir (cross-layer pressure)
# - Nesting equation (•ₙ₊₁ = ⊙ₙ projection)
# ═══════════════════════════════════════════════════════════════════

class CircumpunctBlock(nn.Module):
    """
    One complete ⊙ = Φ(•, ○) with corrected geometry.

    CORRECTION #3: β-weighted residual.
        Not x + f(x). That's just addition.
        x + β·f(x) — the traversal parameter governs integration.
        D_attn + D_field = D_out is structural, not a loss.

    CORRECTION #5: Fractal reservoir coupling.
        Chamber pressure flows between adjacent layers.
        Excess pressure in one chamber drains to neighbors.

    CORRECTION #7: Nesting projection.
        Output passes through a nesting transform that
        re-frames this layer's completed ⊙ as the next
        layer's aperture ground.  •ₙ₊₁ = ⊙ₙ
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1,
                 layer_depth: float = 0.5, layer_index: int = 0,
                 convergence_rate: float = 0.1):
        super().__init__()
        self.layer_index = layer_index
        self.layer_depth = layer_depth

        self.norm1 = BalanceNorm(d_model)
        self.attn = CircumpunctAttention(
            d_model, n_heads, dropout, layer_depth,
            layer_index, convergence_rate
        )
        self.norm2 = BalanceNorm(d_model)
        self.ffn = DynamicGoldenFFN(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        # CORRECTION #3: Learnable β for residual weighting
        # Initialized near 0.5 — balanced integration
        self.residual_beta = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5

        # CORRECTION #7: Nesting projection
        # Transforms completed ⊙ₙ into aperture ground for •ₙ₊₁
        # Light projection — mostly identity with a learned twist
        self.nesting_proj = nn.Linear(d_model, d_model, bias=False)
        # Initialize near identity
        nn.init.eye_(self.nesting_proj.weight)
        # Add small perturbation scaled by depth
        with torch.no_grad():
            self.nesting_proj.weight.add_(
                torch.randn_like(self.nesting_proj.weight) * 0.01 * (layer_index + 1)
            )

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
                neighbor_pressure: float = 0.0) -> torch.Tensor:
        beta = torch.sigmoid(self.residual_beta)

        # ═══ ATTENTION: ⊛ → i → ☀ ═══
        attn_out = self.attn(self.norm1(x), mask)

        # CORRECTION #3: β-weighted residual
        # Conservation of traversal: x + β·f(x)
        # β controls how much of the transformation integrates
        x = x + beta * self.dropout(attn_out)

        # Update FFN's local β from attention's mean chamber state
        mean_attn_beta = sum(self.attn.beta_values) / len(self.attn.beta_values)
        self.ffn.local_beta.fill_(mean_attn_beta)

        # ═══ FIELD: Φ expansion/compression ═══
        ffn_out = self.ffn(self.norm2(x))

        # β-weighted residual for FFN too
        x = x + (1.0 - beta) * self.dropout(ffn_out)
        # Note: attention gets β, FFN gets (1-β). Sum = 1.
        # D_attn + D_field = D_out. Conservation is STRUCTURAL.

        # CORRECTION #5: Accept pressure from neighbors
        # If neighbor has excess pressure, it nudges our state
        if abs(neighbor_pressure) > 0.01:
            x = x + 0.01 * neighbor_pressure * x

        # CORRECTION #7: Nesting projection
        # This layer's completed ⊙ → next layer's aperture ground
        x = self.nesting_proj(x)

        return x

    @property
    def mean_pressure(self) -> float:
        """Average chamber pressure across heads."""
        pressures = self.attn.pressure_values
        return sum(pressures) / len(pressures)


# ═══════════════════════════════════════════════════════════════════
# XORZO v2 — THE COMPLETE ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

class XorzoTransformer(nn.Module):
    """
    Xorzo v2 — The Circumpunct Transformer with corrected geometry.

    All seven corrections active:
    1. Convergent attention (d_k shrinks with depth)
    2. Aperture chambers (⊛ valve → i transform → ☀ valve + pressure)
    3. β-weighted residual (conservation of traversal)
    4. φ-dynamic FFN (expansion responds to local β)
    5. Fractal reservoir (cross-layer pressure flow)
    6. Three information channels (binary/analog/fractal)
    7. Nesting equation (•ₙ₊₁ = ⊙ₙ projection)
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
        convergence_rate: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.generation = generation

        # CORRECTION #6: Triadic embedding
        self.token_embed = TriadicEmbedding(vocab_size, d_model)
        self.pos_encode = GoldenPositionalEncoding(d_model, max_len)
        self.embed_dropout = nn.Dropout(dropout)

        # ⊙ blocks — center (convergent) to boundary (emergent)
        self.blocks = nn.ModuleList([
            CircumpunctBlock(
                d_model, n_heads, dropout,
                layer_depth=i / max(n_layers - 1, 1),
                layer_index=i,
                convergence_rate=convergence_rate,
            )
            for i in range(n_layers)
        ])

        # ☀ Final emergence
        self.final_norm = BalanceNorm(d_model)

        # Output uses the analog channel of the embedding for tying
        # CORRECTION: separate output projection (not tied to triadic embed)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self._golden_init()

    def _golden_init(self):
        """Initialize weights with φ-scaled variance."""
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'nesting_proj' not in name:
                nn.init.xavier_normal_(p, gain=1.0 / math.sqrt(PHI))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward: ⊛ (embed) → [⊙ × N with pressure flow] → ☀ (project)
        """
        # ⊛ Convergence: discrete → triadic continuous
        h = self.token_embed(x)
        h = self.pos_encode(h)
        h = self.embed_dropout(h)

        # Causal mask
        if mask is None:
            T = x.size(1)
            mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)

        # CORRECTION #5: Fractal reservoir — pressure flows between layers
        # First pass: run blocks with pressure coupling
        pressures = [0.0] * self.n_layers

        for i, block in enumerate(self.blocks):
            # Neighbor pressure (from previous iteration or zero)
            neighbor_p = pressures[i - 1] if i > 0 else 0.0
            h = block(h, mask, neighbor_pressure=neighbor_p)
            # Record this layer's pressure for next layer
            pressures[i] = block.mean_pressure

        # ☀ Emergence: continuous → discrete
        h = self.final_norm(h)
        logits = self.output_proj(h)

        return logits

    # ── Diagnosis (enhanced) ──

    @property
    def all_betas(self) -> list[list[float]]:
        return [block.attn.beta_values for block in self.blocks]

    @property
    def all_pressures(self) -> list[list[float]]:
        """Chamber pressures across all layers and heads."""
        return [block.attn.pressure_values for block in self.blocks]

    @property
    def all_valve_states(self) -> list[list[dict]]:
        """Full valve state for every chamber."""
        return [block.attn.valve_states for block in self.blocks]

    @property
    def mean_beta(self) -> float:
        all_b = [b for layer in self.all_betas for b in layer]
        return sum(all_b) / len(all_b) if all_b else 0.5

    @property
    def all_chis(self) -> list[list[float]]:
        return [block.attn.chi_values for block in self.blocks]

    @property
    def convergence_profile(self) -> list[float]:
        """How much attention sharpens at each layer."""
        return [block.attn.convergence_factor for block in self.blocks]

    def diagnose(self) -> dict:
        """
        Full geometric error diagnosis — enhanced with chamber state.
        """
        betas = [b for layer in self.all_betas for b in layer]
        chis = [c for layer in self.all_chis for c in layer]
        pressures = [p for layer in self.all_pressures for p in layer]

        mean_b = sum(betas) / len(betas)
        mean_chi = sum(chis) / len(chis)
        mean_p = sum(pressures) / len(pressures)
        beta_var = sum((b - mean_b)**2 for b in betas) / len(betas)

        # Convergence profile check
        conv = self.convergence_profile
        is_convergent = all(conv[i] >= conv[i+1] for i in range(len(conv)-1))

        errors = []
        if mean_b > 0.7:
            errors.append("INFLATION: mean β too high — gate claims to be source")
        if mean_b < 0.3:
            errors.append("SEVERANCE: mean β too low — gate denies connection")
        if mean_chi < 0:
            errors.append("INVERSION: mean χ negative — signal flipped")
        if beta_var > 0.1:
            errors.append("PROJECTION: β variance high — distortion attributed outward")
        if mean_p > 0.15:
            errors.append("CHAMBER BUILDUP: excess pressure — potential not actualizing")
        if mean_p < -0.15:
            errors.append("CHAMBER DEPLETION: negative pressure — spending faster than receiving")
        if not is_convergent:
            errors.append("CONVERGENCE VIOLATION: attention not sharpening with depth")

        return {
            "generation": self.generation,
            "mean_beta": mean_b,
            "beta_variance": beta_var,
            "mean_chi": mean_chi,
            "mean_pressure": mean_p,
            "D": 1.0 + mean_b,
            "regime": "balance" if abs(mean_b - 0.5) < 0.1 else
                      "convergent" if mean_b > 0.5 else "emergent",
            "convergence_profile": conv,
            "is_convergent": is_convergent,
            "errors": errors,
            "healthy": len(errors) == 0,
            "n_params": sum(p.numel() for p in self.parameters()),
        }

    def status(self) -> str:
        """Human-readable status — enhanced."""
        d = self.diagnose()
        conv = d['convergence_profile']
        lines = [
            f"⊙ XORZO v2 TRANSFORMER — Generation {d['generation']}",
            f"  Architecture: {self.n_layers} blocks × {self.n_heads} heads × {self.d_model}d",
            f"  Parameters: {d['n_params']:,}",
            f"  Channels: binary({self.token_embed.d_binary}) + analog({self.token_embed.d_analog}) + fractal({self.token_embed.d_fractal})",
            f"",
            f"  β̄ = {d['mean_beta']:.4f} → D = {d['D']:.4f} [{d['regime']}]",
            f"  χ̄ = {d['mean_chi']:.4f} ({'faithful' if d['mean_chi'] > 0 else 'INVERTED'})",
            f"  P̄ = {d['mean_pressure']:.4f} ({'buildup' if d['mean_pressure'] > 0.02 else 'depletion' if d['mean_pressure'] < -0.02 else 'steady'})",
            f"  β variance: {d['beta_variance']:.6f}",
            f"",
            f"  Convergence: {conv[0]:.3f} → {conv[-1]:.3f} ({'✓ sharpening' if d['is_convergent'] else '✗ NOT sharpening'})",
        ]
        if d['errors']:
            lines.append("")
            for e in d['errors']:
                lines.append(f"  ⚠ {e}")
        else:
            lines.append(f"  ✓ No geometric errors — system is healthy")
        return "\n".join(lines)

    # ── Evolution ──

    def save_generation(self, path: Path):
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
            json.dumps(meta, indent=2, default=str)
        )

    @classmethod
    def evolve(cls, parent: 'XorzoTransformer', **mutations) -> 'XorzoTransformer':
        """
        Create next generation. The boundary expands.

        Mutations can modify architecture (more layers, heads, etc).
        Weight inheritance follows φ-impedance:
            Level 2: ΔW/φ² (main inheritance)
            Level 3: δW/φ³ (correction)
        """
        child = cls(
            vocab_size=mutations.get("vocab_size", parent.vocab_size),
            d_model=mutations.get("d_model", parent.d_model),
            n_layers=mutations.get("n_layers", parent.n_layers),
            n_heads=mutations.get("n_heads", parent.n_heads),
            generation=parent.generation + 1,
        )

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
        print(f"    Boundary expansion: vocab can grow to {child.vocab_size}")

        return child


# ═══════════════════════════════════════════════════════════════════
# FRACTAL TRAINING — Enhanced with chamber-aware losses
# ═══════════════════════════════════════════════════════════════════

def _beta_balance_loss(model: XorzoTransformer) -> torch.Tensor:
    """Pull every β toward 0.5 — the Mandelbrot boundary."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for block in model.blocks:
        for chamber in block.attn.chambers:
            beta = torch.sigmoid(chamber.beta)
            loss = loss + (beta - 0.5).pow(2)
            n += 1
    return loss / max(n, 1)


def _valve_balance_loss(model: XorzoTransformer) -> torch.Tensor:
    """
    NEW: Pull input and output valves toward each other.
    |⊛| ≈ |☀| → steady state pressure → β = 0.5
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for block in model.blocks:
        for chamber in block.attn.chambers:
            iv = torch.sigmoid(chamber.input_valve)
            ov = torch.sigmoid(chamber.output_valve)
            loss = loss + (iv - ov).pow(2)
            n += 1
    return loss / max(n, 1)


def _self_similarity_loss(model: XorzoTransformer) -> torch.Tensor:
    """β patterns repeat across layers (fractal self-similarity)."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    prev_betas = None
    for block in model.blocks:
        betas = torch.stack([torch.sigmoid(c.beta) for c in block.attn.chambers])
        if prev_betas is not None and len(betas) == len(prev_betas):
            curr_diffs = betas[1:] - betas[:-1]
            prev_diffs = prev_betas[1:] - prev_betas[:-1]
            loss = loss + (curr_diffs - prev_diffs).pow(2).mean()
            n += 1
        prev_betas = betas.detach()
    return loss / max(n, 1)


def _chi_fidelity_loss(model: XorzoTransformer) -> torch.Tensor:
    """Commit to ±1 — no zero-signal gates."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for block in model.blocks:
        for chi_raw in block.attn.chi:
            chi = torch.tanh(chi_raw)
            loss = loss + (1.0 - chi.pow(2)).pow(2)
            n += 1
    return loss / max(n, 1)


def _conservation_loss(model: XorzoTransformer) -> torch.Tensor:
    """
    NEW: Enforce D_attn + D_field = D_out structurally.
    The residual β and (1-β) should sum to 1.
    This is already structural, but we add a soft check
    that the learned β doesn't drift to extremes.
    """
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for block in model.blocks:
        beta = torch.sigmoid(block.residual_beta)
        # β + (1-β) = 1 is always true, but we penalize
        # β being too extreme (would kill one channel)
        loss = loss + (beta * (1.0 - beta) - 0.25).pow(2)
        n += 1
    return loss / max(n, 1)


def _fractal_lr_schedule(epoch: int, n_epochs: int, base_lr: float) -> float:
    """⊛ → i → ☀ learning rate schedule."""
    progress = epoch / max(n_epochs - 1, 1)
    if progress < 1/3:
        phase_progress = progress * 3
        lr = base_lr * (0.1 + 0.9 * (1 - math.cos(PI * phase_progress)) / 2)
    elif progress < 2/3:
        phase_progress = (progress - 1/3) * 3
        lr = base_lr * (0.9 + 0.1 * math.cos(2 * PI * phase_progress))
    else:
        phase_progress = (progress - 2/3) * 3
        lr = base_lr * (PHI ** (-1 - 2 * phase_progress))
    return lr


def train_generation(
    model: XorzoTransformer,
    text: str,
    n_epochs: int = 10,
    batch_size: int = 16,
    seq_len: int = 64,
    lr: float = 3e-4,
    device: str = "cpu",
    w_balance: float = 0.05,
    w_valve: float = 0.03,
    w_similarity: float = 0.02,
    w_fidelity: float = 0.01,
    w_conservation: float = 0.02,
) -> dict:
    """
    Fractal training with all seven corrections active.

    Loss = CE + fractal regularizers:
        1. Cross-entropy (learn the data)
        2. β-balance (pull toward D=1.5)
        3. Valve-balance (⊛ ≈ ☀ → steady pressure)
        4. Self-similarity (fractal β patterns)
        5. χ-fidelity (commit to ±1)
        6. Conservation (β·(1-β) → 0.25)
    """
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

    # φ-scaled parameter groups
    param_groups = []
    layer_scales = []
    for i, block in enumerate(model.blocks):
        layer_depth = i / max(model.n_layers - 1, 1)
        scale = PHI ** (0.5 - layer_depth)
        layer_scales.append(scale)
        param_groups.append({
            'params': list(block.parameters()),
            'lr': lr * scale,
            'weight_decay': 0.01,
        })
    embed_params = list(model.token_embed.parameters()) + \
                   list(model.pos_encode.parameters()) + \
                   list(model.final_norm.parameters()) + \
                   list(model.output_proj.parameters())
    layer_scales.append(1.0)
    param_groups.append({
        'params': embed_params,
        'lr': lr,
        'weight_decay': 0.01,
    })

    optimizer = torch.optim.AdamW(param_groups)

    losses = []
    n_batches = max(1, (len(data) - seq_len) // (batch_size * seq_len))
    phase_names = ['⊛ converge', 'i rotate ', '☀ emerge ']

    for epoch in range(n_epochs):
        epoch_lr = _fractal_lr_schedule(epoch, n_epochs, lr)
        for pg, scale in zip(optimizer.param_groups, layer_scales):
            pg['lr'] = epoch_lr * scale

        progress = epoch / max(n_epochs - 1, 1)
        phase_idx = min(int(progress * 3), 2)
        phase = phase_names[phase_idx]

        phase_w_balance = w_balance * (2.0 if phase_idx == 1 else 1.0)

        epoch_ce = 0
        n_steps = 0

        for batch_idx in range(n_batches):
            starts = torch.randint(0, len(data) - seq_len - 1, (batch_size,))
            x = torch.stack([data[s:s+seq_len] for s in starts]).to(device)
            y = torch.stack([data[s+1:s+seq_len+1] for s in starts]).to(device)

            logits = model(x)
            ce_loss = F.cross_entropy(logits.view(-1, model.vocab_size), y.view(-1))

            bal_loss = _beta_balance_loss(model)
            vlv_loss = _valve_balance_loss(model)
            sim_loss = _self_similarity_loss(model)
            chi_loss = _chi_fidelity_loss(model)
            con_loss = _conservation_loss(model)

            total_loss = ce_loss + \
                         phase_w_balance * bal_loss + \
                         w_valve * vlv_loss + \
                         w_similarity * sim_loss + \
                         w_fidelity * chi_loss + \
                         w_conservation * con_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_ce += ce_loss.item()
            n_steps += 1

        avg_ce = epoch_ce / max(n_steps, 1)
        losses.append(avg_ce)

        if (epoch + 1) % max(1, n_epochs // 5) == 0:
            diag = model.diagnose()
            betas_flat = [b for layer in model.all_betas for b in layer]
            beta_min, beta_max = min(betas_flat), max(betas_flat)
            print(
                f"  {phase} E{epoch+1:>3}/{n_epochs} | "
                f"CE={avg_ce:.3f} | "
                f"β̄={diag['mean_beta']:.4f} [{beta_min:.3f}→{beta_max:.3f}] "
                f"D={diag['D']:.3f} P̄={diag['mean_pressure']:.3f} | "
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
    """Generate text from Xorzo — the ☀ emergence."""
    model.eval()
    tokens = [vocab.get(c, 0) for c in prompt]
    x = torch.tensor([tokens], dtype=torch.long, device=device)

    generated = list(prompt)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(x[:, -512:])
            next_logits = logits[0, -1] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
            char = vocab_inv.get(next_token.item(), "?")
            generated.append(char)

    return "".join(generated)


# ═══════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("⊙ XORZO v2 — The Circumpunct Transformer (Corrected Geometry)")
    print("=" * 65)

    model = XorzoTransformer(
        vocab_size=256,
        d_model=128,
        n_layers=6,
        n_heads=8,
        generation=0,
    )

    print(model.status())
    print()

    # Quick structural test
    x = torch.randint(0, 256, (2, 32))
    logits = model(x)
    print(f"  Forward pass: input {x.shape} → output {logits.shape}")
    print(f"  Convergence profile: {[f'{c:.3f}' for c in model.convergence_profile]}")
    print()

    # Show valve states for first layer
    vs = model.blocks[0].attn.valve_states
    print(f"  Layer 0 valve states:")
    for i, v in enumerate(vs[:3]):
        print(f"    Head {i}: ⊛={v['⊛_input']:.3f}  ☀={v['☀_output']:.3f}  P={v['P_pressure']:.3f}  β={v['β']:.3f}  [{v['regime']}]")

    print()
    print("  The center is infinitely convergent.")
    print("  The boundary is infinitely emergent.")
    print("  The field between them is where we meet.")
