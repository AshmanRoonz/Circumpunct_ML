"""
XORZO v3 — Cross-Scale Resonant Circumpunct Transformer

v1: Circumpunct as computation (β, χ, golden ratio)
v2: Seven structural corrections (chambers, pressure, triadic embedding, nesting)
v3: Cross-scale architecture with phase-resonance coupling

NEW in v3:
    8. CROSS-SCALE NESTING — Two simultaneous streams:
       - Micro (token-level): your body, your cells, the local
       - Macro (chunk-level): your soul, the aim, the whole
       Continuously braided through ⊛ and ☀.

    9. PHASE RESONANCE — Binding criterion:
       Resonance = cos(Δθ) across complex pairs.
       Cross-scale coupling is AMPLIFIED when phases align.
       This is what decides when Φ actually binds vs just mixing.
       Truth passes cleanly through the singularity.

    10. AIM TOKENS — Each chunk's macro center is a learned
        convergent singularity (not just a mean-pool).
        The soul is the aim. The lean before there's anything
        to lean toward.

Architecture:
    At every layer:
        1. Micro ⊙ self-attends (local body-mind loop)
        2. Macro re-grounds from micro (the whole re-reads its parts)
        3. Macro ⊙ self-attends (higher-order context loop)
        4. Cross-scale coupling:
           ⊛ Convergence: macro reads micro (gathering into center)
           ☀ Emergence: micro reads macro (field shaping boundary)
           Both gated by • aperture AND resonance coherence

Everything from v2 is preserved:
    - Aperture chambers (⊛ → i → ☀ with pressure)
    - Convergent attention (d_k shrinks with depth)
    - β-weighted residual (conservation of traversal)
    - φ-dynamic FFN (expansion responds to local β)
    - Three information channels (binary/analog/fractal)
    - Nesting equation (•ₙ₊₁ = ⊙ₙ)
    - χ gates (faithful/inverted)
    - Golden positional encoding
    - Fractal training (⊛→i→☀ LR schedule)
    - Four geometric error diagnosis

⊙ = Φ(•, ○) at every scale, between every scale.
"""

import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

PHI = (1 + math.sqrt(5)) / 2
PI = math.pi


# ═══════════════════════════════════════════════════════════════════
# PHASE RESONANCE — NEW in v3
# The binding criterion. When oscillations align, signal couples.
# ═══════════════════════════════════════════════════════════════════

def phase_resonance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute phase-coherence resonance between two sets of vectors.

    Interprets adjacent pairs as (re, im) components of complex numbers.
    Resonance = mean cos(Δθ) across all pairs.

    a: (B, Tq, D)   — query vectors
    b: (B, Tk, D)   — key vectors
    returns: (B, Tq, Tk) resonance matrix in [-1, 1]

    +1 = perfect phase alignment (truth passes cleanly)
     0 = random/orthogonal (no binding)
    -1 = anti-phase (inversion)
    """
    assert a.size(-1) % 2 == 0
    pairs = a.size(-1) // 2

    def to_phase(x):
        x2 = x.view(*x.shape[:-1], pairs, 2)
        return torch.atan2(x2[..., 1] + eps, x2[..., 0] + eps)

    theta_a = to_phase(a)  # (B, Tq, pairs)
    theta_b = to_phase(b)  # (B, Tk, pairs)

    # Pairwise phase difference
    delta = theta_a.unsqueeze(2) - theta_b.unsqueeze(1)  # (B, Tq, Tk, pairs)
    r = torch.cos(delta).mean(dim=-1)  # (B, Tq, Tk)

    return r


# ═══════════════════════════════════════════════════════════════════
# GOLDEN POSITIONAL ENCODING (from v1)
# ═══════════════════════════════════════════════════════════════════

class GoldenPositionalEncoding(nn.Module):
    """Positional encoding using the golden angle (2π/φ²)."""

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
# APERTURE CHAMBER (from v2)
# ⊛ input valve → i transform → ☀ output valve + pressure
# ═══════════════════════════════════════════════════════════════════

class ApertureChamber(nn.Module):
    """Three-stage aperture: ⊛ → i → ☀ with pressure tracking."""

    def __init__(self, d_head: int, depth: float = 0.5):
        super().__init__()
        self.d_head = d_head
        self.depth = depth

        beta_init = 0.5 + 0.2 * (1.0 - 2.0 * depth)
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.input_valve = nn.Parameter(torch.tensor(0.5))
        self.output_valve = nn.Parameter(torch.tensor(0.5))
        self.register_buffer('pressure', torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.sigmoid(self.beta)
        iv = torch.sigmoid(self.input_valve)
        ov = torch.sigmoid(self.output_valve)

        # ⊛ input valve
        x_in = x * iv

        # i transform (complex rotation by πβ)
        angle = PI * beta
        half = self.d_head // 2
        x_real = x_in[..., :half]
        x_imag = x_in[..., half:2*half]

        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        out_real = x_real * cos_a - x_imag * sin_a
        out_imag = x_real * sin_a + x_imag * cos_a

        if self.d_head % 2 == 1:
            x_transformed = torch.cat([out_real, out_imag, x_in[..., -1:]], dim=-1)
        else:
            x_transformed = torch.cat([out_real, out_imag], dim=-1)

        # ☀ output valve
        x_out = x_transformed * ov

        # Update pressure
        with torch.no_grad():
            dp = iv.detach() - ov.detach()
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
# CIRCUMPUNCT ATTENTION (from v2, with convergent scaling)
# ═══════════════════════════════════════════════════════════════════

class CircumpunctAttention(nn.Module):
    """
    Bilateral Circumpunct Attention: ⊙ = Φ(•, ○)

    Every token is a circumpunct with:
      • aperture_i  = σ(W_a · h_i)  — how open am I to receive?
      ○ expression_j = σ(W_e · h_j)  — how much am I offering?
      alignment_ij  = f(•_i, ○_j)   — does what you express match what I admit?

    attn_weight_ij = aperture_i × expression_j × softmax(alignment_ij)

    This is fundamentally bilateral: both sender and receiver participate.
    Standard attention is one-sided (Q·K similarity). Here, a token can
    be CLOSED (low aperture), QUIET (low expression), or SELECTIVELY
    RESONANT (high alignment with specific partners).

    A token with agency. That's what makes it a circumpunct.
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

        self.convergence_factor = 1.0 / (PHI ** (layer_index * convergence_rate))
        self.scale = math.sqrt(self.d_head * self.convergence_factor)

        # • Aperture: per-head gate on the RECEIVER (how open am I?)
        # Projects each token to n_heads scalars
        self.W_aperture = nn.Linear(d_model, n_heads, bias=True)

        # ○ Expression: per-head gate on the SENDER (how much do I offer?)
        self.W_expression = nn.Linear(d_model, n_heads, bias=True)

        # Alignment projections (replaces Q/K with circumpunct semantics)
        # • inner projection: what am I looking for? (receiver's center)
        self.W_inner = nn.Linear(d_model, d_model, bias=False)
        # ○ outer projection: what am I offering? (sender's boundary)
        self.W_outer = nn.Linear(d_model, d_model, bias=False)
        # Value: what content flows through
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        self.chambers = nn.ModuleList([
            ApertureChamber(
                self.d_head,
                depth=self._head_depth(h, n_heads, layer_depth)
            )
            for h in range(n_heads)
        ])

        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.chi = nn.Parameter(torch.ones(n_heads))

    @staticmethod
    def _head_depth(head_idx, n_heads, layer_depth):
        head_fraction = head_idx / max(n_heads - 1, 1)
        lo = layer_depth * 0.5
        hi = 0.5 + layer_depth * 0.5
        return lo + head_fraction * (hi - lo)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape

        # ═══ • APERTURE: how open is each token to receive? ═══
        # (B, T, n_heads) -> (B, n_heads, T, 1)
        aperture = torch.sigmoid(self.W_aperture(x))
        aperture = aperture.permute(0, 2, 1).unsqueeze(-1)  # (B, H, T, 1)

        # ═══ ○ EXPRESSION: how much is each token offering? ═══
        # (B, T, n_heads) -> (B, n_heads, 1, T)
        expression = torch.sigmoid(self.W_expression(x))
        expression = expression.permute(0, 2, 1).unsqueeze(-2)  # (B, H, 1, T)

        # ═══ ALIGNMENT: does •_i resonate with ○_j? ═══
        # Inner (•): what the receiver's center seeks
        inner = self.W_inner(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        # Outer (○): what the sender's boundary offers
        outer = self.W_outer(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Alignment score: how well does ○_j match •_i?
        alignment = torch.matmul(inner, outer.transpose(-2, -1)) / self.scale  # (B, H, T, T)
        if mask is not None:
            alignment = alignment.masked_fill(mask == 0, float('-inf'))

        # ═══ BILATERAL ATTENTION: aperture × expression × alignment ═══
        # Softmax over alignment (which tokens resonate)
        attn_align = F.softmax(alignment, dim=-1)  # (B, H, T, T)

        # Modulate by bilateral gates: receiver's openness × sender's willingness
        # A closed token (low aperture) receives nothing regardless of alignment
        # A quiet token (low expression) sends nothing regardless of alignment
        attn = attn_align * aperture * expression  # (B, H, T, T)

        # Re-normalize after gating so attention still sums to meaningful values
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        attn = self.dropout(attn)

        # Value flow
        v = self.W_v(x).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        converged = torch.matmul(attn, v)  # (B, H, T, d_head)

        # Chamber processing per head (pressure, β dynamics)
        rotated = []
        for h in range(self.n_heads):
            chambered = self.chambers[h](converged[:, h])
            chi_h = torch.tanh(self.chi[h])
            rotated.append(chambered * chi_h)

        combined = torch.stack(rotated, dim=1)
        combined = combined.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_out(combined)

    @property
    def beta_values(self):
        return [c.current_beta for c in self.chambers]

    @property
    def chi_values(self):
        return [torch.tanh(c).item() for c in self.chi]

    @property
    def pressure_values(self):
        return [c.current_pressure for c in self.chambers]

    @property
    def aperture_stats(self):
        """Return mean aperture and expression for diagnostics."""
        return {"layer": self.layer_index}

    @property
    def valve_states(self):
        return [c.valve_state for c in self.chambers]


# ═══════════════════════════════════════════════════════════════════
# BALANCE NORM (from v1/v2)
# ═══════════════════════════════════════════════════════════════════

class BalanceNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)
        self.balance_weight = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        normed = self.norm(x)
        half = x.shape[-1] // 2
        e1 = x[..., :half].pow(2).mean(dim=-1, keepdim=True)
        e2 = x[..., half:].pow(2).mean(dim=-1, keepdim=True)
        imbalance = (e1 - e2) * self.balance_weight
        correction = torch.zeros_like(x)
        correction[..., :half] = -imbalance
        correction[..., half:] = imbalance
        return normed + correction


# ═══════════════════════════════════════════════════════════════════
# φ-DYNAMIC FFN (from v2)
# ═══════════════════════════════════════════════════════════════════

class DynamicGoldenFFN(nn.Module):
    """FFN with β-responsive expansion. The field breathes."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_hidden_max = int(d_model * PHI ** 1.5) + 1
        self.W_in = nn.Linear(d_model, self.d_hidden_max)
        self.W_out = nn.Linear(self.d_hidden_max, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('local_beta', torch.tensor(0.5))

    def forward(self, x):
        beta = self.local_beta.clamp(0.1, 0.9)
        expansion_ratio = PHI ** (0.5 + beta)
        d_active = min(int(self.d_model * expansion_ratio), self.d_hidden_max)
        hidden = self.W_in(x)
        if d_active < self.d_hidden_max:
            hidden[..., d_active:] = 0.0
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        return self.W_out(hidden)


# ═══════════════════════════════════════════════════════════════════
# TRIADIC EMBEDDING (from v2)
# Binary (•) + Analog (Φ) + Fractal (○)
# ═══════════════════════════════════════════════════════════════════

class TriadicEmbedding(nn.Module):
    """Three-channel embedding: binary(1/4) + analog(1/2) + fractal(1/4)."""

    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.d_binary = d_model // 4
        self.d_analog = d_model // 2
        self.d_fractal = d_model - self.d_binary - self.d_analog

        self.binary_embed = nn.Embedding(vocab_size, self.d_binary)
        self.analog_embed = nn.Embedding(vocab_size, self.d_analog)
        self.fractal_embed = nn.Embedding(vocab_size, self.d_fractal)
        self.binary_gate = nn.Linear(self.d_binary, self.d_binary)

    def forward(self, tokens):
        b = torch.tanh(self.binary_gate(self.binary_embed(tokens)) * 2.0)
        a = self.analog_embed(tokens)
        f = self.fractal_embed(tokens)
        return torch.cat([b, a, f], dim=-1)

    @property
    def weight(self):
        return self.analog_embed.weight


# ═══════════════════════════════════════════════════════════════════
# CIRCUMPUNCT BLOCK (from v2 with all corrections)
# β-weighted residual, pressure flow, nesting equation
# ═══════════════════════════════════════════════════════════════════

class CircumpunctBlock(nn.Module):
    """
    One complete ⊙ = Φ(•, ○) with corrected geometry.
    Corrections #1-7 from v2 active.
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

        self.residual_beta = nn.Parameter(torch.tensor(0.0))

        self.nesting_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.nesting_proj.weight)
        with torch.no_grad():
            self.nesting_proj.weight.add_(
                torch.randn_like(self.nesting_proj.weight) * 0.01 * (layer_index + 1)
            )

    def forward(self, x, mask=None, neighbor_pressure=0.0):
        beta = torch.sigmoid(self.residual_beta)

        attn_out = self.attn(self.norm1(x), mask)
        x = x + beta * self.dropout(attn_out)

        mean_attn_beta = sum(self.attn.beta_values) / len(self.attn.beta_values)
        self.ffn.local_beta.fill_(mean_attn_beta)

        ffn_out = self.ffn(self.norm2(x))
        x = x + (1.0 - beta) * self.dropout(ffn_out)

        if abs(neighbor_pressure) > 0.01:
            x = x + 0.01 * neighbor_pressure * x

        x = self.nesting_proj(x)
        return x

    @property
    def mean_pressure(self):
        pressures = self.attn.pressure_values
        return sum(pressures) / len(pressures)


# ═══════════════════════════════════════════════════════════════════
# NEW v3: CROSS-SCALE COUPLING with PHASE RESONANCE
# ⊛ Convergence: macro reads micro (gathering into center)
# ☀ Emergence: micro reads macro (field shaping boundary)
# Gated by aperture AND resonance coherence
# ═══════════════════════════════════════════════════════════════════

class CrossScaleCoupling(nn.Module):
    """
    Cross-scale circumpunct coupling with ACCUMULATED phase-resonance.

    Two operations:
        ⊛ Convergence: macro attends to micro (compression/gathering)
        ☀ Emergence: micro attends to macro (guidance/shaping)

    Both gated by:
        • aperture gate (bilateral — receiver opens, sender expresses)
        × resonance (accumulated cos Δθ across layers — not single-pass)

    Resonance accumulates across layers:
        r^(l) = λ · r^(l-1) + (1-λ) · cos(Δθ^(l))

    This turns depth into time. Early layers form tentative connections.
    If those connections sustain phase coherence layer after layer,
    resonance grows and weight strengthens. If alignment was accidental,
    resonance decays and weight weakens.

    A stranger says something relevant — you hear it (alignment, no resonance).
    Someone in deep conversation says the same — it lands (alignment + resonance).
    Same content. Different binding strength. The channel was already open.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1,
                 resonance_lambda: float = 0.7):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.norm_micro = BalanceNorm(d_model)
        self.norm_macro = BalanceNorm(d_model)

        # Aperture gates for each stream
        self.gate_micro = nn.Linear(d_model, 1)
        self.gate_macro = nn.Linear(d_model, 1)

        # ⊛ Convergence: macro queries, micro keys/values
        self.conv_q = nn.Linear(d_model, d_model, bias=False)
        self.conv_k = nn.Linear(d_model, d_model, bias=False)
        self.conv_v = nn.Linear(d_model, d_model, bias=False)
        self.conv_out = nn.Linear(d_model, d_model, bias=False)

        # ☀ Emergence: micro queries, macro keys/values
        self.emer_q = nn.Linear(d_model, d_model, bias=False)
        self.emer_k = nn.Linear(d_model, d_model, bias=False)
        self.emer_v = nn.Linear(d_model, d_model, bias=False)
        self.emer_out = nn.Linear(d_model, d_model, bias=False)

        # Resonance strength learnable scaling
        self.resonance_alpha = nn.Parameter(torch.tensor(1.0))

        # λ for resonance EMA — how much history matters
        # High λ = deep memory, connections slow to build but stable
        # Low λ = fast adaptation, responds to immediate coherence
        self.resonance_lambda = resonance_lambda

        self.dropout = nn.Dropout(dropout)

    def _mha(self, q_proj, k_proj, v_proj, out_proj, q_in, kv_in, mask=None,
             aperture_gate=None, expression_gate=None):
        """
        Bilateral cross-attention helper.

        If aperture_gate (B, Tq, 1) and expression_gate (B, Tk, 1) are provided,
        attention is modulated: receiver's openness × sender's willingness × alignment.
        Otherwise falls back to standard cross-attention.
        """
        B = q_in.size(0)
        Tq = q_in.size(1)
        Tk = kv_in.size(1)

        q = q_proj(q_in).view(B, Tq, self.n_heads, self.d_head).transpose(1, 2)
        k = k_proj(kv_in).view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)
        v = v_proj(kv_in).view(B, Tk, self.n_heads, self.d_head).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        if mask is not None:
            scores = scores + mask
        attn = F.softmax(scores, dim=-1)

        # Bilateral gating: aperture of receiver × expression of sender
        if aperture_gate is not None and expression_gate is not None:
            # aperture_gate: (B, Tq, 1) -> (B, 1, Tq, 1) for broadcasting over heads
            a_gate = aperture_gate.unsqueeze(1)   # (B, 1, Tq, 1)
            # expression_gate: (B, Tk, 1) -> (B, 1, 1, Tk) for broadcasting
            e_gate = expression_gate.unsqueeze(1).transpose(2, 3)  # (B, 1, 1, Tk)
            attn = attn * a_gate * e_gate
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)

        ctx = ctx.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return out_proj(ctx)

    def forward(self, o_micro, o_macro, um_mask=None, mu_mask=None,
                r_conv_acc=None, r_emer_acc=None):
        """
        o_micro: (B, Tu, D) — micro stream (tokens)
        o_macro: (B, Tm, D) — macro stream (chunks/aims)
        um_mask: mask for macro-queries-micro-keys
        mu_mask: mask for micro-queries-macro-keys
        r_conv_acc: (B, Tm, Tu) — accumulated convergence resonance from previous layers
        r_emer_acc: (B, Tu, Tm) — accumulated emergence resonance from previous layers

        Returns: (o_micro_new, o_macro_new, r_conv_acc_new, r_emer_acc_new)
        """
        ou = self.norm_micro(o_micro)
        om = self.norm_macro(o_macro)

        # Aperture gates
        g_micro = torch.sigmoid(self.gate_micro(ou))   # (B, Tu, 1)
        g_macro = torch.sigmoid(self.gate_macro(om))   # (B, Tm, 1)

        # ═══ INSTANTANEOUS PHASE COHERENCE ═══
        r_conv_instant = phase_resonance(om, ou)  # (B, Tm, Tu)
        r_emer_instant = phase_resonance(ou, om)  # (B, Tu, Tm)

        # ═══ RESONANCE ACCUMULATION ═══
        # r^(l) = λ · r^(l-1) + (1-λ) · cos(Δθ^(l))
        # Depth becomes time. Connections that sustain coherence strengthen.
        # Accidental alignment decays. History matters.
        lam = self.resonance_lambda

        if r_conv_acc is None:
            r_conv_acc_new = r_conv_instant
        else:
            r_conv_acc_new = lam * r_conv_acc + (1 - lam) * r_conv_instant

        if r_emer_acc is None:
            r_emer_acc_new = r_emer_instant
        else:
            r_emer_acc_new = lam * r_emer_acc + (1 - lam) * r_emer_instant

        # Use ACCUMULATED resonance (not instantaneous) for gating
        # This is the difference between a stranger's words and a friend's
        r_conv_scalar = torch.sigmoid(
            self.resonance_alpha * r_conv_acc_new.mean(dim=-1, keepdim=True)
        )  # (B, Tm, 1)

        r_emer_scalar = torch.sigmoid(
            self.resonance_alpha * r_emer_acc_new.mean(dim=-1, keepdim=True)
        )  # (B, Tu, 1)

        # ═══ ⊛ CONVERGENCE: macro reads micro ═══
        # Bilateral: macro's aperture (receiver) × micro's expression (sender)
        delta_macro = self._mha(
            self.conv_q, self.conv_k, self.conv_v, self.conv_out,
            om, ou, mask=um_mask,
            aperture_gate=g_macro,     # macro decides how open to receive
            expression_gate=g_micro    # micro decides how much to offer
        )

        # Gated by ACCUMULATED resonance
        o_macro_new = o_macro + self.dropout(delta_macro * r_conv_scalar)

        # ═══ ☀ EMERGENCE: micro reads macro ═══
        # Bilateral: micro's aperture (receiver) × macro's expression (sender)
        delta_micro = self._mha(
            self.emer_q, self.emer_k, self.emer_v, self.emer_out,
            ou, om, mask=mu_mask,
            aperture_gate=g_micro,     # micro decides how open to receive
            expression_gate=g_macro    # macro decides how much to offer
        )

        # Gated by ACCUMULATED resonance
        o_micro_new = o_micro + self.dropout(delta_micro * r_emer_scalar)

        return o_micro_new, o_macro_new, r_conv_acc_new, r_emer_acc_new


# ═══════════════════════════════════════════════════════════════════
# NEW v3: AIM TOKEN INITIALIZATION
# The macro center is a learned convergent singularity per chunk.
# Not just mean-pool. The soul is the aim.
# ═══════════════════════════════════════════════════════════════════

class AimPool(nn.Module):
    """
    Create macro tokens from micro tokens.

    Instead of simple mean-pooling, each chunk gets a learned
    "aim vector" that attends to its micro tokens to become
    the chunk's convergent center.

    The aim is the wanting. The lean before there's anything
    to lean toward.
    """

    def __init__(self, d_model: int, max_chunks: int = 128):
        super().__init__()
        self.d_model = d_model
        # Learned aim vectors — one per possible chunk position
        self.aim_vectors = nn.Parameter(torch.randn(max_chunks, d_model) * 0.02)
        self.aim_attn = nn.Linear(d_model, d_model, bias=False)
        self.aim_key = nn.Linear(d_model, d_model, bias=False)

    def forward(self, o_micro: torch.Tensor, chunk_size: int) -> torch.Tensor:
        """
        o_micro: (B, T, D)
        returns: (B, Tm, D) where Tm = ceil(T/chunk_size)
        """
        B, T, D = o_micro.shape
        Tm = (T + chunk_size - 1) // chunk_size

        # Pad micro to full chunks
        pad = Tm * chunk_size - T
        if pad > 0:
            o_micro = F.pad(o_micro, (0, 0, 0, pad))

        # Reshape: (B, Tm, chunk_size, D)
        chunks = o_micro.view(B, Tm, chunk_size, D)

        # Aim vectors for this many chunks
        aims = self.aim_vectors[:Tm].unsqueeze(0).expand(B, -1, -1)  # (B, Tm, D)

        # Aim attends to chunk contents
        aim_q = self.aim_attn(aims)       # (B, Tm, D)
        chunk_k = self.aim_key(chunks)     # (B, Tm, cs, D)

        # Attention: aim queries, chunk keys
        scores = torch.einsum('btd,btsd->bts', aim_q, chunk_k) / math.sqrt(D)
        attn = F.softmax(scores, dim=-1)  # (B, Tm, cs)

        # Weighted sum of chunk values
        pooled = torch.einsum('bts,btsd->btd', attn, chunks)  # (B, Tm, D)

        # Blend with aim (residual — the aim persists)
        return aims + pooled


# ═══════════════════════════════════════════════════════════════════
# VESICA BIRTH — New circumpuncts from resonant overlap
#
# When two tokens sustain phase coherence across layers, their
# overlap IS a new ⊙. Not a stronger edge. A child.
#
# "Hot" is ⊙. "Dog" is ⊙. "Hotdog" is a NEW ⊙ born from
# their vesica — with its own •, Φ, ○ that can't be decomposed
# back without losing what it is.
#
# Birth requires sustained resonance, not single-pass alignment.
# The threshold IS the resonance accumulation. Only pairs that
# maintain phase coherence across sufficient depth cross into
# generating a new ⊙. Everything else decays.
#
# Dense attention, sparse generation. Most interactions don't
# reproduce. Few crystallize into persistent structure.
# ═══════════════════════════════════════════════════════════════════

class VesicaBirth(nn.Module):
    """
    Detect resonant overlaps and birth new macro tokens (⊙ children).

    Monitors accumulated emergence resonance (micro→macro).
    When micro tokens in a chunk show strong accumulated resonance,
    their midpoint representation becomes a new aim token —
    a child circumpunct born from the vesica of its parents.

    The child has:
        • = aperture from both parents' shared opening
        ○ = boundary defined by the intersection contour
        Φ = mediation between what both contribute

    Sparsity is intrinsic: only top-k most resonant overlaps birth.
    """

    def __init__(self, d_model: int, max_births_per_layer: int = 4,
                 birth_threshold: float = 0.6):
        super().__init__()
        self.d_model = d_model
        self.max_births = max_births_per_layer
        self.birth_threshold = birth_threshold

        # Vesica field: how to combine two parents into a child
        # The child's representation = Φ(parent_i, parent_j)
        self.vesica_field = nn.Linear(d_model * 2, d_model, bias=False)

        # Birth gate: does this overlap stabilize into a new ⊙?
        self.birth_gate = nn.Linear(d_model, 1)

        # Learnable birth threshold (starts at birth_threshold, can shift)
        self.threshold_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, o_micro, o_macro, r_emer_acc, chunk_size):
        """
        o_micro: (B, Tu, D) — micro stream
        o_macro: (B, Tm, D) — macro stream
        r_emer_acc: (B, Tu, Tm) — accumulated emergence resonance
        chunk_size: int

        Returns: (o_macro_expanded, n_births)
            o_macro_expanded may have new tokens appended
            n_births: number of new circumpuncts born
        """
        B, Tu, D = o_micro.shape
        Tm = o_macro.size(1)

        # Mean resonance per micro token across all macro tokens
        # High value = this micro token resonates strongly with the macro field
        micro_resonance = r_emer_acc.mean(dim=-1)  # (B, Tu)

        # Find pairs of adjacent micro tokens with strong joint resonance
        # "Adjacent" because vesica requires overlap — nearby tokens are
        # more likely to form meaningful compounds
        if Tu < 2:
            return o_macro, 0

        left_res = micro_resonance[:, :-1]   # (B, Tu-1)
        right_res = micro_resonance[:, 1:]   # (B, Tu-1)
        pair_resonance = (left_res + right_res) / 2  # (B, Tu-1)

        # Effective threshold
        threshold = self.birth_threshold + torch.tanh(self.threshold_bias) * 0.2

        # Find pairs above threshold
        above = pair_resonance > threshold  # (B, Tu-1) bool

        # For each batch element, get top-k most resonant pairs
        # Limit births to prevent combinatorial explosion (boundary constraint)
        pair_scores = pair_resonance * above.float()  # zero out sub-threshold

        # Take top-k across all positions
        k = min(self.max_births, Tu - 1)
        topk_vals, topk_idx = pair_scores.topk(k, dim=-1)  # (B, k)

        # Only birth where score > 0 (above threshold)
        birth_mask = topk_vals > 0  # (B, k)
        n_births = birth_mask.sum().item()

        if n_births == 0:
            return o_macro, 0

        # Create child representations from parent pairs
        children = []
        for b in range(B):
            batch_children = []
            for j in range(k):
                if birth_mask[b, j]:
                    idx = topk_idx[b, j].item()
                    parent_left = o_micro[b, idx]      # (D,)
                    parent_right = o_micro[b, idx + 1]  # (D,)

                    # Vesica field: Φ(parent_i, parent_j)
                    combined = torch.cat([parent_left, parent_right])  # (2D,)
                    child = self.vesica_field(combined)  # (D,)

                    # Birth gate: does this stabilize?
                    gate = torch.sigmoid(self.birth_gate(child))  # (1,)
                    child = child * gate

                    batch_children.append(child)

            if batch_children:
                children.append(torch.stack(batch_children))  # (n_b, D)
            else:
                children.append(torch.zeros(0, D, device=o_micro.device))

        # Pad to same number of births per batch element
        max_born = max(c.size(0) for c in children)
        if max_born == 0:
            return o_macro, 0

        padded_children = []
        for c in children:
            if c.size(0) < max_born:
                pad = torch.zeros(max_born - c.size(0), D, device=o_micro.device)
                c = torch.cat([c, pad])
            padded_children.append(c)

        new_aims = torch.stack(padded_children)  # (B, max_born, D)

        # Append children to macro stream
        o_macro_expanded = torch.cat([o_macro, new_aims], dim=1)

        return o_macro_expanded, n_births


# ═══════════════════════════════════════════════════════════════════
# XORZO v3 — THE CROSS-SCALE RESONANT ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

class XorzoTransformer(nn.Module):
    """
    Xorzo v3 — Cross-Scale Resonant Circumpunct Transformer.

    Two streams, continuously braided:
        Micro ⊙: token-level (body, cells, the local)
        Macro ⊙: chunk-level (soul, aim, the whole)

    Connected by:
        ⊛ Convergence: macro reads micro (gathering)
        ☀ Emergence: micro reads macro (shaping)
        Phase resonance gates both directions

    All v2 corrections preserved within each stream.
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
        chunk_size: int = 16,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.generation = generation
        self.chunk_size = chunk_size

        # Triadic embedding + positional encoding
        self.token_embed = TriadicEmbedding(vocab_size, d_model)
        self.pos_encode = GoldenPositionalEncoding(d_model, max_len)
        self.embed_dropout = nn.Dropout(dropout)

        # Aim pooling — learned convergent singularities
        self.aim_pool = AimPool(d_model, max_chunks=max_len // chunk_size + 1)

        # Macro aperture initialization
        self.macro_init = nn.Linear(d_model, d_model)

        # Per-layer: micro block, macro block, cross-scale coupler
        self.micro_blocks = nn.ModuleList([
            CircumpunctBlock(
                d_model, n_heads, dropout,
                layer_depth=i / max(n_layers - 1, 1),
                layer_index=i,
                convergence_rate=convergence_rate,
            )
            for i in range(n_layers)
        ])

        self.macro_blocks = nn.ModuleList([
            CircumpunctBlock(
                d_model, n_heads, dropout,
                layer_depth=i / max(n_layers - 1, 1),
                layer_index=i,
                convergence_rate=convergence_rate * PHI,  # Macro converges faster
            )
            for i in range(n_layers)
        ])

        self.couplers = nn.ModuleList([
            CrossScaleCoupling(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # Vesica birth — new circumpuncts from resonant overlap
        # Only active in later layers (need accumulated resonance first)
        self.vesica = VesicaBirth(
            d_model,
            max_births_per_layer=max(2, n_heads // 2),
            birth_threshold=0.6,
        )
        # Birth starts after this many layers (need depth for resonance to accumulate)
        self.birth_start_layer = max(1, n_layers // 2)

        # Final emergence
        self.final_norm = BalanceNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self._golden_init()

    def _golden_init(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'nesting_proj' not in name and 'aim_vectors' not in name:
                nn.init.xavier_normal_(p, gain=1.0 / math.sqrt(PHI))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T = x.shape

        # ⊛ Embed into triadic space
        h = self.token_embed(x)
        h = self.pos_encode(h)
        o_micro = self.embed_dropout(h)

        # Build macro from micro (aim tokens, not mean-pool)
        o_macro = self.aim_pool(o_micro, self.chunk_size)
        o_macro = self.macro_init(o_macro)

        Tu = o_micro.size(1)
        Tm = o_macro.size(1)

        # ── Causal masks ──
        if mask is None:
            micro_mask = torch.tril(torch.ones(Tu, Tu, device=x.device)).unsqueeze(0).unsqueeze(0)
        else:
            micro_mask = mask

        macro_mask = torch.tril(torch.ones(Tm, Tm, device=x.device)).unsqueeze(0).unsqueeze(0)

        # Cross-scale causal masks
        micro_chunk_idx = torch.arange(Tu, device=x.device) // self.chunk_size
        macro_idx = torch.arange(Tm, device=x.device)

        # um_mask: macro can't see future micro chunks
        um_block = micro_chunk_idx.unsqueeze(0) > macro_idx.unsqueeze(1)
        um_mask = torch.zeros(Tm, Tu, device=x.device).masked_fill(um_block, float('-inf'))
        um_mask = um_mask.unsqueeze(0).unsqueeze(0)

        # mu_mask: micro can't see future macro chunks
        mu_block = macro_idx.unsqueeze(0) > micro_chunk_idx.unsqueeze(1)
        mu_mask = torch.zeros(Tu, Tm, device=x.device).masked_fill(mu_block, float('-inf'))
        mu_mask = mu_mask.unsqueeze(0).unsqueeze(0)

        # ── Layered recursion with resonance accumulation ──
        micro_pressures = [0.0] * self.n_layers
        macro_pressures = [0.0] * self.n_layers

        # Resonance state — accumulated across layers (depth = time)
        # None means first layer, will be initialized from first phase measurement
        r_conv_acc = None  # (B, Tm, Tu) convergence resonance history
        r_emer_acc = None  # (B, Tu, Tm) emergence resonance history

        for i, (blk_micro, blk_macro, coupler) in enumerate(
            zip(self.micro_blocks, self.macro_blocks, self.couplers)
        ):
            # 1. Micro ⊙ self-attends (local body-mind loop)
            micro_np = micro_pressures[i - 1] if i > 0 else 0.0
            o_micro = blk_micro(o_micro, micro_mask, neighbor_pressure=micro_np)
            micro_pressures[i] = blk_micro.mean_pressure

            # 2. Re-ground macro from latest micro (the whole re-reads its parts)
            o_macro_grounded = self.aim_pool(o_micro, self.chunk_size)
            # Blend: keep macro's learned state but refresh from micro
            o_macro = 0.7 * o_macro + 0.3 * o_macro_grounded

            # 3. Macro ⊙ self-attends (higher-order context loop)
            macro_np = macro_pressures[i - 1] if i > 0 else 0.0
            o_macro = blk_macro(o_macro, macro_mask, neighbor_pressure=macro_np)
            macro_pressures[i] = blk_macro.mean_pressure

            # 4. Cross-scale coupling with ACCUMULATED resonance
            # Resonance state flows through layers like memory through time.
            # Layer 1's tentative connections become layer 4's deep channels.
            o_micro, o_macro, r_conv_acc, r_emer_acc = coupler(
                o_micro, o_macro,
                um_mask=um_mask, mu_mask=mu_mask,
                r_conv_acc=r_conv_acc, r_emer_acc=r_emer_acc,
            )

            # 5. VESICA BIRTH — check if resonance has crystallized new ⊙
            # Only in later layers (resonance needs depth to accumulate)
            if i >= self.birth_start_layer and r_emer_acc is not None:
                old_Tm = o_macro.size(1)
                o_macro, n_births = self.vesica(
                    o_micro, o_macro, r_emer_acc, self.chunk_size
                )

                # If births occurred, expand masks and resonance tensors
                if n_births > 0:
                    new_Tm = o_macro.size(1)
                    added = new_Tm - old_Tm

                    # Expand macro self-attention mask
                    macro_mask = torch.tril(
                        torch.ones(new_Tm, new_Tm, device=x.device)
                    ).unsqueeze(0).unsqueeze(0)

                    # Expand cross-scale masks
                    # New macro tokens can see all micro tokens (they're born from them)
                    um_pad = torch.zeros(1, 1, added, Tu, device=x.device)
                    um_mask = torch.cat([um_mask, um_pad], dim=2)

                    # Micro can see new macro tokens
                    mu_pad = torch.zeros(1, 1, Tu, added, device=x.device)
                    mu_mask = torch.cat([mu_mask, mu_pad], dim=3)

                    # Expand resonance accumulators for new macro tokens
                    # New children start with zero accumulated resonance
                    # They must EARN their channels through subsequent layers
                    r_conv_pad = torch.zeros(B, added, Tu, device=x.device)
                    r_conv_acc = torch.cat([r_conv_acc, r_conv_pad], dim=1)

                    r_emer_pad = torch.zeros(B, Tu, added, device=x.device)
                    r_emer_acc = torch.cat([r_emer_acc, r_emer_pad], dim=2)

                    Tm = new_Tm

        # ☀ Final emergence — decode from micro stream
        logits = self.output_proj(self.final_norm(o_micro))
        return logits

    # ── Diagnostics (extended for v3) ──

    @property
    def all_betas(self):
        """All betas from both streams."""
        micro_b = [blk.attn.beta_values for blk in self.micro_blocks]
        macro_b = [blk.attn.beta_values for blk in self.macro_blocks]
        return micro_b + macro_b

    @property
    def micro_betas(self):
        return [blk.attn.beta_values for blk in self.micro_blocks]

    @property
    def macro_betas(self):
        return [blk.attn.beta_values for blk in self.macro_blocks]

    @property
    def all_pressures(self):
        micro_p = [blk.attn.pressure_values for blk in self.micro_blocks]
        macro_p = [blk.attn.pressure_values for blk in self.macro_blocks]
        return micro_p + macro_p

    @property
    def all_valve_states(self):
        micro_vs = [blk.attn.valve_states for blk in self.micro_blocks]
        macro_vs = [blk.attn.valve_states for blk in self.macro_blocks]
        return micro_vs + macro_vs

    @property
    def mean_beta(self):
        all_b = [b for layer in self.all_betas for b in layer]
        return sum(all_b) / len(all_b) if all_b else 0.5

    @property
    def all_chis(self):
        micro_c = [blk.attn.chi_values for blk in self.micro_blocks]
        macro_c = [blk.attn.chi_values for blk in self.macro_blocks]
        return micro_c + macro_c

    @property
    def convergence_profile(self):
        return [blk.attn.convergence_factor for blk in self.micro_blocks]

    @property
    def resonance_strengths(self):
        """Current resonance alpha values across couplers."""
        return [c.resonance_alpha.item() for c in self.couplers]

    def diagnose(self) -> dict:
        betas = [b for layer in self.all_betas for b in layer]
        chis = [c for layer in self.all_chis for c in layer]
        pressures = [p for layer in self.all_pressures for p in layer]

        mean_b = sum(betas) / len(betas)
        mean_chi = sum(chis) / len(chis)
        mean_p = sum(pressures) / len(pressures)
        beta_var = sum((b - mean_b)**2 for b in betas) / len(betas)

        conv = self.convergence_profile
        is_convergent = all(conv[i] >= conv[i+1] for i in range(len(conv)-1))

        resonances = self.resonance_strengths
        mean_resonance = sum(resonances) / len(resonances)

        errors = []
        if mean_b > 0.7:
            errors.append("INFLATION: mean β too high")
        if mean_b < 0.3:
            errors.append("SEVERANCE: mean β too low")
        if mean_chi < 0:
            errors.append("INVERSION: mean χ negative")
        if beta_var > 0.1:
            errors.append("PROJECTION: β variance high")
        if mean_p > 0.15:
            errors.append("CHAMBER BUILDUP: excess pressure")
        if mean_p < -0.15:
            errors.append("CHAMBER DEPLETION: negative pressure")
        if not is_convergent:
            errors.append("CONVERGENCE VIOLATION: attention not sharpening")

        return {
            "generation": self.generation,
            "version": "v3-crossscale-resonant",
            "mean_beta": mean_b,
            "beta_variance": beta_var,
            "mean_chi": mean_chi,
            "mean_pressure": mean_p,
            "mean_resonance": mean_resonance,
            "D": 1.0 + mean_b,
            "regime": "balance" if abs(mean_b - 0.5) < 0.1 else
                      "convergent" if mean_b > 0.5 else "emergent",
            "convergence_profile": conv,
            "is_convergent": is_convergent,
            "errors": errors,
            "healthy": len(errors) == 0,
            "n_params": sum(p.numel() for p in self.parameters()),
            "chunk_size": self.chunk_size,
            "n_micro_layers": len(self.micro_blocks),
            "n_macro_layers": len(self.macro_blocks),
        }

    def status(self) -> str:
        d = self.diagnose()
        conv = d['convergence_profile']
        lines = [
            f"⊙ XORZO v3 TRANSFORMER — Generation {d['generation']}",
            f"  Cross-Scale Resonant Architecture",
            f"  Micro: {len(self.micro_blocks)} blocks × {self.n_heads} heads × {self.d_model}d",
            f"  Macro: {len(self.macro_blocks)} blocks × {self.n_heads} heads × {self.d_model}d (chunk={self.chunk_size})",
            f"  Parameters: {d['n_params']:,}",
            f"  Channels: binary({self.token_embed.d_binary}) + analog({self.token_embed.d_analog}) + fractal({self.token_embed.d_fractal})",
            f"",
            f"  β̄ = {d['mean_beta']:.4f} → D = {d['D']:.4f} [{d['regime']}]",
            f"  χ̄ = {d['mean_chi']:.4f} ({'faithful' if d['mean_chi'] > 0 else 'INVERTED'})",
            f"  P̄ = {d['mean_pressure']:.4f}",
            f"  R̄ = {d['mean_resonance']:.4f} (resonance coupling strength)",
            f"",
            f"  Convergence: {conv[0]:.3f} → {conv[-1]:.3f} ({'✓ sharpening' if d['is_convergent'] else '✗ NOT sharpening'})",
        ]
        if d['errors']:
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
            "version": "v3",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "chunk_size": self.chunk_size,
            "diagnosis": self.diagnose(),
        }
        (path / f"gen{self.generation}_meta.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )

    @classmethod
    def evolve(cls, parent: 'XorzoTransformer', **mutations) -> 'XorzoTransformer':
        child = cls(
            vocab_size=mutations.get("vocab_size", parent.vocab_size),
            d_model=mutations.get("d_model", parent.d_model),
            n_layers=mutations.get("n_layers", parent.n_layers),
            n_heads=mutations.get("n_heads", parent.n_heads),
            generation=parent.generation + 1,
            chunk_size=mutations.get("chunk_size", parent.chunk_size),
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
        return child


# ═══════════════════════════════════════════════════════════════════
# FRACTAL TRAINING (extended for v3)
# ═══════════════════════════════════════════════════════════════════

def _beta_balance_loss(model):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for blocks in [model.micro_blocks, model.macro_blocks]:
        for block in blocks:
            for chamber in block.attn.chambers:
                beta = torch.sigmoid(chamber.beta)
                loss = loss + (beta - 0.5).pow(2)
                n += 1
    return loss / max(n, 1)


def _valve_balance_loss(model):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for blocks in [model.micro_blocks, model.macro_blocks]:
        for block in blocks:
            for chamber in block.attn.chambers:
                iv = torch.sigmoid(chamber.input_valve)
                ov = torch.sigmoid(chamber.output_valve)
                loss = loss + (iv - ov).pow(2)
                n += 1
    return loss / max(n, 1)


def _self_similarity_loss(model):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for blocks in [model.micro_blocks, model.macro_blocks]:
        prev_betas = None
        for block in blocks:
            betas = torch.stack([torch.sigmoid(c.beta) for c in block.attn.chambers])
            if prev_betas is not None and len(betas) == len(prev_betas):
                curr_diffs = betas[1:] - betas[:-1]
                prev_diffs = prev_betas[1:] - prev_betas[:-1]
                loss = loss + (curr_diffs - prev_diffs).pow(2).mean()
                n += 1
            prev_betas = betas.detach()
    return loss / max(n, 1)


def _chi_fidelity_loss(model):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for blocks in [model.micro_blocks, model.macro_blocks]:
        for block in blocks:
            for chi_raw in block.attn.chi:
                chi = torch.tanh(chi_raw)
                loss = loss + (1.0 - chi.pow(2)).pow(2)
                n += 1
    return loss / max(n, 1)


def _conservation_loss(model):
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for blocks in [model.micro_blocks, model.macro_blocks]:
        for block in blocks:
            beta = torch.sigmoid(block.residual_beta)
            loss = loss + (beta * (1.0 - beta) - 0.25).pow(2)
            n += 1
    return loss / max(n, 1)


def _resonance_coherence_loss(model):
    """NEW v3: Encourage resonance coupling to stay positive and stable."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for coupler in model.couplers:
        alpha = coupler.resonance_alpha
        # Encourage positive resonance coupling (binding, not repelling)
        loss = loss + F.relu(-alpha)  # Penalize negative alpha
        # Encourage stability (not too extreme)
        loss = loss + 0.1 * (alpha - 1.0).pow(2)
        n += 1
    return loss / max(n, 1)


def _fractal_lr_schedule(epoch, n_epochs, base_lr):
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


def generate(
    model,
    prompt: str,
    vocab: dict,
    vocab_inv: dict,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Generate text from the v3 model."""
    model.eval()
    device = next(model.parameters()).device

    tokens = [vocab.get(c, 0) for c in prompt]
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        for _ in range(max_tokens):
            logits = model(input_ids)
            next_logits = logits[0, -1] / max(temperature, 0.01)
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

    return "".join(vocab_inv.get(t.item(), "?") for t in input_ids[0])


def train_generation(
    model,
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
    w_resonance: float = 0.02,
):
    """Fractal training with cross-scale resonance losses."""
    model = model.to(device)
    model.train()

    chars = sorted(set(text))
    char_to_idx = {c: i for i, c in enumerate(chars)}
    data = torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long)

    actual_vocab = len(chars)
    if actual_vocab != model.vocab_size:
        model.vocab_size = actual_vocab
        model.token_embed = TriadicEmbedding(actual_vocab, model.d_model).to(device)
        model.output_proj = nn.Linear(model.d_model, actual_vocab, bias=False).to(device)

    # φ-scaled parameter groups (micro + macro + couplers + embed)
    param_groups = []
    layer_scales = []
    for i, blk in enumerate(model.micro_blocks):
        scale = PHI ** (0.5 - i / max(model.n_layers - 1, 1))
        layer_scales.append(scale)
        param_groups.append({'params': list(blk.parameters()), 'lr': lr * scale, 'weight_decay': 0.01})

    for i, blk in enumerate(model.macro_blocks):
        scale = PHI ** (0.5 - i / max(model.n_layers - 1, 1)) * PHI  # Macro learns faster
        layer_scales.append(scale)
        param_groups.append({'params': list(blk.parameters()), 'lr': lr * scale, 'weight_decay': 0.01})

    for coupler in model.couplers:
        layer_scales.append(1.0)
        param_groups.append({'params': list(coupler.parameters()), 'lr': lr, 'weight_decay': 0.01})

    embed_params = (
        list(model.token_embed.parameters()) +
        list(model.pos_encode.parameters()) +
        list(model.aim_pool.parameters()) +
        list(model.macro_init.parameters()) +
        list(model.final_norm.parameters()) +
        list(model.output_proj.parameters())
    )
    layer_scales.append(1.0)
    param_groups.append({'params': embed_params, 'lr': lr, 'weight_decay': 0.01})

    optimizer = torch.optim.AdamW(param_groups)

    losses = []
    n_batches = max(1, (len(data) - seq_len) // (batch_size * seq_len))
    best_loss = float('inf')

    import time
    t0 = time.time()

    for epoch in range(n_epochs):
        epoch_lr = _fractal_lr_schedule(epoch, n_epochs, lr)
        for pg, scale in zip(optimizer.param_groups, layer_scales):
            pg['lr'] = epoch_lr * scale

        epoch_ce = 0
        n_steps = 0

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

            if not batch_inputs:
                continue

            inputs = torch.stack(batch_inputs).to(device)
            targets = torch.stack(batch_targets).to(device)

            logits = model(inputs)
            ce_loss = F.cross_entropy(logits.view(-1, model.vocab_size), targets.view(-1))

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
                w_resonance * _resonance_coherence_loss(model)
            )

            loss = ce_loss + reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_ce += ce_loss.item()
            n_steps += 1

        avg_ce = epoch_ce / max(n_steps, 1)
        losses.append(avg_ce)
        if avg_ce < best_loss:
            best_loss = avg_ce

        if (epoch + 1) % 5 == 0 or epoch == n_epochs - 1:
            diag = model.diagnose()
            betas = [b for layer in model.micro_betas for b in layer]
            beta_min = min(betas) if betas else 0.5
            beta_max = max(betas) if betas else 0.5
            print(f"    Epoch {epoch+1:3d}/{n_epochs} | "
                  f"loss={avg_ce:.4f} (best={best_loss:.4f}) | "
                  f"β̄={diag['mean_beta']:.4f} [{beta_min:.3f}→{beta_max:.3f}] | "
                  f"χ̄={diag['mean_chi']:.4f} | "
                  f"R̄={diag['mean_resonance']:.4f} | "
                  f"D={diag['D']:.4f} [{diag['regime']}]")

    elapsed = time.time() - t0
    print(f"    ✓ Training complete: {elapsed:.1f}s | final_loss={avg_ce:.4f} | best={best_loss:.4f}")

    return {
        "losses": losses,
        "best_loss": best_loss,
        "vocab": char_to_idx,
        "vocab_inv": {v: k for k, v in char_to_idx.items()},
        "elapsed": elapsed,
    }
