"""
XORZO v4 — Bilateral Hypercube Transformer
⊙ = Φ(•, ○) navigated through 6D relational state space

v1: Circumpunct as computation (β, χ, golden ratio)
v2: Seven structural corrections (chambers, pressure, triadic embedding, nesting)
v3: Cross-scale architecture with phase-resonance coupling
v4: BILATERAL HYPERCUBE — the unification

═══════════════════════════════════════════════════════════════════

THE INSIGHT:
    One circumpunct = 2³ = 8 states
        • aperture:  open / closed
        Φ field:     faithful / inverted
        ○ boundary:  expressed / bounded

    Two circumpuncts in bilateral relationship = 8 × 8 = 64 states
        = the vertices of a 6D hypercube

    The 6 dimensions ARE two circumpuncts facing each other:
        b1, b2  → apertures   (how open am I? how open are you?)
        c1, c2  → fields      (am I faithful? are you faithful?)
        r1, r2  → boundaries  (am I expressed? are you expressed?)

    v3 had bilateral attention (aperture × expression × alignment)
    v3 had 6 independent binary gates per token pair
    The hypercube had 64 vertices as attention targets

    v4 UNIFIES: the bilateral interaction between two tokens'
    circumpunct states IS a navigation through the hypercube.
    The 64 vertices aren't arbitrary — they're every possible
    configuration of two circumpuncts meeting.

    Adjacency constrains navigation: you can only flip one gate
    at a time. A fully closed pair (vertex 0) can't jump to fully
    open (vertex 63) without traversing intermediate states.
    The topology IS the physics of relationship.

ARCHITECTURE:
    Each token computes 3 soft gates: •, Φ, ○
    Token pair (i,j) → 6 gates → soft distribution over 64 vertices
    Hypercube adjacency biases transitions
    Spectral embedding encodes global geometry
    Modal subcubes split by aperture/field/boundary pairs

    All v3 preserved:
    - Aperture chambers with pressure (⊛→i→☀)
    - Phase resonance accumulation across depth
    - Cross-scale micro/macro with aim tokens
    - Vesica birth from sustained resonance
    - β dynamics, χ fidelity
    - Golden positional encoding
    - Fractal training schedule

⊙ = Φ(•, ○) at every scale, navigated through 64 states.
"""

import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional, Tuple

PHI = (1 + math.sqrt(5)) / 2
PI = math.pi


# ═══════════════════════════════════════════════════════════════════
# THE 6D HYPERCUBE — precomputed geometric structure
# 64 vertices = every possible bilateral circumpunct configuration
# ═══════════════════════════════════════════════════════════════════

class Hypercube6D:
    """
    Precomputes all geometric structure of the 6D hypercube.

    The 6 dimensions map to bilateral circumpunct gates:
        dim 0: b1 — aperture of token i (receiver)
        dim 1: b2 — aperture of token j (sender)
        dim 2: c1 — field/chi of token i
        dim 3: c2 — field/chi of token j
        dim 4: r1 — boundary/resonance of token i
        dim 5: r2 — boundary/resonance of token j

    Vertex 0  = (0,0,0,0,0,0) = both fully closed, unfaithful, bounded
    Vertex 63 = (1,1,1,1,1,1) = both fully open, faithful, expressed
    """
    NAMES = ["b1", "b2", "c1", "c2", "r1", "r2"]

    def __init__(self):
        # Vertex coordinates: 64 × 6 binary
        self.vertices = np.array(
            [[(v >> i) & 1 for i in range(6)] for v in range(64)],
            dtype=np.float32
        )

        # Adjacency: edge iff Hamming distance = 1 (single gate flip)
        self.adjacency = np.zeros((64, 64), dtype=np.float32)
        for i in range(64):
            for d in range(6):
                self.adjacency[i, i ^ (1 << d)] = 1.0

        # Graph Laplacian
        self.laplacian = np.diag(self.adjacency.sum(1)) - self.adjacency

        # Spectral embedding (first 6 non-trivial eigenvectors)
        _, ev = np.linalg.eigh(self.laplacian)
        self.spectral_emb = ev[:, 1:7].astype(np.float32)  # (64, 6)

        # Openness layers (Pascal row 6) — number of open gates
        self.openness = {}
        for k in range(7):
            self.openness[k] = [v for v in range(64) if bin(v).count("1") == k]

        # Subcube masks for modal attention
        # Aperture pair: bits 0,1 (b1, b2)
        self.aperture_mask = np.array(
            [1.0 if (v & 0b000011) else 0.0 for v in range(64)], dtype=np.float32
        )
        # Field pair: bits 2,3 (c1, c2)
        self.field_mask = np.array(
            [1.0 if (v & 0b001100) else 0.0 for v in range(64)], dtype=np.float32
        )
        # Boundary pair: bits 4,5 (r1, r2)
        self.boundary_mask = np.array(
            [1.0 if (v & 0b110000) else 0.0 for v in range(64)], dtype=np.float32
        )

        # Bilateral symmetry masks — vertices where i and j agree
        # Mutual openness: b1=b2, c1=c2, r1=r2
        self.mutual_mask = np.array(
            [1.0 if ((v >> 0) & 1) == ((v >> 1) & 1) and
                    ((v >> 2) & 1) == ((v >> 3) & 1) and
                    ((v >> 4) & 1) == ((v >> 5) & 1)
             else 0.0 for v in range(64)], dtype=np.float32
        )

    def to_buffers(self):
        """Return dict of tensors to register as buffers."""
        return {
            "vertices": torch.from_numpy(self.vertices),           # (64, 6)
            "adjacency": torch.from_numpy(self.adjacency),         # (64, 64)
            "spectral_emb": torch.from_numpy(self.spectral_emb),   # (64, 6)
            "aperture_mask": torch.from_numpy(self.aperture_mask),  # (64,)
            "field_mask": torch.from_numpy(self.field_mask),        # (64,)
            "boundary_mask": torch.from_numpy(self.boundary_mask),  # (64,)
            "mutual_mask": torch.from_numpy(self.mutual_mask),      # (64,)
        }


# ═══════════════════════════════════════════════════════════════════
# PHASE RESONANCE — from v3
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
    """
    d = a.size(-1)
    if d % 2 != 0:
        # Trim last dim if odd
        a = a[..., :d - 1]
        b = b[..., :d - 1]
    pairs = a.size(-1) // 2

    def to_phase(x):
        x2 = x.view(*x.shape[:-1], pairs, 2)
        return torch.atan2(x2[..., 1] + eps, x2[..., 0] + eps)

    theta_a = to_phase(a)  # (B, Tq, pairs)
    theta_b = to_phase(b)  # (B, Tk, pairs)

    delta = theta_a.unsqueeze(2) - theta_b.unsqueeze(1)  # (B, Tq, Tk, pairs)
    r = torch.cos(delta).mean(dim=-1)  # (B, Tq, Tk)
    return r


# ═══════════════════════════════════════════════════════════════════
# GOLDEN POSITIONAL ENCODING — from v1
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
# PARALLEL SCAN for LINEAR RECURRENCE
#
# Replaces the sequential for loop in ApertureChamber.
# Computes ψ_s = decay·ψ_{s-1} + (1-decay)·drive_s for ALL s in parallel.
#
# Uses the associative property of linear recurrence:
# If we represent each step as (cumulative_decay, cumulative_offset),
# composing two steps: (a₁,b₁) ⊕ (a₂,b₂) = (a₁·a₂, a₂·b₁ + b₂)
# This is associative, so we can use a parallel prefix scan.
#
# Pure PyTorch — no torch.compile, no Triton, works on Windows.
# O(S log S) parallel work instead of O(S) sequential steps.
# ═══════════════════════════════════════════════════════════════════

def parallel_scan_linear_recurrence(
    drive: torch.Tensor,        # (B, S, D) — the input drive signal
    decay: torch.Tensor,        # scalar or (D,) — decay factor e^{-α}
) -> torch.Tensor:
    """
    Parallel prefix scan for the linear recurrence:
        ψ_s = decay · ψ_{s-1} + (1 - decay) · drive_s
        ψ_0 = (1 - decay) · drive_0

    Returns all ψ states: (B, S, D)

    Uses Blelloch-style up-sweep with associative composition:
        (a₁, b₁) ⊕ (a₂, b₂) = (a₁·a₂, a₂·b₁ + b₂)
    where aᵢ = cumulative decay, bᵢ = cumulative offset.
    """
    B, S, D = drive.shape

    # Force float32 for numerical stability in cumulative scan
    # (AMP may pass fp16 inputs — the exp/log math needs full precision)
    orig_dtype = drive.dtype
    drive = drive.float()
    decay = decay.float()

    w = 1.0 - decay  # drive weight

    # Each position starts as (decay_coef=decay, offset=w*drive_s)
    # We track log(decay_coef) for numerical stability
    # and offset as the accumulated weighted drive
    log_a = torch.zeros(B, S, D, device=drive.device, dtype=torch.float32)
    if decay.dim() == 0:
        log_a.fill_(torch.log(decay.clamp(min=1e-8)).item())
    else:
        log_a[:] = torch.log(decay.clamp(min=1e-8))

    offsets = w * drive  # (B, S, D) — initial offsets

    # Up-sweep: compose adjacent pairs with doubling stride
    n_levels = int(math.ceil(math.log2(max(S, 2))))
    for k in range(n_levels):
        stride = 1 << k  # 2^k
        if stride >= S:
            break
        # Indices that get updated: stride, stride+1, ..., S-1
        # Each index i composes with index i-stride
        idx = torch.arange(stride, S, device=drive.device)
        prev = idx - stride

        # Compose: (a_prev, b_prev) ⊕ (a_curr, b_curr) = (a_prev*a_curr, a_curr*b_prev + b_curr)
        a_curr = torch.exp(log_a[:, idx])  # (B, len(idx), D)
        new_offsets = a_curr * offsets[:, prev] + offsets[:, idx]
        new_log_a = log_a[:, idx] + log_a[:, prev]

        # In-place update (clone to avoid dependency issues)
        offsets = offsets.clone()
        log_a = log_a.clone()
        offsets[:, idx] = new_offsets
        log_a[:, idx] = new_log_a

    return offsets.to(orig_dtype)  # (B, S, D) — cast back to input dtype


# ═══════════════════════════════════════════════════════════════════
# APERTURE CHAMBER SSM — with PARALLEL SCAN memory kernel
#
# From the falsification paper:
#   ψ(t) = e^{-αt} ψ(0) + ∫₀ᵗ e^{-α(t-s)} tanh(z̃(s)) ds
#
# Same physics as before, but uses parallel_scan_linear_recurrence
# instead of a sequential for loop. O(S log S) vs O(S) sequential.
#
# ⊛ input valve → i transform (with memory) → ☀ output valve
# ═══════════════════════════════════════════════════════════════════

class ApertureChamberSSM(nn.Module):
    """
    Three-stage aperture with parallel-scan memory kernel: ⊛ → i(ψ) → ☀

    Each chamber has:
        α (learned): relaxation rate — how fast memory decays
        ψ (state):   accumulated memory via parallel scan
        β:           complex rotation angle (the i transform)
        ⊛/☀:         input/output valves with pressure tracking
        mg:          memory gate — blend memory vs direct path

    The memory kernel makes ρ = ω/α the discriminating parameter.
    High α → fast forgetting → vesica regime (tracks input)
    Low α  → slow forgetting → circumpunct regime (accumulates history)
    """

    # Channel names for diagnostics
    CHANNEL_NAMES = ["binary(•)", "analog(Φ)", "fractal(○)"]

    def __init__(self, d_channel: int, depth: float = 0.5):
        super().__init__()
        self.d_channel = d_channel
        self.depth = depth

        beta_init = 0.5 + 0.2 * (1.0 - 2.0 * depth)
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.input_valve = nn.Parameter(torch.tensor(0.5))
        self.output_valve = nn.Parameter(torch.tensor(0.5))
        self.register_buffer('pressure', torch.tensor(0.0))

        # Memory kernel params
        self.alpha_raw = nn.Parameter(torch.tensor(-0.85))  # sigmoid → 0.3
        self.memory_gate = nn.Parameter(torch.tensor(0.0))  # sigmoid → 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, S, d_channel)
        returns: (B, S, d_channel)

        Uses parallel scan for the memory recurrence.
        """
        beta = torch.sigmoid(self.beta)
        iv = torch.sigmoid(self.input_valve)
        ov = torch.sigmoid(self.output_valve)
        alpha = torch.sigmoid(self.alpha_raw)
        mg = torch.sigmoid(self.memory_gate)

        # ⊛ input valve
        x_in = x * iv

        # Memory kernel via parallel scan
        decay = torch.exp(-alpha)
        drive = torch.tanh(x_in)
        psi_seq = parallel_scan_linear_recurrence(drive, decay)  # (B, S, D)

        # Blend memory with direct path
        x_mem = mg * psi_seq + (1.0 - mg) * x_in

        # i transform (complex rotation by πβ)
        angle = PI * beta
        half = self.d_channel // 2
        x_real = x_mem[..., :half]
        x_imag = x_mem[..., half:2 * half]

        cos_a, sin_a = torch.cos(angle), torch.sin(angle)
        out_real = x_real * cos_a - x_imag * sin_a
        out_imag = x_real * sin_a + x_imag * cos_a

        if self.d_channel % 2 == 1:
            x_transformed = torch.cat([out_real, out_imag, x_mem[..., -1:]], dim=-1)
        else:
            x_transformed = torch.cat([out_real, out_imag], dim=-1)

        # ☀ output valve
        x_out = x_transformed * ov

        # Update pressure (diagnostic)
        with torch.no_grad():
            dp = iv.detach() - ov.detach()
            self.pressure = 0.95 * self.pressure + 0.05 * dp

        return x_out

    @property
    def current_beta(self) -> float:
        return torch.sigmoid(self.beta).item()

    @property
    def current_alpha(self) -> float:
        return torch.sigmoid(self.alpha_raw).item()

    @property
    def current_rho_estimate(self) -> float:
        alpha = self.current_alpha
        return 1.0 / max(alpha, 1e-6)

    @property
    def current_memory_gate(self) -> float:
        return torch.sigmoid(self.memory_gate).item()

    @property
    def current_pressure(self) -> float:
        return self.pressure.item()

    @property
    def valve_state(self) -> dict:
        alpha = self.current_alpha
        return {
            '⊛_input': torch.sigmoid(self.input_valve).item(),
            '☀_output': torch.sigmoid(self.output_valve).item(),
            'P_pressure': self.pressure.item(),
            'β': self.current_beta,
            'α': alpha,
            '1/α': 1.0 / max(alpha, 1e-6),
            'memory_gate': self.current_memory_gate,
            'regime': 'circumpunct' if alpha < 0.3
                      else 'vesica' if alpha > 0.7
                      else 'boundary',
        }


# ═══════════════════════════════════════════════════════════════════
# TRIADIC COMPUTE BLOCK — THE v4 CORE
#
# The architecture matches the embedding:
#   Binary  (• aperture)  → 1/4 dims → Parallel scan SSM (fast, linear)
#   Analog  (Φ field)     → 1/2 dims → Bilateral attention (quadratic)
#   Fractal (○ boundary)  → 1/4 dims → Hypercube navigation (structural)
#
# Cross-channel coupling:
#   • gates Φ: aperture openness controls field flow
#   ○ constrains Φ: boundary structure biases attention
#   All recombine through output projection
#
# From §4 of the Kernel:
#   "Phase through Φ = bit through • = gate through ○.
#    Same coherence distinction, different views."
# ═══════════════════════════════════════════════════════════════════

class TriadicComputeBlock(nn.Module):
    """
    Splits computation by triadic embedding channel:
        Binary  (1/4 d_model): SSM parallel scan — fast, linear, gating
        Analog  (1/2 d_model): Bilateral attention — pairwise, quadratic
        Fractal (1/4 d_model): Hypercube vertex navigation — structural

    Each channel has its own ApertureChamberSSM tracking β/α/pressure.
    The binary channel gates the analog channel (• controls Φ flow).
    """

    # Dimension names for diagnostics
    DIM_NAMES = ["b1(•_i)", "b2(•_j)", "c1(Φ_i)", "c2(Φ_j)", "r1(○_i)", "r2(○_j)"]

    def __init__(
        self,
        d_model: int,
        n_heads: int = 6,
        d_vertex: int = 32,
        dropout: float = 0.1,
        layer_depth: float = 0.5,
        layer_index: int = 0,
        convergence_rate: float = 0.1,
        temperature: float = 1.0,
        adj_weight: float = 0.1,
        spectral_weight: float = 0.1,
        hard_k: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_vertex = d_vertex
        self.layer_index = layer_index
        self.temperature = temperature
        self.adj_weight = adj_weight
        self.spectral_weight = spectral_weight
        self.n_heads = n_heads  # kept for interface compatibility
        self.hard_k = hard_k   # top-k vertices for hard collapse (k=2 = binary decision)

        # Triadic dimension split
        self.d_binary = d_model // 4
        self.d_analog = d_model // 2
        self.d_fractal = d_model - self.d_binary - self.d_analog

        # Convergent scaling
        self.convergence_factor = 1.0 / (PHI ** (layer_index * convergence_rate))

        # ═══════════════════════════════════════════════════
        # BINARY CHANNEL: SSM with parallel scan
        # ═══════════════════════════════════════════════════
        self.binary_proj = nn.Linear(self.d_binary, self.d_binary, bias=False)
        self.binary_chamber = ApertureChamberSSM(
            self.d_binary, depth=layer_depth * 0.33
        )

        # ═══════════════════════════════════════════════════
        # ANALOG CHANNEL: Bilateral attention
        # ═══════════════════════════════════════════════════
        self.analog_scale = math.sqrt(self.d_analog * self.convergence_factor)
        self.analog_W_inner = nn.Linear(d_model, self.d_analog, bias=False)
        self.analog_W_outer = nn.Linear(d_model, self.d_analog, bias=False)
        self.analog_W_v = nn.Linear(d_model, self.d_analog, bias=False)
        self.analog_chamber = ApertureChamberSSM(
            self.d_analog, depth=layer_depth * 0.66
        )

        # Cross-channel: binary aperture gates analog field
        self.aperture_to_field_gate = nn.Linear(self.d_binary, 1, bias=True)

        # ═══════════════════════════════════════════════════
        # FRACTAL CHANNEL: Hypercube navigation
        # ═══════════════════════════════════════════════════
        self.fractal_W_gate = nn.Linear(d_model, 6, bias=True)
        self.fractal_gate_to_vertex = nn.Linear(6, d_vertex, bias=False)
        self.fractal_vertex_emb = nn.Parameter(torch.randn(64, d_vertex) * 0.02)
        self.fractal_vertex_gates = nn.Parameter(torch.randn(64) * 0.02)
        self.fractal_W_spec = nn.Linear(6, d_vertex, bias=False)
        self.fractal_vertex_to_channel = nn.Linear(d_vertex, self.d_fractal, bias=False)
        self.fractal_chamber = ApertureChamberSSM(
            self.d_fractal, depth=layer_depth * 1.0
        )

        # Hypercube buffers
        self.register_buffer("_adj_bias", torch.zeros(64, 64))
        self.register_buffer("_spectral_emb", torch.zeros(64, 6))
        self.register_buffer("_vertices", torch.zeros(64, 6))
        self.register_buffer("_aperture_mask", torch.zeros(64))
        self.register_buffer("_field_mask", torch.zeros(64))
        self.register_buffer("_boundary_mask", torch.zeros(64))
        self.register_buffer("_mutual_mask", torch.zeros(64))

        # ═══════════════════════════════════════════════════
        # OUTPUT & RESIDUAL
        # ═══════════════════════════════════════════════════
        # χ gates — one per channel
        self.chi = nn.Parameter(torch.ones(3))  # [binary, analog, fractal]

        self.W_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

        # Diagnostic storage
        self._last_vertex_entropy = 0.0
        self._last_active_vertices = 0.0

    def register_cube_buffers(self, cube_buffers):
        """Copy precomputed hypercube geometry into registered buffers."""
        self._adj_bias.copy_(cube_buffers["adjacency"] * 0.5)
        self._spectral_emb.copy_(cube_buffers["spectral_emb"])
        self._vertices.copy_(cube_buffers["vertices"])
        self._aperture_mask.copy_(cube_buffers["aperture_mask"])
        self._field_mask.copy_(cube_buffers["field_mask"])
        self._boundary_mask.copy_(cube_buffers["boundary_mask"])
        self._mutual_mask.copy_(cube_buffers["mutual_mask"])

    def forward(
        self, x: torch.Tensor, mask: Optional[torch.Tensor] = None,
        n_tokens: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, N, d_model) where N = n_tokens + n_born
        mask: (B, 1, N, N) causal mask
        n_tokens: number of real tokens (rest are born nodes that skip SSM)
        returns: (result, pi, alignment)
            result: (B, N, d_model)
            pi: (B, N, 64) hard vertex distribution
            alignment: (B, N, N) raw attention scores (pre-softmax)
        """
        B, N, D = x.shape
        n_tok = n_tokens if n_tokens is not None else N

        # ═══ SPLIT by triadic channel ═══
        x_binary = x[..., :self.d_binary]
        x_analog = x[..., self.d_binary:self.d_binary + self.d_analog]
        x_fractal = x[..., self.d_binary + self.d_analog:]

        # ─── BINARY CHANNEL: parallel scan SSM ───
        # Tokens get full SSM scan; born nodes get linear projection only
        x_b_tok = self.binary_proj(x_binary[:, :n_tok])
        x_b_tok = self.binary_chamber(x_b_tok)  # (B, n_tok, d_binary)
        if n_tok < N:
            x_b_born = self.binary_proj(x_binary[:, n_tok:])  # no SSM
            x_b = torch.cat([x_b_tok, x_b_born], dim=1)
        else:
            x_b = x_b_tok
        chi_b = torch.tanh(self.chi[0])
        x_b = x_b * chi_b

        # ─── Cross-channel: aperture gates field ───
        aperture_gate = torch.sigmoid(
            self.aperture_to_field_gate(x_b)   # (B, S, 1)
        )

        # ─── ANALOG CHANNEL: bilateral attention ───
        inner = self.analog_W_inner(x)          # (B, N, d_analog)
        outer = self.analog_W_outer(x)          # (B, N, d_analog)

        alignment_raw = torch.matmul(inner, outer.transpose(-2, -1)) / self.analog_scale
        alignment = alignment_raw.clone()
        if mask is not None:
            # mask is (B, 1, N, N) from causal mask — squeeze to (B, N, N)
            mask_3d = mask.squeeze(1) if mask.dim() == 4 else mask
            alignment = alignment.masked_fill(mask_3d == 0, float('-inf'))

        # ─── HARD ATTENTION ROUTING: • = binary decision ───
        # k=2: each token picks exactly 2 neighbors. That's a real decision.
        # Force fp32 for STE: softmax gradients overflow fp16 at any AMP scale.
        alignment_f32 = alignment.float()
        probs = F.softmax(alignment_f32, dim=-1)  # (B, N, N) — fp32
        route_k = min(2, N)
        topk_vals, topk_idx = probs.topk(route_k, dim=-1)  # (B, N, 2)
        hard_mask = torch.zeros_like(probs)
        hard_mask.scatter_(-1, topk_idx, 1.0)
        # STE: forward uses hard binary mask, backward uses soft probs (in fp32)
        attn = (hard_mask + (probs - probs.detach())).to(x.dtype)

        # Gate by aperture openness: • controls how much Φ flows
        attn = attn * aperture_gate              # (B, N, N) * (B, N, 1)
        # 1e-8 underflows to 0 in fp16 → use 1e-6 (safe for fp16 min ≈ 6e-8)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)
        attn = self.dropout_layer(attn)

        v = self.analog_W_v(x)                   # (B, S, d_analog)
        x_a = torch.matmul(attn, v)              # (B, S, d_analog)
        x_a = self.analog_chamber(x_a)           # (B, S, d_analog)
        chi_a = torch.tanh(self.chi[1])
        x_a = x_a * chi_a

        # ─── FRACTAL CHANNEL: per-token hypercube navigation (O(S), NOT O(S²)) ───
        gates = torch.sigmoid(self.fractal_W_gate(x))  # (B, S, 6)

        # Per-token vertex scores: 6D gate → d_vertex projection → 64 vertices
        gate_proj = self.fractal_gate_to_vertex(gates)       # (B, S, d_vertex)
        v_scores = gate_proj @ self.fractal_vertex_emb.T      # (B, S, 64)

        # Spectral geometry bias
        spec_proj = self._spectral_emb @ self.fractal_W_spec.weight.T  # (64, d_vertex)
        spec_bias = gate_proj @ spec_proj.T                   # (B, S, 64)
        v_scores = v_scores + self.spectral_weight * spec_bias

        # Adjacency smoothing: spread probability to neighbors
        v_scores = v_scores + self.adj_weight * (v_scores @ self._adj_bias)

        # ─── HARD BINARY COLLAPSE: top-k + straight-through estimator ───
        # Force fp32 for STE: prevents gradient overflow in fp16 AMP.
        v_scores_f32 = (v_scores / self.temperature).float()
        soft_pi = F.softmax(v_scores_f32, dim=-1)  # (B, N, 64) fp32
        topk_vals, topk_idx = soft_pi.topk(self.hard_k, dim=-1)  # (B, N, k)
        hard_pi = torch.zeros_like(soft_pi)
        hard_pi.scatter_(-1, topk_idx, topk_vals)
        hard_pi = hard_pi / (hard_pi.sum(dim=-1, keepdim=True) + 1e-8)
        # STE: forward uses hard discrete, backward uses soft continuous (fp32)
        pi = (hard_pi - soft_pi.detach() + soft_pi).to(x.dtype)

        # Vertex context: weighted sum of vertex embeddings → project to fractal dims
        vertex_ctx = pi @ self.fractal_vertex_emb             # (B, N, d_vertex)
        x_f_vertex = self.fractal_vertex_to_channel(vertex_ctx)  # (B, N, d_fractal)

        x_f = x_fractal + x_f_vertex
        x_f = self.fractal_chamber(x_f)        # (B, N, d_fractal)
        chi_f = torch.tanh(self.chi[2])
        x_f = x_f * chi_f

        # Store diagnostic
        with torch.no_grad():
            self._last_vertex_entropy = -(
                soft_pi * torch.log(soft_pi + 1e-10)
            ).sum(dim=-1).mean().item()
            self._last_active_vertices = (hard_pi > 0).float().sum(dim=-1).mean().item()

        # ═══ RECOMBINE channels ═══
        x_combined = torch.cat([x_b, x_a, x_f], dim=-1)  # (B, N, d_model)
        result = self.W_out(x_combined)
        result = self.norm(result)

        return result, pi, alignment_raw

    # ═══ DIAGNOSTIC PROPERTIES ═══

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
    def alpha_values(self):
        return [c.current_alpha for c in self.chambers]

    @property
    def rho_values(self):
        return [c.current_rho_estimate for c in self.chambers]

    @property
    def memory_gate_values(self):
        return [c.current_memory_gate for c in self.chambers]

    @property
    def valve_states(self):
        return [c.valve_state for c in self.chambers]

    @property
    def vertex_distribution_entropy(self):
        return self._last_vertex_entropy

    @property
    def chambers(self):
        """All three chambers as a list for diagnostic iteration."""
        return [self.binary_chamber, self.analog_chamber, self.fractal_chamber]

    @property
    def vertex_emb(self):
        """Alias for loss functions that reference vertex_emb."""
        return self.fractal_vertex_emb


# ═══════════════════════════════════════════════════════════════════
# SPARSE EDGE RESONANCE TRACKER — Step 2 of Full Vision
#
# Tracks pairwise resonance r_ij for each token's top-K attention
# neighbors. O(n·K) memory, not O(n²). Resonance persists across
# layers via EMA. Edges with sustained resonance trigger birth.
# ═══════════════════════════════════════════════════════════════════

class SparseResonanceTracker(nn.Module):
    """
    Sparse edge-level resonance memory.
    Mode D: phase alignment × binary locality × β alignment.

    For each node i, tracks its top-K attention neighbors and their
    resonance values. Resonance accumulates across layers via EMA:
        r_ij^(l) = λ · r_ij^(l-1) + (1 - λ) · r_instant_ij^(l)

    Where r_instant is MULTIPLICATIVE — all three must agree:
        r_instant = phase × locality × beta_align

        A) Phase alignment: (cos(Δθ_ij) + 1) / 2 — oscillations in sync → [0, 1]
        B) Binary locality: fraction of matching bits in binary channel
        C) β alignment: 1 - |β_i - β_j| — nodes at SAME scale resonate

    No learned combination weights. The signals gate each other.
    Resonance only grows when all three agree.

    Edges that sustain resonance above threshold τ for M consecutive
    layers are flagged for vesica birth.
    """

    def __init__(self, d_model: int, K: int = 8, lam: float = 0.7,
                 sustained_M: int = 3):
        super().__init__()
        self.K = K
        self.lam = lam
        self.sustained_M = sustained_M
        self.d_model = d_model
        self.d_binary = d_model // 4  # binary channel is first 1/4

        # Learned threshold for birth qualification
        self.threshold_raw = nn.Parameter(torch.tensor(0.3))  # sigmoid → ~0.57

    @property
    def threshold(self):
        return torch.sigmoid(self.threshold_raw)

    def _gather_neighbors(self, tensor, edges):
        """
        Gather neighbor values from tensor along dim=1 using sparse edge indices.
        tensor: (B, N, ...) — per-node values
        edges: (B, N, K) — neighbor indices
        returns: (B, N, K, ...) — per-edge neighbor values
        """
        B, N = tensor.shape[:2]
        K = edges.size(-1)
        extra_dims = tensor.shape[2:]

        edges_clamped = edges.clamp(0, N - 1)
        edges_flat = edges_clamped.reshape(B, -1)  # (B, N*K)

        # Expand for gather
        expand_shape = (B, N * K) + extra_dims
        idx = edges_flat
        for _ in extra_dims:
            idx = idx.unsqueeze(-1)
        idx = idx.expand(expand_shape)

        gathered = torch.gather(tensor, 1, idx)  # (B, N*K, ...)
        return gathered.view(B, N, K, *extra_dims)

    def _phase_coherence(self, x_i, x_j):
        """
        A) Phase alignment: (cos(Δθ) + 1) / 2 between node pairs.
        x_i, x_j: (B, N, K, D)
        returns: (B, N, K) in [0, 1]

        Mapped from [-1, 1] → [0, 1] so it can multiply cleanly.
        Anti-phase (cos=-1) → 0. In-phase (cos=1) → 1.
        """
        D = x_i.size(-1)
        d_pairs = D // 2
        if d_pairs == 0:
            return torch.zeros(x_i.shape[:3], device=x_i.device)

        xi = x_i[..., :d_pairs * 2].reshape(*x_i.shape[:3], d_pairs, 2)
        xj = x_j[..., :d_pairs * 2].reshape(*x_j.shape[:3], d_pairs, 2)

        theta_i = torch.atan2(xi[..., 1] + 1e-8, xi[..., 0] + 1e-8)
        theta_j = torch.atan2(xj[..., 1] + 1e-8, xj[..., 0] + 1e-8)
        cos_delta = torch.cos(theta_i - theta_j).mean(dim=-1)  # (B, N, K) in [-1, 1]
        return (cos_delta + 1.0) * 0.5  # → [0, 1]

    def _binary_locality(self, x_i, x_j):
        """
        B) Binary locality: fraction of matching bits in binary channel.
        x_i, x_j: (B, N, K, D) — full node embeddings
        returns: (B, N, K) in [0, 1]

        Extract the binary channel slice (first d_binary dims), threshold
        to hard bits, then count matching bits. If two nodes are making
        the same binary decisions, they're structurally aligned.
        """
        # Extract binary channel and threshold to bits
        bits_i = (x_i[..., :self.d_binary] > 0).float()  # (B, N, K, d_binary)
        bits_j = (x_j[..., :self.d_binary] > 0).float()
        same_bits = (bits_i == bits_j).float()             # (B, N, K, d_binary)
        return same_bits.mean(dim=-1)                      # (B, N, K) fraction matching

    def _beta_alignment(self, beta_i, beta_j):
        """
        C) β alignment: nodes at the SAME scale resonate.
        beta_i, beta_j: (B, N, K) — aperture values in [0, 1]
        returns: (B, N, K) in [0, 1]

        NOT mutual openness (√(β_i·β_j)). Instead: gradient alignment.
        Two nodes with β=0.2 and β=0.3 resonate strongly (same scale).
        Two nodes with β=0.1 and β=0.9 don't (different scales).
        This is what makes cross-scale birth meaningful — it only happens
        when nodes converge to the same operating point.
        """
        return 1.0 - (beta_i - beta_j).abs()

    def forward(
        self,
        x: torch.Tensor,                               # (B, N, D)
        attn_scores: torch.Tensor,                      # (B, N, N) raw alignment
        pi: Optional[torch.Tensor] = None,              # (B, N, 64) — unused, kept for API compat
        beta: Optional[torch.Tensor] = None,            # (B, N) aperture openness
        prev_edges: Optional[torch.Tensor] = None,      # (B, N_prev, K)
        prev_resonance: Optional[torch.Tensor] = None,  # (B, N_prev, K)
        prev_sustained: Optional[torch.Tensor] = None,  # (B, N_prev, K) int counter
    ):
        """
        Returns:
            edges: (B, N, K) — top-K neighbor indices per node
            resonance: (B, N, K) — accumulated resonance per edge
            sustained: (B, N, K) — consecutive-layer counter above threshold
        """
        B, N, D = x.shape
        K = min(self.K, N - 1) if N > 1 else 1

        # 1. Find this layer's top-K neighbors from attention scores
        scores = attn_scores.clone()
        diag_mask = torch.eye(N, device=scores.device, dtype=torch.bool).unsqueeze(0)
        scores.masked_fill_(diag_mask, float('-inf'))
        _, edges = scores.topk(K, dim=-1)  # (B, N, K)

        # 2. Compute instantaneous resonance: A × B × C (multiplicative)
        # All three must agree. No learned weights. The signals gate each other.
        x_i = x.unsqueeze(2).expand(-1, -1, K, -1)  # (B, N, K, D)
        x_j = self._gather_neighbors(x, edges)       # (B, N, K, D)

        # A) Phase alignment: (cos(Δθ) + 1) / 2 → [0, 1]
        r_phase = self._phase_coherence(x_i, x_j)    # (B, N, K) in [0, 1]

        # B) Binary locality: fraction of matching bits in binary channel
        r_locality = self._binary_locality(x_i, x_j)  # (B, N, K) in [0, 1]

        # C) β alignment: 1 - |β_i - β_j| → [0, 1]
        if beta is not None:
            beta_i = beta.unsqueeze(2).expand(-1, -1, K)       # (B, N, K)
            beta_j = self._gather_neighbors(beta.unsqueeze(-1), edges).squeeze(-1)
            r_beta = self._beta_alignment(beta_i, beta_j)      # (B, N, K) in [0, 1]
        else:
            r_beta = torch.ones_like(r_phase)

        # Multiplicative: resonance only grows when all three agree
        r_instant = r_phase * r_locality * r_beta     # (B, N, K) in [0, 1]

        # 3. EMA update — match edges from previous layer
        if prev_edges is not None and prev_resonance is not None:
            N_prev = prev_edges.size(1)
            N_min = min(N, N_prev)

            matched_r = torch.zeros(B, N, K, device=x.device)

            if N_min > 0:
                K_prev = prev_edges.size(-1)
                prev_e = prev_edges[:, :N_min].unsqueeze(-1)   # (B, N_min, K_prev, 1)
                curr_e = edges[:, :N_min].unsqueeze(-2)        # (B, N_min, 1, K)
                match_mask = (prev_e == curr_e)                 # (B, N_min, K_prev, K)

                any_match = match_mask.any(dim=-2)              # (B, N_min, K)
                match_idx = match_mask.float().argmax(dim=-2)   # (B, N_min, K)
                prev_r_expanded = torch.gather(
                    prev_resonance[:, :N_min], 2, match_idx
                )

                ema_r = self.lam * prev_r_expanded + (1 - self.lam) * r_instant[:, :N_min]
                matched_r[:, :N_min] = torch.where(any_match, ema_r, r_instant[:, :N_min])
                if N > N_min:
                    matched_r[:, N_min:] = r_instant[:, N_min:]

                # Sustained counter
                prev_s = prev_sustained[:, :N_min] if prev_sustained is not None else torch.zeros(B, N_min, K_prev, device=x.device)
                prev_s_matched = torch.gather(prev_s, 2, match_idx)
                above_threshold = (matched_r[:, :N_min] > self.threshold).float()
                sustained = torch.zeros(B, N, K, device=x.device)
                sustained[:, :N_min] = torch.where(
                    any_match,
                    (prev_s_matched + 1) * above_threshold,
                    above_threshold,
                )
                if N > N_min:
                    sustained[:, N_min:] = (r_instant[:, N_min:] > self.threshold).float()
            else:
                matched_r = r_instant
                sustained = (r_instant > self.threshold).float()

            resonance = matched_r
        else:
            resonance = r_instant
            sustained = (r_instant > self.threshold).float()

        return edges, resonance, sustained


# ═══════════════════════════════════════════════════════════════════
# BALANCE NORM — from v2
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
# φ-DYNAMIC FFN — from v2
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
        # Guard against NaN from upstream gradient explosion
        if torch.isnan(beta):
            beta = torch.tensor(0.5, device=x.device)
        expansion_ratio = PHI ** (0.5 + beta.item())
        d_active = min(int(self.d_model * expansion_ratio), self.d_hidden_max)
        hidden = self.W_in(x)
        if d_active < self.d_hidden_max:
            hidden[..., d_active:] = 0.0
        hidden = F.gelu(hidden)
        hidden = self.dropout(hidden)
        return self.W_out(hidden)


# ═══════════════════════════════════════════════════════════════════
# TRIADIC EMBEDDING — from v2
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
# BILATERAL HYPERCUBE BLOCK
# β-weighted residual, pressure flow, nesting equation
# ═══════════════════════════════════════════════════════════════════

class BilateralHypercubeBlock(nn.Module):
    """
    Single transformer block using triadic compute:
    1. TriadicComputeBlock (binary SSM + analog attention + fractal hypercube)
    2. DynamicGoldenFFN (on full d_model after recombination)
    3. β-weighted residuals + nesting projection
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 6,
        d_vertex: int = 32,
        dropout: float = 0.1,
        layer_depth: float = 0.5,
        layer_index: int = 0,
        convergence_rate: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.layer_index = layer_index
        self.layer_depth = layer_depth

        self.norm1 = BalanceNorm(d_model)
        self.attn = TriadicComputeBlock(
            d_model, n_heads, d_vertex, dropout,
            layer_depth, layer_index, convergence_rate,
            temperature,
        )
        self.norm2 = BalanceNorm(d_model)
        self.ffn = DynamicGoldenFFN(d_model, dropout)
        self.dropout = nn.Dropout(dropout)

        # β-weighted residual from v2
        self.residual_beta = nn.Parameter(torch.tensor(0.0))

        # Nesting equation: •ₙ₊₁ = ⊙ₙ
        self.nesting_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.eye_(self.nesting_proj.weight)
        with torch.no_grad():
            self.nesting_proj.weight.add_(
                torch.randn_like(self.nesting_proj.weight) * 0.01 * (layer_index + 1)
            )

    def register_cube_buffers(self, cube_buffers):
        self.attn.register_cube_buffers(cube_buffers)

    def forward(self, x, mask=None, neighbor_pressure=0.0, n_tokens=None):
        beta = torch.sigmoid(self.residual_beta)

        # Triadic compute: binary SSM + analog attention + fractal hypercube
        attn_out, pi, alignment = self.attn(self.norm1(x), mask, n_tokens=n_tokens)
        x = x + beta * self.dropout(attn_out)

        # Update FFN local β from attention chambers
        mean_attn_beta = sum(self.attn.beta_values) / len(self.attn.beta_values)
        if not (math.isnan(mean_attn_beta) or math.isinf(mean_attn_beta)):
            self.ffn.local_beta.fill_(mean_attn_beta)

        # FFN with complementary residual weight
        ffn_out = self.ffn(self.norm2(x))
        x = x + (1.0 - beta) * self.dropout(ffn_out)

        # Pressure coupling between layers
        if abs(neighbor_pressure) > 0.01:
            x = x + 0.01 * neighbor_pressure * x

        # Nesting: output of this ⊙ becomes the center of the next
        x = self.nesting_proj(x)
        return x, pi, alignment

    @property
    def mean_pressure(self):
        pressures = self.attn.pressure_values
        return sum(pressures) / len(pressures)


# ═══════════════════════════════════════════════════════════════════
# CROSS-SCALE COUPLING with PHASE RESONANCE — from v3
# ⊛ Convergence: macro reads micro
# ☀ Emergence: micro reads macro
# Gated by aperture AND resonance coherence
# ═══════════════════════════════════════════════════════════════════

class CrossScaleCoupling(nn.Module):
    """
    Cross-scale circumpunct coupling with ACCUMULATED phase-resonance.
    Identical to v3 — the cross-scale mechanism is independent of
    whether attention is bilateral, hypercube, or both.
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

        self.gate_micro = nn.Linear(d_model, 1)
        self.gate_macro = nn.Linear(d_model, 1)

        # ⊛ Convergence
        self.conv_q = nn.Linear(d_model, d_model, bias=False)
        self.conv_k = nn.Linear(d_model, d_model, bias=False)
        self.conv_v = nn.Linear(d_model, d_model, bias=False)
        self.conv_out = nn.Linear(d_model, d_model, bias=False)

        # ☀ Emergence
        self.emer_q = nn.Linear(d_model, d_model, bias=False)
        self.emer_k = nn.Linear(d_model, d_model, bias=False)
        self.emer_v = nn.Linear(d_model, d_model, bias=False)
        self.emer_out = nn.Linear(d_model, d_model, bias=False)

        self.resonance_alpha = nn.Parameter(torch.tensor(1.0))
        self.resonance_lambda = resonance_lambda
        self.dropout = nn.Dropout(dropout)

    def _mha(self, q_proj, k_proj, v_proj, out_proj, q_in, kv_in, mask=None,
             aperture_gate=None, expression_gate=None):
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

        if aperture_gate is not None and expression_gate is not None:
            a_gate = aperture_gate.unsqueeze(1)
            e_gate = expression_gate.unsqueeze(1).transpose(2, 3)
            attn = attn * a_gate * e_gate
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-6)

        attn = self.dropout(attn)
        ctx = torch.matmul(attn, v)
        ctx = ctx.transpose(1, 2).contiguous().view(B, Tq, self.d_model)
        return out_proj(ctx)

    def forward(self, o_micro, o_macro, um_mask=None, mu_mask=None,
                r_conv_acc=None, r_emer_acc=None):
        ou = self.norm_micro(o_micro)
        om = self.norm_macro(o_macro)

        g_micro = torch.sigmoid(self.gate_micro(ou))
        g_macro = torch.sigmoid(self.gate_macro(om))

        # Instantaneous phase coherence
        r_conv_instant = phase_resonance(om, ou)
        r_emer_instant = phase_resonance(ou, om)

        # Resonance accumulation
        lam = self.resonance_lambda
        if r_conv_acc is None:
            r_conv_acc_new = r_conv_instant
        else:
            r_conv_acc_new = lam * r_conv_acc + (1 - lam) * r_conv_instant

        if r_emer_acc is None:
            r_emer_acc_new = r_emer_instant
        else:
            r_emer_acc_new = lam * r_emer_acc + (1 - lam) * r_emer_instant

        r_conv_acc_new = r_conv_acc_new.clamp(-2.0, 2.0)
        r_emer_acc_new = r_emer_acc_new.clamp(-2.0, 2.0)

        r_conv_scalar = torch.sigmoid(
            self.resonance_alpha * r_conv_acc_new.mean(dim=-1, keepdim=True)
        )
        r_emer_scalar = torch.sigmoid(
            self.resonance_alpha * r_emer_acc_new.mean(dim=-1, keepdim=True)
        )

        # ⊛ Convergence
        delta_macro = self._mha(
            self.conv_q, self.conv_k, self.conv_v, self.conv_out,
            om, ou, mask=um_mask,
            aperture_gate=g_macro, expression_gate=g_micro
        )
        o_macro_new = o_macro + self.dropout(delta_macro * r_conv_scalar)

        # ☀ Emergence
        delta_micro = self._mha(
            self.emer_q, self.emer_k, self.emer_v, self.emer_out,
            ou, om, mask=mu_mask,
            aperture_gate=g_micro, expression_gate=g_macro
        )
        o_micro_new = o_micro + self.dropout(delta_micro * r_emer_scalar)

        return o_micro_new, o_macro_new, r_conv_acc_new, r_emer_acc_new


# ═══════════════════════════════════════════════════════════════════
# AIM TOKEN INITIALIZATION — from v3
# ═══════════════════════════════════════════════════════════════════

class AimPool(nn.Module):
    """Create macro tokens from micro tokens via learned aim vectors."""

    def __init__(self, d_model: int, max_chunks: int = 128):
        super().__init__()
        self.d_model = d_model
        self.aim_vectors = nn.Parameter(torch.randn(max_chunks, d_model) * 0.02)
        self.aim_attn = nn.Linear(d_model, d_model, bias=False)
        self.aim_key = nn.Linear(d_model, d_model, bias=False)

    def forward(self, o_micro: torch.Tensor, chunk_size: int) -> torch.Tensor:
        B, T, D = o_micro.shape
        Tm = (T + chunk_size - 1) // chunk_size

        pad = Tm * chunk_size - T
        if pad > 0:
            o_micro = F.pad(o_micro, (0, 0, 0, pad))

        chunks = o_micro.view(B, Tm, chunk_size, D)
        aims = self.aim_vectors[:Tm].unsqueeze(0).expand(B, -1, -1)

        aim_q = self.aim_attn(aims)
        chunk_k = self.aim_key(chunks)

        scores = torch.einsum('btd,btsd->bts', aim_q, chunk_k) / math.sqrt(D)
        attn = F.softmax(scores, dim=-1)
        pooled = torch.einsum('bts,btsd->btd', attn, chunks)

        return aims + pooled


# ═══════════════════════════════════════════════════════════════════
# INTERSECTION MLP — for vesica birth embedding
# ○_k = IntersectMLP([○_i, ○_j, ○_i ⊙ ○_j, |○_i - ○_j|])
# ═══════════════════════════════════════════════════════════════════

class IntersectMLP(nn.Module):
    """Computes intersection embedding for vesica birth.

    The child embedding is not a concatenation — it is an intersection:
    the MLP learns to extract what is SHARED between two parent embeddings,
    which is exactly what the vesica piscis represents geometrically.

    Uses tanh(emb_i * emb_j) to bound the multiplicative interaction —
    raw products spike fast and are the primary explosion vector.
    """
    def __init__(self, d_model: int):
        super().__init__()
        # 3 channels: emb_i, emb_j, tanh(emb_i * emb_j)
        self.net = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, emb_i: torch.Tensor, emb_j: torch.Tensor) -> torch.Tensor:
        """
        emb_i, emb_j: (..., D)
        returns: (..., D) — intersection embedding, normalized
        """
        # Bounded multiplicative interaction — tanh prevents magnitude spikes
        prod = torch.tanh(emb_i * emb_j)
        combined = torch.cat([emb_i, emb_j, prod], dim=-1)
        return self.norm(self.net(combined))


# ═══════════════════════════════════════════════════════════════════
# VESICA BIRTH — Full Vision version
# Birth from SUSTAINED resonance on sparse edges.
# Children get intersection embeddings and binary apertures via STE.
# ═══════════════════════════════════════════════════════════════════

class VesicaBirth(nn.Module):
    """
    Detect sustained resonance on sparse edges and birth new nodes.

    A new circumpunct is born when:
    1. Edge (i,j) has resonance r_ij > τ for M consecutive layers
    2. The overlap isn't trivial (parents aren't identical or disjoint)
    3. Birth budget hasn't been exhausted for this layer

    The child gets:
        ○_k = IntersectMLP([○_i, ○_j, ○_i ⊙ ○_j, |○_i - ○_j|])
        •_k = STE_Bernoulli(σ(W_aperture[○_i, ○_j, r_ij]))
    """

    def __init__(self, d_model: int, max_births_per_layer: int = 8,
                 sustained_M: int = 3, diversity_threshold: float = 0.9):
        super().__init__()
        self.d_model = d_model
        self.max_births = max_births_per_layer
        self.sustained_M = sustained_M
        self.diversity_threshold = diversity_threshold

        # Intersection embedding for child boundary (○)
        self.intersect = IntersectMLP(d_model)

        # Binary aperture gate for child (•)
        # Input: parent_i, parent_j, resonance scalar
        self.aperture_proj = nn.Linear(d_model * 2 + 1, d_model)

        # Birth quality gate
        self.birth_gate = nn.Linear(d_model, 1)

        # Track which parent pairs have already birthed (for diversity)
        self._used_pairs = set()

    def reset_birth_memory(self):
        """Reset per-forward-pass birth memory."""
        self._used_pairs = set()

    def forward(
        self,
        x: torch.Tensor,              # (B, N, D) all current nodes
        edges: torch.Tensor,           # (B, N, K) sparse edge indices
        resonance: torch.Tensor,       # (B, N, K) resonance values
        sustained: torch.Tensor,       # (B, N, K) consecutive-layer counter
        node_beta: Optional[torch.Tensor] = None,  # (B, N) per-node β
    ) -> Tuple[Optional[torch.Tensor], int, Optional[torch.Tensor]]:
        """
        Returns:
            new_born: (B, n_born, D) or None if no births
            n_births: total number of births across batch
            child_beta: (B, n_born) β values for born nodes, or None
        """
        B, N, D = x.shape
        K = edges.size(-1)
        device = x.device

        # Find edges with sustained resonance >= M
        qualified = sustained >= self.sustained_M  # (B, N, K)

        if not qualified.any():
            return None, 0, None

        # Score qualified edges by resonance strength
        scores = resonance * qualified.float()  # (B, N, K)

        # Flatten to find global top-B births per batch
        scores_flat = scores.view(B, -1)  # (B, N*K)
        n_candidates = min(self.max_births, scores_flat.size(-1))
        topk_vals, topk_flat_idx = scores_flat.topk(n_candidates, dim=-1)

        # Convert flat indices back to (node_idx, edge_slot)
        node_idx = topk_flat_idx // K   # (B, n_candidates)
        edge_slot = topk_flat_idx % K   # (B, n_candidates)

        birth_mask = topk_vals > 0  # (B, n_candidates)
        n_births = birth_mask.sum().item()

        if n_births == 0:
            return None, 0, None

        # Gather parent embeddings
        parent_i_idx = node_idx  # (B, n_candidates)

        # Vectorized gather: edges[b, node_idx[b,c], edge_slot[b,c]]
        # edges is (B, N, K), we index with node_idx and edge_slot
        # Flatten edges to (B, N*K) and use topk_flat_idx directly
        parent_j_idx = torch.gather(edges.view(B, -1), 1, topk_flat_idx)

        # Gather parent embeddings
        emb_i = torch.gather(x, 1, parent_i_idx.unsqueeze(-1).expand(-1, -1, D))
        emb_j = torch.gather(x, 1, parent_j_idx.clamp(0, N-1).unsqueeze(-1).expand(-1, -1, D))

        # Diversity check: child shouldn't be too similar to either parent
        child_boundary = self.intersect(emb_i, emb_j)  # (B, n_candidates, D)

        cos_i = F.cosine_similarity(child_boundary, emb_i, dim=-1)  # (B, n_candidates)
        cos_j = F.cosine_similarity(child_boundary, emb_j, dim=-1)
        diverse_enough = (cos_i < self.diversity_threshold) & (cos_j < self.diversity_threshold)
        birth_mask = birth_mask & diverse_enough

        n_births = birth_mask.sum().item()
        if n_births == 0:
            return None, 0, None

        # Binary aperture via STE Bernoulli
        r_vals = torch.gather(resonance.view(B, -1), 1, topk_flat_idx)  # (B, n_candidates)
        aperture_input = torch.cat([emb_i, emb_j, r_vals.unsqueeze(-1)], dim=-1)
        aperture_logits = self.aperture_proj(aperture_input)  # (B, n_candidates, D)
        aperture_prob = torch.sigmoid(aperture_logits)
        # STE Bernoulli: hard sample, soft gradient
        aperture_hard = (aperture_prob > 0.5).float()
        aperture = aperture_hard - aperture_prob.detach() + aperture_prob

        # Birth quality gate
        gate = torch.sigmoid(self.birth_gate(child_boundary))  # (B, n_candidates, 1)

        # Final child embedding: boundary gated by aperture
        child_emb = child_boundary * gate * aperture  # (B, n_candidates, D)

        # Zero out non-births
        child_emb = child_emb * birth_mask.unsqueeze(-1).float()

        # ─── Child β: stability-constrained from intersection geometry ───
        # β_child = mean(β_i, β_j) - γ · |β_i - β_j|
        # Vesica curvature is TIGHTER than either parent.
        # Cross-scale children contract, not explore. Scale tension compression.
        # γ = 0.25, clamped to [0.55, 0.70] to stay in convergent regime.
        if node_beta is not None:
            beta_i = torch.gather(node_beta, 1, parent_i_idx)                  # (B, n_candidates)
            beta_j = torch.gather(node_beta, 1, parent_j_idx.clamp(0, N - 1))  # (B, n_candidates)
            beta_mean = (beta_i + beta_j) * 0.5
            beta_diff = (beta_i - beta_j).abs()
            gamma = 0.25
            child_beta = beta_mean - gamma * beta_diff
            child_beta = child_beta.clamp(0.55, 0.70)
            child_beta = child_beta * birth_mask.float()  # zero out non-births
        else:
            child_beta = None

        # Find actual max births per batch element for padding
        births_per_batch = birth_mask.sum(dim=-1)  # (B,)
        max_born = births_per_batch.max().item()

        if max_born == 0:
            return None, 0, None

        # Compact: gather only the born children
        # For simplicity, keep all candidates but zero out non-births
        # (the pruner will clean up zero-norm nodes later)
        new_born = child_emb[:, :max_born]  # (B, max_born, D)
        if child_beta is not None:
            child_beta = child_beta[:, :max_born]  # (B, max_born)

        return new_born, n_births, child_beta


# ═══════════════════════════════════════════════════════════════════
# NODE MATURATION — The Aperture Traversal of Born Nodes
#
# A child node recapitulates the circumpunct traversal during
# its own development:
#
#   Stage D (0.5D): Hard closed. aperture = 0. Visible but silent.
#                   No gradient flows. The bare crossing.
#
#   Stage A (1.0D): Scalar opening. aperture ∈ (0, τ_diff).
#                   Uniform gating — everything opens together.
#                   Binary moments accumulating into continuity.
#
#   Stage C (2.0D): Channel differentiation. aperture ≥ τ_diff.
#                   Binary/analog/fractal gates open independently.
#                   Relation becomes directional.
#
#   Stage B (3.0D): Mature. All channel gates > τ_mature.
#                   Full participation, indistinguishable from tokens.
#                   Fully structural.
#
# Transitions are driven by RESONANCE. Without sustained resonance,
# aperture decays. The child must earn its participation.
# The traversal is ordered: D → A → C → B.
# Regression allowed (any → D). Skipping forbidden.
# ═══════════════════════════════════════════════════════════════════

class NodeMaturation(nn.Module):
    """
    Foveated maturation for born nodes — progressive rendering.

    Like downloading an image: blurry first, then sharpens where you look.
    Tracks per-node aperture (how sharp) and channel gates (which channels).
    Resonance drives aperture (attention = focus = sharpening).
    The model sees everything from birth — just out of focus at first.
    """

    STAGE_D = 0  # blurry — all channels averaged, whole picture at low res
    STAGE_A = 1  # sharpening — channels beginning to separate
    STAGE_C = 2  # differentiated — full channel separation, clear picture
    STAGE_B = 3  # integrated — mature, acts on what it sees
    STAGE_NAMES = ["D(blurry)", "A(sharpening)", "C(clear)", "B(integrated)"]

    def __init__(
        self,
        d_model: int,
        open_rate: float = 0.1,
        decay: float = 0.98,
        differentiation_threshold: float = 0.5,
        maturation_threshold: float = 0.8,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_binary = d_model // 4
        self.d_analog = d_model // 2
        self.d_fractal = d_model - self.d_binary - self.d_analog

        self.open_rate = open_rate
        self.decay = decay
        self.diff_threshold = differentiation_threshold
        self.mat_threshold = maturation_threshold

        # Learned channel split: how aperture distributes across channels.
        # Initialized so binary opens first (fast gate), then analog, then fractal.
        # • is fastest, Φ mediates, ○ is slowest.
        self.channel_split_raw = nn.Parameter(
            torch.tensor([1.0, 0.0, -0.5])  # sigmoid → ~0.73, 0.50, 0.38
        )

    @property
    def channel_split(self):
        return torch.sigmoid(self.channel_split_raw)

    def init_state(self, B: int, n_nodes: int, device: torch.device):
        """Initialize maturation state for newly born nodes."""
        return {
            'aperture': torch.zeros(B, n_nodes, device=device),
            'channel_gates': torch.zeros(B, n_nodes, 3, device=device),
        }

    def concat_state(self, existing, new_state):
        """Concatenate new born node state onto existing."""
        if existing is None:
            return new_state
        return {
            'aperture': torch.cat([existing['aperture'], new_state['aperture']], dim=1),
            'channel_gates': torch.cat([existing['channel_gates'], new_state['channel_gates']], dim=1),
        }

    def prune_state(self, state, keep_mask):
        """Remove pruned nodes from maturation state."""
        if state is None:
            return None
        return {
            'aperture': state['aperture'][:, keep_mask],
            'channel_gates': state['channel_gates'][:, keep_mask],
        }

    def cap_state(self, state, max_n):
        """Hard cap on number of tracked nodes."""
        if state is None:
            return None
        return {
            'aperture': state['aperture'][:, :max_n],
            'channel_gates': state['channel_gates'][:, :max_n],
        }

    def clear_state(self):
        """Return None to signal all nodes pruned."""
        return None

    def update(self, state, born_resonance):
        """
        Update aperture and channel gates based on resonance signal.

        state: dict with 'aperture' (B, n_born) and 'channel_gates' (B, n_born, 3)
        born_resonance: (B, n_born) — max resonance each born node received

        Returns updated state dict.
        """
        if state is None:
            return None

        aperture = state['aperture']
        cg = state['channel_gates']

        # Resonance-driven opening — no threshold.
        # Driving force proportional to resonance itself.
        # Decay vs drive determines steady state naturally:
        #   high sustained resonance → aperture opens
        #   low/zero resonance → decay wins, aperture closes
        # This is how real oscillators work: damping vs driving force.
        drive = born_resonance.clamp(min=0.0) * self.open_rate

        # Decay + drive
        aperture = aperture * self.decay + drive
        aperture = aperture.clamp(0.0, 1.0)

        # Channel differentiation: once aperture > diff_threshold,
        # channel gates move toward aperture * channel_split
        is_diff = (aperture > self.diff_threshold).unsqueeze(-1)  # (B, n, 1)
        split = self.channel_split.unsqueeze(0).unsqueeze(0)      # (1, 1, 3)
        target_gates = aperture.unsqueeze(-1) * split              # (B, n, 3)

        # EMA toward target for differentiating nodes (0.15 per layer)
        cg = torch.where(is_diff, cg * 0.85 + target_gates * 0.15, cg)
        cg = cg.clamp(0.0, 1.0)

        return {
            'aperture': aperture,
            'channel_gates': cg,
        }

    def gate_embeddings(self, born_nodes, state):
        """
        Apply foveated maturation gating to born node embeddings.

        Like progressive image loading / foveated rendering:
        Stage D (aperture == 0): BLURRY — all channels averaged into one
            smeared value. The node is present and participates, just
            without any channel detail. Like a 1-pixel thumbnail.
        Stage A (0 < aperture < diff): SHARPENING — channels beginning
            to separate. Blend between blur and per-channel detail,
            controlled by aperture (= where the model is "looking").
        Stage C (aperture >= diff): CLEAR — full per-channel gating.
            The node sees •, Φ, ○ as distinct streams.
        Stage B (all channels > mat): INTEGRATED — full participation,
            effectively gate ≈ 1 everywhere.

        Returns gated embeddings for graph assembly.
        """
        if state is None or born_nodes is None:
            return born_nodes

        aperture = state['aperture']      # (B, n_born)
        cg = state['channel_gates']       # (B, n_born, 3)

        B, n_born, D = born_nodes.shape
        db, da, df = self.d_binary, self.d_analog, self.d_fractal

        # Align state with current born count
        if aperture.size(1) < n_born:
            pad_a = torch.zeros(B, n_born - aperture.size(1), device=aperture.device)
            pad_cg = torch.zeros(B, n_born - cg.size(1), 3, device=cg.device)
            aperture = torch.cat([aperture, pad_a], dim=1)
            cg = torch.cat([cg, pad_cg], dim=1)
        elif aperture.size(1) > n_born:
            aperture = aperture[:, :n_born]
            cg = cg[:, :n_born]

        # ═══ FOVEATED RENDERING ═══
        # Like downloading an image: blurry first, then sharpens where you look.
        # Total energy stays constant — only detail (channel separation) changes.

        # BLUR: average all channels into one smeared value.
        # The whole picture is there from birth, just out of focus.
        channel_mean = born_nodes.mean(dim=-1, keepdim=True)  # (B, n_born, 1)
        blurred = channel_mean.expand_as(born_nodes)          # uniform blur

        # SHARP: the original signal with full channel detail.
        # For Stage C/B, modulated by learned per-channel gates.
        is_diff = (aperture > self.diff_threshold).unsqueeze(-1)
        channel_gate = torch.cat([
            cg[..., 0:1].expand(-1, -1, db),
            cg[..., 1:2].expand(-1, -1, da),
            cg[..., 2:3].expand(-1, -1, df),
        ], dim=-1)  # (B, n_born, D)

        # Below diff_threshold: sharp = raw original (uniform focus)
        # Above diff_threshold: sharp = channel-gated (selective focus)
        sharp = torch.where(is_diff, born_nodes * channel_gate, born_nodes)

        # Aperture controls blur→sharp blend (foveated rendering).
        # aperture=0 → fully blurry (Stage D, just born)
        # aperture→1 → fully sharp (Stage B, integrated)
        # Resonance drives aperture. Where the model looks, it sharpens.
        blend = aperture.unsqueeze(-1)  # (B, n_born, 1)

        # Linear interpolation preserves total energy:
        # At aperture=0: pure blur (all channels same average)
        # At aperture=1: pure sharp (full channel detail)
        born_for_graph = blurred + blend * (sharp - blurred)

        return born_for_graph

    def get_born_resonance(self, resonance, edges, S, N, n_born, device):
        """
        Extract per-born-node max resonance from tracker output.

        Born nodes sit at indices S..N-1.
        Combines outgoing resonance (born→neighbors) and incoming (tokens→born).
        """
        B = resonance.size(0)
        born_max_res = torch.zeros(B, n_born, device=device)

        if N <= S or n_born == 0:
            return born_max_res

        K = resonance.size(-1)

        # Outgoing: born nodes' resonance to their neighbors
        n_avail = min(n_born, resonance.size(1) - S)
        if n_avail > 0:
            out_res = resonance[:, S:S + n_avail, :].max(dim=-1).values  # (B, n_avail)
            born_max_res[:, :n_avail] = out_res

        # Incoming: tokens pointing to born nodes as neighbors
        # Vectorized: check which token edges point to born indices [S, S+n_born)
        token_edges = edges[:, :S, :]     # (B, S, K)
        token_res = resonance[:, :S, :]   # (B, S, K)

        # For each born index, find max resonance from tokens whose edge points there
        n_check = min(n_born, 16)  # cap for performance
        for born_idx in range(n_check):
            target = S + born_idx
            is_target = (token_edges == target).float()  # (B, S, K)
            if is_target.any():
                incoming = (token_res * is_target).amax(dim=(1, 2))  # (B,)
                born_max_res[:, born_idx] = torch.max(
                    born_max_res[:, born_idx], incoming
                )

        return born_max_res

    def stage_counts(self, state):
        """Diagnostic: count nodes in each maturation stage."""
        if state is None:
            return {name: 0 for name in self.STAGE_NAMES}

        aperture = state['aperture']
        cg = state['channel_gates']

        # Average across batch
        a = aperture.mean(dim=0)  # (n,)
        cg_min = cg.mean(dim=0).min(dim=-1).values  # (n,)

        n_d = (a == 0).sum().item()
        n_a = ((a > 0) & (a < self.diff_threshold)).sum().item()
        n_c = ((a >= self.diff_threshold) & (cg_min < self.mat_threshold)).sum().item()
        n_b = (cg_min >= self.mat_threshold).sum().item()

        return dict(zip(self.STAGE_NAMES, [n_d, n_a, n_c, n_b]))


# ═══════════════════════════════════════════════════════════════════
# XORZO v4 — THE BILATERAL HYPERCUBE ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════

class XorzoV4Transformer(nn.Module):
    """
    Xorzo v4 — Bilateral Hypercube Transformer.

    The unification of v3's bilateral attention with the 6D hypercube.
    Each token is a circumpunct with 3 binary gates (•, Φ, ○).
    Each token pair navigates through 64 relational states on the hypercube.
    Two streams (micro/macro) braided through resonance-gated coupling.

    2³ × 2 = 64. Two circumpuncts facing each other.
    The hypercube IS the topology of bilateral relationship.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 128,
        n_layers: int = 6,
        n_heads: int = 8,
        d_vertex: int = 32,
        max_len: int = 512,
        dropout: float = 0.1,
        generation: int = 0,
        convergence_rate: float = 0.1,
        chunk_size: int = 16,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_vertex = d_vertex
        self.generation = generation
        self.chunk_size = chunk_size
        self.temperature = temperature

        # Build hypercube geometry
        cube = Hypercube6D()
        cube_buffers = cube.to_buffers()

        # Register cube geometry as module buffers
        self.register_buffer("adjacency", cube_buffers["adjacency"])
        self.register_buffer("vertices", cube_buffers["vertices"])
        self.register_buffer("spectral_emb", cube_buffers["spectral_emb"])
        self.register_buffer("aperture_mask", cube_buffers["aperture_mask"])
        self.register_buffer("field_mask", cube_buffers["field_mask"])
        self.register_buffer("boundary_mask", cube_buffers["boundary_mask"])
        self.register_buffer("mutual_mask", cube_buffers["mutual_mask"])

        # Triadic embedding + positional encoding
        self.token_embed = TriadicEmbedding(vocab_size, d_model)
        self.pos_encode = GoldenPositionalEncoding(d_model, max_len)
        self.embed_dropout = nn.Dropout(dropout)

        # Aim pooling
        self.aim_pool = AimPool(d_model, max_chunks=max_len // chunk_size + 1)
        self.macro_init = nn.Linear(d_model, d_model)

        # ═══ MICRO BLOCKS: bilateral hypercube attention ═══
        self.micro_blocks = nn.ModuleList([
            BilateralHypercubeBlock(
                d_model, n_heads, d_vertex, dropout,
                layer_depth=i / max(n_layers - 1, 1),
                layer_index=i,
                convergence_rate=convergence_rate,
                temperature=temperature,
            )
            for i in range(n_layers)
        ])

        # ═══ MACRO BLOCKS: bilateral hypercube attention (converges faster) ═══
        self.macro_blocks = nn.ModuleList([
            BilateralHypercubeBlock(
                d_model, n_heads, d_vertex, dropout,
                layer_depth=i / max(n_layers - 1, 1),
                layer_index=i,
                convergence_rate=convergence_rate * PHI,
                temperature=temperature,
            )
            for i in range(n_layers)
        ])

        # Register cube buffers in all blocks
        all_cube_bufs = {
            "adjacency": self.adjacency,
            "vertices": self.vertices,
            "spectral_emb": self.spectral_emb,
            "aperture_mask": self.aperture_mask,
            "field_mask": self.field_mask,
            "boundary_mask": self.boundary_mask,
            "mutual_mask": self.mutual_mask,
        }
        for block in self.micro_blocks:
            block.register_cube_buffers(all_cube_bufs)
        for block in self.macro_blocks:
            block.register_cube_buffers(all_cube_bufs)

        # Cross-scale couplers (same as v3)
        self.couplers = nn.ModuleList([
            CrossScaleCoupling(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])

        # ═══ SPARSE RESONANCE TRACKER (Step 2) ═══
        self.resonance_tracker = SparseResonanceTracker(
            d_model, K=8, lam=0.7, sustained_M=3,
        )

        # ═══ VESICA BIRTH (Step 3 — Full Vision) ═══
        self.vesica = VesicaBirth(
            d_model,
            max_births_per_layer=max(4, n_heads),
            sustained_M=3,
            diversity_threshold=0.9,
        )
        self.birth_start_layer = max(1, n_layers // 3)

        # ═══ NODE MATURATION (Step 4 — Aperture Traversal) ═══
        self.maturation = NodeMaturation(d_model)

        # ═══ PRUNING (Step 5) ═══
        self.prune_after_layers = max(2, n_layers // 2)  # min layers before pruning
        self.attn_mass_threshold = 0.01  # min attention mass to survive
        self.born_energy_budget = float(d_model)  # max total born norm
        self.max_total_nodes = max_len * 2  # hard cap: 2× sequence length

        # Final emergence
        self.final_norm = BalanceNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Diagnostic state
        self._last_total_births = 0
        self._last_total_pruned = 0
        self._last_n_born = 0
        self._last_maturation_stages = {name: 0 for name in NodeMaturation.STAGE_NAMES}

        self._golden_init()

    def _golden_init(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'nesting_proj' not in name and 'aim_vectors' not in name \
               and 'vertex_emb' not in name and 'vertex_gates' not in name:
                nn.init.xavier_normal_(p, gain=1.0 / math.sqrt(PHI))

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Full Vision forward pass:
        - Tokens and born nodes form a growing active graph
        - Sparse resonance tracked per layer
        - Birth from sustained resonance
        - Pruning of stale born nodes
        - Macro stream derived from tokens (not born nodes)
        """
        B, T = x.shape
        device = x.device

        # ⊛ Embed into triadic space
        h = self.token_embed(x)
        h = self.pos_encode(h)
        o_micro = self.embed_dropout(h)

        # Build macro from micro (aim tokens)
        o_macro = self.aim_pool(o_micro, self.chunk_size)
        o_macro = self.macro_init(o_macro)

        S = o_micro.size(1)   # fixed token count
        Tm = o_macro.size(1)

        # ── Causal masks for tokens ──
        if mask is None:
            token_mask = torch.tril(torch.ones(S, S, device=device)).unsqueeze(0).unsqueeze(0)
        else:
            token_mask = mask

        macro_mask = torch.tril(torch.ones(Tm, Tm, device=device)).unsqueeze(0).unsqueeze(0)

        # Cross-scale causal masks (tokens ↔ macro only)
        micro_chunk_idx = torch.arange(S, device=device) // self.chunk_size
        macro_idx = torch.arange(Tm, device=device)

        um_block = micro_chunk_idx.unsqueeze(0) > macro_idx.unsqueeze(1)
        um_mask = torch.zeros(Tm, S, device=device).masked_fill(um_block, float('-inf'))
        um_mask = um_mask.unsqueeze(0).unsqueeze(0)

        mu_block = macro_idx.unsqueeze(0) > micro_chunk_idx.unsqueeze(1)
        mu_mask = torch.zeros(S, Tm, device=device).masked_fill(mu_block, float('-inf'))
        mu_mask = mu_mask.unsqueeze(0).unsqueeze(0)

        # ── Living graph state ──
        born_nodes = None         # (B, n_born, D) — grows through layers
        n_born = 0

        # Sparse resonance state (persists across layers)
        prev_edges = None
        prev_resonance = None
        prev_sustained = None

        # Born node tracking
        born_attn_mass = None     # (B, n_born) cumulative attention received
        born_stale_count = None   # (B, n_born) consecutive layers with low attention
        born_beta = None          # (B, n_born) per-node β inherited from parents
        mat_state = None          # maturation state: {'aperture': (B, n_born), 'channel_gates': (B, n_born, 3)}

        micro_pressures = [0.0] * self.n_layers
        macro_pressures = [0.0] * self.n_layers
        r_conv_acc = None
        r_emer_acc = None
        Tm_base = Tm

        # Reset birth memory
        self.vesica.reset_birth_memory()

        # Track total births for diagnostics
        total_births = 0
        total_pruned = 0

        for i, (blk_micro, blk_macro, coupler) in enumerate(
            zip(self.micro_blocks, self.macro_blocks, self.couplers)
        ):
          try:
            # ═══════════════════════════════════════════════════
            # 1. ASSEMBLE ACTIVE GRAPH: tokens + born nodes
            # ═══════════════════════════════════════════════════
            if born_nodes is not None and born_nodes.size(1) > 0:
                # ─── MATURATION GATING at assembly ───
                # Nodes are gated by their maturation stage:
                #   Stage D: detached (visible as keys, no gradient)
                #   Stage A: uniform scalar gate
                #   Stage C: per-channel gate (binary/analog/fractal independent)
                #   Stage B: full participation
                born_for_graph = self.maturation.gate_embeddings(born_nodes, mat_state)

                all_nodes = torch.cat([o_micro, born_for_graph], dim=1)  # (B, S+n_born, D)
                N = S + born_for_graph.size(1)

                # Build combined mask: tokens causal + born nodes can see all tokens
                combined_mask = torch.ones(N, N, device=device).unsqueeze(0).unsqueeze(0)
                combined_mask[:, :, :S, :S] = token_mask[:, :, :S, :S]
            else:
                all_nodes = o_micro
                N = S
                combined_mask = token_mask

            # ═══════════════════════════════════════════════════
            # 2. MICRO BLOCK: triadic compute over all nodes
            # ═══════════════════════════════════════════════════
            micro_np = micro_pressures[i - 1] if i > 0 else 0.0
            all_nodes, _pi, alignment = blk_micro(
                all_nodes, combined_mask,
                neighbor_pressure=micro_np,
                n_tokens=S,
            )
            micro_pressures[i] = blk_micro.mean_pressure

            # Split back into tokens and born
            o_micro = all_nodes[:, :S]
            if N > S:
                born_nodes = all_nodes[:, S:]

            # ═══════════════════════════════════════════════════
            # 3. SPARSE RESONANCE TRACKING (Mode D: phase × locality × β_align)
            # ═══════════════════════════════════════════════════
            # Build per-node β: tokens get chamber β, born nodes keep inherited β
            with torch.no_grad():
                token_beta_val = torch.sigmoid(blk_micro.attn.binary_chamber.beta)
                token_beta = token_beta_val.expand(B, S)  # scalar → (B, S)
                n_born_now = N - S
                if born_beta is not None and n_born_now > 0:
                    # Align born_beta size with actual born node count
                    if born_beta.size(1) < n_born_now:
                        born_beta = F.pad(born_beta, (0, n_born_now - born_beta.size(1)), value=0.6)
                    elif born_beta.size(1) > n_born_now:
                        born_beta = born_beta[:, :n_born_now]
                    node_beta = torch.cat([token_beta, born_beta], dim=1)  # (B, N)
                else:
                    node_beta = token_beta  # (B, S) = (B, N) when no born nodes

            edges, resonance, sustained = self.resonance_tracker(
                all_nodes, alignment,
                pi=_pi,              # (B, N, 64) hard vertex distribution
                beta=node_beta,      # (B, N) per-node aperture openness
                prev_edges=prev_edges,
                prev_resonance=prev_resonance,
                prev_sustained=prev_sustained,
            )
            prev_edges = edges.detach()
            prev_resonance = resonance.detach()
            prev_sustained = sustained.detach()

            # ═══════════════════════════════════════════════════
            # 3b. MATURATION UPDATE — aperture opens from resonance
            #      no_grad: maturation state is managed manually,
            #      not through backprop. Without this, the computation
            #      graph holds resonance tensors alive → OOM.
            # ═══════════════════════════════════════════════════
            if mat_state is not None and born_nodes is not None and born_nodes.size(1) > 0:
              with torch.no_grad():
                n_b = born_nodes.size(1)
                # Align mat_state with current born count
                if mat_state['aperture'].size(1) < n_b:
                    pad_state = self.maturation.init_state(B, n_b - mat_state['aperture'].size(1), device)
                    mat_state = self.maturation.concat_state(mat_state, pad_state)
                elif mat_state['aperture'].size(1) > n_b:
                    mat_state = self.maturation.cap_state(mat_state, n_b)

                # Extract per-born-node resonance signal
                born_res = self.maturation.get_born_resonance(
                    resonance, edges, S, N, n_b, device
                )
                mat_state = self.maturation.update(mat_state, born_res)

            # ═══════════════════════════════════════════════════
            # 4. VESICA BIRTH from sustained resonance
            # ═══════════════════════════════════════════════════
            if i >= self.birth_start_layer:
                new_born, n_births, child_beta = self.vesica(
                    all_nodes, edges, resonance, sustained,
                    node_beta=node_beta,
                )
                total_births += n_births

                if new_born is not None and n_births > 0:
                    # Born node magnitude diagnostic
                    with torch.no_grad():
                        max_val = new_born.abs().max().item()
                        if max_val > 50:
                            print(f"    ⚠ Born node magnitude spike: {max_val:.2f} (layer {i})")

                    if born_nodes is not None:
                        born_nodes = torch.cat([born_nodes, new_born], dim=1)
                    else:
                        born_nodes = new_born

                    # Initialize tracking for new born nodes
                    n_new = new_born.size(1)
                    if born_attn_mass is not None:
                        born_attn_mass = torch.cat([
                            born_attn_mass,
                            torch.zeros(B, n_new, device=device)
                        ], dim=1)
                        born_stale_count = torch.cat([
                            born_stale_count,
                            torch.zeros(B, n_new, device=device)
                        ], dim=1)
                    else:
                        born_attn_mass = torch.zeros(B, n_new, device=device)
                        born_stale_count = torch.zeros(B, n_new, device=device)

                    # Track born node β (stability-constrained inheritance)
                    if child_beta is not None:
                        if born_beta is not None:
                            born_beta = torch.cat([born_beta, child_beta], dim=1)
                        else:
                            born_beta = child_beta

                    # Initialize maturation state: all new nodes start at Stage D (closed)
                    new_mat = self.maturation.init_state(B, n_new, device)
                    mat_state = self.maturation.concat_state(mat_state, new_mat)

            # ═══════════════════════════════════════════════════
            # 5. TRACK ATTENTION MASS for born nodes (pruning)
            # ═══════════════════════════════════════════════════
            if born_nodes is not None and born_attn_mass is not None:
                n_b = born_nodes.size(1)
                # Attention mass = sum of attention that tokens pay to born nodes
                # alignment[:, :S, S:] = token→born attention scores
                if N > S:
                    attn_to_born = F.softmax(alignment[:, :S, :], dim=-1)[:, :, S:N]
                    mass = attn_to_born.sum(dim=1)  # (B, n_born_at_attn_time)
                    # Pad if born grew since attention was computed
                    if mass.size(1) < n_b:
                        mass = F.pad(mass, (0, n_b - mass.size(1)))
                    elif mass.size(1) > n_b:
                        mass = mass[:, :n_b]
                    born_attn_mass = born_attn_mass[:, :n_b] + mass[:, :n_b]

                    # Update stale counter
                    stale = (mass[:, :n_b] < self.attn_mass_threshold).float()
                    born_stale_count = born_stale_count[:, :n_b] * stale + stale
                    # Nodes with attention get their counter reset to 0
                    born_stale_count = born_stale_count * stale

            # ═══════════════════════════════════════════════════
            # 6. PRUNE stale born nodes
            # ═══════════════════════════════════════════════════
            if (born_nodes is not None and born_stale_count is not None
                    and i >= self.prune_after_layers
                    and born_nodes.size(1) > 0):
                # Prune nodes stale for too many layers
                keep_mask = born_stale_count.mean(dim=0) < self.prune_after_layers  # per-node across batch
                if not keep_mask.all():
                    n_pruned = (~keep_mask).sum().item()
                    total_pruned += n_pruned
                    born_nodes = born_nodes[:, keep_mask]
                    born_attn_mass = born_attn_mass[:, keep_mask]
                    born_stale_count = born_stale_count[:, keep_mask]
                    if born_beta is not None:
                        born_beta = born_beta[:, keep_mask]
                    mat_state = self.maturation.prune_state(mat_state, keep_mask)

                    if born_nodes.size(1) == 0:
                        born_nodes = None
                        born_attn_mass = None
                        born_stale_count = None
                        born_beta = None
                        mat_state = self.maturation.clear_state()

            # ═══════════════════════════════════════════════════
            # 7. ENERGY CONSERVATION + HARD CAP
            # ═══════════════════════════════════════════════════
            if born_nodes is not None and born_nodes.size(1) > 0:
                # Energy conservation
                total_norm = born_nodes.norm(dim=-1).sum(dim=-1, keepdim=True)  # (B, 1)
                scale = torch.clamp(
                    self.born_energy_budget / (total_norm + 1e-8), max=1.0
                )
                born_nodes = born_nodes * scale.unsqueeze(-1)

                # Hard cap: 2× sequence length. No exceptions.
                max_born = self.max_total_nodes - S
                if born_nodes.size(1) > max_born:
                    born_nodes = born_nodes[:, :max_born]
                    if born_attn_mass is not None:
                        born_attn_mass = born_attn_mass[:, :max_born]
                    if born_stale_count is not None:
                        born_stale_count = born_stale_count[:, :max_born]
                    if born_beta is not None:
                        born_beta = born_beta[:, :max_born]
                    mat_state = self.maturation.cap_state(mat_state, max_born)

            # ═══════════════════════════════════════════════════
            # 8. RE-GROUND MACRO from tokens (not born nodes)
            # ═══════════════════════════════════════════════════
            o_macro_grounded = self.aim_pool(o_micro, self.chunk_size)
            o_macro_base = 0.7 * o_macro[:, :Tm_base] + 0.3 * o_macro_grounded
            if o_macro.size(1) > Tm_base:
                o_macro = torch.cat([o_macro_base, o_macro[:, Tm_base:]], dim=1)
            else:
                o_macro = o_macro_base

            # ═══════════════════════════════════════════════════
            # 9. MACRO BLOCK
            # ═══════════════════════════════════════════════════
            macro_np = macro_pressures[i - 1] if i > 0 else 0.0
            o_macro, _pi_macro, _align_macro = blk_macro(
                o_macro, macro_mask, neighbor_pressure=macro_np
            )
            macro_pressures[i] = blk_macro.mean_pressure

            # ═══════════════════════════════════════════════════
            # 10. CROSS-SCALE COUPLING (tokens ↔ macro)
            # ═══════════════════════════════════════════════════
            o_micro, o_macro, r_conv_acc, r_emer_acc = coupler(
                o_micro, o_macro,
                um_mask=um_mask, mu_mask=mu_mask,
                r_conv_acc=r_conv_acc, r_emer_acc=r_emer_acc,
            )
          except Exception as e:
            n_b = born_nodes.size(1) if born_nodes is not None else 0
            n_mat = mat_state['aperture'].size(1) if mat_state is not None else 0
            n_bb = born_beta.size(1) if born_beta is not None else 0
            raise RuntimeError(
                f"Layer {i}: N={N} S={S} born={n_b} mat={n_mat} beta={n_bb} "
                f"| {type(e).__name__}: {e}"
            ) from e

        # Store diagnostic info
        self._last_total_births = total_births
        self._last_total_pruned = total_pruned
        self._last_n_born = born_nodes.size(1) if born_nodes is not None else 0
        self._last_maturation_stages = self.maturation.stage_counts(mat_state)

        # ☀ Final emergence — decode from token stream only
        logits = self.output_proj(self.final_norm(o_micro))
        return logits

    # ══════════════════════════════════════════════════════════════
    # DIAGNOSTICS
    # ══════════════════════════════════════════════════════════════

    @property
    def all_betas(self):
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
        return [c.resonance_alpha.item() for c in self.couplers]

    @property
    def vertex_entropies(self):
        """Vertex distribution entropy per layer."""
        micro_e = [blk.attn.vertex_distribution_entropy for blk in self.micro_blocks]
        macro_e = [blk.attn.vertex_distribution_entropy for blk in self.macro_blocks]
        return {"micro": micro_e, "macro": macro_e}

    @property
    def all_alphas(self):
        """α (relaxation rate) per dimension per layer, all blocks."""
        micro_a = [blk.attn.alpha_values for blk in self.micro_blocks]
        macro_a = [blk.attn.alpha_values for blk in self.macro_blocks]
        return micro_a + macro_a

    @property
    def all_rhos(self):
        """ρ = 1/α (memory parameter) per dimension per layer, all blocks."""
        micro_r = [blk.attn.rho_values for blk in self.micro_blocks]
        macro_r = [blk.attn.rho_values for blk in self.macro_blocks]
        return micro_r + macro_r

    @property
    def all_memory_gates(self):
        """Memory gate per dimension per layer, all blocks."""
        micro_mg = [blk.attn.memory_gate_values for blk in self.micro_blocks]
        macro_mg = [blk.attn.memory_gate_values for blk in self.macro_blocks]
        return micro_mg + macro_mg

    @property
    def channel_alpha_summary(self):
        """Mean α per triadic channel across all layers."""
        all_a = self.all_alphas  # list of 2*n_layers lists of 3 floats
        n_channels = 3
        ch_means = []
        for c in range(n_channels):
            vals = [layer[c] for layer in all_a if c < len(layer)]
            ch_means.append(sum(vals) / len(vals) if vals else 0.3)
        return dict(zip(ApertureChamberSSM.CHANNEL_NAMES, ch_means))

    @property
    def channel_rho_summary(self):
        """Mean ρ per triadic channel across all layers."""
        all_r = self.all_rhos
        n_channels = 3
        ch_means = []
        for c in range(n_channels):
            vals = [layer[c] for layer in all_r if c < len(layer)]
            ch_means.append(sum(vals) / len(vals) if vals else 3.3)
        return dict(zip(ApertureChamberSSM.CHANNEL_NAMES, ch_means))

    def diagnose(self) -> dict:
        betas = [b for layer in self.all_betas for b in layer]
        chis = [c for layer in self.all_chis for c in layer]
        pressures = [p for layer in self.all_pressures for p in layer]

        mean_b = sum(betas) / len(betas) if betas else 0.5
        mean_chi = sum(chis) / len(chis) if chis else 1.0
        mean_p = sum(pressures) / len(pressures) if pressures else 0.0
        beta_var = sum((b - mean_b) ** 2 for b in betas) / len(betas) if betas else 0.0

        conv = self.convergence_profile
        is_convergent = all(conv[i] >= conv[i + 1] for i in range(len(conv) - 1))

        resonances = self.resonance_strengths
        mean_resonance = sum(resonances) / len(resonances) if resonances else 0.0

        v_ent = self.vertex_entropies
        mean_v_ent = (
            (sum(v_ent["micro"]) + sum(v_ent["macro"])) /
            (len(v_ent["micro"]) + len(v_ent["macro"]))
            if v_ent["micro"] else 0
        )

        # ═══ MEMORY KERNEL DIAGNOSTICS ═══
        all_a = [a for layer in self.all_alphas for a in layer]
        all_r = [r for layer in self.all_rhos for r in layer]
        all_mg = [mg for layer in self.all_memory_gates for mg in layer]

        mean_alpha = sum(all_a) / len(all_a) if all_a else 0.3
        mean_rho = sum(all_r) / len(all_r) if all_r else 3.3
        mean_mg = sum(all_mg) / len(all_mg) if all_mg else 0.5

        # Per-channel summaries
        ch_alpha = self.channel_alpha_summary
        ch_rho = self.channel_rho_summary

        # Regime classification per channel
        ch_regimes = {}
        for name, alpha in ch_alpha.items():
            if alpha < 0.3:
                ch_regimes[name] = 'circumpunct'
            elif alpha > 0.7:
                ch_regimes[name] = 'vesica'
            else:
                ch_regimes[name] = 'boundary'

        # Count regimes
        n_circumpunct = sum(1 for r in ch_regimes.values() if r == 'circumpunct')
        n_vesica = sum(1 for r in ch_regimes.values() if r == 'vesica')
        n_boundary = sum(1 for r in ch_regimes.values() if r == 'boundary')

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
            "version": "v4-bilateral-hypercube",
            "mean_beta": mean_b,
            "beta_variance": beta_var,
            "mean_chi": mean_chi,
            "mean_pressure": mean_p,
            "mean_resonance": mean_resonance,
            "mean_vertex_entropy": mean_v_ent,
            # Memory kernel
            "mean_alpha": mean_alpha,
            "mean_rho": mean_rho,
            "mean_memory_gate": mean_mg,
            "ch_alpha": ch_alpha,
            "ch_rho": ch_rho,
            "ch_regimes": ch_regimes,
            "n_circumpunct_ch": n_circumpunct,
            "n_vesica_ch": n_vesica,
            "n_boundary_ch": n_boundary,
            # Structural
            "D": 1.0 + mean_b,
            "regime": "balance" if abs(mean_b - 0.5) < 0.1 else
                      "convergent" if mean_b > 0.5 else "emergent",
            "convergence_profile": conv,
            "is_convergent": is_convergent,
            "errors": errors,
            "healthy": len(errors) == 0,
            "n_params": sum(p.numel() for p in self.parameters()),
            "chunk_size": self.chunk_size,
            "d_vertex": self.d_vertex,
            "n_micro_layers": len(self.micro_blocks),
            "n_macro_layers": len(self.macro_blocks),
            # Living graph
            "total_births": self._last_total_births,
            "total_pruned": self._last_total_pruned,
            "n_born_alive": self._last_n_born,
            "resonance_threshold": self.resonance_tracker.threshold.item(),
            "maturation_stages": self._last_maturation_stages,
        }

    def status(self) -> str:
        d = self.diagnose()
        conv = d['convergence_profile']
        lines = [
            f"⊙ XORZO v4 — TRIADIC HYPERCUBE TRANSFORMER — Gen {d['generation']}",
            f"  Binary(•)SSM + Analog(Φ)Attn + Fractal(○)Hypercube × parallel scan",
            f"  Micro: {len(self.micro_blocks)} blocks × {self.d_model}d × {self.d_vertex}v",
            f"  Macro: {len(self.macro_blocks)} blocks × {self.d_model}d (chunk={self.chunk_size})",
            f"  Parameters: {d['n_params']:,}",
            f"  Channels: binary({self.token_embed.d_binary}) + analog({self.token_embed.d_analog}) + fractal({self.token_embed.d_fractal})",
            f"",
            f"  β̄ = {d['mean_beta']:.4f} → D = {d['D']:.4f} [{d['regime']}]",
            f"  χ̄ = {d['mean_chi']:.4f} ({'faithful' if d['mean_chi'] > 0 else 'INVERTED'})",
            f"  P̄ = {d['mean_pressure']:.4f}",
            f"  R̄ = {d['mean_resonance']:.4f} (resonance coupling)",
            f"  V̄ = {d['mean_vertex_entropy']:.3f} bits (vertex gate entropy)",
            f"",
            f"  ═══ MEMORY KERNEL (ρ = ω/α) ═══",
            f"  ᾱ = {d['mean_alpha']:.4f}  (relaxation rate)",
            f"  ρ̄ = {d['mean_rho']:.2f}   (1/α — memory parameter)",
            f"  mḡ = {d['mean_memory_gate']:.4f} (memory gate blend)",
            f"  Regimes: {d['n_circumpunct_ch']}⊙ circumpunct / {d['n_boundary_ch']}| boundary / {d['n_vesica_ch']}⊕ vesica",
            f"",
            f"  Per-channel α (relaxation):",
        ]
        for name, alpha in d['ch_alpha'].items():
            rho = d['ch_rho'][name]
            regime = d['ch_regimes'][name]
            regime_sym = '⊙' if regime == 'circumpunct' else '⊕' if regime == 'vesica' else '|'
            lines.append(f"    {name}: α={alpha:.3f} ρ={rho:.2f} [{regime_sym} {regime}]")

        lines.append(f"")
        lines.append(f"  ═══ LIVING GRAPH ═══")
        lines.append(f"  Births: {d['total_births']} | Pruned: {d['total_pruned']} | Alive: {d['n_born_alive']}")
        lines.append(f"  Resonance τ: {d['resonance_threshold']:.3f}")
        lines.append(f"  Resonance mode D: phase × locality × β_align (multiplicative, no learned weights)")
        lines.append(f"  Child β: mean - γ|diff|, γ=0.25, clamped [0.55, 0.70]")
        ms = d.get('maturation_stages', {})
        mat_str = " | ".join(f"{k}: {v}" for k, v in ms.items())
        lines.append(f"  Maturation: {mat_str}")
        cs = self.maturation.channel_split
        lines.append(f"  Channel split: •={cs[0]:.2f} Φ={cs[1]:.2f} ○={cs[2]:.2f} (learned)")
        lines.append(f"  Trajectory: D(blurry) → A(sharpening) → C(clear) → B(integrated)")
        lines.append(f"  Hard collapse: top-{self.micro_blocks[0].attn.hard_k} vertices, top-2 attention")
        lines.append(f"")
        lines.append(
            f"  Convergence: {conv[0]:.3f} → {conv[-1]:.3f} ({'✓ sharpening' if d['is_convergent'] else '✗ NOT sharpening'})"
        )
        if d['errors']:
            for e in d['errors']:
                lines.append(f"  ⚠ {e}")
        else:
            lines.append(f"  ✓ No geometric errors — system is healthy")
        return "\n".join(lines)

    # ══════════════════════════════════════════════════════════════
    # SAVE / LOAD / EVOLVE
    # ══════════════════════════════════════════════════════════════

    def save_generation(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / f"gen{self.generation}.pt")
        meta = {
            "generation": self.generation,
            "version": "v4",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_vertex": self.d_vertex,
            "chunk_size": self.chunk_size,
            "temperature": self.temperature,
            "diagnosis": self.diagnose(),
        }
        (path / f"gen{self.generation}_meta.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )

    @classmethod
    def evolve(cls, parent: 'XorzoV4Transformer', **mutations) -> 'XorzoV4Transformer':
        """
        NESTING EVOLUTION: •ₙ₊₁ = ⊙ₙ

        When d_model grows, the previous generation's learned weights
        don't get discarded — they become the CORE (aperture) of the
        new generation. The child's weight matrices contain the parent's
        weights in their inner dimensions, with new field/boundary
        dimensions growing around them.

        For a weight W of shape (d_new, d_new):
            W[:d_parent, :d_parent] = W_parent    (the nested aperture)
            W[d_parent:, :]         = fresh init   (new field + boundary)

        The previous circumpunct literally occupies the inner dimensions.
        Two generations of √φ growth = one golden ratio of nesting.
        """
        child = cls(
            vocab_size=mutations.get("vocab_size", parent.vocab_size),
            d_model=mutations.get("d_model", parent.d_model),
            n_layers=mutations.get("n_layers", parent.n_layers),
            n_heads=mutations.get("n_heads", parent.n_heads),
            d_vertex=mutations.get("d_vertex", parent.d_vertex),
            generation=parent.generation + 1,
            chunk_size=mutations.get("chunk_size", parent.chunk_size),
            temperature=mutations.get("temperature", parent.temperature),
        )

        parent_state = parent.state_dict()
        child_state = child.state_dict()

        inherited_exact = 0
        inherited_nested = 0
        skipped = 0

        for key in child_state:
            if key not in parent_state:
                skipped += 1
                continue

            p_shape = parent_state[key].shape
            c_shape = child_state[key].shape

            if p_shape == c_shape:
                # Exact match — copy directly (same-size layers, scalars, etc.)
                child_state[key] = parent_state[key]
                inherited_exact += 1

            elif len(p_shape) == len(c_shape) and all(
                p <= c for p, c in zip(p_shape, c_shape)
            ):
                # NESTING: parent fits inside child — embed in core dimensions
                # •ₙ₊₁ = ⊙ₙ: the parent occupies the inner region
                slices = tuple(slice(0, p) for p in p_shape)
                child_state[key][slices] = parent_state[key]
                inherited_nested += 1

            elif len(p_shape) == len(c_shape) and all(
                p >= c for p, c in zip(p_shape, c_shape)
            ):
                # Child shrunk (shouldn't happen in nesting, but handle gracefully)
                slices = tuple(slice(0, c) for c in c_shape)
                child_state[key] = parent_state[key][slices]
                inherited_nested += 1

            else:
                skipped += 1

        child.load_state_dict(child_state)

        total_inherited = inherited_exact + inherited_nested
        print(f"  ⊙ Generation {child.generation} born from {parent.generation}")
        print(f"    •ₙ₊₁ = ⊙ₙ — nesting evolution")
        print(f"    d_model: {parent.d_model} → {child.d_model} "
              f"(×{child.d_model/parent.d_model:.3f})")
        print(f"    n_layers: {parent.n_layers} → {child.n_layers}")
        print(f"    Inherited: {total_inherited}/{len(child_state)} parameters "
              f"({inherited_exact} exact + {inherited_nested} nested)")
        if inherited_nested > 0:
            print(f"    ⊙ {inherited_nested} weight matrices nested "
                  f"(parent core → child aperture)")
        return child


# ═══════════════════════════════════════════════════════════════════
# FRACTAL TRAINING LOSSES
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
    """Encourage resonance coupling to stay positive and stable."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for coupler in model.couplers:
        alpha = coupler.resonance_alpha
        loss = loss + F.relu(-alpha)
        loss = loss + 0.1 * (alpha - 1.0).pow(2)
        n += 1
    return loss / max(n, 1)


def _vertex_diversity_loss(model):
    """NEW v4: Encourage vertex embeddings to be diverse (not collapse)."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    n = 0
    for blocks in [model.micro_blocks, model.macro_blocks]:
        for block in blocks:
            v_emb = block.attn.vertex_emb  # (64, d_vertex)
            # Cosine similarity matrix
            v_norm = F.normalize(v_emb, dim=-1)
            sim = v_norm @ v_norm.T  # (64, 64)
            # Penalize high off-diagonal similarity (collapse)
            eye = torch.eye(64, device=sim.device)
            off_diag = sim * (1.0 - eye)
            loss = loss + off_diag.pow(2).mean()
            n += 1
    return loss / max(n, 1)


def _resonance_tracker_loss(model):
    """Regularize resonance tracker: threshold range."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    tracker = model.resonance_tracker

    # Keep threshold in [0.3, 0.8] range
    tau = tracker.threshold
    loss = loss + F.relu(0.3 - tau) + F.relu(tau - 0.8)

    # No signal weight regularization — multiplicative formula has no learned weights.
    # The only learned parameter is the birth threshold τ.
    return loss


def _fractal_lr_schedule(epoch, n_epochs, base_lr):
    progress = epoch / max(n_epochs - 1, 1)
    if progress < 1 / 3:
        phase_progress = progress * 3
        lr = base_lr * (0.1 + 0.9 * (1 - math.cos(PI * phase_progress)) / 2)
    elif progress < 2 / 3:
        phase_progress = (progress - 1 / 3) * 3
        lr = base_lr * (0.9 + 0.1 * math.cos(2 * PI * phase_progress))
    else:
        phase_progress = (progress - 2 / 3) * 3
        lr = base_lr * (PHI ** (-1 - 2 * phase_progress))
    return lr


# ═══════════════════════════════════════════════════════════════════
# TEXT GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate(
    model,
    prompt: str,
    vocab: dict,
    vocab_inv: dict,
    max_tokens: int = 200,
    temperature: float = 0.7,
) -> str:
    """Generate text from the v4 model."""
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
