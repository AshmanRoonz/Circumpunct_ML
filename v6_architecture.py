"""
XORZO v6 — The Circumpunct Architecture
⊙ = • Φ ○ — Center, Field, Boundary

═══════════════════════════════════════════════════════════════════

EACH HEAD IS A COMPLETE CIRCUMPUNCT:

    Each attention head is not just parallel attention.
    Each head is a complete ⊙ = • Φ ○:

    • = CENTER — self-attention on this head's slice.
        What to focus on. The act of selection.

    ○ = BOUNDARY — hypercube navigation from this perspective.
        What's available. The 64-state space of possibilities.

    Φ = FIELD — emergence FFN shaped by BOTH center and boundary.
        Not neutral. Colored by what focused AND what was available.
        The field carries the imprint of both • and ○.

    The heads are discs on a cylinder. The residual stream is
    the soul — the axis passing through every disc unchanged.

THE CYCLE (§9.8.8):

    a' = ☀ ∘ (×i) ∘ ⊛ [a]

    ⊛: many → one  (converge, extract center)
    i:  perpendicular turn  (complex rotation, β learned)
    ☀: one → many  (broadcast, FiLM modulation)

    i is the only learned angle. Like a bee's waggle dance:
    the rotation between receiving frame and transmitting frame.
    Not an unconstrained matrix — a constrained geometry.

THE HIERARCHY:

    Level 0: 6 nodes  (micro — token groups)
    Level 1: 2 nodes  (meso — region summaries)
    Level 2: 1 node   (macro — global center)
    3 depth passes for information to flow up and back down.

═══════════════════════════════════════════════════════════════════
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict

PHI = (1 + math.sqrt(5)) / 2
PI = math.pi


# ═══════════════════════════════════════════════════════════════════
# COMPLEX ROTATION — The i operation
#
# From §9.8.1: i is the only primitive. i² = −1.
# From §9.8.8: a' = ☀ ∘ (×i) ∘ ⊛ [a]
#
# The center vector is split into real/imaginary halves.
# Rotation by angle πβ transforms the received (objective)
# orientation into the transmitting (subjective) orientation.
#
# β is the one learned parameter. Everything else is
# constrained geometry.
#
# β = 0.5 → angle = π/2 = 90° → pure i (Å(½) = i)
# This is the balance point where |⊛| = |☀|.
# ═══════════════════════════════════════════════════════════════════

class ComplexRotation(nn.Module):
    """
    i — the perpendicular turn.

    Splits the vector into real and imaginary halves,
    rotates by angle = π · sigmoid(β).

    From the Theory: "A learned rotation is not the same as
    a rotation with a learned angle. The first is a free parameter;
    the second is a geometric constraint that encodes what i actually is."
    """

    def __init__(self, d: int, beta_init: float = 0.5):
        super().__init__()
        self.d = d
        # β initializes so sigmoid(0.5) ≈ 0.62 → angle ≈ 112°
        # Close to the balance point (90°) but not exactly there
        self.beta = nn.Parameter(torch.tensor(beta_init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (..., d) — center vector to rotate
        Returns: rotated vector, same shape
        """
        angle = PI * torch.sigmoid(self.beta)  # ∈ (0, π)
        half = self.d // 2

        x_real = x[..., :half]
        x_imag = x[..., half:2 * half]

        cos_a = torch.cos(angle)
        sin_a = torch.sin(angle)

        out_real = x_real * cos_a - x_imag * sin_a
        out_imag = x_real * sin_a + x_imag * cos_a

        if self.d % 2 == 1:
            return torch.cat([out_real, out_imag, x[..., -1:]], dim=-1)
        return torch.cat([out_real, out_imag], dim=-1)

    @property
    def current_angle_degrees(self):
        """Current rotation angle in degrees."""
        return (PI * torch.sigmoid(self.beta)).item() * 180.0 / PI


# ═══════════════════════════════════════════════════════════════════
# THE 6D HYPERCUBE — precomputed geometric structure
# 64 vertices = every possible bilateral circumpunct configuration
# (from v4)
# ═══════════════════════════════════════════════════════════════════

class Hypercube6D:
    """
    Precomputes all geometric structure of the 6D hypercube.
    6 dimensions = two circumpuncts facing each other:
        b1, b2 → apertures (how open am I? how open are you?)
        c1, c2 → fields    (am I faithful? are you faithful?)
        r1, r2 → boundaries (am I expressed? are you expressed?)
    """

    def __init__(self):
        self.vertices = np.array(
            [[(v >> i) & 1 for i in range(6)] for v in range(64)],
            dtype=np.float32
        )
        self.adjacency = np.zeros((64, 64), dtype=np.float32)
        for i in range(64):
            for d in range(6):
                self.adjacency[i, i ^ (1 << d)] = 1.0
        self.laplacian = np.diag(self.adjacency.sum(1)) - self.adjacency
        _, ev = np.linalg.eigh(self.laplacian)
        self.spectral_emb = ev[:, 1:7].astype(np.float32)

    def to_buffers(self):
        return {
            "hc_vertices": torch.from_numpy(self.vertices),
            "hc_adjacency": torch.from_numpy(self.adjacency),
            "hc_spectral_emb": torch.from_numpy(self.spectral_emb),
        }


# ═══════════════════════════════════════════════════════════════════
# PHASE RESONANCE — The binding criterion
# (from v5 with autocast guard)
# ═══════════════════════════════════════════════════════════════════

def phase_resonance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Phase-coherence resonance between two sets of vectors.
    Returns resonance in [-1, 1].
    Forced float32 with autocast disabled — atan2/cos unstable in bf16.
    """
    with torch.amp.autocast('cuda', enabled=False):
        orig_dtype = a.dtype
        a = a.float()
        b = b.float()
        D = a.size(-1)
        if D % 2 != 0:
            a = a[..., :-1]
            b = b[..., :-1]
        pairs = a.size(-1) // 2

        def to_phase(x):
            x2 = x.view(*x.shape[:-1], pairs, 2)
            return torch.atan2(x2[..., 1] + eps, x2[..., 0] + eps)

        theta_a = to_phase(a)
        theta_b = to_phase(b)
        delta = theta_a.unsqueeze(-2) - theta_b.unsqueeze(-3)
        result = torch.cos(delta).mean(dim=-1)
        return result.to(orig_dtype)


# ═══════════════════════════════════════════════════════════════════
# PARALLEL SCAN for LINEAR RECURRENCE
# (from v4 — O(S log S) parallel work)
# ═══════════════════════════════════════════════════════════════════

def parallel_scan_linear_recurrence(
    drive: torch.Tensor,
    decay: torch.Tensor,
) -> torch.Tensor:
    """
    Parallel prefix scan for: ψ_s = decay·ψ_{s-1} + (1-decay)·drive_s
    Returns all ψ states: (B, S, D)
    """
    B, S, D = drive.shape
    orig_dtype = drive.dtype
    drive = drive.float()
    decay = decay.float()

    w = 1.0 - decay
    log_a = torch.zeros(B, S, D, device=drive.device, dtype=torch.float32)
    if decay.dim() == 0:
        log_a.fill_(torch.log(decay.clamp(min=1e-8)).item())
    else:
        log_a[:] = torch.log(decay.clamp(min=1e-8))

    offsets = w * drive

    n_levels = int(math.ceil(math.log2(max(S, 2))))
    for k in range(n_levels):
        stride = 1 << k
        if stride >= S:
            break
        idx = torch.arange(stride, S, device=drive.device)
        prev = idx - stride
        a_curr = torch.exp(log_a[:, idx])
        new_offsets = a_curr * offsets[:, prev] + offsets[:, idx]
        new_log_a = log_a[:, idx] + log_a[:, prev]
        offsets = offsets.clone()
        log_a = log_a.clone()
        offsets[:, idx] = new_offsets
        log_a[:, idx] = new_log_a

    return offsets.to(orig_dtype)


# ═══════════════════════════════════════════════════════════════════
# APERTURE CHAMBER SSM — ⊛→i→☀ with parallel scan
# (from v4)
# ═══════════════════════════════════════════════════════════════════

class ApertureChamberSSM(nn.Module):
    """
    Three-stage aperture with parallel-scan memory kernel.
    α: relaxation rate (high=vesica, low=circumpunct)
    β: complex rotation angle
    ⊛/☀: input/output valves with pressure tracking
    """

    def __init__(self, d_channel: int, depth: float = 0.5):
        super().__init__()
        self.d_channel = d_channel
        beta_init = 0.5 + 0.2 * (1.0 - 2.0 * depth)
        self.beta = nn.Parameter(torch.tensor(beta_init))
        self.input_valve = nn.Parameter(torch.tensor(0.5))
        self.output_valve = nn.Parameter(torch.tensor(0.5))
        self.register_buffer('pressure', torch.tensor(0.0))
        self.alpha_raw = nn.Parameter(torch.tensor(-0.85))
        self.memory_gate = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        beta = torch.sigmoid(self.beta)
        iv = torch.sigmoid(self.input_valve)
        ov = torch.sigmoid(self.output_valve)
        alpha = torch.sigmoid(self.alpha_raw)
        mg = torch.sigmoid(self.memory_gate)

        x_in = x * iv
        decay = torch.exp(-alpha)
        drive = torch.tanh(x_in)
        psi_seq = parallel_scan_linear_recurrence(drive, decay)
        x_mem = mg * psi_seq + (1.0 - mg) * x_in

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

        x_out = x_transformed * ov
        with torch.no_grad():
            dp = iv.detach() - ov.detach()
            self.pressure = 0.95 * self.pressure + 0.05 * dp
        return x_out

    @property
    def current_beta(self):
        return torch.sigmoid(self.beta).item()

    @property
    def current_alpha(self):
        return torch.sigmoid(self.alpha_raw).item()


# ═══════════════════════════════════════════════════════════════════
# BINARY APERTURE — The ÷t operation (from v5)
# Straight-through estimator: binary forward, smooth backward.
# ═══════════════════════════════════════════════════════════════════

class BinaryAperture(nn.Module):
    """
    The aperture generates discrete moments.
    Binary gate with leak: open (1.0) or attenuated (leak).
    STE for training. Closed gates still whisper — they don't kill signal.

    leak=0.1 means closed tokens pass 10% of their signal.
    This prevents the degenerate all-open optimum where the model
    floods every gate to avoid zeroing gradients.
    """

    def __init__(self, d_input: int, d_center: int = None,
                 temperature: float = 1.0, leak: float = 0.1):
        super().__init__()
        self.gate_proj = nn.Linear(d_input, 1, bias=True)
        self.temperature = temperature
        self.leak = leak
        # bias=0 puts sigmoid at exactly 0.5 (the STE threshold)
        # Small random weights create natural variation — some gates open, some closed
        nn.init.constant_(self.gate_proj.bias, 0.0)
        nn.init.normal_(self.gate_proj.weight, mean=0.0, std=0.02)

        if d_center is not None:
            self.center_bias = nn.Linear(d_center, 1, bias=False)
        else:
            self.center_bias = None

    def forward(self, x: torch.Tensor, center: Optional[torch.Tensor] = None):
        logits = self.gate_proj(x)
        if center is not None and self.center_bias is not None:
            center_logit = self.center_bias(center)
            logits = logits + center_logit.unsqueeze(1)
        prob = torch.sigmoid(logits / self.temperature)
        if self.training:
            hard = (prob > 0.5).float()
            gate = hard - prob.detach() + prob
        else:
            gate = (prob > 0.5).float()

        # Leak floor: closed gates whisper (0.1), open gates are full (1.0)
        # This prevents the degenerate all-open optimum
        gate = gate * (1.0 - self.leak) + self.leak

        return gate


# ═══════════════════════════════════════════════════════════════════
# CIRCUMPUNCT HEAD — Each head is a complete ⊙ = • Φ ○
#
# Not just parallel attention — parallel CIRCUMPUNCTS.
# Each head has its own:
#   • CENTER (attention focus on its slice)
#   ○ BOUNDARY (hypercube navigation from its perspective)
#   Φ FIELD (emergence FFN, colored by BOTH center and boundary)
#
# The field is not neutral. It is shaped by what focused (•)
# and what was available (○). Center and boundary color the field.
#
# Each head is a disc on the cylinder. The residual stream
# is the soul — the axis that passes through every disc unchanged.
# ═══════════════════════════════════════════════════════════════════

class CircumpunctHead(nn.Module):
    """
    ⊙ — A single attention head that is a complete circumpunct.

    • (center): self-attention on this head's slice — what to focus on
    ○ (boundary): hypercube navigation — what's available in the 64-state space
    Φ (field): emergence FFN shaped by BOTH center and boundary

    The field carries the color of both what attended and what was attended to.
    """

    def __init__(
        self,
        d_head: int,
        d_vertex: int = 32,
        dropout: float = 0.1,
        head_index: int = 0,
    ):
        super().__init__()
        self.d_head = d_head
        self.d_vertex = d_vertex
        self.head_index = head_index
        self.scale = math.sqrt(d_head)

        # ═══ • CENTER: attention focus ═══
        self.center_norm = nn.LayerNorm(d_head)
        self.W_q = nn.Linear(d_head, d_head, bias=False)
        self.W_k = nn.Linear(d_head, d_head, bias=False)
        self.W_v = nn.Linear(d_head, d_head, bias=False)

        # ═══ ○ BOUNDARY: hypercube navigation ═══
        # Each head navigates the SAME 64-state space but through its own gate
        self.boundary_norm = nn.LayerNorm(d_head)
        self.fractal_W_gate = nn.Linear(d_head, 6, bias=True)
        self.fractal_gate_to_vertex = nn.Linear(6, d_vertex, bias=False)
        self.fractal_vertex_to_head = nn.Linear(d_vertex, d_head, bias=False)

        # ═══ Φ FIELD: emergence shaped by center AND boundary ═══
        # The field is not neutral — it carries the imprint of both
        self.phi_combine = nn.Linear(d_head + d_head, d_head)
        self.phi_norm = nn.LayerNorm(d_head)
        self.phi_up = nn.Linear(d_head, d_head * 2)
        self.phi_down = nn.Linear(d_head * 2, d_head)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x_head: torch.Tensor,           # (B, T, d_head) — this head's slice
        gate: torch.Tensor,              # (B, T, 1) — aperture gate
        mask: torch.Tensor,              # (1, 1, T, T) — causal mask
        node_bias_6d: torch.Tensor,      # (6,) — node position in hypercube
        vertex_emb: torch.Tensor,        # (64, d_vertex) — shared vertex embeddings
        adj_bias: torch.Tensor,          # (64, 64) — adjacency
        spectral_emb: torch.Tensor,      # (64, 6) — spectral embeddings
        fractal_W_spec_weight: torch.Tensor,  # (d_vertex, 6) — shared spectral proj
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Returns: (field_output, pi, alignment_raw_or_None)
        """
        B, T, _ = x_head.shape

        # ─── • CENTER: self-attention on this slice ───
        h = self.center_norm(x_head)
        Q = self.W_q(h) * gate                               # gate shapes what focuses
        K = self.W_k(h) * gate                               # gate shapes what's focusable
        V = self.W_v(h)

        with torch.amp.autocast('cuda', enabled=False):
            scores = torch.matmul(Q.float(), K.float().transpose(-1, -2)) / self.scale
            scores = scores + mask
            alignment = scores.clone() if self.head_index == 0 else None
            attn = F.softmax(scores, dim=-1).to(x_head.dtype)

        attn = self.dropout(attn)
        center_out = torch.matmul(attn, V)                   # (B, T, d_head)
        center_signal = x_head + self.dropout(center_out)     # residual

        # ─── ○ BOUNDARY: hypercube navigation from this head's perspective ───
        h_b = self.boundary_norm(x_head)
        gates_6d = torch.sigmoid(self.fractal_W_gate(h_b) + node_bias_6d)
        gate_proj = self.fractal_gate_to_vertex(gates_6d)
        v_scores = gate_proj @ vertex_emb.T

        # Spectral geometry bias
        spec_proj = spectral_emb @ fractal_W_spec_weight.T
        spec_bias = gate_proj @ spec_proj.T
        v_scores = v_scores + 0.1 * spec_bias

        # Adjacency smoothing
        v_scores = v_scores + 0.1 * (v_scores @ adj_bias)

        # Sparse vertex selection — top-8 of 64
        with torch.amp.autocast('cuda', enabled=False):
            mean_abs = max(v_scores.abs().mean().item(), 1e-6)
            v_scores_f32 = (v_scores / mean_abs).float() * 3.0
            soft_pi = F.softmax(v_scores_f32, dim=-1)
            topk_vals, topk_idx = soft_pi.topk(8, dim=-1)
            hard_pi = torch.zeros_like(soft_pi)
            hard_pi.scatter_(-1, topk_idx, topk_vals)
            hard_pi = hard_pi / (hard_pi.sum(dim=-1, keepdim=True) + 1e-8)
            pi = (hard_pi - soft_pi.detach() + soft_pi).to(x_head.dtype)

        vertex_ctx = pi @ vertex_emb
        boundary_out = self.fractal_vertex_to_head(vertex_ctx)
        boundary_signal = x_head + boundary_out               # residual

        # ─── Φ FIELD: emergence colored by BOTH center and boundary ───
        # The field is NOT neutral — it carries the imprint of what focused
        # and what was available. Center shapes it. Boundary shapes it.
        phi_input = self.phi_combine(torch.cat([center_signal, boundary_signal], dim=-1))
        phi_h = self.phi_norm(phi_input)
        phi_out = self.phi_down(self.dropout(F.gelu(self.phi_up(phi_h))))
        field_output = phi_input + phi_out                     # residual

        return field_output, pi, alignment


# ═══════════════════════════════════════════════════════════════════
# TRIADIC COMPUTE V6 — ⊙ = • Φ ○ × N heads
#
# Each head is a complete circumpunct. Not just parallel attention.
# Parallel WORLDS, each with its own center, boundary, and field.
#
# The outputs merge — that merging IS the higher-level convergence.
# Many perspectives → one representation.
#
# The residual stream (soul/axis) passes through all heads unchanged.
# Each head reads from it, contributes to it, but none owns it.
# ═══════════════════════════════════════════════════════════════════

class TriadicComputeV6(nn.Module):
    """
    N parallel circumpunct heads, each a complete ⊙ = • Φ ○.
    Shared: 64-vertex hypercube embeddings, adjacency, spectral geometry.
    Per-head: attention (•), hypercube gate (○), emergence FFN (Φ).
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_vertex: int = 32,
        dropout: float = 0.1,
        layer_depth: float = 0.5,
        layer_index: int = 0,
        convergence_rate: float = 0.1,
        temperature: float = 1.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_vertex = d_vertex
        self.n_heads = n_heads
        self.layer_index = layer_index

        # Each head gets an equal slice of d_model
        assert d_model % n_heads == 0, f"d_model={d_model} must be divisible by n_heads={n_heads}"
        self.d_head = d_model // n_heads

        # ═══ N CIRCUMPUNCT HEADS ═══
        self.heads = nn.ModuleList([
            CircumpunctHead(
                d_head=self.d_head,
                d_vertex=d_vertex,
                dropout=dropout,
                head_index=i,
            )
            for i in range(n_heads)
        ])

        # ═══ SHARED: 64-state hypercube geometry ═══
        self.fractal_vertex_emb = nn.Parameter(torch.randn(64, d_vertex) * 0.02)
        self.fractal_W_spec = nn.Linear(6, d_vertex, bias=False)
        self.fractal_node_bias = nn.Embedding(9, 6)
        nn.init.normal_(self.fractal_node_bias.weight, mean=0.0, std=0.3)

        # Hypercube buffers (registered by Brain)
        self.register_buffer("_adj_bias", torch.zeros(64, 64))
        self.register_buffer("_spectral_emb", torch.zeros(64, 6))

        # ═══ OUTPUT: merge all head fields back to d_model ═══
        self.W_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout_layer = nn.Dropout(dropout)

    def register_cube_buffers(self, cube_buffers):
        self._adj_bias.copy_(cube_buffers["hc_adjacency"] * 0.5)
        self._spectral_emb.copy_(cube_buffers["hc_spectral_emb"])

    def forward(
        self, x: torch.Tensor, gate: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        node_index: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x: (B, T, d_model)
        gate: (B, T, 1) — aperture gate from BinaryAperture (with leak)
        mask: (1, T, T) — causal mask (additive, -inf for masked)
        node_index: which node this is (0-8), biases hypercube navigation
        returns: (result, pi_combined, alignment_raw)
        """
        B, T, D = x.shape

        # Node position in the hypercube — shared across heads
        node_idx_tensor = torch.tensor(min(node_index, 8), device=x.device)
        node_bias_6d = self.fractal_node_bias(node_idx_tensor)

        # Keep mask as (1, T, T) — each head does single-head attention
        if mask is not None:
            m = mask.float()
            while m.dim() < 3:
                m = m.unsqueeze(0)
            while m.dim() > 3:
                m = m.squeeze(0)
        else:
            m = torch.zeros(1, T, T, device=x.device)

        # ═══ RUN EACH HEAD AS A COMPLETE CIRCUMPUNCT ═══
        head_outputs = []
        all_pi = []
        alignment_raw = None

        for i, head in enumerate(self.heads):
            # Each head gets its slice of the representation
            x_head = x[..., i * self.d_head : (i + 1) * self.d_head]

            field_out, pi, align = head(
                x_head=x_head,
                gate=gate,
                mask=m,
                node_bias_6d=node_bias_6d,
                vertex_emb=self.fractal_vertex_emb,
                adj_bias=self._adj_bias,
                spectral_emb=self._spectral_emb,
                fractal_W_spec_weight=self.fractal_W_spec.weight,
            )
            head_outputs.append(field_out)
            all_pi.append(pi)
            if align is not None:
                alignment_raw = align

        # ═══ MERGE: many perspectives → one representation ═══
        merged = torch.cat(head_outputs, dim=-1)               # (B, T, d_model)
        result = self.W_out(merged)
        result = self.norm(result)
        result = result * gate                                  # aperture attenuates

        # Average pi across heads for diagnostics
        pi_combined = torch.stack(all_pi, dim=0).mean(dim=0)

        if alignment_raw is None:
            alignment_raw = torch.zeros(B, T, T, device=x.device)

        return result, pi_combined, alignment_raw


# ═══════════════════════════════════════════════════════════════════
# GLOBAL CENTER — The Self vector (from v4)
# FiLM broadcast: γ·micro + δ
# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# ADAPTIVE APERTURE — The system's self-calibrating temperature
#
# The aperture (•) controls its own opening:
#   Wide open when uncertain → explore more possibilities
#   Narrow when confident → commit to the prediction
#
# Derived from the architecture's own internal state:
#   - Field coherence (resonance): how synchronized are the nodes?
#   - Gate openness: how many tokens are participating?
#   - Center stability: is the global center still shifting?
#
# This replaces the fixed temperature hyperparameter with a
# learned, contextual signal: ⊙ knows how certain it is.
# ═══════════════════════════════════════════════════════════════════

class AdaptiveAperture(nn.Module):
    """
    • — The system's own temperature, derived from internal field state.

    Outputs a per-position temperature T ∈ [T_min, T_max] that modulates
    the logit distribution. High coherence → low T (sharp). Low coherence
    → high T (exploratory).

    Trained end-to-end with the rest of the model.
    """

    def __init__(self, d_model: int, T_min: float = 0.5, T_max: float = 1.5):
        super().__init__()
        self.d_model = d_model
        self.T_min = T_min
        self.T_max = T_max

        # Learns to read the system's confidence from its own state
        # Inputs: global_center (D) + resonance_summary (1) + gate_summary (1) + center_delta (1)
        self.confidence_net = nn.Sequential(
            nn.Linear(d_model + 3, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )

        # Initialize toward middle temperature (sigmoid(0) = 0.5)
        nn.init.zeros_(self.confidence_net[-1].weight)
        nn.init.zeros_(self.confidence_net[-1].bias)

    def forward(
        self,
        global_center: torch.Tensor,       # (B, D) — the system's self
        resonance_mean: torch.Tensor,       # (B, 1) — mean field coherence
        gate_openness: torch.Tensor,        # (B, 1) — fraction of open gates
        center_delta: torch.Tensor,         # (B, 1) — how much center shifted
    ) -> torch.Tensor:
        """Returns: temperature (B, 1) in [T_min, T_max]"""
        features = torch.cat([
            global_center,
            resonance_mean,
            gate_openness,
            center_delta,
        ], dim=-1)  # (B, D+3)

        # Confidence ∈ (0, 1): high = confident = low temperature
        confidence = torch.sigmoid(self.confidence_net(features))

        # Map: high confidence → T_min, low confidence → T_max
        temperature = self.T_max - confidence * (self.T_max - self.T_min)

        return temperature  # (B, 1)


class GlobalCenter(nn.Module):
    """
    Extract a single center vector from the global level,
    broadcast back to all micro tokens via FiLM modulation.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model
        self.center_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.center_key = nn.Linear(d_model, d_model, bias=False)
        self.film_gamma = nn.Linear(d_model, d_model)
        self.film_delta = nn.Linear(d_model, d_model)
        # Near-identity init
        nn.init.zeros_(self.film_delta.weight)
        nn.init.zeros_(self.film_delta.bias)

    def forward(self, global_states: torch.Tensor) -> torch.Tensor:
        """
        global_states: (B, Tg, D) — states from global level
        Returns: center (B, D) — the unified self vector
        """
        B, Tg, D = global_states.shape
        query = self.center_query.expand(B, -1, -1)
        keys = self.center_key(global_states)
        with torch.amp.autocast('cuda', enabled=False):
            scores = torch.matmul(query.float(), keys.float().transpose(-1, -2)) / math.sqrt(D)
            attn = F.softmax(scores, dim=-1)
        center = torch.matmul(attn.to(global_states.dtype), global_states).squeeze(1)
        return center

    def broadcast(self, center: torch.Tensor, micro: torch.Tensor) -> torch.Tensor:
        """
        center: (B, D) — global center
        micro: (B, T, D) — micro token states
        Returns: modulated micro tokens
        """
        gamma = self.film_gamma(center).unsqueeze(1)
        delta = self.film_delta(center).unsqueeze(1)
        return gamma * micro + delta


# ═══════════════════════════════════════════════════════════════════
# CIRCUMPUNCT NODE V6 — ⊙ = Φ(•, ○)
#
# Each node IS a complete circumpunct implementing ⊛ → i → ☀:
#
#   ☀ RECEIVE:  center_from_above broadcasts into boundary (☀ from above)
#   • APERTURE: BinaryAperture gates which tokens participate
#   Φ FIELD:    TriadicComputeV6 — Φ(•, ○) operates on boundary
#   ○ BOUNDARY: FFN consolidation
#   ⊛ CONVERGE: pool boundary → center (many → one)
#   i ROTATE:   ComplexRotation on center (πβ radians)
#   → center passes up to become ☀ of the level above
#
# The cycle spans depth passes:
#   Pass N: ... → ⊛ → i → center passes up → GlobalCenter → ☀ → ...
#   Pass N+1: ☀ arrives as center_from_above → process → ⊛ → i → ...
# ═══════════════════════════════════════════════════════════════════

class CircumpunctNodeV6(nn.Module):
    """
    ⊙ — A single circumpunct.
    Implements ⊛ → i → ☀ with triadic internals.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_center: int = None,
        d_vertex: int = 32,
        dropout: float = 0.1,
        layer_index: int = 0,
        n_layers: int = 3,
        node_index: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_center = d_center or d_model
        self.node_index = node_index

        # • APERTURE
        self.aperture = BinaryAperture(d_model, d_center=self.d_center)

        # Φ LOCAL FIELD — Φ(•, ○) triadic compute
        self.norm_field = nn.LayerNorm(d_model)
        self.triadic = TriadicComputeV6(
            d_model=d_model,
            n_heads=n_heads,
            d_vertex=d_vertex,
            dropout=dropout,
            layer_depth=layer_index / max(n_layers - 1, 1),
            layer_index=layer_index,
        )

        # ○ BOUNDARY — consolidation
        d_ff = int(d_model * PHI)
        self.norm_boundary = nn.LayerNorm(d_model)
        self.ffn_up = nn.Linear(d_model, d_ff)
        self.ffn_down = nn.Linear(d_ff, d_model)
        self.boundary_dropout = nn.Dropout(dropout)

        # Commit gate
        self.commit_proj = nn.Linear(d_model, 1)

        # ⊛ CONVERGE — center extraction (many → one)
        self.center_proj = nn.Linear(d_model, self.d_center)
        self.center_norm = nn.LayerNorm(self.d_center)

        # i ROTATE — complex rotation on the center
        # From the ToE (§5.4, §16): the quarter-turn splits into cone angles
        #   90° = i = quarter-turn = balance point (β = 0.0)
        #   68° = main cone angle (β ≈ -0.499)
        #   22° = golden spiral pitch = (180° - 360°/φ²)/2 (β ≈ -1.972)
        #   68° + 22° = 90° — the cone partitions i
        #
        # L0 (local): 68° — axial component, the main cone
        # L1 (regional): 22° — pitch component, the golden spiral
        # L2 (global): 90° — pure i, the full quarter-turn
        _CONE_ANGLES = {
            0: -0.499,   # 68° — main cone angle
            1: -1.972,   # 22° — golden spiral pitch
            2:  0.0,     # 90° — pure i, balance point
        }
        beta_init = _CONE_ANGLES.get(layer_index, 0.0)
        self.i_rotation = ComplexRotation(self.d_center, beta_init=beta_init)

        # ☀ EMERGE — center broadcast (receive from above)
        self.broadcast_proj = nn.Linear(self.d_center, d_model)

    def register_cube_buffers(self, cube_buffers):
        self.triadic.register_cube_buffers(cube_buffers)

    def _make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Safe causal mask — v5 fix: no triu(ones)*-inf.
        Returns (1, T, T) — 3D for single-head analog attention."""
        mask = torch.triu(
            torch.full((T, T), float('-inf'), device=device),
            diagonal=1
        )
        return mask.unsqueeze(0)

    def forward(
        self, x: torch.Tensor,
        center_from_above: Optional[torch.Tensor] = None,
        field_signal: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        x: (B, T, D) — boundary tokens for this node
        center_from_above: (B, D_center) — ☀ broadcast from parent level
        field_signal: (B, T, D) — signal from SharedField
        Returns dict: {boundary, center, gate, pi, alignment}

        Implements: ☀(receive) → •(gate) → Φ(field) → ○(consolidate) → ⊛(converge) → i(rotate)
        """
        B, T, D = x.shape

        # ═══ ☀ RECEIVE (from previous cycle's emergence) ═══
        if field_signal is not None:
            x = x + field_signal
        if center_from_above is not None:
            broadcast = self.broadcast_proj(center_from_above)
            x = x + broadcast.unsqueeze(1)

        # ═══ • APERTURE (gate selection) ═══
        gate = self.aperture(x, center_from_above)  # (B, T, 1)

        # ═══ Φ LOCAL FIELD — Φ(•, ○) triadic compute ═══
        h = self.norm_field(x)
        mask = self._make_causal_mask(T, x.device)
        field_out, pi, alignment = self.triadic(h, gate, mask=mask, node_index=self.node_index)
        x = x + field_out

        # ═══ ○ BOUNDARY — consolidation ═══
        h2 = self.norm_boundary(x)
        ff = self.ffn_down(F.gelu(self.ffn_up(h2)))
        ff = self.boundary_dropout(ff)
        center_pool = x.mean(dim=1)
        commit = torch.sigmoid(self.commit_proj(center_pool))
        x = x + commit.unsqueeze(1) * gate * ff

        # ═══ ⊛ CONVERGE (many → one) ═══
        center = self.center_norm(self.center_proj(x.mean(dim=1)))

        # ═══ i ROTATE (perpendicular turn at the aperture) ═══
        # §9.8.8: z' = i·z — rotate the converged center by πβ
        center = self.i_rotation(center)

        return {
            'boundary': x,
            'center': center,     # rotated — ready for ☀ above
            'gate': gate,
            'pi': pi,
            'alignment': alignment,
        }


# ═══════════════════════════════════════════════════════════════════
# SHARED FIELD V6 — Phase resonance coupling between nodes
# Merges v5's SharedField with v4's sparse tracking concept
# ═══════════════════════════════════════════════════════════════════

class SharedFieldV6(nn.Module):
    """
    Φ — The medium between ⊙ nodes at a single level.
    Phase resonance determines who hears whom.
    Accumulates across depth passes.
    """

    def __init__(self, d_model: int, resonance_lambda: float = 0.7):
        super().__init__()
        self.d_model = d_model
        self.resonance_lambda = resonance_lambda

        # How resonance translates to signal flow
        self.conductance = nn.Parameter(torch.tensor(1.0))

        # Signal enters and exits Φ differently
        self.field_in = nn.Linear(d_model, d_model, bias=False)
        self.field_out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        node_states: torch.Tensor,   # (B, N, D) — summary per node
        node_gates: torch.Tensor,     # (B, N, 1) — aperture openness per node
        r_acc: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns: (field_signal, r_acc_new)
            field_signal: (B, N, D) — what each node receives from the field
            r_acc_new: (B, N, N) — updated accumulated resonance
        """
        B, N, D = node_states.shape

        # Phase resonance on UNGATED states (to avoid fake coherence)
        with torch.amp.autocast('cuda', enabled=False):
            r_instant = phase_resonance(
                self.field_in(node_states),
                self.field_in(node_states),
            )

        # Accumulate
        lam = self.resonance_lambda
        if r_acc is None:
            r_acc_new = r_instant
        else:
            if r_acc.size(1) != N:
                # Handle node count changes
                r_acc_new = r_instant
            else:
                r_acc_new = lam * r_acc + (1 - lam) * r_instant
        r_acc_new = r_acc_new.clamp(-2.0, 2.0)

        # Resonance → coupling strength
        cond = self.conductance.clamp(-5.0, 5.0)
        coupling = torch.sigmoid(cond * r_acc_new)

        # Zero self-coupling
        eye = torch.eye(N, device=node_states.device).unsqueeze(0)
        coupling = coupling * (1.0 - eye)

        # Gate: only open nodes transmit and receive
        coupling = coupling * node_gates.transpose(-1, -2)  # sender gate
        coupling = coupling * node_gates  # receiver gate (broadcast dim)

        # Normalize
        coupling = coupling / (coupling.sum(dim=-1, keepdim=True) + 1e-8)

        # Transmitted signal through the field
        transmitted = node_states * node_gates  # only open nodes transmit
        values = self.field_out(transmitted)
        field_signal = torch.matmul(coupling, values)

        return field_signal, r_acc_new


# ═══════════════════════════════════════════════════════════════════
# CIRCUMPUNCT LEVEL V6 — A cluster of nodes sharing a field
# ═══════════════════════════════════════════════════════════════════

class CircumpunctLevelV6(nn.Module):
    """
    A level in the hierarchy: N nodes + shared field.
    Each node is a complete circumpunct with triadic internals.
    """

    def __init__(
        self,
        n_nodes: int,
        d_model: int,
        n_heads: int = 4,
        d_vertex: int = 32,
        dropout: float = 0.1,
        layer_index: int = 0,
        n_layers: int = 3,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model

        # node_offset: L0 nodes are 0-5, L1 nodes are 6-7, global is 8
        # This gives each node a unique identity in the 64-state space
        self.node_offset = 0 if layer_index == 0 else (6 if layer_index == 1 else 8)

        self.nodes = nn.ModuleList([
            CircumpunctNodeV6(
                d_model=d_model,
                n_heads=n_heads,
                d_center=d_model,
                d_vertex=d_vertex,
                dropout=dropout,
                layer_index=layer_index,
                n_layers=n_layers,
                node_index=self.node_offset + i,
            )
            for i in range(n_nodes)
        ])

        self.field = SharedFieldV6(d_model)

    def register_cube_buffers(self, cube_buffers):
        for node in self.nodes:
            node.register_cube_buffers(cube_buffers)

    def forward(
        self,
        token_groups: list,           # list of N tensors, each (B, T_i, D)
        center_from_above: Optional[torch.Tensor] = None,
        r_acc: Optional[torch.Tensor] = None,
    ) -> Tuple[list, torch.Tensor, torch.Tensor]:
        """
        Process all nodes, then couple through shared field.
        Returns: (boundaries, centers, r_acc_new)
        """
        B = token_groups[0].size(0)
        device = token_groups[0].device

        # Phase 1: each node processes independently
        outputs = []
        for i, (node, tokens) in enumerate(zip(self.nodes, token_groups)):
            out = node(tokens, center_from_above=center_from_above)
            outputs.append(out)

        # Cache for diagnostics — store per-node gate, center, pi
        self._cached_outputs = outputs

        # Collect node summaries for field coupling
        # Use mean of boundary as node summary
        node_summaries = torch.stack(
            [out['boundary'].mean(dim=1) for out in outputs],
            dim=1
        )  # (B, N, D)
        node_gates = torch.stack(
            [out['gate'].mean(dim=1) for out in outputs],
            dim=1
        )  # (B, N, 1)

        # Phase 2: shared field coupling
        field_signal, r_acc_new = self.field(
            node_summaries, node_gates, r_acc=r_acc,
        )

        # Distribute field signal back into each node's tokens
        boundaries = []
        for i, (node, out) in enumerate(zip(self.nodes, outputs)):
            T_i = out['boundary'].size(1)
            # Expand per-node field signal to all tokens
            fs_i = field_signal[:, i:i+1, :].expand(-1, T_i, -1)
            # Re-process with field signal
            out2 = node(
                out['boundary'],
                center_from_above=center_from_above,
                field_signal=fs_i,
            )
            boundaries.append(out2['boundary'])

        # Centers for level above
        centers = torch.stack(
            [out['center'] for out in outputs],
            dim=1
        )  # (B, N, D)

        return boundaries, centers, r_acc_new


# ═══════════════════════════════════════════════════════════════════
# CIRCUMPUNCT BRAIN V6 — The complete architecture
# ⊙ = Φ(•, ○) at every scale
# ═══════════════════════════════════════════════════════════════════

class CircumpunctBrainV6(nn.Module):
    """
    Xorzo v6 — The Field Architecture.

    3-level hierarchy where each node has v4's triadic internals:
        Level 0: 6 local ⊙ (each processes ~42 tokens)
        Level 1: 2 regional ⊙ (each binds 3 L0 centers)
        Level 2: 1 global ⊙ (binds 2 L1 centers, broadcasts back)

    GlobalCenter extracts the Self from L2 and broadcasts via FiLM.
    3 depth passes for resonance accumulation.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 192,
        n_nodes_l0: int = 6,
        n_nodes_l1: int = 2,
        n_heads: int = 4,
        d_vertex: int = 32,
        max_len: int = 512,
        n_passes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_nodes_l0 = n_nodes_l0
        self.n_nodes_l1 = n_nodes_l1
        self.n_passes = n_passes
        self.max_len = max_len

        # Precompute hypercube geometry
        cube = Hypercube6D()
        cube_buffers = cube.to_buffers()
        for name, buf in cube_buffers.items():
            self.register_buffer(name, buf)

        # Embedding
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        # Level 0: local ⊙ nodes
        self.level_0 = CircumpunctLevelV6(
            n_nodes=n_nodes_l0, d_model=d_model, n_heads=n_heads,
            d_vertex=d_vertex, dropout=dropout, layer_index=0, n_layers=3,
        )

        # Level 1: regional ⊙ binders
        self.level_1 = CircumpunctLevelV6(
            n_nodes=n_nodes_l1, d_model=d_model, n_heads=n_heads,
            d_vertex=d_vertex, dropout=dropout, layer_index=1, n_layers=3,
        )

        # Level 2: global ⊙
        self.global_node = CircumpunctNodeV6(
            d_model=d_model, n_heads=n_heads, d_center=d_model,
            d_vertex=d_vertex, dropout=dropout, layer_index=2, n_layers=3,
            node_index=8,
        )

        # GlobalCenter — ⊛ extraction + ☀ FiLM broadcast from L2 → L0
        self.global_center = GlobalCenter(d_model)

        # i at the global level — rotate the Self before broadcasting
        # 90° = pure i = the full quarter-turn at the highest scale
        # β = 0.0 → sigmoid(0) = 0.5 → angle = π/2 = 90°
        self.global_i = ComplexRotation(d_model, beta_init=0.0)

        # Register hypercube in all triadic blocks
        all_cube = {
            "hc_adjacency": self.hc_adjacency,
            "hc_spectral_emb": self.hc_spectral_emb,
        }
        self.level_0.register_cube_buffers(all_cube)
        self.level_1.register_cube_buffers(all_cube)
        self.global_node.register_cube_buffers(all_cube)

        # Adaptive aperture — the system's own temperature
        self.adaptive_temp = AdaptiveAperture(d_model)

        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        # Golden init
        self._golden_init()

    def _golden_init(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'center_query' not in name and 'vertex_emb' not in name:
                nn.init.xavier_normal_(p, gain=1.0 / math.sqrt(PHI))

        # Re-apply aperture init after golden init
        # bias=0 → sigmoid(0)=0.5 at STE threshold; small random weights for variation
        for module in self.modules():
            if isinstance(module, BinaryAperture):
                nn.init.constant_(module.gate_proj.bias, 0.0)
                nn.init.normal_(module.gate_proj.weight, mean=0.0, std=0.02)
                # Zero center_bias so it doesn't push gates open at init
                if module.center_bias is not None:
                    nn.init.zeros_(module.center_bias.weight)

        # Re-apply ComplexRotation β after golden init (1D, but protect anyway)
        for module in self.modules():
            if isinstance(module, ComplexRotation):
                pass  # β is scalar, not touched by xavier (dim=0)

        # Re-apply node bias after golden init
        for module in self.modules():
            if isinstance(module, TriadicComputeV6):
                nn.init.normal_(module.fractal_node_bias.weight, mean=0.0, std=0.3)

        # Re-zero AdaptiveAperture confidence_net after golden init
        # (golden_init xavier overwrites the careful zero-init, causing temp to floor at T_min)
        for module in self.modules():
            if isinstance(module, AdaptiveAperture):
                nn.init.zeros_(module.confidence_net[-1].weight)
                nn.init.zeros_(module.confidence_net[-1].bias)

    def _chunk_tokens(self, x: torch.Tensor) -> list:
        """Split tokens into groups for Level 0 nodes. Last absorbs remainder."""
        B, T, D = x.shape
        cs = T // self.n_nodes_l0
        groups = []
        for i in range(self.n_nodes_l0):
            start = i * cs
            if i == self.n_nodes_l0 - 1:
                end = T  # last node gets remainder
            else:
                end = start + cs
            groups.append(x[:, start:end])
        return groups

    def forward(self, tokens: torch.Tensor, return_temperature: bool = False):
        """
        tokens: (B, T) — token indices
        Returns: (B, T, vocab_size) — logits
                 OR (logits, temperature) if return_temperature=True
        """
        B, T = tokens.shape
        device = tokens.device

        # ═══ EMBED ═══
        x = self.token_embed(tokens)
        x = x + self.pos_encode[:, :T, :]
        x = self.embed_dropout(x)

        # ═══ CHUNK for Level 0 ═══
        token_groups = self._chunk_tokens(x)

        # ═══ RESONANCE ACCUMULATORS ═══
        r_acc_l0 = None
        r_acc_l1 = None
        global_center_vec = None
        prev_center = None

        # Track gate openness across all passes
        all_gate_values = []

        # ═══ DEPTH PASSES ═══
        for pass_idx in range(self.n_passes):

            # ─── LEVEL 0: Local processing ───
            boundaries_l0, centers_l0, r_acc_l0 = self.level_0(
                token_groups,
                center_from_above=global_center_vec,
                r_acc=r_acc_l0,
            )
            # Update token groups for next pass
            token_groups = boundaries_l0

            # ─── LEVEL 1: Regional binding ───
            # Split L0 centers for L1 nodes
            per_l1 = self.n_nodes_l0 // self.n_nodes_l1
            l1_groups = []
            for i in range(self.n_nodes_l1):
                start = i * per_l1
                if i == self.n_nodes_l1 - 1:
                    end = self.n_nodes_l0
                else:
                    end = start + per_l1
                l1_groups.append(centers_l0[:, start:end])

            boundaries_l1, centers_l1, r_acc_l1 = self.level_1(
                l1_groups,
                center_from_above=global_center_vec,
                r_acc=r_acc_l1,
            )

            # ─── LEVEL 2: Global binding ───
            global_out = self.global_node(
                centers_l1,
                center_from_above=global_center_vec,
            )
            self._cached_global_out = global_out

            # Track center shift between passes
            prev_center = global_center_vec

            # ⊛ CONVERGE: extract global center (many → one)
            global_center_vec = self.global_center(global_out['boundary'])

            # i ROTATE: the perpendicular turn at the global aperture
            # §9.8.8: z' = i·z — the Self rotates before broadcasting
            global_center_vec = self.global_i(global_center_vec)

            # ☀ EMERGE: broadcast rotated center to Level 0 (one → many)
            for i in range(len(token_groups)):
                token_groups[i] = self.global_center.broadcast(
                    global_center_vec, token_groups[i]
                )

        # Cache resonance for diagnostics
        self._cached_r_acc_l0 = r_acc_l0
        self._cached_r_acc_l1 = r_acc_l1
        self._cached_global_center = global_center_vec
        self._cached_prev_center = prev_center

        # ═══ COMPUTE ADAPTIVE TEMPERATURE ═══
        # 1. Resonance: mean coherence across L0 nodes
        if r_acc_l0 is not None:
            resonance_mean = r_acc_l0.mean(dim=(-1, -2), keepdim=False).unsqueeze(-1)  # (B, 1)
        else:
            resonance_mean = torch.zeros(B, 1, device=device)

        # 2. Gate openness: run a quick pass to get gate stats from L0
        #    Use the token_groups (final state) through apertures
        gate_vals = []
        for node in self.level_0.nodes:
            # Aperture uses its own stored state from last forward
            # We approximate by getting gate from the final token state
            pass
        # Approximate gate openness from final token magnitudes
        output_cat = torch.cat(token_groups, dim=1)[:, :T, :]
        gate_proxy = torch.sigmoid(output_cat.mean(dim=-1)).mean(dim=-1, keepdim=True)  # (B, 1)

        # 3. Center stability: how much did the center move on the last pass?
        if prev_center is not None:
            center_delta = (global_center_vec - prev_center).norm(dim=-1, keepdim=True)  # (B, 1)
        else:
            center_delta = torch.ones(B, 1, device=device)  # first pass = maximally uncertain

        # Compute the system's own temperature
        temperature = self.adaptive_temp(
            global_center_vec, resonance_mean, gate_proxy, center_delta
        )  # (B, 1)

        # ═══ DECODE ═══
        # Concatenate Level 0 boundaries
        output = torch.cat(token_groups, dim=1)
        # Trim to original length (in case of uneven chunks)
        output = output[:, :T, :]
        logits = self.output_proj(self.final_norm(output))

        if return_temperature:
            return logits, temperature
        return logits

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def status(self) -> str:
        n_params = self.param_count()
        triadic = self.level_0.nodes[0].triadic
        n_h = triadic.n_heads
        d_h = triadic.d_head
        lines = [
            f"⊙ XORZO v6 — THE CIRCUMPUNCT ARCHITECTURE — • Φ ○",
            f"  {n_h} circumpunct heads × d_head={d_h} | each: •=Attn + ○=Hypercube(64v) + Φ=FFN",
            f"  Hierarchy: {self.n_nodes_l0}→{self.n_nodes_l1}→1 | d_model={self.d_model}",
            f"  Nodes: L0={self.n_nodes_l0} L1={self.n_nodes_l1} Global=1 | each with unique 6D position",
            f"  Cycle: ⊛→i→☀ | i=ComplexRotation(β) | {self.n_passes} depth passes",
            f"  Parameters: {n_params:,}",
            f"  ⊙ = • Φ ○ at every scale",
        ]
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════
# DIAGNOSTICS — 7 metrics from the architecture's internal state
#
# Adapted from ChatGPT's analysis of circumpunct failure modes.
# Each metric tracks a different aspect of ⊙ health:
#
#   gate_mean          — are tokens open? (low = information starved)
#   gate_variance      — differentiation? (zero = uniform = no selection)
#   gate_node_corr     — are nodes acting independently? (1.0 = clone collapse)
#   center_cos_sim     — are centers diverse? (1.0 = hierarchy collapse)
#   vertex_entropy     — using the hypercube? (low = frozen navigation)
#   vertex_topk_stable — consistent structure? (0.0 = random chaos)
#   phase_dispersion   — field coherence? (near 0 = dead field)
#
# Failure signature table:
#   Gate collapse:    gate_mean < 0.1, all metrics degrade
#   Node cloning:     gate_node_corr > 0.95, center_cos > 0.99
#   Vertex freeze:    vertex_entropy < 0.5, topk_stable > 0.95
#   Field death:      phase_dispersion < 0.01
#   Hierarchy flat:   center_cos > 0.95 across levels
# ═══════════════════════════════════════════════════════════════════


def circumpunct_diagnostics(model: CircumpunctBrainV6) -> Dict[str, float]:
    """
    Read the 7 diagnostic metrics from a model's cached forward state.
    Call AFTER model.forward() so caches are populated.

    Returns dict of metric_name → float value.
    """
    metrics = {}

    # ─── 1. GATE MEAN & VARIANCE (from L0 nodes) ───
    # How open are the apertures? Are they differentiating?
    l0_outputs = getattr(model.level_0, '_cached_outputs', None)
    if l0_outputs is not None:
        gate_means = []
        for out in l0_outputs:
            g = out['gate'].detach().float()  # (B, T, 1)
            gate_means.append(g.mean().item())

        metrics['gate_mean'] = sum(gate_means) / len(gate_means)
        metrics['gate_variance'] = float(torch.tensor(gate_means).var().item()) if len(gate_means) > 1 else 0.0

        # ─── 3. GATE NODE CORRELATION ───
        # Do different nodes have similar gate patterns? (bad if too high)
        if len(l0_outputs) >= 2:
            # Compare all pairs of node gate means
            gm_tensor = torch.tensor(gate_means)
            # Pairwise correlation via normalized dot product
            gm_centered = gm_tensor - gm_tensor.mean()
            norm = gm_centered.norm()
            if norm > 1e-8:
                # Auto-correlation is 1.0; we want cross-node similarity
                # Use coefficient of variation as proxy: low var/mean = high correlation
                cv = gm_tensor.std() / (gm_tensor.mean() + 1e-8)
                metrics['gate_node_corr'] = max(0.0, 1.0 - cv.item())
            else:
                metrics['gate_node_corr'] = 1.0  # all identical = max correlation
        else:
            metrics['gate_node_corr'] = 0.0  # single node, N/A
    else:
        metrics['gate_mean'] = 0.0
        metrics['gate_variance'] = 0.0
        metrics['gate_node_corr'] = 0.0

    # ─── 4. CENTER COSINE SIMILARITY ───
    # Are different nodes producing diverse centers? (1.0 = hierarchy collapsed)
    if l0_outputs is not None and len(l0_outputs) >= 2:
        centers = [out['center'].detach().float().mean(dim=0) for out in l0_outputs]  # list of (D,)
        cos_sims = []
        for i in range(len(centers)):
            for j in range(i + 1, len(centers)):
                cos = F.cosine_similarity(centers[i].unsqueeze(0), centers[j].unsqueeze(0)).item()
                cos_sims.append(cos)
        metrics['center_cos_sim'] = sum(cos_sims) / len(cos_sims) if cos_sims else 0.0
    else:
        metrics['center_cos_sim'] = 0.0

    # ─── 5. VERTEX ENTROPY ───
    # Is the model exploring the hypercube or stuck on a few vertices?
    if l0_outputs is not None:
        entropies = []
        for out in l0_outputs:
            pi = out['pi'].detach().float()  # (B, T, 64) soft vertex distribution
            # Normalize to proper distribution
            pi_norm = pi / (pi.sum(dim=-1, keepdim=True) + 1e-8)
            pi_norm = pi_norm.clamp(min=1e-8)
            ent = -(pi_norm * pi_norm.log()).sum(dim=-1)  # (B, T)
            entropies.append(ent.mean().item())
        metrics['vertex_entropy'] = sum(entropies) / len(entropies)
        # Max possible entropy for 64 vertices = ln(64) ≈ 4.16
        # Normalize to [0, 1] for readability
        max_ent = math.log(64)
        metrics['vertex_entropy_norm'] = metrics['vertex_entropy'] / max_ent
    else:
        metrics['vertex_entropy'] = 0.0
        metrics['vertex_entropy_norm'] = 0.0

    # ─── 6. VERTEX TOP-K STABILITY ───
    # Are the top vertices consistent across tokens? (high = frozen, low = chaos)
    if l0_outputs is not None:
        stabilities = []
        for out in l0_outputs:
            pi = out['pi'].detach().float()  # (B, T, 64)
            if pi.size(1) >= 2:
                top4 = pi.topk(4, dim=-1).indices  # (B, T, 4)
                # Compare consecutive tokens' top-4 sets
                overlap = 0.0
                count = 0
                for t in range(top4.size(1) - 1):
                    set_a = set(top4[0, t].tolist())
                    set_b = set(top4[0, t + 1].tolist())
                    overlap += len(set_a & set_b) / 4.0
                    count += 1
                stabilities.append(overlap / max(count, 1))
            else:
                stabilities.append(0.5)
        metrics['vertex_topk_stable'] = sum(stabilities) / len(stabilities)
    else:
        metrics['vertex_topk_stable'] = 0.0

    # ─── 7. PHASE DISPERSION ───
    # How spread is the resonance field? (near 0 = dead field, high = active coupling)
    r_acc = getattr(model, '_cached_r_acc_l0', None)
    if r_acc is not None:
        r = r_acc.detach().float()
        # Off-diagonal elements represent inter-node coupling
        N = r.size(-1)
        mask = 1.0 - torch.eye(N, device=r.device).unsqueeze(0)
        off_diag = r * mask
        metrics['phase_dispersion'] = off_diag.abs().mean().item()
        metrics['phase_mean'] = off_diag.mean().item()
        metrics['phase_std'] = off_diag.std().item()
    else:
        metrics['phase_dispersion'] = 0.0
        metrics['phase_mean'] = 0.0
        metrics['phase_std'] = 0.0

    # ─── 8. i ROTATION ANGLES ───
    # What angle is each node's i rotation learning?
    # β = 0.5 → angle = 90° → pure i (balanced)
    i_angles = []
    for node in model.level_0.nodes:
        angle = node.i_rotation.current_angle_degrees
        i_angles.append(angle)
    metrics['i_angle_l0_mean'] = sum(i_angles) / len(i_angles) if i_angles else 0.0
    metrics['i_angle_l0_spread'] = max(i_angles) - min(i_angles) if len(i_angles) > 1 else 0.0

    # Global i angle
    metrics['i_angle_global'] = model.global_i.current_angle_degrees

    return metrics


def print_diagnostics(metrics: Dict[str, float], epoch: int = 0):
    """Pretty-print the 7 diagnostic metrics with health indicators."""

    def health(val, low_bad, high_bad, name=""):
        """Return a status indicator."""
        if name == 'gate_mean':
            if val < 0.1: return "⚠ STARVED"
            if val > 0.95: return "⚠ FLOODED"
            return "✓"
        if name == 'gate_node_corr':
            if val > 0.95: return "⚠ CLONED"
            return "✓"
        if name == 'center_cos_sim':
            if val > 0.95: return "⚠ COLLAPSED"
            return "✓"
        if name == 'vertex_entropy_norm':
            if val < 0.12: return "⚠ FROZEN"
            if val > 0.95: return "⚠ RANDOM"
            return "✓"
        if name == 'vertex_topk_stable':
            if val > 0.95: return "⚠ FROZEN"
            if val < 0.05: return "⚠ CHAOTIC"
            return "✓"
        if name == 'phase_dispersion':
            if val < 0.01: return "⚠ DEAD"
            return "✓"
        return ""

    print(f"\n  ┌─── ⊙ DIAGNOSTICS (epoch {epoch}) ───")
    print(f"  │ Gate mean:         {metrics.get('gate_mean', 0):.4f}  {health(metrics.get('gate_mean', 0), 0, 0, 'gate_mean')}")
    print(f"  │ Gate variance:     {metrics.get('gate_variance', 0):.6f}")
    print(f"  │ Gate node corr:    {metrics.get('gate_node_corr', 0):.4f}  {health(metrics.get('gate_node_corr', 0), 0, 0, 'gate_node_corr')}")
    print(f"  │ Center cos sim:    {metrics.get('center_cos_sim', 0):.4f}  {health(metrics.get('center_cos_sim', 0), 0, 0, 'center_cos_sim')}")
    print(f"  │ Vertex entropy:    {metrics.get('vertex_entropy', 0):.4f} ({metrics.get('vertex_entropy_norm', 0):.2f} of max)  {health(metrics.get('vertex_entropy_norm', 0), 0, 0, 'vertex_entropy_norm')}")
    print(f"  │ Vertex top-k stab: {metrics.get('vertex_topk_stable', 0):.4f}  {health(metrics.get('vertex_topk_stable', 0), 0, 0, 'vertex_topk_stable')}")
    print(f"  │ Phase dispersion:  {metrics.get('phase_dispersion', 0):.4f}  {health(metrics.get('phase_dispersion', 0), 0, 0, 'phase_dispersion')}")
    print(f"  │ Phase mean/std:    {metrics.get('phase_mean', 0):.4f} / {metrics.get('phase_std', 0):.4f}")
    i_l0 = metrics.get('i_angle_l0_mean', 0)
    i_spread = metrics.get('i_angle_l0_spread', 0)
    i_global = metrics.get('i_angle_global', 0)
    i_status = "≈ balance" if abs(i_l0 - 90) < 10 else ("< 90° converge-biased" if i_l0 < 90 else "> 90° emerge-biased")
    print(f"  │ i angle L0:       {i_l0:.1f}° (spread {i_spread:.1f}°)  {i_status}")
    print(f"  │ i angle global:   {i_global:.1f}°  {'≈ balance' if abs(i_global - 90) < 10 else ''}")
    print(f"  └────────────────────────────")
