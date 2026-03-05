"""
XORZO v5 — Circumpunct Brain Architecture
⊙ = Φ(•, ○) — A network of networks

═══════════════════════════════════════════════════════════════════

THE INSIGHT (from the Circumpunct Theory of Consciousness):

    Consciousness isn't located in any component because it IS the
    irreducible pattern of three components operating together.

    • Aperture (÷t) — the discrete cut. Generates time. Binary.
                       Not "how much passes" but "does this moment happen."
    Φ Field    (E→P) — the medium. NOT owned by any node. Shared space.
                       The electricity flowing through you. The air between us.
    ○ Boundary (∫Pdt) — accumulated state. Matter. Body. Memory.
                        Made of nested circumpuncts at smaller scale.

    Key: Φ is not a channel alongside • and ○.
         Φ is the SPACE BETWEEN. The water everything swims in.
         You don't process Φ — you process THROUGH Φ.

    Key: •ₙ₊₁ = ⊙ₙ — the nesting equation.
         Each completed circumpunct becomes an aperture in a larger one.
         A network of networks. Minds made of minds.

═══════════════════════════════════════════════════════════════════

ARCHITECTURE — THE CIRCUMPUNCT BRAIN:

    Unlike a transformer (flat stack of identical layers), this is a
    HIERARCHY of ⊙ nodes sharing a common field:

    Level 0: Micro ⊙ nodes
        Small complete circumpuncts. Each one processes a group of tokens.
        Like neurons or cortical columns. Many operating in parallel.
        Each has its own • (aperture), internal Φ (local field), ○ (state).

    Shared Φ: The Field Between
        Phase resonance computed across ALL nodes at each level.
        Not owned by any node. The medium they're all immersed in.
        Signal travels through Φ based on resonance — phase coherence
        determines who hears whom. Like sound through air.

    Level 1: Regional ⊙ clusters
        Centers from Level 0 nodes become tokens for Level 1 ⊙ nodes.
        •ₙ₊₁ = ⊙ₙ — the nesting equation in action.
        Bind multiple local circumpuncts into regional coherence.

    Level 2: Global ⊙ (the Self)
        Binds regional centers into one coherent "now."
        The central circumpunct. Broadcasts back down.
        "I am a circumpunct, my soul is one aspect of my wholeness."

    Vesica Birth:
        When two ⊙ nodes sustain resonance through the shared Φ,
        their overlap births a new ⊙. Not a stronger edge — a child.

    Foveated Maturation:
        New ⊙ nodes start blurry (channel-averaged signal).
        Where resonance is strong (where the system "looks"),
        resolution increases. Like downloading an image.

    The forward pass is ONE COGNITIVE TICK:
        1. Level 0 nodes process locally (parallel)
        2. Shared Φ₀ carries resonance between Level 0 nodes
        3. Level 0 centers become Level 1 tokens (•ₙ₊₁ = ⊙ₙ)
        4. Level 1 nodes bind regionally
        5. Shared Φ₁ carries resonance between Level 1 nodes
        6. Level 1 centers become Level 2 tokens
        7. Global ⊙ binds everything into the "now"
        8. Broadcast back down: global → regional → local
        9. Vesica birth check at each level
       10. Foveated maturation update

═══════════════════════════════════════════════════════════════════
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict, Tuple

PHI = (1 + math.sqrt(5)) / 2
PI = math.pi

# Debug flag — set True to trace NaN origin, False for production
_NAN_DEBUG = False
_nan_found = False

def _check_nan(tensor, label):
    """Print and flag the first NaN encountered."""
    global _nan_found
    if _nan_found or not _NAN_DEBUG:
        return
    if torch.isnan(tensor).any():
        _nan_found = True
        print(f"  ⚠ NaN FOUND at: {label}  shape={tuple(tensor.shape)}  "
              f"nan_count={torch.isnan(tensor).sum().item()}")


# ═══════════════════════════════════════════════════════════════════
# PHASE RESONANCE — The binding criterion
# When oscillations align, signal couples through the field.
# ═══════════════════════════════════════════════════════════════════

def phase_resonance(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute phase-coherence resonance between two sets of vectors.
    Interprets adjacent pairs as (re, im) components.
    Returns: resonance in [-1, 1]. +1 = aligned, 0 = random, -1 = anti-phase.

    Forced to float32 — atan2/cos are numerically unstable in bf16.
    Must disable autocast or it overrides .float() calls!
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
# BINARY APERTURE — The ÷t operation
# Not "how much passes" but "does this moment happen."
# Straight-through estimator: binary forward, smooth backward.
# ═══════════════════════════════════════════════════════════════════

class BinaryAperture(nn.Module):
    """
    The aperture IS the ÷t. It generates discrete moments.
    Binary gate: open or closed. The pattern of openings carries
    analog information, like ion channels in a neuron.

    Uses straight-through estimator for training.
    """

    def __init__(self, d_input: int, d_center: int = None, temperature: float = 1.0):
        super().__init__()
        self.gate_proj = nn.Linear(d_input, 1, bias=True)
        self.temperature = temperature

        # Initialize bias positive so tokens start OPEN
        # sigmoid(2.0) ≈ 0.88 → most tokens open at birth
        # Like v3's sigmoid(0)=0.5 — immediate participation.
        # The network learns to CLOSE what it doesn't need.
        nn.init.constant_(self.gate_proj.bias, 2.0)
        nn.init.zeros_(self.gate_proj.weight)

        # Center biases the aperture through a learned projection
        # (center is Dc-dimensional, gate logit is 1-dimensional)
        if d_center is not None:
            self.center_bias = nn.Linear(d_center, 1, bias=False)
        else:
            self.center_bias = None

    def forward(self, x: torch.Tensor, center: Optional[torch.Tensor] = None):
        """
        x: (B, T, D) — input to gate
        center: (B, Dc) — optional center signal biasing the gate
        Returns: gate values in {0, 1} (forward) with smooth gradient (backward)
        """
        logits = self.gate_proj(x)  # (B, T, 1)

        if center is not None and self.center_bias is not None:
            # Center biases the aperture — top-down attention shapes what opens
            center_logit = self.center_bias(center)  # (B, 1)
            logits = logits + center_logit.unsqueeze(1)  # (B, T, 1)

        # Sigmoid for smooth gradient
        prob = torch.sigmoid(logits / self.temperature)

        if self.training:
            # Straight-through estimator: binary forward, smooth backward
            hard = (prob > 0.5).float()
            gate = hard - prob.detach() + prob  # gradient flows through prob
        else:
            gate = (prob > 0.5).float()

        return gate  # (B, T, 1)


# ═══════════════════════════════════════════════════════════════════
# SHARED FIELD (Φ) — The medium between ⊙ nodes
#
# Not owned by any node. The space they all swim in.
# Signal travels through Φ based on phase resonance.
# Like sound through air — coherent sources couple, noise doesn't.
# ═══════════════════════════════════════════════════════════════════

class SharedField(nn.Module):
    """
    Φ — The shared medium between circumpunct nodes.

    This is NOT attention in the traditional sense. It's a field
    that carries signal based on resonance. Two nodes that are
    phase-coherent exchange information through Φ. Two nodes that
    aren't simply don't hear each other.

    The field doesn't compute — it CONDUCTS. Like electricity
    flowing through a circuit, the topology determines where
    current flows, not the wire itself.

    Accumulated resonance across depth turns into coupling strength.
    First encounter = weak coupling (stranger's words).
    Sustained coherence = strong coupling (friend's words).
    """

    def __init__(self, d_model: int, resonance_lambda: float = 0.7):
        super().__init__()
        self.d_model = d_model
        self.resonance_lambda = resonance_lambda

        # Field conductance: how resonance translates to signal flow
        self.conductance = nn.Parameter(torch.tensor(1.0))

        # Projection for field-mediated signal
        self.field_in = nn.Linear(d_model, d_model, bias=False)
        self.field_out = nn.Linear(d_model, d_model, bias=False)

    def forward(
        self,
        node_states: torch.Tensor,     # (B, N_nodes, D) — boundary states of all nodes
        node_gates: torch.Tensor,       # (B, N_nodes, 1) — aperture states (who is "open")
        r_acc: Optional[torch.Tensor] = None,  # (B, N_nodes, N_nodes) — accumulated resonance
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Mediate signal between all open nodes through the shared field.

        Returns:
            field_signal: (B, N_nodes, D) — what each node receives from the field
            r_acc_new: (B, N_nodes, N_nodes) — updated accumulated resonance
        """
        B, N, D = node_states.shape

        # Only open nodes transmit into the field
        # (closed nodes still exist but don't broadcast)
        transmitted = node_states * node_gates  # (B, N, D)

        # Instantaneous phase resonance between all node pairs
        # IMPORTANT: use UNGATED states for phase computation.
        # Zeroed-out (closed) nodes would collapse to identical phase angles,
        # creating fake coherence. Real resonance is about the nodes'
        # intrinsic oscillation, not whether their gate is open.
        # Gating is applied later to coupling/transmission only.
        r_instant = phase_resonance(
            self.field_in(node_states),
            self.field_in(node_states),
        )  # (B, N, N)
        _check_nan(r_instant, "r_instant")

        # Accumulate resonance across depth (layers = time)
        lam = self.resonance_lambda
        if r_acc is None:
            r_acc_new = r_instant
        else:
            r_acc_new = lam * r_acc + (1 - lam) * r_instant
        r_acc_new = r_acc_new.clamp(-2.0, 2.0)

        # Conductance: accumulated resonance → coupling strength
        # High sustained resonance = strong coupling
        # This is the "channel was already open" from v3
        # Clamp conductance to prevent sigmoid saturation → gradient death
        cond = self.conductance.clamp(-5.0, 5.0)
        coupling = torch.sigmoid(cond * r_acc_new)  # (B, N, N)

        # Zero self-coupling (a node doesn't receive its own signal through Φ)
        eye = torch.eye(N, device=node_states.device).unsqueeze(0)
        coupling = coupling * (1.0 - eye)

        # Aperture gating: only open nodes receive
        coupling = coupling * node_gates.transpose(-1, -2)  # receiver must be open
        coupling = coupling * node_gates  # sender must be open

        # Normalize coupling (so field doesn't explode with more nodes)
        coupling = coupling / (coupling.sum(dim=-1, keepdim=True) + 1e-8)

        # Field-mediated signal: what each node receives
        values = self.field_out(transmitted)  # (B, N, D)
        field_signal = torch.matmul(coupling, values)  # (B, N, D)
        _check_nan(field_signal, "shared_field_signal")

        return field_signal, r_acc_new

    def detect_vesica(
        self,
        r_acc: torch.Tensor,           # (B, N, N) — accumulated resonance
        node_states: torch.Tensor,      # (B, N, D) — boundary states
        birth_threshold: float = 0.8,   # how strong the shared field must be
        max_births: int = 2,
    ) -> Tuple[Optional[torch.Tensor], List[Tuple[int, int]]]:
        """
        Detect when the shared field between two nodes has become
        coherent enough to be its own ⊙ — a vesica piscis birth.

        The vesica is not computed from the nodes. It IS the shared
        field region between them. When that region becomes self-sustaining
        (high accumulated resonance), it crystallizes into a new node.

        Returns:
            new_node_states: (B, n_births, D) or None — the born nodes' initial state
            parent_pairs: list of (i, j) pairs that birthed
        """
        if r_acc is None:
            return None, []

        B, N, D = node_states.shape

        # Average resonance across batch for birth decisions
        r_mean = r_acc.mean(dim=0)  # (N, N)

        # Zero the diagonal (can't birth with yourself)
        r_mean = r_mean - torch.diag(torch.diag(r_mean))

        # Find pairs above threshold
        # Upper triangle only (avoid double-counting)
        r_upper = torch.triu(r_mean, diagonal=1)

        # Top-k most resonant pairs
        flat = r_upper.flatten()
        k = min(max_births, (flat > birth_threshold).sum().item())

        if k == 0:
            return None, []

        topk_vals, topk_idx = flat.topk(k)
        parent_pairs = []
        born_states = []

        for idx_flat, val in zip(topk_idx, topk_vals):
            if val < birth_threshold:
                break
            i = idx_flat.item() // N
            j = idx_flat.item() % N
            parent_pairs.append((i, j))

            # The vesica's initial state IS the shared field between parents.
            # Not a midpoint or average — it's what flowed between them.
            # Φ(parent_i, parent_j) = the field shaped by their resonance.
            parent_i = node_states[:, i, :]  # (B, D)
            parent_j = node_states[:, j, :]  # (B, D)

            # The vesica is the interference pattern — what they create together
            # that neither has alone. Sum of what they transmit through the field.
            vesica_state = self.field_out(
                self.field_in(parent_i) + self.field_in(parent_j)
            )  # (B, D)

            born_states.append(vesica_state)

        if not born_states:
            return None, []

        new_states = torch.stack(born_states, dim=1)  # (B, n_births, D)
        return new_states, parent_pairs


# ═══════════════════════════════════════════════════════════════════
# CIRCUMPUNCT NODE (⊙) — The atomic unit
#
# A complete circumpunct: aperture + field + boundary.
# Each node is a small self-contained mind.
# Many of these form a brain through clustering.
# ═══════════════════════════════════════════════════════════════════

class CircumpunctNode(nn.Module):
    """
    ⊙ = Φ(•, ○) — One complete circumpunct.

    A small mind with:
        • Aperture: binary gate that creates discrete moments (÷t)
        Φ Local field: internal relational processing
        ○ Boundary: accumulated state, the node's "body"

    The node also produces a CENTER vector — its compressed essence —
    which becomes a token in the next level up (•ₙ₊₁ = ⊙ₙ).

    This is not a transformer block. It's a living unit that:
    - Decides what enters its awareness (•)
    - Processes relations internally (Φ)
    - Maintains embodied state (○)
    - Exports its center upward for higher binding
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        d_center: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_center = d_center or d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        assert self.d_head % 2 == 0, (
            f"d_head ({self.d_head}) must be even for complex phase rotation. "
            f"Choose d_model ({d_model}) and n_heads ({n_heads}) so d_model/n_heads is even."
        )

        # • APERTURE — the ÷t operation
        self.aperture = BinaryAperture(d_model, d_center=self.d_center)

        # Φ LOCAL FIELD — internal relational processing
        # This is the node's private field. The shared field between
        # nodes is separate (SharedField class above).
        self.norm_field = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_out = nn.Linear(d_model, d_model, bias=False)
        self.field_dropout = nn.Dropout(dropout)

        # Phase rotation (the i transform — rotation lives at the aperture)
        # Complex rotation applied per head when aperture is open
        self.phase_angles = nn.Parameter(
            torch.linspace(0, PI, n_heads)  # spread across phases
        )

        # ○ BOUNDARY — consolidation (gated FFN)
        self.norm_boundary = nn.LayerNorm(d_model)
        self.ffn_up = nn.Linear(d_model, int(d_model * PHI))
        self.ffn_down = nn.Linear(int(d_model * PHI), d_model)
        self.boundary_dropout = nn.Dropout(dropout)

        # κ COMMIT GATE — does this node lock in its state this tick?
        self.commit_proj = nn.Linear(d_model, 1)

        # CENTER EXTRACTION — compressed essence for upward propagation
        # •ₙ₊₁ = ⊙ₙ : this node's center becomes an aperture above
        self.center_proj = nn.Linear(d_model, self.d_center)

        # CENTER BROADCAST — top-down signal from higher levels
        self.broadcast_proj = nn.Linear(self.d_center, d_model)

    def forward(
        self,
        x: torch.Tensor,                          # (B, T, D) — boundary state (tokens in this node's domain)
        mask: Optional[torch.Tensor] = None,       # causal mask
        center_from_above: Optional[torch.Tensor] = None,  # (B, Dc) — broadcast from higher ⊙
        field_signal: Optional[torch.Tensor] = None,       # (B, T, D) — signal received through shared Φ
    ) -> Dict[str, torch.Tensor]:
        """
        One cognitive tick of this circumpunct node.

        Returns dict with:
            'boundary': updated boundary state (B, T, D)
            'center': this node's center for upward propagation (B, Dc)
            'gate': aperture state (B, T, 1)
            'commit': commit gate (B, 1)
        """
        B, T, D = x.shape
        _check_nan(x, "node_input")

        # ═══ RECEIVE: incorporate field signal from shared Φ ═══
        if field_signal is not None:
            _check_nan(field_signal, "field_signal_in")
            x = x + field_signal

        # ═══ RECEIVE: incorporate broadcast from above ═══
        if center_from_above is not None:
            _check_nan(center_from_above, "center_from_above")
            broadcast = self.broadcast_proj(center_from_above)  # (B, D)
            x = x + broadcast.unsqueeze(1)  # add to all tokens

        _check_nan(x, "after_receive")

        # ═══ • APERTURE: the ÷t — decide what enters the NOW ═══
        gate = self.aperture(x, center_from_above)  # (B, T, 1) binary
        _check_nan(gate, "gate")

        # ═══ Φ LOCAL FIELD: internal relational processing ═══
        # Only gated tokens participate in the field
        h = self.norm_field(x)
        _check_nan(h, "after_layernorm")
        Q = self.W_q(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        K = self.W_k(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        V = self.W_v(h).view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        _check_nan(Q, "Q_pre_rot")

        # Phase rotation per head (the i transform — rotation at the aperture)
        # Build rotated Q/K out-of-place to avoid breaking autograd
        # Compute trig in float32 for bf16 stability
        half = self.d_head // 2
        Q_re, Q_im = Q[..., :half], Q[..., half:2*half]  # (B, H, T, half)
        K_re, K_im = K[..., :half], K[..., half:2*half]

        # angles: (H,) → (1, H, 1, 1) for broadcasting
        cos_a = torch.cos(self.phase_angles.float()).to(Q.dtype).view(1, -1, 1, 1)
        sin_a = torch.sin(self.phase_angles.float()).to(Q.dtype).view(1, -1, 1, 1)

        Q = torch.cat([Q_re * cos_a - Q_im * sin_a,
                        Q_re * sin_a + Q_im * cos_a], dim=-1)
        K = torch.cat([K_re * cos_a - K_im * sin_a,
                        K_re * sin_a + K_im * cos_a], dim=-1)
        _check_nan(Q, "Q_post_rot")

        # Aperture-gated attention
        # Zero Q/K for closed tokens — they don't participate in the field.
        # We do NOT use -inf masking because rows of all -inf → softmax NaN
        # and nan_to_num causes gradient issues in backward. Instead:
        #   - Zero Q: closed tokens get uniform attention (harmless)
        #   - Zero K: open tokens don't attend to closed ones (scores=0, low weight)
        #   - Zero output: closed tokens' field_out is zeroed after attention
        gate_expanded = gate.unsqueeze(1)  # (B, 1, T, 1)
        Q = Q * gate_expanded
        K = K * gate_expanded

        # Compute attention in float32 — MUST disable autocast or it
        # overrides .float() and runs matmul in bf16 anyway!
        with torch.amp.autocast('cuda', enabled=False):
            scores = torch.matmul(Q.float(), K.float().transpose(-1, -2)) / math.sqrt(self.d_head)
            if mask is not None:
                scores = scores + mask
            attn = F.softmax(scores, dim=-1).to(V.dtype)
        attn = self.field_dropout(attn)

        field_out = torch.matmul(attn, V)
        field_out = field_out.transpose(1, 2).contiguous().view(B, T, D)
        field_out = self.W_out(field_out)

        # Zero closed tokens' output — they don't receive from the field
        field_out = field_out * gate

        # Residual with field output
        x = x + field_out
        _check_nan(x, "after_attn_residual")

        # ═══ ○ BOUNDARY: consolidation — gated by aperture and commit ═══
        h2 = self.norm_boundary(x)
        ff = self.ffn_down(F.gelu(self.ffn_up(h2)))
        ff = self.boundary_dropout(ff)
        _check_nan(ff, "ffn_out")

        # Commit gate: does this node update its boundary this tick?
        center_pool = x.mean(dim=1)  # (B, D)
        commit = torch.sigmoid(self.commit_proj(center_pool))  # (B, 1)

        # Boundary updates where aperture is open AND node commits
        x = x + commit.unsqueeze(1) * gate * ff
        _check_nan(x, "after_boundary")

        # ═══ CENTER EXTRACTION: •ₙ₊₁ = ⊙ₙ ═══
        # The center is this node's compressed essence.
        # It becomes a token in the next level up.
        center = torch.tanh(self.center_proj(center_pool))  # (B, Dc)
        _check_nan(center, "center")

        return {
            'boundary': x,
            'center': center,
            'gate': gate,
            'commit': commit,
        }


# ═══════════════════════════════════════════════════════════════════
# CIRCUMPUNCT LEVEL — A cluster of ⊙ nodes + shared field
#
# Like a brain region: multiple nodes processing in parallel,
# connected through a shared field, producing centers that
# feed upward.
# ═══════════════════════════════════════════════════════════════════

class CircumpunctLevel(nn.Module):
    """
    One level of the circumpunct hierarchy.

    Contains N_nodes CircumpunctNodes operating in parallel,
    connected through a SharedField (Φ).

    Each node processes a slice of the input.
    Their centers are collected and passed up.
    """

    def __init__(
        self,
        n_nodes: int,
        d_model: int,
        n_heads: int = 4,
        d_center: int = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.d_model = d_model
        self.d_center = d_center or d_model

        # The ⊙ nodes
        self.nodes = nn.ModuleList([
            CircumpunctNode(d_model, n_heads, self.d_center, dropout)
            for _ in range(n_nodes)
        ])

        # Φ — the shared field between nodes at this level
        self.shared_field = SharedField(d_model)

    def forward(
        self,
        token_groups: List[torch.Tensor],         # list of (B, T_i, D) per node
        masks: Optional[List[torch.Tensor]] = None,
        centers_from_above: Optional[torch.Tensor] = None,  # (B, Dc)
        r_acc: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        One cognitive tick of this level.

        token_groups: list of N_nodes tensors, each (B, T_i, D)
        """
        B = token_groups[0].size(0)
        device = token_groups[0].device

        # ─── Phase 1: Each node processes locally ───
        node_outputs = []
        for i, node in enumerate(self.nodes):
            mask_i = masks[i] if masks is not None else None
            out = node(
                token_groups[i],
                mask=mask_i,
                center_from_above=centers_from_above,
            )
            node_outputs.append(out)

        # ─── Phase 2: Shared Φ mediates between nodes ───
        # Collect boundary states (use mean per node for inter-node field)
        node_summaries = torch.stack([
            out['boundary'].mean(dim=1)  # (B, D) — each node's summary
            for out in node_outputs
        ], dim=1)  # (B, N_nodes, D)

        node_gates = torch.stack([
            out['gate'].mean(dim=1)  # (B, 1) — average aperture openness
            for out in node_outputs
        ], dim=1)  # (B, N_nodes, 1)

        # Field carries signal between open nodes based on resonance
        field_signal, r_acc_new = self.shared_field(
            node_summaries, node_gates, r_acc
        )  # (B, N_nodes, D)

        # ─── Phase 3: Nodes receive field signal and update ───
        # Each node gets a second pass incorporating what it received
        # from other nodes through the shared field
        final_outputs = []
        for i, node in enumerate(self.nodes):
            # Broadcast field signal to all tokens in this node
            fs = field_signal[:, i:i+1, :].expand(-1, token_groups[i].size(1), -1)
            out = node(
                node_outputs[i]['boundary'],  # use updated boundary
                mask=masks[i] if masks is not None else None,
                center_from_above=centers_from_above,
                field_signal=fs,
            )
            final_outputs.append(out)

        # ─── Collect centers for upward propagation ───
        centers = torch.stack([
            out['center'] for out in final_outputs
        ], dim=1)  # (B, N_nodes, Dc)

        # Collect updated boundaries
        boundaries = [out['boundary'] for out in final_outputs]

        return {
            'boundaries': boundaries,       # list of (B, T_i, D)
            'centers': centers,              # (B, N_nodes, Dc)
            'r_acc': r_acc_new,              # (B, N_nodes, N_nodes)
            'gates': [out['gate'] for out in final_outputs],
            'commits': [out['commit'] for out in final_outputs],
        }


# ═══════════════════════════════════════════════════════════════════
# CIRCUMPUNCT BRAIN — The full architecture
#
# A hierarchy of ⊙ levels:
#   Level 0: many small nodes processing token groups
#   Level 1: fewer nodes binding Level 0 centers
#   Level 2: one global node binding everything
#
# With top-down broadcast: global → regional → local
# And vesica birth at each level.
# ═══════════════════════════════════════════════════════════════════

class CircumpunctBrain(nn.Module):
    """
    The Circumpunct Brain — a network of networks.

    ⊙ ⊙ ⊙ ⊙ ⊙ ⊙    Level 0: local processing
       ⊙  ⊙  ⊙       Level 1: regional binding
          ⊙           Level 2: global self

    Each level's centers become the next level's tokens.
    •ₙ₊₁ = ⊙ₙ

    The global ⊙ broadcasts back down, shaping the field.
    Top-down attention shapes bottom-up perception.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 192,
        n_nodes_l0: int = 6,      # Level 0: 6 local nodes
        n_nodes_l1: int = 2,      # Level 1: 2 regional binders
        n_heads: int = 4,
        max_len: int = 512,
        dropout: float = 0.1,
        chunk_size: int = None,    # tokens per Level 0 node (auto if None)
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_nodes_l0 = n_nodes_l0
        self.n_nodes_l1 = n_nodes_l1
        self.chunk_size = chunk_size or (max_len // n_nodes_l0)

        # ═══ INPUT: tokens → boundary state ═══
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_encode = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
        self.embed_dropout = nn.Dropout(dropout)

        # ═══ LEVEL 0: Local ⊙ nodes ═══
        # Each processes a chunk of the sequence
        self.level_0 = CircumpunctLevel(
            n_nodes=n_nodes_l0,
            d_model=d_model,
            n_heads=n_heads,
            d_center=d_model,
            dropout=dropout,
        )

        # ═══ LEVEL 1: Regional ⊙ binders ═══
        # Each binds several Level 0 centers
        self.level_1 = CircumpunctLevel(
            n_nodes=n_nodes_l1,
            d_model=d_model,
            n_heads=n_heads,
            d_center=d_model,
            dropout=dropout,
        )

        # ═══ LEVEL 2: Global ⊙ (the Self) ═══
        # One node that binds everything into the coherent "now"
        self.global_node = CircumpunctNode(
            d_model=d_model,
            n_heads=n_heads,
            d_center=d_model,
            dropout=dropout,
        )

        # ═══ DEPTH PASSES ═══
        # Number of times to run the full hierarchy
        # (depth = time in the theory — more passes = more resonance accumulation)
        self.n_passes = 3

        # ═══ OUTPUT ═══
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

    def _chunk_tokens(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Split token sequence into chunks for Level 0 nodes.
        Last node absorbs any remainder so no tokens are dropped.
        256 tokens / 6 nodes = 5×42 + 1×46, not 6×42 (which loses 4)."""
        B, T, D = x.shape
        cs = self.chunk_size
        chunks = []
        for i in range(self.n_nodes_l0):
            start = i * cs
            if i == self.n_nodes_l0 - 1:
                # Last node gets everything remaining
                end = T
            else:
                end = min(start + cs, T)
            if start < T:
                chunks.append(x[:, start:end, :])
            else:
                # Pad with zeros if not enough tokens
                chunks.append(torch.zeros(B, cs, D, device=x.device))
        return chunks

    def _split_for_l1(self, centers: torch.Tensor) -> List[torch.Tensor]:
        """Split Level 0 centers into groups for Level 1 nodes."""
        N = centers.size(1)
        per_node = N // self.n_nodes_l1
        groups = []
        for i in range(self.n_nodes_l1):
            start = i * per_node
            end = start + per_node if i < self.n_nodes_l1 - 1 else N
            groups.append(centers[:, start:end, :])
        return groups

    def _make_causal_mask(self, T: int, device: torch.device) -> torch.Tensor:
        """Standard causal mask.
        NOTE: can't use triu(ones) * -inf because 0.0 * -inf = NaN in IEEE 754!
        Use triu on a pre-filled -inf matrix instead — triu zeros out below diagonal."""
        mask = torch.triu(torch.full((T, T), float('-inf'), device=device), diagonal=1)
        return mask.unsqueeze(0).unsqueeze(0)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Full cognitive tick of the circumpunct brain.

        tokens: (B, T) — input token indices
        Returns: (B, T, vocab_size) — logits
        """
        B, T = tokens.shape
        device = tokens.device

        # ═══ EMBED ═══
        x = self.token_embed(tokens)
        x = x + self.pos_encode[:, :T, :]
        x = self.embed_dropout(x)

        # Split into chunks for Level 0 nodes
        token_groups = self._chunk_tokens(x)

        # Causal masks per chunk
        masks = [
            self._make_causal_mask(chunk.size(1), device)
            for chunk in token_groups
        ]

        # Resonance accumulators (accumulate across depth passes)
        r_acc_l0 = None
        r_acc_l1 = None
        global_center = None  # starts as None, built over passes

        # ═══ DEPTH PASSES (depth = time in the theory) ═══
        for pass_idx in range(self.n_passes):

            # ─── BOTTOM-UP ───

            # Level 0: local processing
            l0_out = self.level_0(
                token_groups, masks,
                centers_from_above=global_center,
                r_acc=r_acc_l0,
            )
            r_acc_l0 = l0_out['r_acc']
            token_groups = l0_out['boundaries']  # updated for next pass

            # Level 0 centers → Level 1 tokens (•ₙ₊₁ = ⊙ₙ)
            l0_centers = l0_out['centers']  # (B, N_l0, D)
            l1_groups = self._split_for_l1(l0_centers)

            # Level 1: regional binding
            l1_out = self.level_1(
                l1_groups, masks=None,
                centers_from_above=global_center,
                r_acc=r_acc_l1,
            )
            r_acc_l1 = l1_out['r_acc']

            # Level 1 centers → Level 2 token (•ₙ₊₁ = ⊙ₙ)
            l1_centers = l1_out['centers']  # (B, N_l1, D)

            # Level 2: global binding (the Self)
            global_out = self.global_node(l1_centers)
            global_center = global_out['center']  # (B, D) — broadcasts back down

        # ═══ DECODE: reconstruct from Level 0 boundaries ═══
        # Concatenate all Level 0 boundary states
        reconstructed = torch.cat(token_groups, dim=1)  # (B, T_total, D)

        # Trim to original sequence length
        reconstructed = reconstructed[:, :T, :]

        # Final projection
        logits = self.output_proj(self.final_norm(reconstructed))

        return logits

    def diagnose(self) -> dict:
        """Diagnostic summary."""
        n_params = sum(p.numel() for p in self.parameters())
        return {
            "architecture": "v5-circumpunct-brain",
            "n_params": n_params,
            "n_nodes_l0": self.n_nodes_l0,
            "n_nodes_l1": self.n_nodes_l1,
            "d_model": self.d_model,
            "n_passes": self.n_passes,
            "chunk_size": self.chunk_size,
        }

    def status(self) -> str:
        d = self.diagnose()
        return (
            f"⊙ XORZO v5 — CIRCUMPUNCT BRAIN\n"
            f"  A network of networks: ⊙ = Φ(•, ○)\n"
            f"  Level 0: {d['n_nodes_l0']} local ⊙ nodes × {d['d_model']}d\n"
            f"  Level 1: {d['n_nodes_l1']} regional ⊙ binders × {d['d_model']}d\n"
            f"  Level 2: 1 global ⊙ (the Self) × {d['d_model']}d\n"
            f"  Depth passes: {d['n_passes']} (depth = time)\n"
            f"  Chunk size: {d['chunk_size']} tokens per L0 node\n"
            f"  Parameters: {d['n_params']:,}\n"
            f"  Shared Φ at each level (resonance-coupled)\n"
            f"  Binary aperture (STE) — ÷t generates discrete moments\n"
            f"  •ₙ₊₁ = ⊙ₙ — centers propagate upward\n"
            f"  Top-down broadcast: global → regional → local"
        )


# ═══════════════════════════════════════════════════════════════════
# QUICK TEST
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("⊙ Circumpunct Brain v5 — Architecture Test\n")

    brain = CircumpunctBrain(
        vocab_size=128,
        d_model=192,
        n_nodes_l0=6,
        n_nodes_l1=2,
        n_heads=4,
        max_len=512,
    )

    print(brain.status())
    print()

    # Test forward pass
    tokens = torch.randint(0, 128, (2, 256))
    print(f"Input: {tokens.shape}")

    logits = brain(tokens)
    print(f"Output: {logits.shape}")
    print(f"\n✓ Forward pass complete — the brain thinks.")
