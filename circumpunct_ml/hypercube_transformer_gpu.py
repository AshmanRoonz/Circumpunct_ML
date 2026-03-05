"""
HYPERCUBE TRANSFORMER — PyTorch GPU Port
6D State Space Navigation as Attention

Original: Ashman Roonz, 2026 (NumPy prototype)
GPU port: for RTX 4070 training and head-to-head comparison with Xorzo v3

The core idea: attention isn't a scalar weight between tokens.
It's a probability distribution over 64 vertices of a 6D hypercube.
Each vertex is a relational state defined by 6 binary gates:
    b1, b2 (aperture)  — how open is the connection?
    c1, c2 (boundary)  — how structured is the interface?
    r1, r2 (resonance) — how phase-coherent is the coupling?

Attention = navigation through a graph of possible relationships.
Adjacent vertices (one bit flip) are naturally favored — you can't
jump from fully closed to fully open without passing through
intermediate states. The topology constrains the dynamics.

The hypercube IS the space of possible ⊙ configurations.
"""

import math
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Optional

PHI = (1 + math.sqrt(5)) / 2
PI = math.pi


# ═══════════════════════════════════════════════════════════════════
# THE 6D HYPERCUBE — precomputed geometric structure
# 64 vertices, 192 edges, Laplacian spectrum follows Pascal row 6
# ═══════════════════════════════════════════════════════════════════

class Hypercube6D:
    """
    Precomputes all geometric structure of the 6D hypercube.
    Returns torch buffers ready to register in nn.Modules.
    """
    NAMES = ["b1", "b2", "c1", "c2", "r1", "r2"]

    def __init__(self):
        # Vertex coordinates: 64 × 6 binary
        self.vertices = np.array([[(v >> i) & 1 for i in range(6)] for v in range(64)], dtype=np.float32)

        # Adjacency: edge iff Hamming distance = 1
        self.adjacency = np.zeros((64, 64), dtype=np.float32)
        for i in range(64):
            for d in range(6):
                self.adjacency[i, i ^ (1 << d)] = 1.0

        # Graph Laplacian
        self.laplacian = np.diag(self.adjacency.sum(1)) - self.adjacency

        # Spectral embedding (first 6 non-trivial eigenvectors)
        _, ev = np.linalg.eigh(self.laplacian)
        self.spectral_emb = ev[:, 1:7].astype(np.float32)  # (64, 6)

        # Openness layers (Pascal row 6)
        self.openness = {}
        for k in range(7):
            self.openness[k] = [v for v in range(64) if bin(v).count("1") == k]

        # Subcube masks for modal attention
        # Aperture vertices: bits 0,1 active
        self.aperture_mask = np.array([1 if (v & 0b000011) else 0 for v in range(64)], dtype=np.float32)
        # Boundary vertices: bits 2,3 active
        self.boundary_mask = np.array([1 if (v & 0b001100) else 0 for v in range(64)], dtype=np.float32)
        # Resonance vertices: bits 4,5 active
        self.resonance_mask = np.array([1 if (v & 0b110000) else 0 for v in range(64)], dtype=np.float32)

    def to_buffers(self):
        """Return dict of tensors to register as buffers."""
        return {
            "adjacency": torch.from_numpy(self.adjacency),
            "spectral_emb": torch.from_numpy(self.spectral_emb),
            "aperture_mask": torch.from_numpy(self.aperture_mask),
            "boundary_mask": torch.from_numpy(self.boundary_mask),
            "resonance_mask": torch.from_numpy(self.resonance_mask),
        }


# ═══════════════════════════════════════════════════════════════════
# TRIADIC EMBEDDING — aperture + field + boundary channels
# ═══════════════════════════════════════════════════════════════════

class TriadicEmbedding(nn.Module):
    """
    Three-channel embedding: aperture(1/3) + field(1/3) + boundary(1/3).
    Positional encoding added to boundary channel (position = boundary).
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 2048):
        super().__init__()
        self.d_model = d_model
        self.d_aperture = d_model // 3
        self.d_field = d_model // 3
        self.d_boundary = d_model - self.d_aperture - self.d_field

        self.embed_a = nn.Embedding(vocab_size, self.d_aperture)
        self.embed_f = nn.Embedding(vocab_size, self.d_field)
        self.embed_b = nn.Embedding(vocab_size, self.d_boundary)

        # Sinusoidal PE on boundary channel
        pe = torch.zeros(max_len, self.d_boundary)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_boundary, 2, dtype=torch.float) *
                        -(math.log(10000.0) / self.d_boundary))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:self.d_boundary // 2 + self.d_boundary % 2])
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_boundary)

        self.norm = nn.LayerNorm(d_model)

    def forward(self, tokens):
        B, S = tokens.shape
        a = self.embed_a(tokens)
        f = self.embed_f(tokens)
        b = self.embed_b(tokens) + self.pe[:, :S]
        return self.norm(torch.cat([a, f, b], dim=-1))


# ═══════════════════════════════════════════════════════════════════
# HYPERCUBE ATTENTION — the core mechanism
#
# Every token pair produces a distribution over 64 relational states.
# Adjacency bias constrains navigation (can't jump 6 bits at once).
# Spectral embedding encodes global geometry.
# Modal attention splits output by aperture/boundary/resonance subcubes.
# ═══════════════════════════════════════════════════════════════════

class HypercubeAttention(nn.Module):
    """
    Attention as navigation through a 6D hypercube of relational states.

    For each token pair (i, j):
        1. Project to vertex space: q_i, k_j → interaction tensor
        2. Score all 64 vertices via vertex embeddings + spectral + adjacency bias
        3. Softmax → π(v | i, j): probability of each relational state
        4. Gate values by vertex-specific gates
        5. Split into modal streams (aperture/boundary/resonance subcubes)
        6. Recombine
    """

    def __init__(self, d_model: int, d_vertex: int = 32, temperature: float = 1.0,
                 dropout: float = 0.1, adj_weight: float = 0.1, spectral_weight: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.d_vertex = d_vertex
        self.temperature = temperature
        self.adj_weight = adj_weight
        self.spectral_weight = spectral_weight

        # Q/K projections into vertex interaction space
        self.W_q = nn.Linear(d_model, d_vertex, bias=False)
        self.W_k = nn.Linear(d_model, d_vertex, bias=False)

        # Vertex embeddings: 64 relational states in vertex space
        self.vertex_emb = nn.Parameter(torch.randn(64, d_vertex) * 0.02)

        # Vertex gates: each state modulates the value flow differently
        self.vertex_gates = nn.Parameter(torch.randn(64, d_model) * 0.02)

        # Value projection
        self.W_v = nn.Linear(d_model, d_model, bias=False)

        # Spectral projection: map 6D spectral coords → vertex space
        self.W_spec = nn.Linear(6, d_vertex, bias=False)

        # Modal attention projections (aperture/boundary/resonance subcubes)
        d_third = d_model // 3
        self.W_aperture = nn.Linear(d_model, d_third, bias=False)
        self.W_boundary = nn.Linear(d_model, d_third, bias=False)
        self.W_resonance = nn.Linear(d_model, d_model - 2 * d_third, bias=False)

        # Output projection
        self.W_out = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Hypercube buffers — registered properly so they move with .to(device)
        # Initialized as empty, set by register_cube_buffers()
        self.register_buffer("_adj_bias", torch.zeros(64, 64))
        self.register_buffer("_spectral_emb", torch.zeros(64, 6))
        self.register_buffer("_aperture_mask", torch.zeros(64))
        self.register_buffer("_boundary_mask", torch.zeros(64))
        self.register_buffer("_resonance_mask", torch.zeros(64))

    def register_cube_buffers(self, cube_buffers):
        """Copy precomputed hypercube geometry into registered buffers."""
        self._adj_bias.copy_(cube_buffers["adjacency"] * 0.5)
        self._spectral_emb.copy_(cube_buffers["spectral_emb"])
        self._aperture_mask.copy_(cube_buffers["aperture_mask"])
        self._boundary_mask.copy_(cube_buffers["boundary_mask"])
        self._resonance_mask.copy_(cube_buffers["resonance_mask"])

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        x: (B, S, D)
        mask: (B, S, S) causal mask (1 = attend, 0 = mask)
        returns: (B, S, D), pi: (B, S, S, 64) vertex distributions
        """
        B, S, D = x.shape

        # Project to vertex interaction space
        q = self.W_q(x)  # (B, S, d_vertex)
        k = self.W_k(x)  # (B, S, d_vertex)

        # Interaction tensor: outer product in vertex space
        # (B, S, 1, dv) * (B, 1, S, dv) → (B, S, S, dv)
        inter = q.unsqueeze(2) * k.unsqueeze(1)

        # Score each vertex: interaction · vertex_embedding
        # (B, S, S, dv) @ (64, dv).T → (B, S, S, 64)
        v_scores = inter @ self.vertex_emb.T

        # Spectral bias: encode global hypercube geometry
        # spectral_emb: (64, 6) @ W_spec: (6, dv) → (64, dv)
        spec_proj = self._spectral_emb @ self.W_spec.weight.T  # (64, dv)
        spec_bias = inter @ spec_proj.T  # (B, S, S, 64)
        v_scores = v_scores + self.spectral_weight * spec_bias

        # Adjacency bias: favor navigation along edges
        # v_scores: (B, S, S, 64) @ adj: (64, 64) → (B, S, S, 64)
        v_scores = v_scores + self.adj_weight * (v_scores @ self._adj_bias)

        # Softmax over vertices → relational state distribution
        pi = F.softmax(v_scores / self.temperature, dim=-1)  # (B, S, S, 64)

        # Causal masking
        if mask is not None:
            pi = pi * mask.unsqueeze(-1)

        # Value flow gated by vertex states — MEMORY OPTIMIZED
        # Instead of materializing (B, S, S, D) weighted_gates,
        # collapse π over vertices first: (B, S, S, 64) @ (64, D) → collapse per-vertex
        # Then apply to values via standard attention pattern.
        #
        # Approach: reduce to scalar attention weights per (i,j) by averaging
        # gate contributions, then use standard (B, S, S) attention.
        gates = torch.sigmoid(self.vertex_gates)  # (64, D)
        gate_mean = gates.mean(dim=-1)  # (64,) — mean gate per vertex

        # Scalar attention: π(v|i,j) · gate_mean(v) → (B, S, S)
        scalar_attn = (pi * gate_mean.view(1, 1, 1, 64)).sum(dim=-1)  # (B, S, S)
        scalar_attn = scalar_attn / (scalar_attn.sum(dim=-1, keepdim=True) + 1e-8)
        scalar_attn = self.dropout(scalar_attn)

        # Apply to values: (B, S, S) @ (B, S, D) → (B, S, D)
        V = self.W_v(x)  # (B, S, D)
        out = torch.bmm(scalar_attn, V)

        # Modal attention: split by subcube
        def modal_attn(subcube_mask, W_modal):
            # Sum π over subcube vertices → modal attention weights
            # subcube_mask: (64,) → select vertices
            modal_pi = (pi * subcube_mask.view(1, 1, 1, 64)).sum(dim=-1)  # (B, S, S)
            modal_pi = modal_pi / (modal_pi.sum(dim=-1, keepdim=True) + 1e-8)
            if mask is not None:
                modal_pi = modal_pi * mask
                modal_pi = modal_pi / (modal_pi.sum(dim=-1, keepdim=True) + 1e-8)
            modal_pi = self.dropout(modal_pi)
            # (B, S, S) @ (B, S, d_modal) → (B, S, d_modal)
            return torch.bmm(modal_pi, W_modal(x))

        m_aperture = modal_attn(self._aperture_mask, self.W_aperture)
        m_boundary = modal_attn(self._boundary_mask, self.W_boundary)
        m_resonance = modal_attn(self._resonance_mask, self.W_resonance)

        modal_out = torch.cat([m_aperture, m_boundary, m_resonance], dim=-1)

        # Combine vertex attention + modal attention
        combined = out + modal_out
        result = self.W_out(combined)
        return self.norm(result), pi

    @property
    def vertex_distribution_entropy(self):
        """Mean entropy of vertex gates (diagnostic)."""
        gates = torch.sigmoid(self.vertex_gates)
        p = gates / (gates.sum(dim=0, keepdim=True) + 1e-8)
        ent = -(p * torch.log2(p + 1e-10)).sum(dim=0).mean()
        return ent.item()


# ═══════════════════════════════════════════════════════════════════
# FFN
# ═══════════════════════════════════════════════════════════════════

class HypercubeFFN(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        d_ff = d_model * 4
        self.W1 = nn.Linear(d_model, d_ff)
        self.W2 = nn.Linear(d_ff, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = F.gelu(self.W1(x))
        h = self.dropout(h)
        return self.norm(self.W2(h))


# ═══════════════════════════════════════════════════════════════════
# BLOCK — triadic decomposition + hypercube attention + FFN
# ═══════════════════════════════════════════════════════════════════

class HypercubeBlock(nn.Module):
    def __init__(self, d_model: int, d_vertex: int = 32, temperature: float = 1.0,
                 dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        d_a = d_model // 3
        d_f = d_model // 3
        d_b = d_model - d_a - d_f

        # Triadic decomposition projections
        self.W_decompose_a = nn.Linear(d_model, d_a, bias=False)
        self.W_decompose_f = nn.Linear(d_model, d_f, bias=False)
        self.W_decompose_b = nn.Linear(d_model, d_b, bias=False)

        self.attn = HypercubeAttention(d_model, d_vertex, temperature, dropout)
        self.ffn = HypercubeFFN(d_model, dropout)

    def register_cube_buffers(self, cube_buffers):
        self.attn.register_cube_buffers(cube_buffers)

    def forward(self, x, mask=None):
        # Triadic decomposition (for diagnostics / future use)
        a = self.W_decompose_a(x)
        f = self.W_decompose_f(x)
        b = self.W_decompose_b(x)

        # Attention with residual
        attn_out, pi = self.attn(x, mask)
        x = x + attn_out

        # FFN with residual
        x = x + self.ffn(x)

        return x, pi


# ═══════════════════════════════════════════════════════════════════
# HYPERCUBE TRANSFORMER — full model
# ═══════════════════════════════════════════════════════════════════

class HypercubeTransformerGPU(nn.Module):
    """
    Hypercube Transformer — 6D State Space Navigation.

    64 relational states as a graph. Attention is navigation.
    The hypercube IS the topology of possible relationships.
    """

    def __init__(
        self,
        vocab_size: int = 256,
        d_model: int = 192,
        n_layers: int = 6,
        d_vertex: int = 32,
        max_len: int = 512,
        dropout: float = 0.1,
        temperature: float = 1.0,
        generation: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.d_vertex = d_vertex
        self.generation = generation
        self.temperature = temperature

        # Build hypercube geometry
        cube = Hypercube6D()
        cube_buffers = cube.to_buffers()

        # Register cube geometry as module buffers (moves with .to(device))
        self.register_buffer("adjacency", cube_buffers["adjacency"])
        self.register_buffer("spectral_emb", cube_buffers["spectral_emb"])
        self.register_buffer("aperture_mask", cube_buffers["aperture_mask"])
        self.register_buffer("boundary_mask", cube_buffers["boundary_mask"])
        self.register_buffer("resonance_mask", cube_buffers["resonance_mask"])

        # Embedding
        self.embedding = TriadicEmbedding(vocab_size, d_model, max_len)
        self.embed_dropout = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            HypercubeBlock(d_model, d_vertex, temperature, dropout)
            for _ in range(n_layers)
        ])

        # Register cube buffers in each block's attention
        for block in self.blocks:
            block.register_cube_buffers({
                "adjacency": self.adjacency,
                "spectral_emb": self.spectral_emb,
                "aperture_mask": self.aperture_mask,
                "boundary_mask": self.boundary_mask,
                "resonance_mask": self.resonance_mask,
            })

        # Output
        self.final_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and 'vertex_emb' not in name and 'vertex_gates' not in name:
                nn.init.xavier_normal_(p, gain=1.0 / math.sqrt(PHI))

    def forward(self, x: torch.Tensor, return_pi: bool = False):
        """
        x: (B, S) token indices
        returns: (B, S, vocab_size) logits
        """
        B, S = x.shape

        # Embed
        h = self.embedding(x)
        h = self.embed_dropout(h)

        # Causal mask
        mask = torch.tril(torch.ones(S, S, device=x.device)).unsqueeze(0).expand(B, -1, -1)

        # Forward through blocks
        pis = []
        for block in self.blocks:
            h, pi = block(h, mask)
            pis.append(pi)

        # Output
        logits = self.output_proj(self.final_norm(h))

        if return_pi:
            return logits, pis
        return logits

    # ── Diagnostics ──

    def diagnose(self) -> dict:
        """Analyze the hypercube attention structure."""
        n_params = sum(p.numel() for p in self.parameters())

        # Vertex gate analysis
        all_entropies = []
        for block in self.blocks:
            all_entropies.append(block.attn.vertex_distribution_entropy)

        mean_entropy = sum(all_entropies) / len(all_entropies) if all_entropies else 0

        # Adjacency utilization — how much do vertex embeddings
        # respect the graph structure?
        v_emb = self.blocks[0].attn.vertex_emb.detach()
        v_sim = F.cosine_similarity(v_emb.unsqueeze(0), v_emb.unsqueeze(1), dim=-1)
        adj = self.adjacency
        adj_sim = (v_sim * adj).sum() / adj.sum()
        non_adj = 1.0 - adj - torch.eye(64, device=adj.device)
        non_adj_sim = (v_sim * non_adj).sum() / non_adj.sum()

        return {
            "generation": self.generation,
            "version": "hypercube-6d",
            "n_params": n_params,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_vertex": self.d_vertex,
            "mean_gate_entropy": mean_entropy,
            "adj_vertex_similarity": adj_sim.item(),
            "non_adj_vertex_similarity": non_adj_sim.item(),
            "locality_ratio": adj_sim.item() / (non_adj_sim.item() + 1e-8),
            "healthy": True,
        }

    def status(self) -> str:
        d = self.diagnose()
        d_a = self.d_model // 3
        d_f = self.d_model // 3
        d_b = self.d_model - d_a - d_f
        lines = [
            f"⬡ HYPERCUBE TRANSFORMER — Generation {d['generation']}",
            f"  6D State Space Navigation (64 vertices × 192 edges)",
            f"  d_model: {d['d_model']} = {d_a}(aperture) + {d_f}(field) + {d_b}(boundary)",
            f"  {d['n_layers']} layers × d_vertex={d['d_vertex']}",
            f"  Parameters: {d['n_params']:,}",
            f"",
            f"  Gate entropy: {d['mean_gate_entropy']:.3f} / 6.0 bits",
            f"  Vertex locality: {d['locality_ratio']:.2f}x (adj vs non-adj similarity)",
            f"  Adjacent similarity: {d['adj_vertex_similarity']:.4f}",
            f"  Non-adjacent similarity: {d['non_adj_vertex_similarity']:.4f}",
        ]
        return "\n".join(lines)

    # ── Save / Load / Evolve ──

    def save_generation(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / f"gen{self.generation}.pt")
        meta = {
            "generation": self.generation,
            "version": "hypercube-6d",
            "vocab_size": self.vocab_size,
            "d_model": self.d_model,
            "n_layers": self.n_layers,
            "d_vertex": self.d_vertex,
            "diagnosis": self.diagnose(),
        }
        (path / f"gen{self.generation}_meta.json").write_text(
            json.dumps(meta, indent=2, default=str)
        )

    @classmethod
    def evolve(cls, parent: 'HypercubeTransformerGPU', **mutations) -> 'HypercubeTransformerGPU':
        child = cls(
            vocab_size=mutations.get("vocab_size", parent.vocab_size),
            d_model=mutations.get("d_model", parent.d_model),
            n_layers=mutations.get("n_layers", parent.n_layers),
            d_vertex=mutations.get("d_vertex", parent.d_vertex),
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
        print(f"  ⬡ Generation {child.generation} born from {parent.generation}")
        print(f"    Inherited {inherited}/{len(child_state)} parameters")
        return child


# ═══════════════════════════════════════════════════════════════════
# TEXT GENERATION
# ═══════════════════════════════════════════════════════════════════

def generate(model, prompt, vocab, vocab_inv, max_tokens=200, temperature=0.7, device="cpu"):
    """Generate text from a trained model."""
    model.eval()
    tokens = [vocab.get(c, 0) for c in prompt]
    with torch.no_grad():
        for _ in range(max_tokens):
            x = torch.tensor([tokens[-512:]], dtype=torch.long, device=device)
            logits = model(x)
            next_logits = logits[0, -1] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()
            tokens.append(next_token)
            char = vocab_inv.get(next_token, "?")
            if char == "\n" and len(tokens) > len(prompt) + 50:
                break
    return "".join(vocab_inv.get(t, "?") for t in tokens)
