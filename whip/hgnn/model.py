"""Hamiltonian Graph Neural Network for bead chain dynamics.

The network learns the potential energy V(q) via message passing on the
chain graph. Kinetic energy T = |p|^2 / (2m) is hardcoded with known masses.
Total Hamiltonian H = T + V.

Equations of motion are obtained via autograd:
    dq/dt =  dH/dp
    dp/dt = -dH/dq

Rigid rod constraints are enforced exactly at each integration step via
Lagrange multiplier projection (following Finzi et al., NeurIPS 2020).
"""

import torch
import torch.nn as nn
import numpy as np


class ResMLPBlock(nn.Module):
    """MLP with LayerNorm and residual connection."""

    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden_dim) if in_dim != hidden_dim else nn.Identity()
        self.norm = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

    def forward(self, x):
        x = self.proj(x)
        return x + self.mlp(self.norm(x))


class PotentialEnergyGNN(nn.Module):
    """GNN that computes potential energy V(q) via message passing.

    Uses explicit left/right neighbor concatenation (not scatter_add_)
    to preserve directional information along the chain. Each node sees
    [self | left_neighbor | right_neighbor] at every message-passing round,
    matching the U-Net GNN's approach.

    Endpoints use zero-masking for the missing neighbor.
    """

    def __init__(self, n_beads=16, hidden_dim=64, n_message_passes=3):
        super().__init__()
        self.n_beads = n_beads
        self.hidden_dim = hidden_dim
        self.n_message_passes = n_message_passes

        # Node encoder: position (2) + mass (1) + is_fixed (1) = 4
        self.node_encoder = ResMLPBlock(4, hidden_dim)

        # Edge feature encoder: relative position (2) + distance (1) + rest_length (1) = 4
        self.edge_encoder = ResMLPBlock(4, hidden_dim)

        # Message passing: each node sees [self | left_msg | right_msg]
        # where each msg = edge_mlp([neighbor_node | edge_feat])
        self.msg_mlps = nn.ModuleList([
            ResMLPBlock(hidden_dim * 2, hidden_dim)  # [neighbor_h | edge_h] -> message
            for _ in range(n_message_passes)
        ])
        self.node_mlps = nn.ModuleList([
            ResMLPBlock(hidden_dim * 3, hidden_dim)  # [self | left_msg | right_msg]
            for _ in range(n_message_passes)
        ])

        # Per-node energy readout
        self.energy_readout = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # Precompute chain adjacency (set in _build_chain_adj)
        self._chain_adj = None

    def _build_chain_adj(self, n_beads, device):
        """Build (N, 2) adjacency: [left_neighbor, right_neighbor], -1 = missing."""
        adj = torch.full((n_beads, 2), -1, dtype=torch.long, device=device)
        for i in range(n_beads):
            if i > 0:
                adj[i, 0] = i - 1
            if i < n_beads - 1:
                adj[i, 1] = i + 1
        return adj

    def _compute_edge_and_neighbor_features(self, q, rest_lengths, mass, fixed_mask):
        """Compute per-node left/right neighbor and edge features.

        Missing neighbors (bead 0 has no left, bead N-1 has no right) are
        represented as ghost beads: mass=0, same position as the endpoint,
        so relative position=0, distance=0, rest_length=0. This gives the
        MLP physically meaningful "nothing here" features rather than
        arbitrary zeros.

        Returns:
            left_neighbor_feat: (B, N, 4) [q_left(2) | mass(1) | fixed(1)]
            right_neighbor_feat: (B, N, 4)
            left_edge_feat: (B, N, 4) [dq(2) | dist(1) | rest_len(1)]
            right_edge_feat: (B, N, 4)
        """
        B, N, _ = q.shape
        adj = self._chain_adj

        left_idx = adj[:, 0].clamp(min=0)
        right_idx = adj[:, 1].clamp(min=0)
        has_left = (adj[:, 0] >= 0)     # (N,) bool
        has_right = (adj[:, 1] >= 0)    # (N,) bool

        # --- Neighbor node features ---
        # Ghost bead: position = same as self, mass = 0, fixed = 0
        mass_expanded = mass.unsqueeze(0).unsqueeze(-1).expand(B, N, 1)
        fixed_expanded = fixed_mask.float().unsqueeze(0).unsqueeze(-1).expand(B, N, 1)

        # Left neighbor features
        left_q = torch.where(has_left.unsqueeze(0).unsqueeze(-1), q[:, left_idx, :], q)
        left_m = torch.where(has_left.unsqueeze(0).unsqueeze(-1), mass_expanded[:, left_idx, :],
                             torch.zeros_like(mass_expanded[:, :1, :]).expand(B, N, 1))
        left_f = torch.where(has_left.unsqueeze(0).unsqueeze(-1), fixed_expanded[:, left_idx, :],
                             torch.zeros_like(fixed_expanded[:, :1, :]).expand(B, N, 1))
        left_neighbor_feat = torch.cat([left_q, left_m, left_f], dim=-1)  # (B, N, 4)

        # Right neighbor features
        right_q = torch.where(has_right.unsqueeze(0).unsqueeze(-1), q[:, right_idx, :], q)
        right_m = torch.where(has_right.unsqueeze(0).unsqueeze(-1), mass_expanded[:, right_idx, :],
                              torch.zeros_like(mass_expanded[:, :1, :]).expand(B, N, 1))
        right_f = torch.where(has_right.unsqueeze(0).unsqueeze(-1), fixed_expanded[:, right_idx, :],
                              torch.zeros_like(fixed_expanded[:, :1, :]).expand(B, N, 1))
        right_neighbor_feat = torch.cat([right_q, right_m, right_f], dim=-1)  # (B, N, 4)

        # --- Edge features ---
        # Ghost edge: dq=0, dist=0, rest_length=0
        dq_left = left_q - q     # (B, N, 2) — zero for ghost
        dist_left = dq_left.norm(dim=-1, keepdim=True)
        dq_right = right_q - q
        dist_right = dq_right.norm(dim=-1, keepdim=True)

        # Rest lengths: rod i connects node i to i+1
        # Left rod of node i is rod (i-1), right rod is rod i
        rl_left = torch.zeros(N, device=q.device)
        rl_left[1:] = rest_lengths   # node 0 has no left rod -> 0
        rl_right = torch.zeros(N, device=q.device)
        rl_right[:N-1] = rest_lengths  # node N-1 has no right rod -> 0

        left_edge_feat = torch.cat([
            dq_left, dist_left,
            rl_left.unsqueeze(0).unsqueeze(-1).expand(B, N, 1),
        ], dim=-1)  # (B, N, 4)

        right_edge_feat = torch.cat([
            dq_right, dist_right,
            rl_right.unsqueeze(0).unsqueeze(-1).expand(B, N, 1),
        ], dim=-1)  # (B, N, 4)

        return left_neighbor_feat, right_neighbor_feat, left_edge_feat, right_edge_feat

    def forward(self, q, edge_index, rest_lengths, mass, fixed_mask):
        """Compute potential energy V(q).

        Args:
            q: (B, N, 2) positions
            edge_index: (2, E) bidirectional edge indices (unused, kept for API compat)
            rest_lengths: (n_rods,) rest length per rod
            mass: (N,) mass per node
            fixed_mask: (N,) bool, True = fixed node

        Returns:
            V: (B,) total potential energy
        """
        B, N, _ = q.shape

        # Build chain adjacency on first call (or if device changes)
        if self._chain_adj is None or self._chain_adj.device != q.device:
            self._chain_adj = self._build_chain_adj(N, q.device)

        adj = self._chain_adj
        left_idx = adj[:, 0].clamp(min=0)
        right_idx = adj[:, 1].clamp(min=0)

        # --- Node features ---
        node_feat = torch.cat([
            q,                                              # (B, N, 2)
            mass.unsqueeze(0).unsqueeze(-1).expand(B, N, 1),
            fixed_mask.float().unsqueeze(0).unsqueeze(-1).expand(B, N, 1),
        ], dim=-1)  # (B, N, 4)
        h_nodes = self.node_encoder(node_feat)  # (B, N, H)

        # --- Neighbor and edge features ---
        # Ghost beads (mass=0, same position) for missing neighbors
        left_nf, right_nf, left_ef, right_ef = \
            self._compute_edge_and_neighbor_features(q, rest_lengths, mass, fixed_mask)

        h_left_edge = self.edge_encoder(left_ef)    # (B, N, H)
        h_right_edge = self.edge_encoder(right_ef)  # (B, N, H)

        # Encode neighbor node features for message input
        h_left_neighbor = self.node_encoder(left_nf)    # (B, N, H)
        h_right_neighbor = self.node_encoder(right_nf)  # (B, N, H)

        # --- Message passing ---
        for k in range(self.n_message_passes):
            # Update neighbor hidden states from current h_nodes
            # (ghost beads keep their initial encoding — mass=0 features)
            h_left_cur = h_nodes[:, left_idx, :]    # (B, N, H)
            h_right_cur = h_nodes[:, right_idx, :]  # (B, N, H)

            # For endpoints, use the ghost-bead encoding instead
            has_left = (adj[:, 0] >= 0).unsqueeze(0).unsqueeze(-1)   # (1, N, 1)
            has_right = (adj[:, 1] >= 0).unsqueeze(0).unsqueeze(-1)  # (1, N, 1)
            h_left_cur = torch.where(has_left, h_left_cur, h_left_neighbor)
            h_right_cur = torch.where(has_right, h_right_cur, h_right_neighbor)

            # Left message: [left_neighbor_h | left_edge_h]
            left_msg = self.msg_mlps[k](torch.cat([h_left_cur, h_left_edge], dim=-1))

            # Right message: [right_neighbor_h | right_edge_h]
            right_msg = self.msg_mlps[k](torch.cat([h_right_cur, h_right_edge], dim=-1))

            # Node update: [self | left_msg | right_msg]
            h_nodes = self.node_mlps[k](torch.cat([h_nodes, left_msg, right_msg], dim=-1))

        # --- Energy readout ---
        per_node_energy = self.energy_readout(h_nodes).squeeze(-1)  # (B, N)
        V = per_node_energy.sum(dim=-1)  # (B,)

        return V


class HamiltonianGNN(nn.Module):
    """Full Hamiltonian: H(q, p) = T(p) + V(q).

    T is hardcoded as |p|^2 / (2m).
    V is learned by PotentialEnergyGNN.
    """

    def __init__(self, n_beads=16, hidden_dim=64, n_message_passes=3,
                 mass=None, rest_lengths=None, edge_index=None, fixed_mask=None):
        super().__init__()
        self.n_beads = n_beads

        self.potential_net = PotentialEnergyGNN(n_beads, hidden_dim, n_message_passes)

        # Register known physical quantities as buffers (not parameters)
        if mass is not None:
            self.register_buffer("mass", torch.tensor(mass, dtype=torch.float32))
        if rest_lengths is not None:
            self.register_buffer("rest_lengths", torch.tensor(rest_lengths, dtype=torch.float32))
        if edge_index is not None:
            self.register_buffer("edge_index", torch.tensor(edge_index, dtype=torch.long))
        if fixed_mask is not None:
            self.register_buffer("fixed_mask", torch.tensor(fixed_mask, dtype=torch.bool))

    def kinetic_energy(self, p):
        """T = sum_i |p_i|^2 / (2 * m_i).  Shape: (B,)."""
        # p: (B, N, 2), mass: (N,)
        return (0.5 * (p ** 2).sum(dim=-1) / self.mass.unsqueeze(0)).sum(dim=-1)

    def hamiltonian(self, q, p):
        """H(q, p) = T(p) + V(q).  Shape: (B,)."""
        T = self.kinetic_energy(p)
        V = self.potential_net(q, self.edge_index, self.rest_lengths, self.mass, self.fixed_mask)
        return T + V

    def time_derivatives(self, q, p):
        """Compute dq/dt, dp/dt via Hamilton's equations using autograd.

        dq/dt =  dH/dp
        dp/dt = -dH/dq
        """
        q = q.detach().requires_grad_(True)
        p = p.detach().requires_grad_(True)

        H = self.hamiltonian(q, p).sum()

        dH_dq, dH_dp = torch.autograd.grad(H, [q, p], create_graph=self.training)

        dq_dt = dH_dp    # Hamilton's equation
        dp_dt = -dH_dq   # Hamilton's equation

        return dq_dt, dp_dt


class ConstrainedDynamics:
    """Applies Lagrange multiplier projection to enforce rigid rod constraints.

    Given unconstrained dynamics (dq/dt, dp/dt) from the Hamiltonian,
    projects them onto the constraint manifold so that rod lengths are
    maintained exactly.
    """

    def __init__(self, edge_pairs, rest_lengths, mass, fixed_mask):
        """
        Args:
            edge_pairs: (n_rods, 2) undirected rod connectivity
            rest_lengths: (n_rods,) rest length per rod
            mass: (N,) mass per node
            fixed_mask: (N,) bool
        """
        if not isinstance(edge_pairs, torch.Tensor):
            edge_pairs = torch.tensor(edge_pairs, dtype=torch.long)
        self.edge_pairs = edge_pairs       # (n_rods, 2)
        self.rest_lengths = rest_lengths   # (n_rods,)
        self.mass = mass                   # (N,)
        self.fixed_mask = fixed_mask       # (N,)
        # Pre-split for indexing
        self.i_idx = edge_pairs[:, 0]
        self.j_idx = edge_pairs[:, 1]

    def constraint_violations(self, q):
        """Phi(q) = ||q_j - q_i||^2 - L0^2 for each rod.  Shape: (B, C)."""
        i_idx, j_idx = self.i_idx, self.j_idx
        dq = q[:, j_idx, :] - q[:, i_idx, :]   # (B, C, 2)
        dist_sq = (dq ** 2).sum(dim=-1)          # (B, C)
        L0_sq = self.rest_lengths ** 2            # (C,)
        return dist_sq - L0_sq

    def project_derivatives(self, q, p, dq_dt, dp_dt):
        """Project (dq_dt, dp_dt) onto the constraint tangent space.

        Uses the SHAKE-style projection:
            dz/dt_projected = dz/dt - J @ DPhi @ lambda

        where lambda is chosen so that d/dt(Phi) = 0.
        """
        B, N, d = q.shape
        i_idx, j_idx = self.i_idx, self.j_idx

        dq_ij = q[:, j_idx, :] - q[:, i_idx, :]  # (B, C, 2)

        # Constraint Jacobian dPhi/dq: for constraint c, node i and j
        # dPhi_c/dq_i = -2 * (q_j - q_i) = -2 * dq_ij
        # dPhi_c/dq_j = +2 * (q_j - q_i) = +2 * dq_ij

        # Time derivative of constraints:
        # dot_Phi_c = 2 * (q_j - q_i) . (dq_j/dt - dq_i/dt)
        # We want this to be zero after projection.

        # For the symplectic projection, we need:
        # DPhi^T @ J @ DPhi and DPhi^T @ J @ gradH
        # But for our chain it's simpler to work directly:

        # The velocity constraint: (q_j - q_i) . (v_j - v_i) = 0
        # where v = dq/dt = dH/dp = p/m
        # After applying a correction lambda_c along (q_j - q_i):
        #   v_i -> v_i + lambda_c * (q_j - q_i) / m_i
        #   v_j -> v_j - lambda_c * (q_j - q_i) / m_j

        inv_mass = torch.where(
            self.fixed_mask,
            torch.zeros_like(self.mass),
            1.0 / self.mass,
        )  # (N,)

        # Current velocity from dq_dt
        dv_ij = dq_dt[:, j_idx, :] - dq_dt[:, i_idx, :]  # (B, C, 2)

        # dot_Phi = 2 * dq_ij . dv_ij  (we drop the factor of 2, it cancels)
        dot_phi = (dq_ij * dv_ij).sum(dim=-1)  # (B, C)

        # Denominator: dq_ij . dq_ij * (1/m_i + 1/m_j)
        w_sum = inv_mass[i_idx] + inv_mass[j_idx]  # (C,)
        denom = (dq_ij ** 2).sum(dim=-1) * w_sum.unsqueeze(0)  # (B, C)
        denom = denom.clamp(min=1e-12)

        # Lagrange multipliers
        lam = dot_phi / denom  # (B, C)

        # Apply corrections (vectorized, no Python loop over constraints)
        # correction per constraint: lam * dq_ij  (B, C, 2)
        correction = lam.unsqueeze(-1) * dq_ij  # (B, C, 2)

        # For non-fixed nodes: m * inv_mass = 1, so dp correction = correction
        # dq correction = inv_mass * correction
        inv_mass_i = inv_mass[i_idx].unsqueeze(0).unsqueeze(-1)  # (1, C, 1)
        inv_mass_j = inv_mass[j_idx].unsqueeze(0).unsqueeze(-1)  # (1, C, 1)

        dq_dt_proj = dq_dt.clone()
        dp_dt_proj = dp_dt.clone()

        # Scatter corrections to node i (+) and node j (-)
        # Use index_add_ for vectorized accumulation
        corr_qi = (inv_mass_i * correction).to(dq_dt.dtype)  # (B, C, 2)
        corr_qj = (inv_mass_j * correction).to(dq_dt.dtype)  # (B, C, 2)

        for b in range(B):
            dq_dt_proj[b].index_add_(0, i_idx, corr_qi[b])
            dq_dt_proj[b].index_add_(0, j_idx, -corr_qj[b])
            dp_dt_proj[b].index_add_(0, i_idx, correction[b])
            dp_dt_proj[b].index_add_(0, j_idx, -correction[b])

        return dq_dt_proj, dp_dt_proj


def leapfrog_step(hamiltonian_gnn, constraints, q, p, dt):
    """One leapfrog (Störmer-Verlet) integration step.

    Symplectic integrator that preserves the Hamiltonian structure:
        p_{1/2}  = p_n     + (dt/2) * dp/dt(q_n)
        q_{n+1}  = q_n     + dt * dq/dt(p_{1/2})
        p_{n+1}  = p_{1/2} + (dt/2) * dp/dt(q_{n+1})

    Constraint projection is applied after position and momentum updates.
    """
    # Half-step momentum
    dq_dt, dp_dt = hamiltonian_gnn.time_derivatives(q, p)
    if constraints is not None:
        dq_dt, dp_dt = constraints.project_derivatives(q, p, dq_dt, dp_dt)
    p_half = p + 0.5 * dt * dp_dt

    # Full-step position
    dq_dt2, _ = hamiltonian_gnn.time_derivatives(q, p_half)
    if constraints is not None:
        dq_dt2, _ = constraints.project_derivatives(q, p_half, dq_dt2, torch.zeros_like(dp_dt))
    q_new = q + dt * dq_dt2

    # Half-step momentum again
    _, dp_dt2 = hamiltonian_gnn.time_derivatives(q_new, p_half)
    if constraints is not None:
        _, dp_dt2 = constraints.project_derivatives(q_new, p_half, dq_dt2, dp_dt2)
    p_new = p_half + 0.5 * dt * dp_dt2

    # Re-pin fixed nodes
    fixed = hamiltonian_gnn.fixed_mask
    q_new[:, fixed, :] = q[:, fixed, :]
    p_new[:, fixed, :] = 0.0

    return q_new, p_new


def integrate_trajectory(hamiltonian_gnn, constraints, q0, p0, dt, n_steps):
    """Integrate for n_steps using leapfrog, returning all states.

    Returns:
        qs: (B, n_steps+1, N, 2)
        ps: (B, n_steps+1, N, 2)
    """
    qs = [q0]
    ps = [p0]
    q, p = q0, p0

    for _ in range(n_steps):
        q, p = leapfrog_step(hamiltonian_gnn, constraints, q, p, dt)
        qs.append(q)
        ps.append(p)

    return torch.stack(qs, dim=1), torch.stack(ps, dim=1)


def build_edge_index(n_beads):
    """Build bidirectional edge index for the chain graph."""
    fwd = np.array([[i, i + 1] for i in range(n_beads - 1)])
    bwd = fwd[:, ::-1].copy()
    edges = np.concatenate([fwd, bwd], axis=0).T  # (2, 2*(n-1))
    return edges


def build_edge_pairs(n_beads):
    """Build undirected edge pairs for constraints."""
    return np.array([[i, i + 1] for i in range(n_beads - 1)])
