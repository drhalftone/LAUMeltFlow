"""Flux-form (conservative) variant of the heat-shield surrogate.

White-paper Sec 4.3/6: the direct per-cell state-delta model (gnn_model.py)
carries a per-step bias on quiescent cells and fails to generalize across
forcings on the stiff gas case. This model predicts FLUXES on cell faces and
takes their divergence, plus a separate per-cell SOURCE term:

    delta_i = -(Phi(i, i+1) + Phi(i, i-1))  +  source_i
              [------ conservative transport ------]   [-- local --]

where Phi(i, j) is the flux LEAVING cell i across the face toward neighbor j.
Phi is built ANTISYMMETRIC, Phi(a, b) = -Phi(b, a), via a directed half-flux:

    Phi(a, b) = psi(h_a, h_b, g) - psi(h_b, h_a, g)

Two properties hold BY CONSTRUCTION:
  1. Conservation: cell i's outgoing flux across a face equals the negative of
     its neighbor's outgoing flux across that same face, so what leaves i enters
     its neighbor. (Both cells compute the same psi on the same encoded states.)
  2. Zero baseline: a uniform field has h_a = h_b, so Phi = 0. Quiescent cells
     get zero transport change -- the bias channel is removed.

Local change that is NOT neighbor-gradient-driven (pyrolysis: a local Arrhenius
source depending on the cell's own temperature) flows through source_i, the
physically correct place for it.

Same per-cell forward interface as HeatMPGNN, so the train/rollout plumbing is
reused. Each cell reconstructs its neighbors' ABSOLUTE state from [self, relative
edge] using the normalization buffers, so conservation holds even though the
forward runs one cell at a time. The output is the delta in y_std units with NO
mean offset (y_mean buffer stays 0), so the zero baseline survives denormalization.
"""

import torch
import torch.nn as nn

from .gnn_model import ResMLPBlock


class HeatFluxMPGNN(nn.Module):
    """Conservative flux + local source per-cell predictor. Weight-shared across
    cells (any mesh length), drop-in for HeatMPGNN's forward/step_mesh."""

    def __init__(self, node_dim, edge_dim, output_dim,
                 hidden_dim=64, n_message_passes=1):
        super().__init__()
        self.node_dim = node_dim
        self.edge_dim = edge_dim                  # = node_dim + 1 (rel state + dx)
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_message_passes = n_message_passes  # kept for ckpt compat (K=1 reach)

        self.node_encoder = ResMLPBlock(node_dim, hidden_dim)
        self.geom_encoder = ResMLPBlock(1, hidden_dim)        # per-face dx (symmetric)
        self.flux_mlp = nn.Sequential(                        # directed half-flux psi
            ResMLPBlock(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        self.source_mlp = nn.Sequential(                      # local per-cell source
            ResMLPBlock(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )

        self.register_buffer("node_mean", torch.zeros(node_dim))
        self.register_buffer("node_std", torch.ones(node_dim))
        self.register_buffer("edge_mean", torch.zeros(edge_dim))
        self.register_buffer("edge_std", torch.ones(edge_dim))
        self.register_buffer("y_mean", torch.zeros(output_dim))   # stays 0 (zero baseline)
        self.register_buffer("y_std", torch.ones(output_dim))

    def _half_flux(self, h_self, h_nbr, g):
        """psi(h_self, h_nbr, g) -> (B, output_dim) directed half-flux."""
        return self.flux_mlp(torch.cat([h_self, h_nbr, g], dim=-1))

    def _face_flux(self, h_self, h_nbr, g):
        """Antisymmetric flux LEAVING self toward nbr: Phi = psi(s,n) - psi(n,s)."""
        return self._half_flux(h_self, h_nbr, g) - self._half_flux(h_nbr, h_self, g)

    def _neighbor_node_feat(self, node_feat, edge):
        """Reconstruct the neighbor's normalized node features from this cell's
        normalized node features and the normalized relative edge.

        edge (normalized) = [(nbr - self)_phys, dx_phys] scaled by edge stats.
        Returns the neighbor's features in the same normalized space node_feat is in."""
        F = self.node_dim
        self_abs = node_feat * self.node_std + self.node_mean
        rel_phys = edge[..., :F] * self.edge_std[:F] + self.edge_mean[:F]
        nbr_abs = self_abs + rel_phys
        return (nbr_abs - self.node_mean) / self.node_std

    def forward(self, node_feat, left_edge, right_edge, has_left, has_right):
        """Per-cell forward. Inputs normalized (as in HeatMPGNN). Returns the
        normalized (zero-mean) per-cell delta (B, output_dim)."""
        F = self.node_dim
        h_self = self.node_encoder(node_feat)
        h_left = self.node_encoder(self._neighbor_node_feat(node_feat, left_edge))
        h_right = self.node_encoder(self._neighbor_node_feat(node_feat, right_edge))
        g_left = self.geom_encoder(left_edge[..., F:F + 1])
        g_right = self.geom_encoder(right_edge[..., F:F + 1])

        flux_right = self._face_flux(h_self, h_right, g_right)   # leaving self -> right
        flux_left = self._face_flux(h_self, h_left, g_left)      # leaving self -> left
        zeros = torch.zeros_like(flux_right)
        flux_right = torch.where(has_right.unsqueeze(-1), flux_right, zeros)
        flux_left = torch.where(has_left.unsqueeze(-1), flux_left, zeros)

        source = self.source_mlp(h_self)
        return -(flux_right + flux_left) + source               # net accumulation + source

    @torch.no_grad()
    def step_mesh(self, node_state, dx, out_idx):
        """One surrogate step over a full mesh (rollout mode). Identical contract
        to HeatMPGNN.step_mesh: advance interior cells 1..m-2; ghosts left to the
        caller. y_mean is 0 here, so the denorm is just * y_std."""
        device = node_state.device
        m = node_state.shape[0]
        interior = torch.arange(1, m - 1, device=device)

        self_f = node_state[interior]
        left_f = node_state[interior - 1]
        right_f = node_state[interior + 1]
        dx_int = dx[interior]

        left_edge = torch.cat([left_f - self_f, dx_int.unsqueeze(-1)], dim=-1)
        right_edge = torch.cat([right_f - self_f, dx_int.unsqueeze(-1)], dim=-1)
        has = torch.ones(interior.shape[0], dtype=torch.bool, device=device)

        nf = (self_f - self.node_mean) / self.node_std
        le = (left_edge - self.edge_mean) / self.edge_std
        re = (right_edge - self.edge_mean) / self.edge_std

        delta = self.forward(nf, le, re, has, has) * self.y_std + self.y_mean

        new_interior = self_f.clone()
        new_interior[:, out_idx] = self_f[:, out_idx] + delta
        return new_interior
