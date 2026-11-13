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


class HeatConductanceMPGNN(HeatFluxMPGNN):
    """Stable-by-construction flux: DOWN-GRADIENT conductance transport + source.

    The free half-flux psi in HeatFluxMPGNN conserves energy but can be
    ANTI-diffusive (heat flowing up-gradient), which conserves while GROWING an
    oscillation -- the checkerboard blow-up (negative temperatures) seen on the
    long fine-cadence aw2 rollout. A merely non-negative down-gradient flux is
    not enough either: at a super-step ~200x the solver's CFL limit, even a
    correct-sign but too-LARGE conductance overshoots (explicit diffusion number
    >> 1/2) and diverges. So here each face conductance is BOUNDED so the update
    is a discrete convex combination of the cell and its neighbors:

        Phi_c(self -> nbr) = D_c * (s_self,c - s_nbr,c),   D_c = cap * sigmoid(.)

    with cap = 1 / (max interior degree) = 1/2 in 1D. Then
        T_i^new = (1 - sum_j D)*T_i + sum_j D*T_j + source_i,
    a weighted average (transport) plus a local source. Because sum_j D <= 1 and
    every weight >= 0, the transported part can never leave [min, max] of the
    stencil -- a discrete maximum principle -> UNCONDITIONALLY STABLE at any dt,
    no CFL limit. D_c is SYMMETRIC in (self, nbr), so the flux stays
    antisymmetric (conservation) and vanishes for s_self = s_nbr (zero baseline).
    A per-cell source (local pyrolysis) is untouched. Same forward/step contract.

    The price of stability at K=1 reach is under-diffusion (one step averages
    only with immediate neighbors), i.e. a bias, not a blow-up; wider reach
    (n_message_passes) trades that back."""

    def __init__(self, node_dim, edge_dim, output_dim,
                 hidden_dim=64, n_message_passes=1, out_in_cols=None,
                 face_cap=0.5, uniform_cond=False, physical_source=False,
                 t_in_col=0, rho_in_col=1, rhov_bulk=1.0, rhoc_bulk=1.0,
                 fixed_cond=None, kinetics=None):
        super().__init__(node_dim, edge_dim, output_dim,
                         hidden_dim=hidden_dim, n_message_passes=n_message_passes)
        # conductance MLP replaces the free flux_mlp; source_mlp is inherited.
        self.cond_mlp = nn.Sequential(
            ResMLPBlock(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim),
        )
        # Spatially-uniform (state-independent) per-channel conductance. A
        # state-dependent MLP conductance can switch itself off in cool cells,
        # freezing the diffusion front; a per-channel scalar cannot, so the
        # front grows as sqrt(t) like Fourier conduction. It still gives zero
        # flux in uniform regions (flux ~ grad), so it never over-smooths the
        # quiescent interior. Anchored near the physical diffusion number.
        self.uniform_cond = uniform_cond
        self.cond_logit = nn.Parameter(torch.zeros(output_dim))
        self.face_cap = face_cap                   # per-face weight cap (1/degree)
        # "Learnable around the physical anchor": if a uniform (learnable) cond is
        # paired with an anchor (fixed_cond values), start the logit AT the anchor
        # so training relaxes the physical value rather than starting cold at 0.25.
        if uniform_cond and fixed_cond is not None:
            frac = (torch.tensor(fixed_cond, dtype=torch.float32) / face_cap
                    ).clamp(1e-4, 1 - 1e-4)
            with torch.no_grad():
                self.cond_logit.copy_(torch.log(frac / (1 - frac)))
        # Fixed (non-learnable) per-channel conductance, anchored at the physical
        # diffusion number. A learned conductance always collapses to under-
        # diffusion because a short BPTT window cannot see the boundary layer
        # accumulate over thousands of steps; fixing it makes the front grow
        # correctly by construction, leaving only the source to learn.
        # When uniform_cond is set, fixed_cond is only the init anchor (above), and
        # the learnable logit governs conductance -- so do not take the fixed path.
        self.has_fixed_cond = (fixed_cond is not None) and not uniform_cond
        if self.has_fixed_cond:
            self.register_buffer("cond_fixed",
                                 torch.tensor(fixed_cond, dtype=torch.float32))
        if out_in_cols is None:
            out_in_cols = list(range(output_dim))
        self.register_buffer("out_in_cols",
                             torch.tensor(out_in_cols, dtype=torch.long))

        # --- Physically-structured pyrolysis source ---
        # A free per-cell source MLP learns the WRONG shape (largest for cold
        # cells, no Arrhenius onset) and, unconstrained, drives runaway. Replace
        # it with a source whose nonlinearity is built in: a per-cell reaction
        # rate that turns on with temperature (a soft Arrhenius gate) and is
        # self-limiting (proportional to remaining virgin mass, so it stops once
        # a cell is fully charred -> cannot run away). One shared rate drives
        # every predicted channel through a learned coefficient, which couples
        # temperature, char species and gas the way real pyrolysis does.
        #     rate = sigmoid((T - T0)/w) * clamp((rho - rhoc)/(rhov - rhoc), 0, 1)
        #     source_c = coef_c * rate
        self.physical_source = physical_source
        self.pyro_T0 = nn.Parameter(torch.tensor(800.0))       # onset temperature
        self.pyro_logw = nn.Parameter(torch.tensor(4.0))       # onset width (softplus)
        self.pyro_coef = nn.Parameter(torch.zeros(output_dim))  # per-channel coupling
        self.register_buffer("t_in_col", torch.tensor(int(t_in_col)))
        self.register_buffer("rho_in_col", torch.tensor(int(rho_in_col)))
        self.register_buffer("rhov_bulk", torch.tensor(float(rhov_bulk)))
        self.register_buffer("rhoc_bulk", torch.tensor(float(rhoc_bulk)))

        # Real per-species Arrhenius kinetics (the solver's DECOMP), aligned to
        # the OUTPUT channels: char channels decompose exactly, non-char channels
        # (A=0, is_char=False) are driven by the total decomposition rate through
        # a learned coupling. This replaces the crude single sigmoid gate, whose
        # onset the accumulation-blind loss could not pin down.
        self.real_kinetics = kinetics is not None
        if self.real_kinetics:
            def _b(key):
                return torch.tensor(kinetics[key], dtype=torch.float32)
            self.register_buffer("kin_A", _b("A"))
            self.register_buffer("kin_E", _b("E"))
            self.register_buffer("kin_psi", _b("psi"))
            self.register_buffer("kin_Tmin", _b("Tmin"))
            self.register_buffer("kin_rhov", _b("rhov"))
            self.register_buffer("kin_rhoc", _b("rhoc"))
            self.register_buffer("kin_is_char", _b("is_char") > 0.5)
            self.register_buffer("dt_sub", torch.tensor(float(kinetics["dt_sub"])))

    def _pyro_source(self, node_feat):
        """Structured pyrolysis source (B, output_dim), in forward (y_std) units."""
        self_abs = node_feat * self.node_std + self.node_mean   # de-normalize
        if not self.real_kinetics:
            # Crude single sigmoid gate (kept for comparison).
            Tp = self_abs[..., self.t_in_col]
            rhop = self_abs[..., self.rho_in_col]
            w = nn.functional.softplus(self.pyro_logw) + 1.0
            gate = torch.sigmoid((Tp - self.pyro_T0) / w)
            virgin = ((rhop - self.rhoc_bulk)
                      / (self.rhov_bulk - self.rhoc_bulk)).clamp(0.0, 1.0)
            return self.pyro_coef.unsqueeze(0) * (gate * virgin).unsqueeze(-1)

        # Real per-species Arrhenius decomposition (mirrors pyrolysis.step_species).
        # These kinetics have NO learnable parameters, so compute them WITHOUT a
        # gradient -- the W^(1-psi) term has exploding gradients (W^-3) for nearly
        # charred cells and would NaN through BPTT. Only the T/gas coupling
        # coefficient (pyro_coef) is learned, applied outside the no-grad block.
        with torch.no_grad():
            T = self_abs[..., self.t_in_col].unsqueeze(-1)          # (B, 1)
            rho_ch = self_abs[..., self.out_in_cols]                # (B, output_dim)
            span = (self.kin_rhov - self.kin_rhoc).clamp(min=1e-6)
            W = ((rho_ch - self.kin_rhoc) / span).clamp(1e-6, 1.0)  # virgin fraction
            arg = self.kin_A * self.dt_sub * torch.exp(-self.kin_E / T)
            onemp = 1.0 - self.kin_psi
            W_psi1 = W * torch.exp(-arg)                            # psi == 1 branch
            term = (W.pow(onemp) + (self.kin_psi - 1.0) * arg).clamp(min=1e-6)
            W_psin = term.pow(1.0 / onemp)                         # psi != 1 branch
            W_new = torch.where(self.kin_psi == 1.0, W_psi1, W_psin)
            active = (T > self.kin_Tmin) & self.kin_is_char
            W_new = torch.where(active, W_new, W)
            d_rho = (W_new - W) * span                             # <=0 for char, 0 else
            dm = (-d_rho * self.kin_is_char.float()).sum(-1, keepdim=True)  # (B,1)>=0
        # char channels take their exact delta; T/gas take a learned coupling to dm.
        phys = torch.where(self.kin_is_char, d_rho,
                           self.pyro_coef.unsqueeze(0) * dm)
        return phys / self.y_std

    def _conductance(self, h_self, h_nbr, g):
        """Symmetric per-face conductance in [0, face_cap] (bounded so the
        per-cell update is a stable convex combination)."""
        if self.has_fixed_cond:
            return self.cond_fixed.expand(h_self.shape[0], -1)
        if self.uniform_cond:
            D = self.face_cap * torch.sigmoid(self.cond_logit)   # (output_dim,)
            return D.expand(h_self.shape[0], -1)
        a = self.cond_mlp(torch.cat([h_self, h_nbr, g], dim=-1))
        b = self.cond_mlp(torch.cat([h_nbr, h_self, g], dim=-1))
        return self.face_cap * torch.sigmoid(a + b)

    def forward(self, node_feat, left_edge, right_edge, has_left, has_right):
        F = self.node_dim
        nbr_l = self._neighbor_node_feat(node_feat, left_edge)
        nbr_r = self._neighbor_node_feat(node_feat, right_edge)
        h_self = self.node_encoder(node_feat)
        h_left = self.node_encoder(nbr_l)
        h_right = self.node_encoder(nbr_r)
        g_left = self.geom_encoder(left_edge[..., F:F + 1])
        g_right = self.geom_encoder(right_edge[..., F:F + 1])

        s_self = node_feat[..., self.out_in_cols]
        # Down-gradient flux LEAVING self toward each neighbor.
        flux_left = self._conductance(h_self, h_left, g_left) * (
            s_self - nbr_l[..., self.out_in_cols])
        flux_right = self._conductance(h_self, h_right, g_right) * (
            s_self - nbr_r[..., self.out_in_cols])
        zeros = torch.zeros_like(flux_right)
        flux_right = torch.where(has_right.unsqueeze(-1), flux_right, zeros)
        flux_left = torch.where(has_left.unsqueeze(-1), flux_left, zeros)

        source = self._pyro_source(node_feat) if self.physical_source \
            else self.source_mlp(h_self)
        return -(flux_right + flux_left) + source
