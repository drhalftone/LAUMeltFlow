"""Multi-step (rollout) training for the heat-shield surrogate.

Single-step training (train_gnn.py) only ever asks "given the TRUE state, predict
one step." This trainer instead unrolls the model M steps -- feeding its own
predictions back in -- and puts the loss on the whole short rollout. The model
therefore optimizes the thing we actually care about (trajectory accuracy) and
learns to cope with its own imperfect inputs. We backprop through the unroll
(backprop-through-time), so the gradient "knows" how an early-step error hurts
later steps.

Because a cell's next value depends on its neighbors' next values, the step must
advance the WHOLE mesh together (not independent cells like single-step training).
The boundary forcing (ghost cells) is re-imposed from ground truth each step --
the BC is a known input, not something the model predicts.

Warm-starts from a stable single-step model (the noise-trained one) and
fine-tunes; multi-step from random init can diverge.

    python -m heat_python.gnn.train_rollout --data heat_python/data/aw1_forcing_dataset.npz \
        --init heat_python/models/heat_mpgnn_noise.pt --rollout-steps 4 \
        --out heat_python/models/heat_mpgnn_rollout.pt
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .graph import to_neighbor_samples
from .gnn_model import HeatMPGNN
from .gnn_flux_model import HeatFluxMPGNN, HeatConductanceMPGNN
from .train_gnn import select_out_cols
from ..case import load_case
from ..pyrolysis import load_solid


def mesh_step(model, state, dx, stats, consts, cols):
    """Advance a batch of full meshes one step (differentiable). Returns the
    new INTERIOR state (B, n, F); the caller re-imposes ghosts from the BC.

    state (B, m, F) includes the two ghost cells at indices 0 and m-1.
    Built without in-place edits so autograd can backprop through it.

    Column-agnostic: predicts a delta for `cols.out` (e.g. [T, rho_i.., pg,
    mdotf]); derives rho (= sum rho_w*rho_i) and porosity; leaves inert species
    and any other columns fixed. Works for aw1 (gas off) and aw2 (gas on)."""
    node_mean, node_std, edge_mean, edge_std, y_mean, y_std = stats
    rho_w, phi, phi_c, rhov_b, rhoc_b = consts
    B, m, F = state.shape
    n = m - 2

    self_f = state[:, 1:m - 1, :]
    left_f = state[:, 0:m - 2, :]
    right_f = state[:, 2:m, :]
    dx_int = dx[1:m - 1].view(1, n, 1).expand(B, n, 1)

    left_edge = torch.cat([left_f - self_f, dx_int], dim=-1)    # (B, n, F+1)
    right_edge = torch.cat([right_f - self_f, dx_int], dim=-1)

    nf = ((self_f - node_mean) / node_std).reshape(B * n, F)
    le = ((left_edge - edge_mean) / edge_std).reshape(B * n, F + 1)
    re = ((right_edge - edge_mean) / edge_std).reshape(B * n, F + 1)
    has = torch.ones(B * n, dtype=torch.bool, device=state.device)

    delta = model(nf, le, re, has, has) * y_std + y_mean       # (B*n, |out|)
    delta = delta.reshape(B, n, -1)

    # Predicted columns: self + delta. Everything else starts at self.
    delta_for = {c: delta[:, :, j:j + 1] for j, c in enumerate(cols.out)}
    parts = []
    for c in range(F):
        if c in delta_for:
            parts.append(self_f[:, :, c:c + 1] + delta_for[c])
        else:
            parts.append(self_f[:, :, c:c + 1])               # fixed (e.g. inert)
    # Recompute rho and porosity from the (now updated) species columns.
    rho = sum(rho_w[k] * parts[c] for k, c in enumerate(cols.rhoi))
    beta = ((rhov_b - rho) / max(rhov_b - rhoc_b, 1e-30)).clamp(0, 1)
    por = (phi + (phi_c - phi) * beta).clamp(min(phi, phi_c), max(phi, phi_c))
    parts[cols.rho] = rho
    parts[cols.por] = por
    return torch.cat(parts, dim=-1)


class _Cols:
    """Resolved feature-column indices for mesh_step."""
    def __init__(self, names, predict_gas=True):
        self.out = select_out_cols(names, predict_gas=predict_gas)
        self.rho = names.index("rho")
        self.por = names.index("porosity")
        self.rhoi = [i for i, nm in enumerate(names) if nm.startswith("rho_i")]


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = dict(np.load(args.data, allow_pickle=True))
    node_in = d["node_in"]
    feat_names = [str(x) for x in d["feature_names"]]
    cols = _Cols(feat_names, predict_gas=not args.no_pred_gas)
    OUT_COLS = cols.out
    # Gas fields that are held from truth (not predicted): re-imposed each step.
    aux_cols = [i for i, nm in enumerate(feat_names)
                if nm in ("pg", "mdotf") and i not in OUT_COLS]
    ns = int(d["n_species"]); m = node_in.shape[1]; F = node_in.shape[2]

    # Per-trajectory ordered snapshot sequences. Single-trajectory datasets
    # (no traj_id) are treated as one sequence.
    if "traj_id" in d:
        traj_id = d["traj_id"]
        ntraj = int(traj_id.max()) + 1
        seqs = np.stack([node_in[traj_id == t] for t in range(ntraj)])  # (T,S,m,F)
    else:
        ntraj = 1
        seqs = node_in[None]                                  # (1, S, m, F)
    S = seqs.shape[1]
    dx_cell = np.empty(m); dx_cell[:-1] = d["edge_attr"][::2, 0]
    dx_cell[-1] = dx_cell[-2]

    # Normalization stats (same definitions as single-step training).
    s = to_neighbor_samples(node_in, d["target_delta"], dx_cell, OUT_COLS)
    nfa = torch.from_numpy(s["node_feat"]); ea = torch.from_numpy(
        np.concatenate([s["left_edge"], s["right_edge"]]))
    ta = torch.from_numpy(s["target"])
    stats = [t.to(device) for t in (
        nfa.mean(0), nfa.std(0).clamp(min=1e-8),
        ea.mean(0), ea.std(0).clamp(min=1e-8),
        ta.mean(0), ta.std(0).clamp(min=1e-8))]
    if args.flux or args.conductance:
        stats[4] = torch.zeros_like(stats[4])               # flux: zero baseline
    node_std_out = stats[1][OUT_COLS]                       # for the loss scale
    if args.conductance:
        # The conductance flux D*(s_self - s_nbr) uses ABSOLUTE-state-normalized
        # differences, so its delta must be denormalized by the absolute-state
        # std, not the (tiny) per-step delta std -- otherwise the effective
        # diffusion is throttled by y_std/node_std (~30x) and the front can never
        # penetrate no matter how large D is. This makes D<=0.5 a genuine
        # physical convex combination.
        stats[5] = node_std_out.clone()

    # Physical constants for the derived quantities (from the case material).
    repo = Path(__file__).resolve().parents[1]
    case_dir = repo / args.case_dir
    case = load_case(case_dir / "heat.case")
    solid = load_solid(case_dir / case.solid_file, case.solid_kats)
    consts = (torch.tensor(solid.rho_w, dtype=torch.float32, device=device),
              solid.phi, case.phi_c, solid.rhov_bulk, solid.rhoc_bulk)

    kinetics = None
    if args.real_kinetics:
        rhoi0 = feat_names.index("rho_i0")
        dt_gap = float(np.asarray(d.get("dt_gap", 0.0184)).mean())
        kinetics = {k: [] for k in
                    ("A", "E", "psi", "Tmin", "rhov", "rhoc", "is_char")}
        for incol in OUT_COLS:
            if feat_names[incol].startswith("rho_i"):
                k = incol - rhoi0                          # species index
                kinetics["A"].append(float(solid.A[k]))
                kinetics["E"].append(float(solid.E[k]))
                kinetics["psi"].append(float(solid.psi[k]))
                kinetics["Tmin"].append(float(solid.T_min[k]))
                kinetics["rhov"].append(float(solid.rho_v[k]))
                kinetics["rhoc"].append(float(solid.rho_c[k]))
                kinetics["is_char"].append(1.0)
            else:
                for key, v in (("A", 0.0), ("E", 1.0), ("psi", 1.0),
                               ("Tmin", 1e9), ("rhov", 1.0), ("rhoc", 0.0),
                               ("is_char", 0.0)):
                    kinetics[key].append(v)
        kinetics["dt_sub"] = dt_gap / max(args.substeps, 1)

    if args.conductance:
        model = HeatConductanceMPGNN(
            node_dim=F, edge_dim=F + 1, output_dim=len(OUT_COLS),
            hidden_dim=args.hidden, n_message_passes=args.K,
            out_in_cols=OUT_COLS, uniform_cond=args.uniform_cond,
            physical_source=args.physical_source,
            t_in_col=feat_names.index("T"), rho_in_col=feat_names.index("rho"),
            rhov_bulk=solid.rhov_bulk, rhoc_bulk=solid.rhoc_bulk,
            fixed_cond=([float(x) for x in args.fixed_cond.split(",")]
                        if args.fixed_cond else None),
            kinetics=kinetics).to(device)
    else:
        Model = HeatFluxMPGNN if args.flux else HeatMPGNN
        model = Model(node_dim=F, edge_dim=F + 1, output_dim=len(OUT_COLS),
                      hidden_dim=args.hidden, n_message_passes=args.K).to(device)
    names = ["node_mean", "node_std", "edge_mean", "edge_std", "y_mean", "y_std"]
    for nm, buf in zip(names, stats):
        getattr(model, nm).copy_(buf)
    if args.init:
        ck = torch.load(args.init, map_location=device, weights_only=False)
        # strict=False: warm-starting across architecture tweaks (e.g. the
        # uniform-conductance scalar) reuses the shared encoder/source weights.
        model.load_state_dict(ck["state_dict"], strict=False)
        print(f"warm-started from {args.init}")

    seqs_t = torch.from_numpy(seqs.astype(np.float32)).to(device)   # (T,S,m,F)
    dx_t = torch.from_numpy(dx_cell.astype(np.float32)).to(device)
    M = args.rollout_steps
    n_starts = S - M                                        # valid window starts
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train/val split. Multi-trajectory: hold out every 5th trajectory.
    # Single-trajectory: hold out every 5th window start within the sequence.
    # Fine-cadence trajectories have thousands of near-identical overlapping
    # windows; stride subsamples them so an epoch is affordable.
    starts = range(0, n_starts, args.stride)
    if ntraj > 1:
        val_pairs = [(t, st) for t in range(ntraj) if t % 5 == 0 for st in starts]
        train_pairs = [(t, st) for t in range(ntraj) if t % 5 != 0 for st in starts]
    else:
        val_pairs = [(0, st) for st in starts if (st // args.stride) % 5 == 0]
        train_pairs = [(0, st) for st in starts if (st // args.stride) % 5 != 0]
    val_t = torch.tensor([p[0] for p in val_pairs], device=device)
    val_s = torch.tensor([p[1] for p in val_pairs], device=device)
    train_t = torch.tensor([p[0] for p in train_pairs], device=device)
    train_s = torch.tensor([p[1] for p in train_pairs], device=device)

    def run_windows(tj, st, train_mode):
        if train_mode:
            order = torch.randperm(tj.numel(), device=device)
            tj, st = tj[order], st[order]
        tot, cnt = 0.0, 0
        for b in range(0, tj.numel(), args.batch_size):
            bt, bs_ = tj[b:b + args.batch_size], st[b:b + args.batch_size]
            state = seqs_t[bt, bs_]                         # (B, m, F)
            loss = 0.0
            for k in range(M):
                truth_k = seqs_t[bt, bs_ + k]               # ghosts = forcing
                # One super-step (data cadence) = `substeps` internal sub-steps,
                # forcing (ghosts + held gas) held constant within the super-step.
                for _ in range(args.substeps):
                    interior = mesh_step(model, state, dx_t, stats, consts, cols)
                    if aux_cols:                             # hold gas fields from truth
                        interior = interior.clone()
                        interior[:, :, aux_cols] = truth_k[:, 1:m - 1][:, :, aux_cols]
                    state = torch.cat([truth_k[:, 0:1], interior,
                                       truth_k[:, m - 1:m]], dim=1)
                truth_next = seqs_t[bt, bs_ + k + 1][:, 1:m - 1]
                err = ((state[:, 1:m - 1][:, :, OUT_COLS]
                        - truth_next[:, :, OUT_COLS]) / node_std_out)
                se = err ** 2
                if args.loss_weight > 0 or args.temporal_weight > 0:
                    Tt = truth_next[:, :, 0]                  # true temperature
                    w = torch.ones_like(Tt)
                    if args.loss_weight > 0:
                        # Spatial: upweight the hot near-surface boundary layer,
                        # a small fraction of cells but where the diffusion the
                        # model must learn lives. Safe here: the maximum principle
                        # bounds the model, so focusing the loss cannot destabilize.
                        w = w + args.loss_weight * ((Tt - 298.0) / 1000.0).clamp(0, 3)
                    if args.temporal_weight > 0:
                        # Temporal: upweight cells changing fast per super-step,
                        # i.e. the sharp flux on/off transients where the surrogate
                        # runs hot. Concentrates training on the transient -- the
                        # PINN transfer-learning idea applied to a supervised loss.
                        dT = (truth_next[:, :, 0] - truth_k[:, 1:m - 1, 0]).abs()
                        w = w + args.temporal_weight * (dT / 100.0).clamp(0, 3)
                    se = se * w.unsqueeze(-1)
                loss = loss + se.mean()
            loss = loss / M
            if train_mode:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * bt.numel(); cnt += bt.numel()
        return tot / max(cnt, 1)

    print(f"device={device}  rollout M={M}  warm-start={'yes' if args.init else 'no'}"
          f"  predict={[feat_names[c] for c in OUT_COLS]}")
    for epoch in range(args.epochs):
        model.train(); tr = run_windows(train_t, train_s, True)
        model.eval()
        with torch.no_grad():
            va = run_windows(val_t, val_s, False)
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train {tr:.4e}  val {va:.4e}")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "out_cols": OUT_COLS,
                "feature_names": [str(x) for x in d["feature_names"]],
                "hidden": args.hidden, "K": args.K, "node_dim": F,
                "edge_dim": F + 1, "output_dim": len(OUT_COLS),
                "substeps": args.substeps, "uniform_cond": args.uniform_cond,
                "physical_source": args.physical_source,
                "t_in_col": feat_names.index("T"),
                "rho_in_col": feat_names.index("rho"),
                "rhov_bulk": float(solid.rhov_bulk),
                "rhoc_bulk": float(solid.rhoc_bulk),
                "fixed_cond": ([float(x) for x in args.fixed_cond.split(",")]
                               if args.fixed_cond else None),
                "kinetics": kinetics,
                "model_type": ("conductance" if args.conductance
                               else "flux" if args.flux else "direct")}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--init", default="heat_python/models/heat_mpgnn_noise.pt")
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw1",
                   help="case providing the material constants (rho_w, phi, phi_c)")
    p.add_argument("--no-pred-gas", action="store_true",
                   help="hold pg/mdotf from truth instead of predicting them")
    p.add_argument("--rollout-steps", type=int, default=4)
    p.add_argument("--substeps", type=int, default=1,
                   help="internal stable sub-steps per data super-step (reach): "
                        "the model advances the field in `substeps` bounded "
                        "convex-combination updates, forcing held constant")
    p.add_argument("--stride", type=int, default=1,
                   help="subsample rollout-window starts by this stride "
                        "(fine-cadence windows overlap heavily)")
    p.add_argument("--loss-weight", type=float, default=0.0,
                   help="upweight hot near-surface cells: w = 1 + lw*clamp((T-298)/1000,0,3)")
    p.add_argument("--temporal-weight", type=float, default=0.0,
                   help="upweight cells changing fast per super-step (the flux "
                        "on/off transients): w += tw*clamp(|dT_step|/100,0,3)")
    p.add_argument("--epochs", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--flux", action="store_true",
                   help="multi-step training for the conservative flux-form model")
    p.add_argument("--conductance", action="store_true",
                   help="multi-step training for the stable conductance-form model")
    p.add_argument("--uniform-cond", action="store_true",
                   help="spatially-uniform per-channel conductance (scalar), so "
                        "the diffusion front cannot freeze; anchors near physical")
    p.add_argument("--physical-source", action="store_true",
                   help="structured Arrhenius pyrolysis source (T-gated, self-"
                        "limiting) instead of a free source MLP -> nonlinear + stable")
    p.add_argument("--fixed-cond", default=None,
                   help="comma-separated per-channel FIXED conductance (physical "
                        "anchor), e.g. 0.375,0,0,0.3,0.3 for T,rhoi0,rhoi1,pg,mdotf")
    p.add_argument("--real-kinetics", action="store_true",
                   help="use the material's real per-species Arrhenius pyrolysis "
                        "(exact char front) instead of the crude learned gate")
    p.add_argument("--out", default="heat_python/models/heat_mpgnn_rollout.pt")
    train(p.parse_args())
