"""Rollout evaluation: the real test of the heat-shield surrogate.

Take a held-out heating scenario, get the solver's ground-truth trajectory,
then roll the GNN forward FROM ITS OWN PREDICTIONS step by step (re-imposing
the known boundary forcing via the ghost cells each step, and deriving
rho/porosity from the predicted species densities). Compare the GNN trajectory
against the solver's. One-step loss can look great while rollout drifts -- this
is what tells us if the surrogate is actually usable.

    python -m heat_python.eval_rollout --model heat_python/models/heat_mpgnn.pt
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from .solver import run
from .case import load_case
from .pyrolysis import load_solid
from .gas import _porosity_array
from .io_files import TimeTable
from .gnn_model import HeatMPGNN
from .gnn_flux_model import HeatFluxMPGNN
from .make_dataset import _forcing


def held_out_forcing(t_final=60.0, init_temp=298.15):
    """Reproduce trajectory 0's forcing (the first val trajectory, seed 0)."""
    rng = np.random.default_rng(0)
    t_peak = float(rng.uniform(700.0, 2000.0))
    t_ramp = float(rng.uniform(0.5, 15.0))
    cool = bool(rng.uniform() < 0.4)
    t_hold_end = float(rng.uniform(t_ramp + 5.0, t_final - 2.0))
    return _forcing(t_ramp, t_peak, t_hold_end, init_temp, t_final, cool), \
        (t_peak, t_ramp, cool)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(__file__).resolve().parents[1]
    case_dir = repo / "heat_2026-04-11_1837" / "examples" / "aw1"
    case = load_case(case_dir / "heat.case")
    solid = load_solid(case_dir / case.solid_file, case.solid_kats)
    rho_w = torch.tensor(solid.rho_w, dtype=torch.float32, device=device)
    rhov_b, rhoc_b = solid.rhov_bulk, solid.rhoc_bulk
    phi, phi_c = solid.phi, case.phi_c

    # --- Ground-truth solver trajectory for a held-out forcing ---
    tt, (t_peak, t_ramp, cool) = held_out_forcing()
    print(f"held-out forcing: peak={t_peak:.0f}K ramp={t_ramp:.1f}s cool={cool}")
    rec_kw = (dict(record_dt=args.record_dt) if args.record_dt is not None
              else dict(record_every=10))
    res = run(case_dir, verbose=False, lgas=False,
              write_con=False, time_table=tt, nbrn=args.nbrn, **rec_kw)
    if args.nbrn is not None:
        msg = f"mesh override: NBRN={args.nbrn} (model trained at 100)"
        if args.record_dt is not None:
            msg += f", fixed snapshot dt={args.record_dt:.5f}s"
        print(msg)
    tr = res["trajectory"]
    T_gt = tr["T"]               # (S, m) ground truth, with ghosts
    rho_i_gt = tr["rho_i"]       # (S, ns, m)
    S, ns, m = rho_i_gt.shape
    n = m - 2
    dx = torch.tensor(tr["dx"], dtype=torch.float32, device=device)

    # --- Load model ---
    ck = torch.load(args.model, map_location=device, weights_only=False)
    Model = HeatFluxMPGNN if ck.get("model_type") == "flux" else HeatMPGNN
    model = Model(node_dim=ck["node_dim"], edge_dim=ck["edge_dim"],
                  output_dim=ck["output_dim"], hidden_dim=ck["hidden"],
                  n_message_passes=ck["K"]).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    out_idx = torch.tensor(ck["out_cols"], device=device)  # [T, rho_i0, rho_i1]

    # Build the node-feature stack [T, rho, rho_i0, rho_i1, rho_i2, porosity]
    # for every ground-truth snapshot (so we can pull ghost forcing each step).
    def feats(T_row, rhoi_row):
        # T_row (m,), rhoi_row (ns, m)
        rho = (rho_w[:, None].cpu().numpy() * rhoi_row).sum(0)  # (m,)
        por = _porosity_array(rho, rhov_b, rhoc_b, case, solid)
        cols = [T_row, rho] + [rhoi_row[k] for k in range(ns)] + [por]
        return np.stack(cols, axis=1)  # (m, F)

    gt_feats = np.stack([feats(T_gt[s], rho_i_gt[s]) for s in range(S)])  # (S,m,F)
    gt_feats = torch.tensor(gt_feats, dtype=torch.float32, device=device)

    # --- Roll the GNN forward ---
    state = gt_feats[0].clone()                # (m, F) start from GT snapshot 0
    pred_T = [state[:, 0].cpu().numpy()]
    pred_rho = [state[:, 1].cpu().numpy()]
    for k in range(S - 1):
        # Impose the known forcing: ghost cells from the GT snapshot at step k.
        state[0] = gt_feats[k, 0]
        state[m - 1] = gt_feats[k, m - 1]
        new_int = model.step_mesh(state, dx, out_idx)   # (n, F), out cols updated
        # Derive rho_i2 (inert, unchanged), rho, porosity consistently.
        rho = (rho_w[None, :] * new_int[:, 2:2 + ns]).sum(1)
        new_int[:, 1] = rho
        beta = ((rhov_b - rho) / max(rhov_b - rhoc_b, 1e-30)).clamp(0, 1)
        por = phi + (phi_c - phi) * beta
        new_int[:, 5] = por.clamp(min(phi, phi_c), max(phi, phi_c)).clamp(1e-6, 0.99)
        state = state.clone()
        state[1:m - 1] = new_int
        pred_T.append(state[:, 0].cpu().numpy())
        pred_rho.append(state[:, 1].cpu().numpy())

    pred_T = np.stack(pred_T)                   # (S, m)
    pred_rho = np.stack(pred_rho)               # (S, m) GNN density (for char front)
    err = np.abs(pred_T[:, 1:n + 1] - T_gt[:, 1:n + 1])   # interior T error
    # error vs depth-into-time
    print(f"\nROLLOUT vs solver ({S} GNN steps over {tr['time'][-1]:.0f}s):")
    print(f"  interior T: mean |err| {err.mean():.2f} K   max {err.max():.2f} K")
    print(f"  final-step interior T: mean {err[-1].mean():.2f} K  max {err[-1].max():.2f} K")
    for frac, lbl in [(0.1, "10%"), (0.5, "50%"), (1.0, "end")]:
        s = min(int(frac * (S - 1)), S - 1)
        print(f"  t={tr['time'][s]:5.1f}s ({lbl}): mean {err[s].mean():6.2f} K  "
              f"max {err[s].max():6.2f} K")

    if args.save:
        out = Path(args.save)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, time=tr["time"], T_gt=T_gt, T_pred=pred_T,
                            err=err, rho_gt=tr["rho"], rho_pred=pred_rho,
                            rhov_bulk=rhov_b, rhoc_bulk=rhoc_b)
        print(f"  saved rollout -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="heat_python/models/heat_mpgnn.pt")
    p.add_argument("--save", default="heat_python/data/rollout_aw1.npz")
    p.add_argument("--nbrn", type=int, default=None,
                   help="override mesh resolution (default: case value, 100)")
    p.add_argument("--record-dt", type=float, default=None,
                   help="fixed physical snapshot dt (resolution-independent); "
                        "training value is 0.11513s")
    main(p.parse_args())
