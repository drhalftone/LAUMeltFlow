"""Adaptive-cadence rollout for the aw2 (gas) surrogate.

The model is trained dt-aware (see graph.build_adaptive_dataset): it takes the
time step dt as an input feature, so one model can take a SMALL step near the
fast surface transients (flux on/off) and a LARGE step on the slow plateau. This
keeps the per-step jump small everywhere (the failure cause) while keeping the
total step count low (the accumulation cause).

Schedule: fine (gap=1) when within `--fine-window` seconds of a forcing event
(t=0 turn-on, t=t_off turn-off); coarse (gap=10) otherwise. Ghosts and ground
truth are taken from the recorded full-resolution trajectory at each landed step.

    python -m heat_python.eval_adaptive \
        --traj heat_python/data/aw2_traj.npz \
        --model heat_python/models/heat_mpgnn_aw2_adaptive_rollout.pt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from .graph import node_features
from .gnn_model import HeatMPGNN


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(__file__).resolve().parents[1]
    traj = dict(np.load(repo / args.traj, allow_pickle=False))

    X, names = node_features(traj)               # (S, m, F) physical, no dt
    time = traj["time"]
    S, m, F = X.shape
    n = m - 2
    snap_dt = float(np.median(np.diff(time)))
    rhov_b, rhoc_b = float(traj["rhov_bulk"]), float(traj["rhoc_bulk"])
    phi, phi_c = 0.8, 0.85
    rho_col = names.index("rho"); por_col = names.index("porosity")
    rhoi_cols = [i for i, nm in enumerate(names) if nm.startswith("rho_i")]
    dx = torch.tensor(traj["dx"], dtype=torch.float32, device=device)

    ck = torch.load(repo / args.model, map_location=device, weights_only=False)
    model = HeatMPGNN(node_dim=ck["node_dim"], edge_dim=ck["edge_dim"],
                      output_dim=ck["output_dim"], hidden_dim=ck["hidden"],
                      n_message_passes=ck["K"]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    out_idx = torch.tensor(ck["out_cols"], device=device)
    feat_names = [str(x) for x in ck["feature_names"]]
    dt_col = feat_names.index("dt")              # the dt input column
    rho_w = torch.tensor(_rho_w(X, rhoi_cols), dtype=torch.float32, device=device)

    # append a dt column to the physical features (set per step during rollout)
    gtX = np.concatenate([X, np.zeros((S, m, 1))], axis=2)
    gt = torch.tensor(gtX, dtype=torch.float32, device=device)   # (S, m, F+1)

    # --- forcing events for the schedule (turn-on at 0, turn-off at t_off) ---
    t_off = args.t_off
    coarse_gap = args.coarse_gap
    fine_w = args.fine_window

    def gap_at(t):
        near = (t <= fine_w) or (abs(t - t_off) <= fine_w)
        return 1 if near else coarse_gap

    # --- adaptive rollout ---
    state = gt[0].clone()
    pred_T = [state[:, 0].cpu().numpy()]; pred_t = [time[0]]
    idx_list = [0]
    k = 0
    while k < S - 1:
        g = gap_at(time[k])
        g = min(g, S - 1 - k)                    # don't overrun
        dt_phys = g * snap_dt
        state[:, dt_col] = dt_phys               # tell the model the step size
        state[0] = gt[k, 0]; state[m - 1] = gt[k, m - 1]         # ghosts (forcing) at k
        state[0, dt_col] = dt_phys; state[m - 1, dt_col] = dt_phys
        new_int = model.step_mesh(state, dx, out_idx)            # (n, F+1)
        rho = (rho_w[None, :] * new_int[:, rhoi_cols]).sum(1)
        new_int[:, rho_col] = rho
        beta = ((rhov_b - rho) / max(rhov_b - rhoc_b, 1e-30)).clamp(0, 1)
        por = (phi + (phi_c - phi) * beta).clamp(min(phi, phi_c), max(phi, phi_c)).clamp(1e-6, 0.99)
        new_int[:, por_col] = por
        k_next = k + g
        state = state.clone()
        state[1:m - 1] = new_int
        # re-impose ghosts from truth at the LANDED step
        state[0] = gt[k_next, 0]; state[m - 1] = gt[k_next, m - 1]
        pred_T.append(state[:, 0].cpu().numpy()); pred_t.append(time[k_next])
        idx_list.append(k_next)
        k = k_next

    pred_T = np.stack(pred_T)                     # (Ksteps, m)
    landed = np.array(idx_list)
    T_gt_land = X[landed, :, 0]                   # truth at landed steps
    err = np.abs(pred_T[:, 1:n + 1] - T_gt_land[:, 1:n + 1])
    nsteps = len(landed) - 1
    print(f"\nADAPTIVE ROLLOUT vs solver ({nsteps} steps over {time[landed[-1]]:.0f}s, "
          f"{n} cells; fine within {fine_w}s of t=0 and t={t_off}):")
    print(f"  interior T: mean |err| {err.mean():.2f} K   max {err.max():.2f} K")
    tl = time[landed]
    for frac, lbl in [(0.1, "10%"), (0.5, "50%"), (1.0, "end")]:
        s = min(int(frac * nsteps), nsteps)
        print(f"  t={tl[s]:6.1f}s ({lbl}): mean {err[s].mean():7.2f} K  max {err[s].max():8.2f} K")
    print(f"  (vs fixed-coarse 652 steps and fixed-fine 6515 steps)")

    if args.save:
        out = repo / args.save
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, time=tl, T_gt=T_gt_land, T_pred=pred_T, err=err)
        print(f"  saved -> {out}")


def _rho_w(X, rhoi_cols):
    ns = len(rhoi_cols)
    A = X[:, :, rhoi_cols].reshape(-1, ns)
    b = X[:, :, 1].reshape(-1)
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return w


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--t-off", type=float, default=60.0, help="flux turn-off time")
    p.add_argument("--fine-window", type=float, default=2.0,
                   help="seconds around each forcing event to step fine")
    p.add_argument("--coarse-gap", type=int, default=10)
    p.add_argument("--save", default=None)
    main(p.parse_args())
