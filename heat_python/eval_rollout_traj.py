"""Generic trajectory-based rollout evaluation (works for aw1 and aw2/gas).

Unlike eval_rollout (which re-runs the aw1 solver with a hard-coded held-out
forcing), this rolls the surrogate forward over a RECORDED trajectory: it starts
from snapshot 0, re-imposes the ghost cells from the recorded trajectory each
step (so the true boundary forcing is supplied), advances the interior with the
model, derives rho/porosity from the predicted species, and compares the rolled
temperature field against the recorded ground truth.

    python -m heat_python.eval_rollout_traj \
        --traj heat_python/data/aw2_traj.npz \
        --model heat_python/models/heat_mpgnn_aw2_noise.pt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from .graph import node_features
from .gnn_model import HeatMPGNN
from .gnn_flux_model import HeatFluxMPGNN


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(__file__).resolve().parents[1]
    traj = dict(np.load(repo / args.traj, allow_pickle=False))

    X, names = node_features(traj)               # (S, m, F) incl. ghosts
    S, m, F = X.shape
    n = m - 2
    rhov_b, rhoc_b = float(traj["rhov_bulk"]), float(traj["rhoc_bulk"])
    # porosity blend params (TACOT defaults used in graph._porosity)
    phi, phi_c = 0.8, 0.85
    rho_col = names.index("rho")
    por_col = names.index("porosity")
    rhoi_cols = [i for i, nm in enumerate(names) if nm.startswith("rho_i")]
    dx = torch.tensor(traj["dx"], dtype=torch.float32, device=device)

    gt = torch.tensor(X, dtype=torch.float32, device=device)   # (S, m, F)

    ck = torch.load(repo / args.model, map_location=device, weights_only=False)
    Model = HeatFluxMPGNN if ck.get("model_type") == "flux" else HeatMPGNN
    model = Model(node_dim=ck["node_dim"], edge_dim=ck["edge_dim"],
                  output_dim=ck["output_dim"], hidden_dim=ck["hidden"],
                  n_message_passes=ck["K"]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    out_idx = torch.tensor(ck["out_cols"], device=device)
    # Gas fields the model does NOT predict are held from truth each step.
    aux_cols = [i for i, nm in enumerate(names)
                if nm in ("pg", "mdotf") and i not in list(ck["out_cols"])]
    # rho_w to rebuild bulk rho from species (sum_k rho_w_k * rho_i_k)
    rho_w = torch.tensor(_rho_w_from_traj(traj, rhoi_cols, X), dtype=torch.float32,
                         device=device)

    state = gt[0].clone()
    pred_T = [state[:, 0].cpu().numpy()]
    pred_rho = [state[:, rho_col].cpu().numpy()]
    for k in range(S - 1):
        state[0] = gt[k, 0]                       # re-impose ghosts from truth
        state[m - 1] = gt[k, m - 1]
        new_int = model.step_mesh(state, dx, out_idx)
        # derive bulk rho + porosity from predicted species (consistency)
        rho = (rho_w[None, :] * new_int[:, rhoi_cols]).sum(1)
        new_int[:, rho_col] = rho
        beta = ((rhov_b - rho) / max(rhov_b - rhoc_b, 1e-30)).clamp(0, 1)
        por = (phi + (phi_c - phi) * beta).clamp(min(phi, phi_c), max(phi, phi_c)).clamp(1e-6, 0.99)
        new_int[:, por_col] = por
        for c in aux_cols:                        # held gas fields from truth
            new_int[:, c] = gt[k + 1, 1:m - 1, c]
        state = state.clone()
        state[1:m - 1] = new_int
        pred_T.append(state[:, 0].cpu().numpy())
        pred_rho.append(state[:, rho_col].cpu().numpy())

    pred_T = np.stack(pred_T)
    pred_rho = np.stack(pred_rho)
    T_gt = X[:, :, 0]
    err = np.abs(pred_T[:, 1:n + 1] - T_gt[:, 1:n + 1])
    t = traj["time"]
    print(f"\nROLLOUT vs solver ({S} steps over {t[-1]:.0f}s, {n} cells):")
    print(f"  interior T: mean |err| {err.mean():.2f} K   max {err.max():.2f} K")
    # Hot-band metric: isolates the steep near-surface region the GNN undershoots.
    # signed bias < 0 means the surrogate runs cooler than the solver there.
    Ti = T_gt[:, 1:n + 1]
    Pi = pred_T[:, 1:n + 1]
    hot = Ti > 800.0
    if hot.any():
        signed = (Pi - Ti)[hot]
        print(f"  hot band (T>800K): mean |err| {np.abs(Pi - Ti)[hot].mean():.2f} K"
              f"   signed bias {signed.mean():+.2f} K   ({int(hot.sum())} samples)")
    for frac, lbl in [(0.1, "10%"), (0.5, "50%"), (1.0, "end")]:
        s = min(int(frac * (S - 1)), S - 1)
        print(f"  t={t[s]:6.1f}s ({lbl}): mean {err[s].mean():7.2f} K  max {err[s].max():8.2f} K")

    if args.save:
        out = repo / args.save
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, time=t, T_gt=T_gt, T_pred=pred_T, err=err,
                            rho_gt=X[:, :, rho_col], rho_pred=pred_rho,
                            rhov_bulk=rhov_b, rhoc_bulk=rhoc_b)
        print(f"  saved -> {out}")


def _rho_w_from_traj(traj, rhoi_cols, X):
    """Recover the species weights rho_w so that sum_k rho_w_k * rho_i_k = rho.
    Solve the least-squares mapping from species columns to bulk rho over all
    cells/snapshots (exact since rho is a fixed linear combination)."""
    ns = len(rhoi_cols)
    A = X[:, :, rhoi_cols].reshape(-1, ns)        # (N, ns)
    b = X[:, :, 1].reshape(-1)                    # rho column
    w, *_ = np.linalg.lstsq(A, b, rcond=None)
    return w


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--save", default=None)
    main(p.parse_args())
