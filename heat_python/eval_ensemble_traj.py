"""Ensemble trajectory rollout: average several models' per-step predictions.

Same contract as eval_rollout_traj, but loads N checkpoints and averages their
step_mesh() interior predictions each step. Targets init-sensitive rollout
instability (averaging out the per-model variance measured across random inits).

    python -m heat_python.eval_ensemble_traj --traj heat_python/data/aw2_holdout_traj.npz \
        --models heat_python/models/heat_flux_aw2.pt,heat_python/models/heat_flux_aw2_r2.pt,heat_python/models/heat_flux_aw2_r3.pt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from .graph import node_features
from .gnn_model import HeatMPGNN
from .gnn_flux_model import HeatFluxMPGNN
from .eval_rollout_traj import _rho_w_from_traj


def _load(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    Model = HeatFluxMPGNN if ck.get("model_type") == "flux" else HeatMPGNN
    model = Model(node_dim=ck["node_dim"], edge_dim=ck["edge_dim"],
                  output_dim=ck["output_dim"], hidden_dim=ck["hidden"],
                  n_message_passes=ck["K"]).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    return model, ck


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(__file__).resolve().parents[1]
    traj = dict(np.load(repo / args.traj, allow_pickle=False))

    X, names = node_features(traj)
    S, m, F = X.shape
    n = m - 2
    rhov_b, rhoc_b = float(traj["rhov_bulk"]), float(traj["rhoc_bulk"])
    phi, phi_c = 0.8, 0.85
    rho_col = names.index("rho")
    por_col = names.index("porosity")
    rhoi_cols = [i for i, nm in enumerate(names) if nm.startswith("rho_i")]
    dx = torch.tensor(traj["dx"], dtype=torch.float32, device=device)
    gt = torch.tensor(X, dtype=torch.float32, device=device)

    paths = [p.strip() for p in args.models.split(",")]
    models, cks = zip(*[_load(repo / p, device) for p in paths])
    ck = cks[0]
    out_idx = torch.tensor(ck["out_cols"], device=device)
    aux_cols = [i for i, nm in enumerate(names)
                if nm in ("pg", "mdotf") and i not in list(ck["out_cols"])]
    rho_w = torch.tensor(_rho_w_from_traj(traj, rhoi_cols, X), dtype=torch.float32,
                         device=device)

    state = gt[0].clone()
    pred_T = [state[:, 0].cpu().numpy()]
    for k in range(S - 1):
        state[0] = gt[k, 0]
        state[m - 1] = gt[k, m - 1]
        preds = torch.stack([mdl.step_mesh(state, dx, out_idx) for mdl in models])
        new_int = preds.mean(0)
        rho = (rho_w[None, :] * new_int[:, rhoi_cols]).sum(1)
        new_int[:, rho_col] = rho
        beta = ((rhov_b - rho) / max(rhov_b - rhoc_b, 1e-30)).clamp(0, 1)
        por = (phi + (phi_c - phi) * beta).clamp(min(phi, phi_c), max(phi, phi_c)).clamp(1e-6, 0.99)
        new_int[:, por_col] = por
        for c in aux_cols:
            new_int[:, c] = gt[k + 1, 1:m - 1, c]
        state = state.clone()
        state[1:m - 1] = new_int
        pred_T.append(state[:, 0].cpu().numpy())

    pred_T = np.stack(pred_T)
    T_gt = X[:, :, 0]
    err = np.abs(pred_T[:, 1:n + 1] - T_gt[:, 1:n + 1])
    t = traj["time"]
    print(f"\nENSEMBLE ({len(models)} models) ROLLOUT vs solver ({S} steps over {t[-1]:.0f}s, {n} cells):")
    print(f"  interior T: mean |err| {err.mean():.2f} K   max {err.max():.2f} K")
    for frac, lbl in [(0.1, "10%"), (0.5, "50%"), (1.0, "end")]:
        s = min(int(frac * (S - 1)), S - 1)
        print(f"  t={t[s]:6.1f}s ({lbl}): mean {err[s].mean():7.2f} K  max {err[s].max():8.2f} K")

    if args.save:
        out = repo / args.save
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out, time=t, T_gt=T_gt, T_pred=pred_T, err=err,
                            rho_gt=X[:, :, rho_col], rhov_bulk=rhov_b, rhoc_bulk=rhoc_b)
        print(f"  saved -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True)
    p.add_argument("--models", required=True, help="comma-separated checkpoint paths")
    p.add_argument("--save", default=None)
    main(p.parse_args())
