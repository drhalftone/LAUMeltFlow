"""Wall-time speedup: GNN surrogate rollout vs. the reference solver.

Fair comparison: both produce the SAME 60 s aw1 trajectory on the same held-out
forcing. The solver runs its full CFL-limited time loop; the surrogate rolls
forward at the snapshot cadence (its super-step), which is the source of the
speedup. We time the solver (CPU, as it runs) and the GNN rollout on CPU and,
if available, GPU. Reported: wall times and ratios.

    python -m heat_python.measure_speedup --model heat_python/models/heat_mpgnn_rollout.pt
"""
from __future__ import annotations
import time
from pathlib import Path

import numpy as np
import torch

from .solver import run
from .case import load_case
from .pyrolysis import load_solid
from .gas import _porosity_array
from .gnn_model import HeatMPGNN
from .eval_rollout import held_out_forcing


def _build_gt_feats(tr, rho_w, rhov_b, rhoc_b, case, solid):
    T_gt, rho_i_gt = tr["T"], tr["rho_i"]
    S, ns, m = rho_i_gt.shape

    def feats(T_row, rhoi_row):
        rho = (rho_w[:, None] * rhoi_row).sum(0)
        por = _porosity_array(rho, rhov_b, rhoc_b, case, solid)
        cols = [T_row, rho] + [rhoi_row[k] for k in range(ns)] + [por]
        return np.stack(cols, axis=1)

    return np.stack([feats(T_gt[s], rho_i_gt[s]) for s in range(S)]), S, ns, m


def _time_rollout(model, gt_feats, dx, out_idx, rho_w, rhov_b, rhoc_b,
                  phi, phi_c, ns, m, device, reps=3):
    gt = torch.tensor(gt_feats, dtype=torch.float32, device=device)
    dxt = torch.tensor(dx, dtype=torch.float32, device=device)
    rho_w_t = torch.tensor(rho_w, dtype=torch.float32, device=device)
    S = gt.shape[0]
    best = float("inf")
    for _ in range(reps):
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        state = gt[0].clone()
        for k in range(S - 1):
            state[0] = gt[k, 0]
            state[m - 1] = gt[k, m - 1]
            new_int = model.step_mesh(state, dxt, out_idx)
            rho = (rho_w_t[None, :] * new_int[:, 2:2 + ns]).sum(1)
            new_int[:, 1] = rho
            beta = ((rhov_b - rho) / max(rhov_b - rhoc_b, 1e-30)).clamp(0, 1)
            por = (phi + (phi_c - phi) * beta).clamp(min(phi, phi_c), max(phi, phi_c)).clamp(1e-6, 0.99)
            new_int[:, 5] = por
            state = state.clone()
            state[1:m - 1] = new_int
        if device == "cuda":
            torch.cuda.synchronize()
        best = min(best, time.perf_counter() - t0)
    return best, S


def main(args):
    repo = Path(__file__).resolve().parents[1]
    case_dir = repo / "heat_2026-04-11_1837" / "examples" / "aw1"
    case = load_case(case_dir / "heat.case")
    solid = load_solid(case_dir / case.solid_file, case.solid_kats)
    rho_w = np.asarray(solid.rho_w, dtype=np.float32)
    rhov_b, rhoc_b = solid.rhov_bulk, solid.rhoc_bulk
    phi, phi_c = solid.phi, case.phi_c

    tt, (t_peak, t_ramp, cool) = held_out_forcing()
    print(f"held-out forcing: peak={t_peak:.0f}K ramp={t_ramp:.1f}s cool={cool}")

    # --- time the solver (full run, with recording so we get the trajectory) ---
    t0 = time.perf_counter()
    res = run(case_dir, verbose=False, lgas=False, record_every=10,
              write_con=False, time_table=tt)
    t_solver = time.perf_counter() - t0
    tr = res["trajectory"]
    n_solver_steps = res["n_steps"]

    gt_feats, S, ns, m = _build_gt_feats(tr, rho_w, rhov_b, rhoc_b, case, solid)
    dx = tr["dx"]

    # --- load model ---
    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    out_idx_np = ck["out_cols"]

    def make_model(device):
        mdl = HeatMPGNN(node_dim=ck["node_dim"], edge_dim=ck["edge_dim"],
                        output_dim=ck["output_dim"], hidden_dim=ck["hidden"],
                        n_message_passes=ck["K"]).to(device)
        mdl.load_state_dict(ck["state_dict"]); mdl.eval()
        return mdl, torch.tensor(out_idx_np, device=device)

    print(f"\nSolver: {n_solver_steps} CFL steps -> {t_solver:.3f} s wall")
    print(f"GNN rollout: {S-1} super-steps (snapshot cadence = 10 solver steps each)")

    results = []
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    for dev in devices:
        model, out_idx = make_model(dev)
        t_gnn, _ = _time_rollout(model, gt_feats, dx, out_idx, rho_w, rhov_b,
                                 rhoc_b, phi, phi_c, ns, m, dev)
        results.append((dev, t_gnn))
        print(f"  GNN ({dev:4s}): {t_gnn:.4f} s wall   ->  {t_solver / t_gnn:6.1f}x faster than solver")

    print("\nSummary (aw1, 60 s trajectory):")
    print(f"  reference solver (CPU):  {t_solver*1e3:8.1f} ms")
    for dev, t in results:
        print(f"  GNN surrogate ({dev:4s}):  {t*1e3:8.1f} ms   ({t_solver/t:.0f}x)")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="heat_python/models/heat_mpgnn_rollout.pt")
    main(p.parse_args())
