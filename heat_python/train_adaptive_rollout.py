"""Multi-step (rollout) training for the dt-aware ADAPTIVE aw2 surrogate.

Like train_rollout.py, but builds short rollout windows at MULTIPLE cadences
(strides) from the raw trajectory, sets the dt input feature for each, and
unrolls M steps at that cadence. Training contractivity at both fine and coarse
dt is what the adaptive rollout (eval_adaptive.py) needs: fine steps near the
fast transients, coarse steps on the slow plateau, all from one model.

Warm-starts from the dt-aware single-step (+noise) model.

    python -m heat_python.train_adaptive_rollout \
        --traj heat_python/data/aw2_traj.npz \
        --init heat_python/models/heat_mpgnn_aw2_adaptive_noise.pt \
        --case-dir heat_2026-04-11_1837/examples/aw2_tc21 \
        --out heat_python/models/heat_mpgnn_aw2_adaptive_rollout.pt
"""
from __future__ import annotations
from pathlib import Path

import numpy as np
import torch

from .graph import node_features, mesh_graph
from .gnn_model import HeatMPGNN
from .train_rollout import mesh_step, _Cols
from .case import load_case
from .pyrolysis import load_solid


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    repo = Path(__file__).resolve().parents[1]
    traj = dict(np.load(repo / args.traj, allow_pickle=False))

    X, names = node_features(traj)                  # (S, m, F)
    names = list(names) + ["dt"]
    S, m, F0 = X.shape
    F = F0 + 1                                       # with dt column
    n = m - 2
    snap_dt = float(np.median(np.diff(traj["time"])))
    cols = _Cols(names, predict_gas=True)            # out_cols exclude dt
    gaps = [int(g) for g in args.gaps]
    M = args.rollout_steps

    Xt = torch.tensor(X, dtype=torch.float32, device=device)        # (S, m, F0)
    dx_t = torch.tensor(traj["dx"], dtype=torch.float32, device=device)

    # material constants for the derived rho/porosity
    case_dir = repo / args.case_dir
    case = load_case(case_dir / "heat.case")
    solid = load_solid(case_dir / case.solid_file, case.solid_kats)
    consts = (torch.tensor(solid.rho_w, dtype=torch.float32, device=device),
              solid.phi, case.phi_c, solid.rhov_bulk, solid.rhoc_bulk)

    # normalization stats from the saved dt-aware dataset's model buffers (init)
    ck = torch.load(repo / args.init, map_location=device, weights_only=False)
    model = HeatMPGNN(node_dim=F, edge_dim=F + 1, output_dim=len(cols.out),
                      hidden_dim=ck["hidden"], n_message_passes=ck["K"]).to(device)
    model.load_state_dict(ck["state_dict"])
    print(f"warm-started from {args.init}; gaps={gaps} M={M}")
    stats = (model.node_mean, model.node_std, model.edge_mean, model.edge_std,
             model.y_mean, model.y_std)
    node_std_out = model.node_std[cols.out]

    # build (gap, start) windows: a window needs M*gap+1 snapshots
    windows = []
    for g in gaps:
        for s in range(0, S - M * g):
            windows.append((g, s))
    windows = np.array(windows)
    rng = np.random.default_rng(0)
    val_mask = (np.arange(len(windows)) % 5 == 0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def add_dt(state2d, g):
        dt_col = torch.full((state2d.shape[0], 1), g * snap_dt, device=device)
        return torch.cat([state2d, dt_col], dim=-1)

    def run(idxs, train_mode):
        order = rng.permutation(idxs) if train_mode else idxs
        tot, cnt = 0.0, 0
        for b in range(0, len(order), args.batch_size):
            bw = windows[order[b:b + args.batch_size]]
            # group by gap so a batch shares one stride (windows mix gaps across batches)
            loss = 0.0; nb = 0
            for g in gaps:
                sel = bw[bw[:, 0] == g][:, 1]
                if len(sel) == 0:
                    continue
                st = torch.tensor(sel, device=device)
                # state0 with dt feature
                s0 = Xt[st, :, :]                       # (B, m, F0)
                state = torch.cat([s0, torch.full((s0.shape[0], m, 1), g * snap_dt,
                                                   device=device)], dim=-1)
                lg = 0.0
                for k in range(M):
                    interior = mesh_step(model, state, dx_t, stats, consts, cols)
                    nxt = Xt[st + (k + 1) * g]             # (B, m, F0) truth at landed
                    # re-impose ghosts (forcing) + dt feature
                    dtcol0 = torch.full((nxt.shape[0], 1, 1), g * snap_dt, device=device)
                    gh0 = torch.cat([nxt[:, 0:1], dtcol0], dim=-1)
                    ghN = torch.cat([nxt[:, m - 1:m], dtcol0], dim=-1)
                    state = torch.cat([gh0, interior, ghN], dim=1)
                    err = ((state[:, 1:m - 1][:, :, cols.out]
                            - nxt[:, 1:m - 1][:, :, cols.out]) / node_std_out)
                    lg = lg + (err ** 2).mean()
                loss = loss + lg / M; nb += 1
            if nb == 0:
                continue
            loss = loss / nb
            if train_mode:
                opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(bw); cnt += len(bw)
        return tot / max(cnt, 1)

    tr_idx = np.where(~val_mask)[0]; va_idx = np.where(val_mask)[0]
    for epoch in range(args.epochs):
        model.train(); tr = run(tr_idx, True)
        model.eval()
        with torch.no_grad():
            va = run(va_idx, False)
        if epoch % 2 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train {tr:.4e}  val {va:.4e}")

    out = Path(repo / args.out); out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "out_cols": cols.out,
                "feature_names": names, "hidden": ck["hidden"], "K": ck["K"],
                "node_dim": F, "edge_dim": F + 1, "output_dim": len(cols.out)}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--traj", required=True)
    p.add_argument("--init", required=True)
    p.add_argument("--case-dir", default="heat_2026-04-11_1837/examples/aw2_tc21")
    p.add_argument("--gaps", type=int, nargs="+", default=[1, 10])
    p.add_argument("--rollout-steps", type=int, default=4)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--out", default="heat_python/models/heat_mpgnn_aw2_adaptive_rollout.pt")
    train(p.parse_args())
