"""Train the heat-shield state-surrogate MPGNN on a forcing-sweep dataset.

Loads a full-mesh dataset (heat_python/make_dataset.py), converts to per-cell
neighbor samples (graph.to_neighbor_samples), normalizes, splits train/val by
TRAJECTORY (no leakage between heating scenarios), and fits HeatMPGNN to the
per-cell next-state delta of the independent DOF [T, rho_i0, rho_i1].

    python -m heat_python.train_gnn --data heat_python/data/aw1_forcing_dataset.npz \
        --epochs 40 --out heat_python/models/heat_mpgnn.pt
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .graph import to_neighbor_samples
from .gnn_model import HeatMPGNN
from .gnn_flux_model import HeatFluxMPGNN

# Node feature order: [T, rho, rho_i0, rho_i1, rho_i2, porosity, (pg, mdotf)].
# Predict the independent, non-trivial DOF: T and the two reactive species,
# plus the gas state (pg, mdotf) when the dataset is a gas run. rho, porosity,
# and the inert species rho_i2 are derived, not predicted.
PREDICTED = {"T", "rho_i0", "rho_i1", "pg", "mdotf"}
GAS_FIELDS = {"pg", "mdotf"}


def select_out_cols(feat_names, predict_gas=True):
    """Indices of the columns the model predicts a delta for, by name. Works for
    both gas-off (aw1: [T, rho_i0, rho_i1]) and gas (aw2: + [pg, mdotf]).

    predict_gas=False drops pg/mdotf from the predicted set: the model then
    predicts only the thermal/chemical state [T, rho_i] and pg/mdotf are treated
    as known auxiliary inputs (re-imposed from truth during rollout)."""
    pred = PREDICTED if predict_gas else (PREDICTED - GAS_FIELDS)
    return [i for i, nm in enumerate(feat_names) if nm in pred]


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    d = dict(np.load(args.data, allow_pickle=True))
    node_in = d["node_in"]            # (P, m, F)
    target_delta = d["target_delta"]  # (P, n, F)
    dx = d["edge_attr"][::2, 0]       # interface spacings -> approx per-node dx
    # dx above is per-edge; rebuild a per-cell dx of length m from the mesh.
    m = node_in.shape[1]
    dx_cell = np.empty(m)
    dx_cell[:-1] = d["edge_attr"][::2, 0]
    dx_cell[-1] = dx_cell[-2]
    traj_id = d.get("traj_id")

    feat_names = [str(x) for x in d["feature_names"]]
    out_cols = select_out_cols(feat_names, predict_gas=not args.no_pred_gas)
    s = to_neighbor_samples(node_in, target_delta, dx_cell, out_cols, traj_id)
    node_feat = torch.from_numpy(s["node_feat"])
    left_edge = torch.from_numpy(s["left_edge"])
    right_edge = torch.from_numpy(s["right_edge"])
    target = torch.from_numpy(s["target"])
    samp_traj = s.get("sample_traj_id")

    print(f"device={device}  samples={node_feat.shape[0]:,}  "
          f"node_dim={node_feat.shape[1]} edge_dim={left_edge.shape[1]} "
          f"out_dim={target.shape[1]} (predict {[feat_names[c] for c in out_cols]})")

    # --- Normalization (edges use combined left+right stats) ---
    node_mean, node_std = node_feat.mean(0), node_feat.std(0).clamp(min=1e-8)
    edges = torch.cat([left_edge, right_edge], 0)
    edge_mean, edge_std = edges.mean(0), edges.std(0).clamp(min=1e-8)
    y_mean, y_std = target.mean(0), target.std(0).clamp(min=1e-8)
    # Flux-form needs a ZERO mean offset so a uniform field denormalizes to a
    # zero delta (the conservative zero baseline); only the y_std scale is used.
    if args.flux:
        y_mean = torch.zeros_like(y_mean)

    nf = ((node_feat - node_mean) / node_std).to(device)
    le = ((left_edge - edge_mean) / edge_std).to(device)
    re = ((right_edge - edge_mean) / edge_std).to(device)
    y = ((target - y_mean) / y_std).to(device)
    has = torch.ones(nf.shape[0], dtype=torch.bool, device=device)

    # --- Train/val split by trajectory (no leakage) ---
    if samp_traj is not None:
        st = torch.from_numpy(samp_traj.astype(np.int64))
        ntraj = int(st.max()) + 1
        val_trajs = set(range(ntraj))  # hold out ~20% of trajectories
        val_trajs = set(t for t in range(ntraj) if t % 5 == 0)
        val_mask = torch.tensor([int(t) in val_trajs for t in st], device=device)
    else:
        val_mask = torch.zeros(nf.shape[0], dtype=torch.bool, device=device)
        val_mask[::5] = True
    train_idx = (~val_mask).nonzero(as_tuple=True)[0]
    val_idx = val_mask.nonzero(as_tuple=True)[0]
    print(f"train samples={train_idx.numel():,}  val samples={val_idx.numel():,}")

    Model = HeatFluxMPGNN if args.flux else HeatMPGNN
    model = Model(node_dim=nf.shape[1], edge_dim=le.shape[1],
                  output_dim=y.shape[1], hidden_dim=args.hidden,
                  n_message_passes=args.K).to(device)
    for name, buf in [("node_mean", node_mean), ("node_std", node_std),
                      ("edge_mean", edge_mean), ("edge_std", edge_std),
                      ("y_mean", y_mean), ("y_std", y_std)]:
        getattr(model, name).copy_(buf.to(device))

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()
    bs = args.batch_size

    # --- Noise-injection setup (error-accumulation cure) ---
    # Perturb the input state by eps on the predicted columns, propagate that
    # to the relative edge features (edge = neighbor - self, so self += eps =>
    # edge -= eps), and adjust the delta target (next - noisy_self = delta -
    # eps). The model thus learns to predict the delta that maps a slightly-
    # WRONG state back to the TRUE next state -> contractive / self-correcting.
    out_idx = torch.tensor(out_cols, device=device)          # node/edge cols
    node_std_o = node_std.to(device)[out_idx]                # (out_dim,)
    edge_std_o = edge_std.to(device)[out_idx]
    y_std_d = y_std.to(device)
    sigma_phys = args.noise * node_std_o                     # physical-unit std
    if args.noise > 0:
        print(f"noise injection: sigma = {args.noise} * feature-std "
              f"(phys {sigma_phys.cpu().numpy().round(3)})")

    # --- Stay-put regularizer: cold, untouched cells must predict ~zero change.
    # The model otherwise carries a small per-step bias on quiescent cold cells
    # (verified ~1.5 K/step), which accumulates into bulk drift over a rollout.
    # Penalize the model's CLEAN-input prediction toward physical-zero delta on
    # the coldest training cells (T within `stayput_band` K of the min).
    zero_norm = (-y_mean / y_std).to(device)                 # phys-0 in y-space
    T_raw = node_feat[:, 0]
    cold_all = (T_raw < T_raw.min() + args.stayput_band).nonzero(as_tuple=True)[0].to(device)
    cold_idx = cold_all[torch.isin(cold_all, train_idx)]
    if args.stayput > 0:
        print(f"stay-put reg: weight={args.stayput} on {cold_idx.numel():,} cold cells "
              f"(T<{float(T_raw.min())+args.stayput_band:.1f}K) -> zero delta")

    for epoch in range(args.epochs):
        model.train()
        perm = train_idx[torch.randperm(train_idx.numel(), device=device)]
        tot = 0.0
        for b in range(0, perm.numel(), bs):
            idx = perm[b:b + bs]
            nfi, lei, rei, yi = nf[idx], le[idx], re[idx], y[idx]
            if args.noise > 0:
                eps = torch.randn(idx.numel(), out_idx.numel(),
                                  device=device) * sigma_phys      # physical
                nfi = nfi.clone(); nfi[:, out_idx] += eps / node_std_o
                lei = lei.clone(); lei[:, out_idx] -= eps / edge_std_o
                rei = rei.clone(); rei[:, out_idx] -= eps / edge_std_o
                yi = yi - eps / y_std_d
            opt.zero_grad()
            pred = model(nfi, lei, rei, has[idx], has[idx])
            loss = loss_fn(pred, yi)
            if args.stayput > 0 and cold_idx.numel() > 0:
                cb = cold_idx[torch.randint(cold_idx.numel(), (min(bs, cold_idx.numel()),),
                                            device=device)]
                pc = model(nf[cb], le[cb], re[cb], has[cb], has[cb])   # clean inputs
                loss = loss + args.stayput * loss_fn(pc, zero_norm.expand_as(pc))
            loss.backward()
            opt.step()
            tot += loss.item() * idx.numel()
        train_loss = tot / perm.numel()

        model.eval()
        with torch.no_grad():
            vtot = 0.0
            for b in range(0, val_idx.numel(), bs):
                idx = val_idx[b:b + bs]
                pred = model(nf[idx], le[idx], re[idx], has[idx], has[idx])
                vtot += loss_fn(pred, y[idx]).item() * idx.numel()
            val_loss = vtot / max(val_idx.numel(), 1)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"  epoch {epoch:3d}  train {train_loss:.4e}  val {val_loss:.4e}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "out_cols": out_cols, "feature_names": feat_names,
                "hidden": args.hidden, "K": args.K,
                "node_dim": nf.shape[1], "edge_dim": le.shape[1],
                "output_dim": y.shape[1],
                "model_type": "flux" if args.flux else "direct"}, out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--K", type=int, default=1)
    p.add_argument("--noise", type=float, default=0.0,
                   help="input-noise std as a fraction of each feature's std "
                        "(error-accumulation cure; try 0.02-0.05)")
    p.add_argument("--no-pred-gas", action="store_true",
                   help="do not predict pg/mdotf; treat them as known inputs "
                        "(re-imposed from truth during rollout)")
    p.add_argument("--stayput", type=float, default=0.0,
                   help="weight of the stay-put regularizer (cold cells -> zero "
                        "delta); damps bulk drift. try 0.5-2.0")
    p.add_argument("--stayput-band", type=float, default=5.0,
                   help="cells within this many K of the min T count as cold")
    p.add_argument("--flux", action="store_true",
                   help="use the conservative flux-form model (HeatFluxMPGNN): "
                        "predict face fluxes + a local source, zero baseline")
    p.add_argument("--out", default="heat_python/models/heat_mpgnn.pt")
    train(p.parse_args())
