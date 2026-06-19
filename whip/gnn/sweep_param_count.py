"""Sweep BeadMPGNN hidden_dim to measure accuracy vs. parameter count.

Trains a fresh model at each hidden_dim value, saves checkpoint, then
runs rollout evaluation on a small set of chain lengths. Aggregates
results into a single npz + plot.

Usage:
    python sweep_param_count.py
    python sweep_param_count.py --hidden_dims 4,8,16,32,64,128 --epochs 200
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from bead_mpgnn import BeadMPGNN

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from simulation import create_chain, compute_forces, compute_energy


_DATASET_CACHE = {}


def _load_dataset_to_device(data_path, device):
    """Load + normalize the data once, move to device, cache by (path, device)."""
    key = (os.path.abspath(data_path), str(device))
    if key in _DATASET_CACHE:
        return _DATASET_CACHE[key]

    d = np.load(data_path)
    node_feat = torch.from_numpy(d["node_feat"])
    left_edge_feat = torch.from_numpy(d["left_edge_feat"])
    right_edge_feat = torch.from_numpy(d["right_edge_feat"])
    has_left = torch.from_numpy(d["has_left"])
    has_right = torch.from_numpy(d["has_right"])
    Y = torch.from_numpy(d["Y"])

    node_mean = node_feat.mean(dim=0)
    node_std = node_feat.std(dim=0).clamp(min=1e-8)
    all_edges = torch.cat([left_edge_feat, right_edge_feat], dim=0)
    edge_mean = all_edges.mean(dim=0)
    edge_std = all_edges.std(dim=0).clamp(min=1e-8)
    y_mean = Y.mean(dim=0)
    y_std = Y.std(dim=0).clamp(min=1e-8)

    nf_n = ((node_feat - node_mean) / node_std).to(device)
    le_n = ((left_edge_feat - edge_mean) / edge_std).to(device)
    re_n = ((right_edge_feat - edge_mean) / edge_std).to(device)
    Y_n = ((Y - y_mean) / y_std).to(device)
    hl = has_left.to(device).bool()
    hr = has_right.to(device).bool()

    n_total = nf_n.shape[0]
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx = perm[:n_train].to(device)
    val_idx = perm[n_train:].to(device)

    bundle = dict(
        nf=nf_n, le=le_n, re=re_n, hl=hl, hr=hr, Y=Y_n,
        train_idx=train_idx, val_idx=val_idx,
        n_train=n_train, n_val=n_val,
        node_mean=node_mean, node_std=node_std,
        edge_mean=edge_mean, edge_std=edge_std,
        y_mean=y_mean, y_std=y_std,
        node_dim=node_feat.shape[1],
        edge_dim=left_edge_feat.shape[1],
        output_dim=Y.shape[1],
    )
    _DATASET_CACHE[key] = bundle
    return bundle


def train_one(hidden_dim, data_path, epochs, batch_size, lr,
              n_message_passes, device, out_dir):
    """Train a single BeadMPGNN config and save best checkpoint.

    Data is held GPU-resident; batches are formed by indexing into device
    tensors. Eliminates DataLoader overhead and host->device transfers.
    """
    b = _load_dataset_to_device(data_path, device)

    model = BeadMPGNN(
        node_dim=b["node_dim"], edge_dim=b["edge_dim"],
        output_dim=b["output_dim"],
        hidden_dim=hidden_dim, n_message_passes=n_message_passes,
    ).to(device)
    model.node_mean.copy_(b["node_mean"].to(model.node_mean.device))
    model.node_std.copy_(b["node_std"].to(model.node_std.device))
    model.edge_mean.copy_(b["edge_mean"].to(model.edge_mean.device))
    model.edge_std.copy_(b["edge_std"].to(model.edge_std.device))
    model.y_mean.copy_(b["y_mean"].to(model.y_mean.device))
    model.y_std.copy_(b["y_std"].to(model.y_std.device))

    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    crit = nn.MSELoss()

    os.makedirs(out_dir, exist_ok=True)
    best_val = float("inf")
    train_losses, val_losses = [], []
    t0 = time.time()

    train_idx = b["train_idx"]
    val_idx = b["val_idx"]
    n_train = b["n_train"]
    n_val = b["n_val"]

    for epoch in range(1, epochs + 1):
        # Shuffle once per epoch (on device)
        shuffle = train_idx[torch.randperm(n_train, device=device)]
        model.train()
        tr_loss = torch.zeros(1, device=device)
        for start in range(0, n_train, batch_size):
            idx = shuffle[start:start + batch_size]
            nf = b["nf"][idx]; le = b["le"][idx]; re = b["re"][idx]
            hl = b["hl"][idx]; hr = b["hr"][idx]; y = b["Y"][idx]
            pred = model(nf, le, re, hl, hr)
            loss = crit(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.detach() * nf.shape[0]
        tr_loss = (tr_loss / n_train).item()
        train_losses.append(tr_loss)

        model.eval()
        v_loss = torch.zeros(1, device=device)
        with torch.no_grad():
            for start in range(0, n_val, batch_size):
                idx = val_idx[start:start + batch_size]
                nf = b["nf"][idx]; le = b["le"][idx]; re = b["re"][idx]
                hl = b["hl"][idx]; hr = b["hr"][idx]; y = b["Y"][idx]
                pred = model(nf, le, re, hl, hr)
                v_loss += crit(pred, y) * nf.shape[0]
        v_loss = (v_loss / n_val).item()
        val_losses.append(v_loss)
        sched.step()

        if v_loss < best_val:
            best_val = v_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": v_loss,
                "args": dict(hidden_dim=hidden_dim,
                             n_message_passes=n_message_passes,
                             epochs=epochs, batch_size=batch_size, lr=lr),
                "node_mean": b["node_mean"], "node_std": b["node_std"],
                "edge_mean": b["edge_mean"], "edge_std": b["edge_std"],
                "y_mean": b["y_mean"], "y_std": b["y_std"],
            }, os.path.join(out_dir, "mpgnn_best.pt"))

        if epoch == 1 or epoch % 25 == 0 or epoch == epochs:
            print(f"    epoch {epoch:4d}/{epochs}  "
                  f"train={tr_loss:.4e}  val={v_loss:.4e}  "
                  f"best={best_val:.4e}  ({time.time()-t0:.0f}s)",
                  flush=True)

    np.savez(os.path.join(out_dir, "losses.npz"),
             train=np.array(train_losses), val=np.array(val_losses))
    return dict(n_params=n_params, best_val_loss=best_val,
                final_val_loss=val_losses[-1],
                train_time_s=time.time() - t0)


def load_model_from_ckpt(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args = ckpt["args"]
    model = BeadMPGNN(
        node_dim=4, edge_dim=6, output_dim=4,
        hidden_dim=args["hidden_dim"],
        n_message_passes=args.get("n_message_passes", 1),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def step_gt(state, dt, gravity):
    forces = compute_forces(state, gravity)
    acc = forces / state.mass[:, None]
    acc[state.fixed] = 0.0
    state.vel += dt * acc
    state.vel[state.fixed] = 0.0
    state.pos += dt * state.vel


def step_gnn(state, model, device):
    pos = torch.from_numpy(state.pos).float().to(device)
    vel = torch.from_numpy(state.vel).float().to(device)
    mass = torch.from_numpy(state.mass).float().to(device)
    is_fixed = torch.from_numpy(state.fixed).to(device)
    edges = torch.from_numpy(state.edges).long().to(device)
    rest_lengths = torch.from_numpy(state.rest_lengths).float().to(device)
    with torch.no_grad():
        pos_new, vel_new = model.step_chain(pos, vel, mass, is_fixed,
                                            edges, rest_lengths)
    state.pos = pos_new.cpu().numpy()
    state.vel = vel_new.cpu().numpy()


def rollout_eval(n_beads, model, device, n_steps, dt, gravity, taper_ratio):
    params = dict(n_nodes=n_beads, total_length=2.0, total_mass=0.5,
                  stiffness=1e4, damping=0.5, drag=0.02,
                  taper_ratio=taper_ratio)
    state_gt = create_chain(**params)
    state_gnn = create_chain(**params)
    anchor = state_gt.pos[0].copy()

    sample_int = max(1, n_steps // 100)
    pos_errors, rod_errs = [], []
    for step in range(1, n_steps + 1):
        step_gt(state_gt, dt, gravity)
        state_gt.pos[0] = anchor; state_gt.vel[0] = 0.0
        step_gnn(state_gnn, model, device)
        state_gnn.pos[0] = anchor; state_gnn.vel[0] = 0.0
        if step % sample_int == 0:
            err = np.linalg.norm(state_gnn.pos - state_gt.pos, axis=-1).mean()
            pos_errors.append(err)
            rod_e = []
            for e in range(len(state_gnn.edges)):
                i, j = state_gnn.edges[e]
                d = np.linalg.norm(state_gnn.pos[j] - state_gnn.pos[i])
                rod_e.append(abs(d - state_gnn.rest_lengths[e])
                             / state_gnn.rest_lengths[e])
            rod_errs.append(np.max(rod_e))
    return dict(pos_err_final=pos_errors[-1],
                pos_err_mean=float(np.mean(pos_errors)),
                rod_err_max=float(np.max(rod_errs)),
                pos_errors=pos_errors)


def main(args):
    hidden_dims = [int(h) for h in args.hidden_dims.split(",")]
    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eval_device = torch.device("cpu")
    chain_lengths = [int(n) for n in args.chain_lengths.split(",")]

    print(f"Sweep over hidden_dim = {hidden_dims}")
    print(f"Train device: {train_device} | Eval device: {eval_device}")
    print(f"Eval chain lengths: {chain_lengths}, n_steps={args.eval_steps}")
    print()

    sweep_dir = os.path.join(args.output_dir, "sweep")
    os.makedirs(sweep_dir, exist_ok=True)

    results = []
    for h in hidden_dims:
        run_dir = os.path.join(sweep_dir, f"h{h:03d}")
        print(f"=== hidden_dim={h}  ->  {run_dir} ===", flush=True)

        ckpt_path = os.path.join(run_dir, "mpgnn_best.pt")
        if os.path.exists(ckpt_path) and not args.force_retrain:
            ckpt = torch.load(ckpt_path, map_location="cpu",
                              weights_only=False)
            # Count params from a fresh model instance with same hidden_dim
            tmp = BeadMPGNN(node_dim=4, edge_dim=6, output_dim=4,
                            hidden_dim=h,
                            n_message_passes=args.n_message_passes)
            n_params = sum(p.numel() for p in tmp.parameters())
            train_info = dict(
                n_params=n_params,
                best_val_loss=float(ckpt["val_loss"]),
                final_val_loss=float(ckpt["val_loss"]),
                train_time_s=0.0,
            )
            print(f"  found existing checkpoint, skipping train: "
                  f"{n_params:,} params, val_loss={ckpt['val_loss']:.4e}",
                  flush=True)
        else:
            train_info = train_one(
                hidden_dim=h,
                data_path=args.data,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                n_message_passes=args.n_message_passes,
                device=train_device,
                out_dir=run_dir,
            )
            print(f"  trained: {train_info['n_params']:,} params, "
                  f"best_val={train_info['best_val_loss']:.4e}, "
                  f"time={train_info['train_time_s']:.0f}s", flush=True)

        model = load_model_from_ckpt(os.path.join(run_dir, "mpgnn_best.pt"),
                                     eval_device)
        chain_results = {}
        for n in chain_lengths:
            t0 = time.time()
            r = rollout_eval(n, model, eval_device, args.eval_steps,
                             dt=0.0001, gravity=9.81,
                             taper_ratio=args.taper_ratio)
            chain_results[n] = r
            print(f"  eval N={n}: pos_err_final={r['pos_err_final']:.4e}, "
                  f"rod_err_max={r['rod_err_max']:.4e}, "
                  f"({time.time()-t0:.0f}s)")

        entry = dict(hidden_dim=h, **train_info,
                     chain_results={int(k): v for k, v in chain_results.items()})
        results.append(entry)

        # Incremental save in case run is interrupted
        with open(os.path.join(sweep_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2, default=lambda o: o.tolist()
                      if hasattr(o, "tolist") else float(o))

    # Plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({'font.size': 14, 'axes.titlesize': 16,
                             'axes.labelsize': 14, 'legend.fontsize': 12})

        params_arr = np.array([r["n_params"] for r in results])
        val_arr = np.array([r["best_val_loss"] for r in results])

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        ax = axes[0]
        ax.loglog(params_arr, val_arr, "o-", linewidth=2, markersize=10,
                  color="tab:blue")
        for r in results:
            ax.annotate(f"h={r['hidden_dim']}",
                        (r["n_params"], r["best_val_loss"]),
                        textcoords="offset points", xytext=(8, -4),
                        fontsize=11)
        ax.set_xlabel("Parameter count")
        ax.set_ylabel("Best validation MSE (normalized)")
        ax.set_title("Training accuracy vs. model size")
        ax.grid(True, which="both", alpha=0.3)

        ax = axes[1]
        for n in chain_lengths:
            errs = [r["chain_results"][n]["pos_err_final"] for r in results]
            ax.loglog(params_arr, errs, "o-", label=f"N={n}", linewidth=2,
                      markersize=8)
        ax.set_xlabel("Parameter count")
        ax.set_ylabel("Final rollout position error (m)")
        ax.set_title(f"Rollout error vs. model size ({args.eval_steps} steps)")
        ax.legend()
        ax.grid(True, which="both", alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(sweep_dir, "param_sweep.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"\nResults JSON: {os.path.join(sweep_dir, 'results.json')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="volume_data_v2.npz")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--hidden_dims", type=str, default="4,8,16,32,64,128")
    p.add_argument("--n_message_passes", type=int, default=1)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--chain_lengths", type=str, default="8,12,16,24")
    p.add_argument("--eval_steps", type=int, default=10000)
    p.add_argument("--taper_ratio", type=float, default=10.0)
    p.add_argument("--force_retrain", action="store_true",
                   help="Retrain even if a checkpoint already exists.")
    args = p.parse_args()
    main(args)
