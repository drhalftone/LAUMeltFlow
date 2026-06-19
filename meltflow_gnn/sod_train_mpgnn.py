"""Train SodMPGNN on uniform-grid Roe-flux training data.

Mirrors the bead_train_mpgnn structure but for the Sod problem. Uses the
GPU-resident data path from the bead param sweep.

Usage:
    python sod_train_mpgnn.py
    python sod_train_mpgnn.py --data data/sod_flux_data.npz --hidden_dim 32
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from sod_mpgnn import SodMPGNN


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load grid_sampler output
    print(f"Loading {args.data}...")
    d = np.load(args.data, allow_pickle=True)
    X = torch.from_numpy(d["inputs"].astype(np.float32))   # (N, 6)
    Y = torch.from_numpy(d["outputs"].astype(np.float32))  # (N, 3)
    print(f"  {X.shape[0]:,} samples")

    # Split into node features
    node_L = X[:, :3]   # (N, 3)
    node_R = X[:, 3:6]  # (N, 3)
    # Edge: constant dx (uniform mesh). The 1D Sod default in meltflow is
    # dx = 0.01. We still pass it through the encoder so the model has the
    # architectural slot, but with one constant value the encoder learns a
    # bias that the message MLP picks up.
    edge_feat = torch.full((X.shape[0], 1), float(args.dx))

    # Normalization stats: use the same mean/std for node_L and node_R since
    # they live in the same state space.
    all_nodes = torch.cat([node_L, node_R], dim=0)
    node_mean = all_nodes.mean(dim=0)
    node_std = all_nodes.std(dim=0).clamp(min=1e-8)

    edge_mean = edge_feat.mean(dim=0)
    edge_std = edge_feat.std(dim=0).clamp(min=1e-8)

    y_mean = Y.mean(dim=0)
    y_std = Y.std(dim=0).clamp(min=1e-8)

    # Pre-normalize and move to device
    node_L_n = ((node_L - node_mean) / node_std).to(device)
    node_R_n = ((node_R - node_mean) / node_std).to(device)
    edge_n = ((edge_feat - edge_mean) / edge_std).to(device)
    Y_n = ((Y - y_mean) / y_std).to(device)

    n_total = X.shape[0]
    n_val = max(1, n_total // 10)
    n_train = n_total - n_val
    perm = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx = perm[:n_train].to(device)
    val_idx = perm[n_train:].to(device)

    # Model
    model = SodMPGNN(
        node_dim=3, edge_dim=1, output_dim=3,
        hidden_dim=args.hidden_dim,
        n_message_passes=args.n_message_passes,
    ).to(device)
    model.node_mean.copy_(node_mean.to(model.node_mean.device))
    model.node_std.copy_(node_std.to(model.node_std.device))
    model.edge_mean.copy_(edge_mean.to(model.edge_mean.device))
    model.edge_std.copy_(edge_std.to(model.edge_std.device))
    model.y_mean.copy_(y_mean.to(model.y_mean.device))
    model.y_std.copy_(y_std.to(model.y_std.device))

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: SodMPGNN, hidden_dim={args.hidden_dim}, "
          f"K={args.n_message_passes}, {n_params:,} params")
    print(f"  Train: {n_train:,}, Val: {n_val:,}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        shuffle = train_idx[torch.randperm(n_train, device=device)]
        model.train()
        tr_loss = torch.zeros(1, device=device)
        for start in range(0, n_train, args.batch_size):
            idx = shuffle[start:start + args.batch_size]
            nl = node_L_n[idx]
            nr = node_R_n[idx]
            ef = edge_n[idx]
            y = Y_n[idx]
            pred = model.forward_pair(nl, nr, ef)
            loss = crit(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.detach() * nl.shape[0]
        tr_loss = (tr_loss / n_train).item()
        train_losses.append(tr_loss)

        model.eval()
        v_loss = torch.zeros(1, device=device)
        with torch.no_grad():
            for start in range(0, n_val, args.batch_size):
                idx = val_idx[start:start + args.batch_size]
                nl = node_L_n[idx]
                nr = node_R_n[idx]
                ef = edge_n[idx]
                y = Y_n[idx]
                pred = model.forward_pair(nl, nr, ef)
                v_loss += crit(pred, y) * nl.shape[0]
        v_loss = (v_loss / n_val).item()
        val_losses.append(v_loss)
        sched.step()

        if v_loss < best_val:
            best_val = v_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": v_loss,
                "args": vars(args),
                "node_mean": node_mean, "node_std": node_std,
                "edge_mean": edge_mean, "edge_std": edge_std,
                "y_mean": y_mean, "y_std": y_std,
            }, os.path.join(args.output_dir, "sod_mpgnn_best.pt"))

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            lr = opt.param_groups[0]["lr"]
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"train={tr_loss:.4e}  val={v_loss:.4e}  "
                  f"best={best_val:.4e}  lr={lr:.1e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    np.savez(os.path.join(args.output_dir, "sod_losses.npz"),
             train=np.array(train_losses), val=np.array(val_losses))

    # Plot training curves
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(train_losses, label="Train", alpha=0.85)
        ax.semilogy(val_losses, label="Val", alpha=0.85)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized MSE")
        ax.set_title(f"SodMPGNN training (h={args.hidden_dim}, K={args.n_message_passes})")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        out = os.path.join(args.output_dir, "sod_training.png")
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved to {out}")
        plt.close(fig)
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"\nBest val: {best_val:.4e}")
    print(f"Saved to {args.output_dir}/")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/sod_flux_data.npz")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--n_message_passes", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dx", type=float, default=0.01,
                   help="Cell width for the constant edge feature.")
    args = p.parse_args()
    train(args)
