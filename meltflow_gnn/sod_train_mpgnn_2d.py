"""Train SodMPGNN_2D on the 2D Sod (5-cell stencil) Roe-flux dataset.

Usage:
    python sod_train_mpgnn_2d.py
    python sod_train_mpgnn_2d.py --hidden_dim 32 --epochs 200
"""

import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

from sod_mpgnn_2d import SodMPGNN_2D


# Cell ordering in the input vector (matches grid_sampler_2d):
#   W (0:4), C (4:8), E (8:12), S (12:16), N (16:20)
# Edge ordering in the output vector:
#   F_w (0:4), F_e (4:8), G_s (8:12), G_n (12:16)
# In the model we order edges as: C-W, C-E, C-S, C-N (same as output ordering).

# Edge attrs: [dx_or_dy, normal_x, normal_y]
# C-W: (dx, -1, 0)
# C-E: (dx, +1, 0)
# C-S: (dy,  0, -1)
# C-N: (dy,  0, +1)


def build_edge_attrs(n_samples, dx, dy):
    """Return (n_samples, 4, 3) tensor of edge attrs for a uniform grid."""
    edges = torch.tensor([
        [dx, -1.0, 0.0],
        [dx,  1.0, 0.0],
        [dy,  0.0, -1.0],
        [dy,  0.0,  1.0],
    ], dtype=torch.float32)
    return edges.unsqueeze(0).expand(n_samples, -1, -1).contiguous()


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading {args.data}...")
    d = np.load(args.data, allow_pickle=True)
    X = torch.from_numpy(d["inputs"].astype(np.float32))   # (N, 20)
    Y = torch.from_numpy(d["outputs"].astype(np.float32))  # (N, 16)
    N = X.shape[0]
    print(f"  {N:,} samples")

    # Reshape to (N, 5, 4) and (N, 4, 4)
    nodes = X.reshape(N, 5, 4)
    fluxes = Y.reshape(N, 4, 4)

    # Build constant edge attrs for the uniform mesh
    edges = build_edge_attrs(N, args.dx, args.dy)  # (N, 4, 3)

    # Normalization: nodes share stats across cells; edges/Y separately
    node_mean = nodes.reshape(-1, 4).mean(dim=0)
    node_std = nodes.reshape(-1, 4).std(dim=0).clamp(min=1e-8)
    edge_mean = edges.reshape(-1, 3).mean(dim=0)
    edge_std = edges.reshape(-1, 3).std(dim=0).clamp(min=1e-8)
    y_mean = fluxes.reshape(-1, 4).mean(dim=0)
    y_std = fluxes.reshape(-1, 4).std(dim=0).clamp(min=1e-8)

    nodes_n = ((nodes - node_mean) / node_std).to(device)
    edges_n = ((edges - edge_mean) / edge_std).to(device)
    Y_n = ((fluxes - y_mean) / y_std).to(device)

    n_val = max(1, N // 10)
    n_train = N - n_val
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(42))
    train_idx = perm[:n_train].to(device)
    val_idx = perm[n_train:].to(device)

    model = SodMPGNN_2D(
        node_dim=4, edge_dim=3, output_dim=4,
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
    print(f"  Model: SodMPGNN_2D, h={args.hidden_dim}, K={args.n_message_passes}, "
          f"{n_params:,} params")
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
            n = nodes_n[idx]
            e = edges_n[idx]
            y = Y_n[idx]
            pred = model.forward_stencil(n, e)
            loss = crit(pred, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.detach() * n.shape[0]
        tr_loss = (tr_loss / n_train).item()
        train_losses.append(tr_loss)

        model.eval()
        v_loss = torch.zeros(1, device=device)
        with torch.no_grad():
            for start in range(0, n_val, args.batch_size):
                idx = val_idx[start:start + args.batch_size]
                n = nodes_n[idx]
                e = edges_n[idx]
                y = Y_n[idx]
                pred = model.forward_stencil(n, e)
                v_loss += crit(pred, y) * n.shape[0]
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
            }, os.path.join(args.output_dir, "sod_mpgnn_2d_best.pt"))

        if epoch == 1 or epoch % 25 == 0 or epoch == args.epochs:
            lr = opt.param_groups[0]["lr"]
            print(f"  epoch {epoch:4d}/{args.epochs}  "
                  f"train={tr_loss:.4e}  val={v_loss:.4e}  "
                  f"best={best_val:.4e}  lr={lr:.1e}  "
                  f"({time.time()-t0:.0f}s)", flush=True)

    np.savez(os.path.join(args.output_dir, "sod_2d_losses.npz"),
             train=np.array(train_losses), val=np.array(val_losses))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.semilogy(train_losses, label="Train", alpha=0.85)
        ax.semilogy(val_losses, label="Val", alpha=0.85)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Normalized MSE")
        ax.set_title(f"SodMPGNN_2D (h={args.hidden_dim}, K={args.n_message_passes})")
        ax.legend()
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        out = os.path.join(args.output_dir, "sod_2d_training.png")
        plt.savefig(out, dpi=150)
        print(f"\nPlot saved to {out}")
        plt.close(fig)
    except Exception as e:
        print(f"Plot skipped: {e}")

    print(f"\nBest val: {best_val:.4e}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/sod2d_flux_data.npz")
    p.add_argument("--output_dir", type=str, default="outputs")
    p.add_argument("--hidden_dim", type=int, default=32)
    p.add_argument("--n_message_passes", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dx", type=float, default=0.02)
    p.add_argument("--dy", type=float, default=0.02)
    args = p.parse_args()
    train(args)
