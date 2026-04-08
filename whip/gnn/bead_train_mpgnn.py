"""Train the message-passing GNN on variable-length volume-sampled data.

Usage:
    python bead_train_mpgnn.py
    python bead_train_mpgnn.py --data volume_data_v2.npz --epochs 200 --hidden_dim 64
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from bead_mpgnn import BeadMPGNN


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print(f"Loading {args.data}...")
    d = np.load(args.data)
    node_feat = torch.from_numpy(d["node_feat"])            # (N, 4)
    left_edge_feat = torch.from_numpy(d["left_edge_feat"])  # (N, 6)
    right_edge_feat = torch.from_numpy(d["right_edge_feat"])  # (N, 6)
    has_left = torch.from_numpy(d["has_left"])              # (N,) bool
    has_right = torch.from_numpy(d["has_right"])            # (N,) bool
    Y = torch.from_numpy(d["Y"])                            # (N, 4)

    print(f"  {node_feat.shape[0]:,} samples")
    print(f"  Node features: {node_feat.shape[1]}, Edge features: {left_edge_feat.shape[1]}")

    # --- Compute normalization stats (separate for node/edge) ---
    node_mean = node_feat.mean(dim=0)
    node_std = node_feat.std(dim=0).clamp(min=1e-8)

    # Edge stats computed over both left and right (same feature space)
    all_edges = torch.cat([left_edge_feat, right_edge_feat], dim=0)
    edge_mean = all_edges.mean(dim=0)
    edge_std = all_edges.std(dim=0).clamp(min=1e-8)

    y_mean = Y.mean(dim=0)
    y_std = Y.std(dim=0).clamp(min=1e-8)

    # Normalize
    node_feat_norm = (node_feat - node_mean) / node_std
    left_edge_norm = (left_edge_feat - edge_mean) / edge_std
    right_edge_norm = (right_edge_feat - edge_mean) / edge_std
    Y_norm = (Y - y_mean) / y_std

    # --- Train/val split ---
    dataset = TensorDataset(node_feat_norm, left_edge_norm, right_edge_norm,
                            has_left.float(), has_right.float(), Y_norm)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"  Train: {n_train:,}, Val: {n_val:,}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0, pin_memory=True)

    # --- Model ---
    model = BeadMPGNN(
        node_dim=node_feat.shape[1],
        edge_dim=left_edge_feat.shape[1],
        output_dim=Y.shape[1],
        hidden_dim=args.hidden_dim,
        n_message_passes=args.n_message_passes,
    ).to(device)

    # Store normalization stats as buffers
    model.node_mean.copy_(node_mean)
    model.node_std.copy_(node_std)
    model.edge_mean.copy_(edge_mean)
    model.edge_std.copy_(edge_std)
    model.y_mean.copy_(y_mean)
    model.y_std.copy_(y_std)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: BeadMPGNN, {args.n_message_passes} message pass(es), "
          f"hidden_dim={args.hidden_dim}, {n_params:,} params")

    # --- Training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            nf, le, re, hl, hr, y = [b.to(device) for b in batch]

            pred = model(nf, le, re, hl.bool(), hr.bool())
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * nf.shape[0]

        epoch_loss /= n_train
        train_losses.append(epoch_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                nf, le, re, hl, hr, y = [b.to(device) for b in batch]
                pred = model(nf, le, re, hl.bool(), hr.bool())
                val_loss += criterion(pred, y).item() * nf.shape[0]

        val_loss /= n_val
        val_losses.append(val_loss)
        scheduler.step()

        # --- Logging ---
        lr = optimizer.param_groups[0]["lr"]
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"Epoch {epoch:4d}/{args.epochs}  "
                  f"train={epoch_loss:.4e}  val={val_loss:.4e}  lr={lr:.1e}")

        # --- Save best ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
                "node_mean": node_mean,
                "node_std": node_std,
                "edge_mean": edge_mean,
                "edge_std": edge_std,
                "y_mean": y_mean,
                "y_std": y_std,
            }, os.path.join(args.output_dir, "mpgnn_best.pt"))

    # --- Save final ---
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_loss": val_losses[-1],
        "args": vars(args),
        "node_mean": node_mean,
        "node_std": node_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }, os.path.join(args.output_dir, "mpgnn_final.pt"))

    np.savez(
        os.path.join(args.output_dir, "mpgnn_losses.npz"),
        train=np.array(train_losses),
        val=np.array(val_losses),
    )

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Loss curves
        ax = axes[0]
        ax.semilogy(train_losses, label="Train", alpha=0.8)
        ax.semilogy(val_losses, label="Val", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss (normalized)")
        ax.set_title("MPGNN Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sample predictions vs targets (denormalized)
        ax = axes[1]
        model.eval()
        with torch.no_grad():
            batch = next(iter(val_loader))
            nf, le, re, hl, hr, y = [b[:200].to(device) for b in batch]
            pred = model(nf, le, re, hl.bool(), hr.bool()).cpu()
            y = y[:200].cpu()

        y_true = y * y_std + y_mean
        y_pred = pred * y_std + y_mean

        ax.scatter(y_true[:, 2].numpy(), y_pred[:, 2].numpy(),
                   s=2, alpha=0.5, c="tab:blue")
        lims = [y_true[:, 2].min().item(), y_true[:, 2].max().item()]
        ax.plot(lims, lims, "r--", linewidth=1, label="Perfect")
        ax.set_xlabel("True d_vel_x")
        ax.set_ylabel("Predicted d_vel_x")
        ax.set_title("Velocity Delta (x) — 200 val samples")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "mpgnn_training.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print(f"\nBest val loss: {best_val_loss:.4e}")
    print(f"Models saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train message-passing GNN")
    parser.add_argument("--data", type=str, default="volume_data_v2.npz")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_message_passes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()
    train(args)
