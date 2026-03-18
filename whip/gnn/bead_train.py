"""Train the per-bead GNN on volume-sampled data.

Usage:
    python bead_train.py
    python bead_train.py --data volume_data.npz --epochs 200 --hidden_dim 64 --lr 1e-3
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from bead_model import BeadGNN


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    print(f"Loading {args.data}...")
    d = np.load(args.data)
    X = torch.from_numpy(d["X"])  # (N, 16)
    Y = torch.from_numpy(d["Y"])  # (N, 4)
    print(f"  {X.shape[0]:,} samples, {X.shape[1]} input features, {Y.shape[1]} targets")

    # --- Compute normalization stats ---
    x_mean = X.mean(dim=0)
    x_std = X.std(dim=0).clamp(min=1e-8)
    y_mean = Y.mean(dim=0)
    y_std = Y.std(dim=0).clamp(min=1e-8)

    X_norm = (X - x_mean) / x_std
    Y_norm = (Y - y_mean) / y_std

    # --- Train/val split ---
    dataset = TensorDataset(X_norm, Y_norm)
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
    model = BeadGNN(
        input_dim=X.shape[1],
        output_dim=Y.shape[1],
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model: {args.n_layers} ResMLPBlocks, hidden_dim={args.hidden_dim}, {n_params:,} params")

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
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            pred = model(x_batch)
            loss = criterion(pred, y_batch)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * x_batch.shape[0]

        epoch_loss /= n_train
        train_losses.append(epoch_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                pred = model(x_batch)
                val_loss += criterion(pred, y_batch).item() * x_batch.shape[0]

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
                "x_mean": x_mean,
                "x_std": x_std,
                "y_mean": y_mean,
                "y_std": y_std,
            }, os.path.join(args.output_dir, "bead_best.pt"))

    # --- Save final ---
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_loss": val_losses[-1],
        "args": vars(args),
        "x_mean": x_mean,
        "x_std": x_std,
        "y_mean": y_mean,
        "y_std": y_std,
    }, os.path.join(args.output_dir, "bead_final.pt"))

    np.savez(
        os.path.join(args.output_dir, "bead_losses.npz"),
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
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Sample predictions vs targets (denormalized)
        ax = axes[1]
        model.eval()
        with torch.no_grad():
            # Grab a batch from val set
            x_sample, y_sample = next(iter(val_loader))
            x_sample = x_sample[:200].to(device)
            y_sample = y_sample[:200]
            pred_sample = model(x_sample).cpu()

        # Denormalize
        y_true = y_sample * y_std + y_mean
        y_pred = pred_sample * y_std + y_mean

        # Plot d_vel_x predicted vs true (largest dynamic range)
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
        plot_path = os.path.join(args.output_dir, "bead_training.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print(f"\nBest val loss: {best_val_loss:.4e}")
    print(f"Models saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train per-bead GNN")
    parser.add_argument("--data", type=str, default="volume_data.npz")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=5)
    args = parser.parse_args()
    train(args)
