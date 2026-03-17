"""Train the U-Net GNN to learn SHAKE constraint projection.

Usage:
    python train.py
    python train.py --data_dir ../data --epochs 200 --hidden_dim 32 --lr 1e-3
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import BeadChainDataset
from model import BeadChainUNet, build_chain_adj, build_tree_children


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Dataset ---
    dataset = BeadChainDataset(args.data_dir)
    n_beads = dataset.n_beads

    # Train/val split (90/10)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # Graph structure
    chain_adj = torch.from_numpy(build_chain_adj(n_beads)).long().to(device)
    tree_children = build_tree_children(n_beads)

    # --- Model ---
    model = BeadChainUNet(
        state_dim=7,
        output_dim=4,
        hidden_dim=args.hidden_dim,
        n_beads=n_beads,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Training ---
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    criterion = nn.MSELoss()

    os.makedirs(args.output_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")

    n_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader, 1):
            # x_batch is (B, 31, 7) but we only need beads (B, 16, 7)
            bead_states = x_batch[:, :n_beads, :].to(device)
            y_batch = y_batch.to(device)

            pred = model(bead_states, chain_adj, tree_children)

            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * bead_states.shape[0]

            if batch_idx % max(1, n_batches // 5) == 0:
                print(f"\r  Epoch {epoch}/{args.epochs}  "
                      f"batch {batch_idx}/{n_batches}  "
                      f"loss={loss.item():.2e}", end="", flush=True)

        epoch_loss /= n_train
        train_losses.append(epoch_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                bead_states = x_batch[:, :n_beads, :].to(device)
                y_batch = y_batch.to(device)
                pred = model(bead_states, chain_adj, tree_children)
                val_loss += criterion(pred, y_batch).item() * bead_states.shape[0]

        val_loss /= n_val
        val_losses.append(val_loss)

        scheduler.step()

        # --- Logging ---
        lr = optimizer.param_groups[0]["lr"]
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"\rEpoch {epoch:4d}/{args.epochs}  "
                  f"train={epoch_loss:.2e}  val={val_loss:.2e}  lr={lr:.1e}"
                  f"                    ")

        # --- Save best ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "args": vars(args),
            }, os.path.join(args.output_dir, "best_model.pt"))

    # --- Save final ---
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "val_loss": val_losses[-1],
        "args": vars(args),
    }, os.path.join(args.output_dir, "final_model.pt"))

    np.savez(
        os.path.join(args.output_dir, "losses.npz"),
        train=np.array(train_losses),
        val=np.array(val_losses),
    )

    # --- Plot ---
    try:
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        ax1.semilogy(train_losses, label="Train", alpha=0.8)
        ax1.semilogy(val_losses, label="Val", alpha=0.8)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("MSE Loss")
        ax1.set_title("Training Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        model.eval()
        with torch.no_grad():
            x_sample, y_sample = dataset[len(dataset) // 2]
            bead_sample = x_sample[:n_beads, :].unsqueeze(0).to(device)
            y_sample = y_sample.numpy()
            pred_sample = model(bead_sample, chain_adj, tree_children)
            pred_sample = pred_sample[0].cpu().numpy()

        beads = np.arange(n_beads)
        ax2.plot(beads, y_sample[:, 0], "b-o", markersize=3, label="Target dx")
        ax2.plot(beads, pred_sample[:, 0], "r--x", markersize=3, label="Pred dx")
        ax2.plot(beads, y_sample[:, 1], "g-o", markersize=3, label="Target dy")
        ax2.plot(beads, pred_sample[:, 1], "m--x", markersize=3, label="Pred dy")
        ax2.set_xlabel("Bead index")
        ax2.set_ylabel("Position correction (m)")
        ax2.set_title("Sample Prediction vs Target")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "training_results.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
        plt.close(fig)

    except Exception as e:
        print(f"Plotting skipped: {e}")

    print(f"\nBest val loss: {best_val_loss:.2e}")
    print(f"Models saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net GNN on bead chain data")
    parser.add_argument("--data_dir", type=str, default="../data")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()

    train(args)
