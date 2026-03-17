"""Train the Hamiltonian GNN on conservative bead chain trajectories.

Loss: MAE between predicted and ground-truth trajectories over short
rollouts (chunk_len steps).

Usage:
    python train.py
    python train.py --hidden_dim 64 --epochs 200 --lr 3e-3
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split

from model import (
    HamiltonianGNN, ConstrainedDynamics, integrate_trajectory,
    build_edge_index, build_edge_pairs,
)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load data ---
    data_path = os.path.join(os.path.dirname(__file__), "data", "trajectories.npz")
    data = np.load(data_path)

    all_q = torch.tensor(data["q"], dtype=torch.float32)   # (N_chunks, C, N, 2)
    all_p = torch.tensor(data["p"], dtype=torch.float32)
    mass = data["mass"]                                     # (N,)
    rest_lengths = data["rest_lengths"]                      # (n_rods,)
    n_nodes = int(data["n_nodes"])
    dt_save = float(data["dt_save"])
    gravity = float(data["gravity"])

    print(f"Loaded {len(all_q)} chunks, shape: {all_q.shape}")
    print(f"n_nodes={n_nodes}, dt_save={dt_save:.6f}, gravity={gravity}")

    # --- Graph structure ---
    edge_index = build_edge_index(n_nodes)
    edge_pairs = build_edge_pairs(n_nodes)
    fixed_mask = np.zeros(n_nodes, dtype=bool)
    fixed_mask[0] = True

    # --- Dataset ---
    dataset = TensorDataset(all_q, all_p)
    n_val = max(1, len(dataset) // 10)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    print(f"Train: {n_train}, Val: {n_val}")

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size,
                            shuffle=False, num_workers=0)

    # --- Model ---
    model = HamiltonianGNN(
        n_beads=n_nodes,
        hidden_dim=args.hidden_dim,
        n_message_passes=args.n_message_passes,
        mass=mass,
        rest_lengths=rest_lengths,
        edge_index=edge_index,
        fixed_mask=fixed_mask,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")

    # --- Constraints ---
    constraints = ConstrainedDynamics(
        edge_pairs=edge_pairs,
        rest_lengths=torch.tensor(rest_lengths, dtype=torch.float32).to(device),
        mass=torch.tensor(mass, dtype=torch.float32).to(device),
        fixed_mask=torch.tensor(fixed_mask, dtype=torch.bool).to(device),
    )

    # --- Training ---
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs,
    )

    os.makedirs(args.output_dir, exist_ok=True)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    chunk_len = all_q.shape[1]  # number of frames per chunk

    n_batches = len(train_loader)
    for epoch in range(1, args.epochs + 1):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for batch_idx, (q_chunk, p_chunk) in enumerate(train_loader, 1):
            q_chunk = q_chunk.to(device)  # (B, C, N, 2)
            p_chunk = p_chunk.to(device)

            B = q_chunk.shape[0]

            # Initial state
            q0 = q_chunk[:, 0]  # (B, N, 2)
            p0 = p_chunk[:, 0]

            # Integrate forward
            pred_q, pred_p = integrate_trajectory(
                model, constraints, q0, p0, dt_save, chunk_len - 1
            )
            # pred_q: (B, C, N, 2)

            # Loss: MAE on positions and momenta
            loss_q = (pred_q - q_chunk).abs().mean()
            loss_p = (pred_p - p_chunk).abs().mean()
            loss = loss_q + args.momentum_weight * loss_p

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            epoch_loss += loss.item() * B
            n_samples += B

            if batch_idx % max(1, n_batches // 5) == 0:
                print(f"\r  Epoch {epoch}/{args.epochs}  "
                      f"batch {batch_idx}/{n_batches}  "
                      f"loss={loss.item():.4e}", end="", flush=True)

        epoch_loss /= n_samples
        train_losses.append(epoch_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        n_val_samples = 0

        # NOTE: can't use torch.no_grad() here because time_derivatives
        # needs autograd to compute dH/dq and dH/dp internally.
        # model.eval() already disables create_graph in the grad call.
        for q_chunk, p_chunk in val_loader:
            q_chunk = q_chunk.to(device)
            p_chunk = p_chunk.to(device)
            B = q_chunk.shape[0]

            q0 = q_chunk[:, 0]
            p0 = p_chunk[:, 0]

            pred_q, pred_p = integrate_trajectory(
                model, constraints, q0, p0, dt_save, chunk_len - 1
            )

            loss_q = (pred_q - q_chunk).abs().mean()
            loss_p = (pred_p - p_chunk).abs().mean()
            loss = loss_q + args.momentum_weight * loss_p

            val_loss += loss.item() * B
            n_val_samples += B

        val_loss /= n_val_samples
        val_losses.append(val_loss)

        scheduler.step()

        # --- Logging ---
        lr = optimizer.param_groups[0]["lr"]
        if epoch % args.log_every == 0 or epoch == 1:
            print(f"\rEpoch {epoch:4d}/{args.epochs}  "
                  f"train={epoch_loss:.4e}  val={val_loss:.4e}  lr={lr:.1e}"
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

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.semilogy(train_losses, label="Train", alpha=0.8)
        ax.semilogy(val_losses, label="Val", alpha=0.8)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (MAE)")
        ax.set_title("HGNN Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(args.output_dir, "training_results.png")
        plt.savefig(plot_path, dpi=150)
        print(f"\nPlot saved to {plot_path}")
        plt.close(fig)
    except Exception as e:
        print(f"Plotting skipped: {e}")

    print(f"\nBest val loss: {best_val_loss:.4e}")
    print(f"Models saved to {args.output_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Hamiltonian GNN")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--n_message_passes", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--momentum_weight", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--log_every", type=int, default=1)
    args = parser.parse_args()

    train(args)
