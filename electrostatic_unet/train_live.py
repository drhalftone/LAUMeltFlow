"""
Training script with live MSE updates and plotting.

Usage:
    python -m electrostatic_unet.train_live --epochs 100 --n-train 10000
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from electrostatic_unet.h2gnn import H2GNN
from electrostatic_unet.dataset import ElectrostaticDataset
from electrostatic_unet.quadtree import Quadtree


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


def train_epoch(model, dataset, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for i in range(len(dataset)):
        particles, E_true = dataset[i]
        particles = particles.to(device)
        E_true = E_true.to(device)

        # Build quadtree from positions
        positions = particles[:, :2]
        quadtree = Quadtree(positions, max_depth=4)

        # Forward pass
        optimizer.zero_grad()
        E_pred = model(particles, quadtree)

        loss = criterion(E_pred, E_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset)


def validate(model, dataset, device):
    """Validate and return average loss."""
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for i in range(len(dataset)):
            particles, E_true = dataset[i]
            particles = particles.to(device)
            E_true = E_true.to(device)

            # Build quadtree from positions
            positions = particles[:, :2]
            quadtree = Quadtree(positions, max_depth=4)
            E_pred = model(particles, quadtree)

            loss = criterion(E_pred, E_true)
            total_loss += loss.item()

    return total_loss / len(dataset)


def main():
    parser = argparse.ArgumentParser(description='Train H2GNN with live updates')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--n-val', type=int, default=500)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--save-plot', type=str, default='training_curve.png')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flush_print("=" * 60)
    flush_print("H2GNN Training with Live Updates")
    flush_print("=" * 60)
    flush_print(f"Device: {device}")
    flush_print(f"Epochs: {args.epochs}")
    flush_print(f"Training samples: {args.n_train}")
    flush_print(f"Validation samples: {args.n_val}")
    flush_print(f"Hidden dim: {args.hidden_dim}")
    flush_print(f"Learning rate: {args.lr}")
    flush_print()

    # Generate datasets
    flush_print("Generating datasets...")
    train_dataset = ElectrostaticDataset(n_samples=args.n_train, n_particles_range=(10, 100))
    val_dataset = ElectrostaticDataset(n_samples=args.n_val, n_particles_range=(10, 100))
    flush_print(f"  Training: {len(train_dataset)} samples")
    flush_print(f"  Validation: {len(val_dataset)} samples")
    flush_print()

    # Create model
    model = H2GNN(
        particle_dim=3,  # x, y, q
        hidden_dim=args.hidden_dim,
        output_dim=2,  # Ex, Ey
        max_depth=args.depth,
        n_mlp_layers=2
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    flush_print(f"Model parameters: {n_params:,}")
    flush_print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    # Create checkpoint directory
    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    # Header
    flush_print(f"{'Epoch':>6} | {'Train MSE':>12} | {'Val MSE':>12} | {'LR':>10} | {'Time':>8} | Status")
    flush_print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        # Train
        train_loss = train_epoch(model, train_dataset, optimizer, device)

        # Validate
        val_loss = validate(model, val_dataset, device)

        # Update scheduler
        scheduler.step(val_loss)

        # Record history
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']

        # Status
        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            status = "* best"
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f"{args.checkpoint_dir}/h2gnn_best.pt")

        flush_print(f"{epoch:>6} | {train_loss:>12.4f} | {val_loss:>12.4f} | {lr:>10.2e} | {elapsed:>7.1f}s | {status}")

        # Save plot every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, 'b-', label='Train MSE', linewidth=2)
            plt.plot(val_losses, 'r-', label='Val MSE', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('MSE Loss', fontsize=12)
            plt.title(f'H2GNN Training Progress (Epoch {epoch})', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            plt.close()
            flush_print(f"       | Plot saved to {args.save_plot}")

        # Save checkpoint every 20 epochs
        if epoch % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, f"{args.checkpoint_dir}/h2gnn_epoch{epoch}.pt")
            flush_print(f"       | Checkpoint saved: h2gnn_epoch{epoch}.pt")

    flush_print("-" * 70)
    flush_print(f"Training complete! Best validation MSE: {best_val_loss:.4f}")
    flush_print(f"Final plot saved to: {args.save_plot}")


if __name__ == '__main__':
    main()
