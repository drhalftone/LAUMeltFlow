"""
Training script for H2GNN electrostatic field predictor.
"""

import torch
import torch.nn.functional as F
import argparse
import os
import time
from typing import Optional

from .quadtree import Quadtree
from .h2gnn import H2GNN
from .dataset import ElectrostaticDataset


def train(
    n_epochs: int = 100,
    n_train: int = 1000,
    n_val: int = 200,
    n_particles_range: tuple = (10, 100),
    max_depth: int = 4,
    hidden_dim: int = 64,
    n_mlp_layers: int = 2,
    lr: float = 1e-4,
    weight_decay: float = 0.0,
    checkpoint_dir: str = "checkpoints",
    checkpoint_every: int = 10,
    device: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True
):
    """
    Train H2GNN model.

    Args:
        n_epochs: number of training epochs
        n_train: number of training samples
        n_val: number of validation samples
        n_particles_range: (min, max) particles per sample
        max_depth: quadtree depth
        hidden_dim: hidden dimension for MLPs
        n_mlp_layers: number of layers in each MLP
        lr: learning rate
        weight_decay: L2 regularization
        checkpoint_dir: directory for saving checkpoints
        checkpoint_every: save checkpoint every N epochs
        device: 'cuda', 'mps', 'cpu', or None for auto-detect
        seed: random seed
        verbose: print training progress
    """
    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    device = torch.device(device)

    if verbose:
        print(f"Training H2GNN on {device}")
        print(f"  Epochs: {n_epochs}")
        print(f"  Training samples: {n_train}")
        print(f"  Validation samples: {n_val}")
        print(f"  Particles per sample: {n_particles_range}")
        print(f"  Quadtree depth: {max_depth}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  MLP layers: {n_mlp_layers}")
        print(f"  Learning rate: {lr}")
        print()

    # Create datasets
    if verbose:
        print("Generating datasets...")
    train_dataset = ElectrostaticDataset(
        n_samples=n_train,
        n_particles_range=n_particles_range,
        seed=seed,
        device=device
    )
    val_dataset = ElectrostaticDataset(
        n_samples=n_val,
        n_particles_range=n_particles_range,
        seed=seed + 1000,
        device=device
    )
    if verbose:
        print(f"  Training: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print()

    # Create model
    model = H2GNN(
        max_depth=max_depth,
        hidden_dim=hidden_dim,
        n_mlp_layers=n_mlp_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"Model parameters: {n_params:,}")
        print()

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=verbose
    )

    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_count = 0

        for particles, E_true in train_dataset:
            # Build quadtree
            positions = particles[:, :2]
            quadtree = Quadtree(positions, max_depth=max_depth)

            # Forward pass
            E_pred = model(particles, quadtree)

            # Loss: MSE on E field
            loss = F.mse_loss(E_pred, E_true)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        train_loss /= train_count

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for particles, E_true in val_dataset:
                positions = particles[:, :2]
                quadtree = Quadtree(positions, max_depth=max_depth)
                E_pred = model(particles, quadtree)
                val_loss += F.mse_loss(E_pred, E_true).item()
                val_count += 1

        val_loss /= val_count

        # Update scheduler
        scheduler.step(val_loss)

        # Record losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start

        if verbose:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1:3d}/{n_epochs}: "
                  f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                  f"lr={current_lr:.2e}, time={epoch_time:.1f}s")

        # Save checkpoint
        if (epoch + 1) % checkpoint_every == 0:
            checkpoint_path = os.path.join(
                checkpoint_dir, f"h2gnn_epoch{epoch+1}.pt"
            )
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'max_depth': max_depth,
                    'hidden_dim': hidden_dim,
                    'n_mlp_layers': n_mlp_layers,
                }
            }, checkpoint_path)
            if verbose:
                print(f"  Saved checkpoint: {checkpoint_path}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, "h2gnn_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'max_depth': max_depth,
                    'hidden_dim': hidden_dim,
                    'n_mlp_layers': n_mlp_layers,
                }
            }, best_path)

    if verbose:
        print()
        print(f"Training complete!")
        print(f"Best validation loss: {best_val_loss:.6f}")

    return model, train_losses, val_losses


def main():
    parser = argparse.ArgumentParser(description='Train H2GNN')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-train', type=int, default=1000)
    parser.add_argument('--n-val', type=int, default=200)
    parser.add_argument('--min-particles', type=int, default=10)
    parser.add_argument('--max-particles', type=int, default=100)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--mlp-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    train(
        n_epochs=args.epochs,
        n_train=args.n_train,
        n_val=args.n_val,
        n_particles_range=(args.min_particles, args.max_particles),
        max_depth=args.depth,
        hidden_dim=args.hidden_dim,
        n_mlp_layers=args.mlp_layers,
        lr=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
