"""
Leaf-level autoencoder training.

Tests that scatter → MLP → gather can reconstruct particle features.
This verifies the basic operations work before adding hierarchy.

Usage:
    python -m electrostatic_unet.train_autoencoder --epochs 100 --n-train 10000
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from electrostatic_unet.dataset import ElectrostaticDataset
from electrostatic_unet.quadtree import Quadtree


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


class LeafAutoencoder(nn.Module):
    """
    Simple autoencoder using scatter/gather at leaf level only.

    Architecture:
        1. Particle features (x, y, q) → encoder MLP → hidden
        2. Scatter (sum) to leaf voxels
        3. Leaf MLP processes aggregated features
        4. Gather (broadcast) back to particles
        5. Decoder MLP → reconstructed (x, y, q)
    """

    def __init__(self, particle_dim=3, hidden_dim=64, n_layers=2):
        super().__init__()

        self.particle_dim = particle_dim
        self.hidden_dim = hidden_dim

        # Encoder: particle features → hidden
        encoder_layers = [nn.Linear(particle_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            encoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.encoder = nn.Sequential(*encoder_layers)

        # Leaf MLP: processes aggregated voxel features
        leaf_layers = [nn.Linear(hidden_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            leaf_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        self.leaf_mlp = nn.Sequential(*leaf_layers)

        # Decoder: hidden + particle features → reconstructed features
        decoder_layers = [nn.Linear(hidden_dim + particle_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            decoder_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        decoder_layers.append(nn.Linear(hidden_dim, particle_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, particles: torch.Tensor, quadtree: Quadtree) -> torch.Tensor:
        """
        Forward pass: encode → scatter → leaf MLP → gather → decode.

        Args:
            particles: (N, 3) particle features (x, y, q)
            quadtree: Quadtree structure

        Returns:
            reconstructed: (N, 3) reconstructed particle features
        """
        N = particles.shape[0]
        device = particles.device

        # Encode particles
        h_particles = self.encoder(particles)  # (N, hidden)

        # Scatter to leaf voxels (sum aggregation)
        particle_to_leaf = quadtree.particle_to_leaf  # (N,) tensor of indices
        n_leaves = len(quadtree.active_voxels[quadtree.max_depth])

        # Sum scatter
        h_leaves = torch.zeros(n_leaves, self.hidden_dim, device=device)
        particle_to_leaf_tensor = torch.tensor(particle_to_leaf, device=device, dtype=torch.long)
        h_leaves.scatter_add_(0, particle_to_leaf_tensor.unsqueeze(1).expand(-1, self.hidden_dim), h_particles)

        # Leaf MLP
        h_leaves = self.leaf_mlp(h_leaves)  # (n_leaves, hidden)

        # Gather back to particles (broadcast)
        h_gathered = h_leaves[particle_to_leaf_tensor]  # (N, hidden)

        # Decode with skip connection to original features
        h_combined = torch.cat([h_gathered, particles], dim=-1)  # (N, hidden + 3)
        reconstructed = self.decoder(h_combined)  # (N, 3)

        return reconstructed


def train_epoch(model, dataset, optimizer, device):
    """Train for one epoch, return average loss."""
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for i in range(len(dataset)):
        particles, _ = dataset[i]  # Ignore E_true, we're doing reconstruction
        particles = particles.to(device)

        # Build quadtree from positions
        positions = particles[:, :2]
        quadtree = Quadtree(positions, max_depth=4)

        # Forward pass
        optimizer.zero_grad()
        reconstructed = model(particles, quadtree)

        loss = criterion(reconstructed, particles)
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
            particles, _ = dataset[i]
            particles = particles.to(device)

            positions = particles[:, :2]
            quadtree = Quadtree(positions, max_depth=4)
            reconstructed = model(particles, quadtree)

            loss = criterion(reconstructed, particles)
            total_loss += loss.item()

    return total_loss / len(dataset)


def main():
    parser = argparse.ArgumentParser(description='Train leaf-level autoencoder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--n-val', type=int, default=500)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-plot', type=str, default='autoencoder_curve.png')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flush_print("=" * 60)
    flush_print("Leaf-Level Autoencoder Training")
    flush_print("=" * 60)
    flush_print(f"Device: {device}")
    flush_print(f"Epochs: {args.epochs}")
    flush_print(f"Training samples: {args.n_train}")
    flush_print(f"Validation samples: {args.n_val}")
    flush_print(f"Hidden dim: {args.hidden_dim}")
    flush_print(f"MLP layers: {args.n_layers}")
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
    model = LeafAutoencoder(
        particle_dim=3,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
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

        train_loss = train_epoch(model, train_dataset, optimizer, device)
        val_loss = validate(model, val_dataset, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        elapsed = time.time() - start_time
        lr = optimizer.param_groups[0]['lr']

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            status = "* best"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f"{args.checkpoint_dir}/autoencoder_best.pt")

        flush_print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {lr:>10.2e} | {elapsed:>7.1f}s | {status}")

        # Save plot every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            plt.figure(figsize=(10, 6))
            plt.plot(train_losses, 'b-', label='Train MSE', linewidth=2)
            plt.plot(val_losses, 'r-', label='Val MSE', linewidth=2)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('MSE Loss', fontsize=12)
            plt.title(f'Leaf Autoencoder Training (Epoch {epoch})', fontsize=14)
            plt.legend(fontsize=11)
            plt.grid(True, alpha=0.3)
            plt.yscale('log')
            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            plt.close()
            flush_print(f"       | Plot saved to {args.save_plot}")

    flush_print("-" * 70)
    flush_print(f"Training complete! Best validation MSE: {best_val_loss:.6f}")
    flush_print(f"Final plot saved to: {args.save_plot}")


if __name__ == '__main__':
    main()
