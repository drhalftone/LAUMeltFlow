"""
Phase 1: Leaf-Level Autoencoder Training

Trains the basic scatter/gather operations at the leaf level.
Particles → Encoder → Scatter → Hyperedge MLP → Gather → Decoder → Reconstruct

These weights will be transferred to the full H²GNN in later phases.

Usage:
    python -m electrostatic_unet.train_phase1 --epochs 100 --n-train 10000
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
    Phase 1: Leaf-level autoencoder.

    Architecture:
        Encoder: particles (x,y,q) → hidden representation
        Scatter: sum particle hidden states into leaf voxels
        Hyperedge MLP: process aggregated leaf features
        Gather: broadcast leaf states back to particles
        Decoder: (gathered + skip) → reconstructed particles

    The encoder, hyperedge MLP, and decoder weights will transfer
    to the full H²GNN in later phases.
    """

    def __init__(self, particle_dim: int = 3, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()

        self.particle_dim = particle_dim
        self.hidden_dim = hidden_dim

        # Encoder: particle features → hidden
        self.encoder = self._make_mlp(particle_dim, hidden_dim, hidden_dim, n_layers)

        # Hyperedge MLP: processes aggregated voxel features
        self.hyperedge_mlp = self._make_mlp(hidden_dim, hidden_dim, hidden_dim, n_layers)

        # Decoder: (gathered hidden + original particle features) → reconstructed
        self.decoder = self._make_mlp(hidden_dim + particle_dim, hidden_dim, particle_dim, n_layers)

    def _make_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
        """Build MLP with GELU activations."""
        layers = []

        # Input layer
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dim, out_dim))

        return nn.Sequential(*layers)

    def forward(self, particles: torch.Tensor, quadtree: Quadtree) -> torch.Tensor:
        """
        Forward pass.

        Args:
            particles: (N, 3) particle features (x, y, q)
            quadtree: Quadtree with particle-to-leaf mapping

        Returns:
            reconstructed: (N, 3) reconstructed particle features
        """
        device = particles.device
        N = particles.shape[0]

        # === ENCODER ===
        h_particles = self.encoder(particles)  # (N, hidden)

        # === SCATTER to leaf voxels (sum aggregation) ===
        particle_to_leaf = quadtree.particle_to_leaf.to(device)  # (N,)
        n_leaves = len(quadtree.active_voxels[quadtree.max_depth])

        # Initialize leaf hidden states
        h_leaves = torch.zeros(n_leaves, self.hidden_dim, device=device)

        # Sum scatter: aggregate particles into their leaf voxels
        h_leaves.scatter_add_(
            0,
            particle_to_leaf.unsqueeze(1).expand(-1, self.hidden_dim),
            h_particles
        )

        # === HYPEREDGE MLP ===
        h_leaves = self.hyperedge_mlp(h_leaves)  # (n_leaves, hidden)

        # === GATHER back to particles (broadcast) ===
        h_gathered = h_leaves[particle_to_leaf]  # (N, hidden)

        # === DECODER with skip connection ===
        h_combined = torch.cat([h_gathered, particles], dim=-1)  # (N, hidden + 3)
        reconstructed = self.decoder(h_combined)  # (N, 3)

        return reconstructed


def train_epoch(model, dataset, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for i in range(len(dataset)):
        particles, _ = dataset[i]  # Ignore E_true
        particles = particles.to(device)

        # Build quadtree
        positions = particles[:, :2]
        quadtree = Quadtree(positions, max_depth=4)

        # Forward
        optimizer.zero_grad()
        reconstructed = model(particles, quadtree)

        # Loss
        loss = criterion(reconstructed, particles)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset)


def validate(model, dataset, device):
    """Validate and return loss + per-feature errors."""
    model.eval()
    total_loss = 0.0
    total_x_err = 0.0
    total_y_err = 0.0
    total_q_err = 0.0
    n_particles = 0
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

            # Per-feature errors
            err = (reconstructed - particles).abs()
            total_x_err += err[:, 0].sum().item()
            total_y_err += err[:, 1].sum().item()
            total_q_err += err[:, 2].sum().item()
            n_particles += particles.shape[0]

    return {
        'mse': total_loss / len(dataset),
        'x_mae': total_x_err / n_particles,
        'y_mae': total_y_err / n_particles,
        'q_mae': total_q_err / n_particles,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 1: Leaf-level autoencoder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--n-val', type=int, default=500)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-plot', type=str, default='phase1_curve.png')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flush_print("=" * 70)
    flush_print("Phase 1: Leaf-Level Autoencoder")
    flush_print("=" * 70)
    flush_print(f"Device: {device}")
    flush_print(f"Epochs: {args.epochs}")
    flush_print(f"Training samples: {args.n_train}")
    flush_print(f"Validation samples: {args.n_val}")
    flush_print(f"Hidden dim: {args.hidden_dim}")
    flush_print(f"MLP layers: {args.n_layers}")
    flush_print(f"Learning rate: {args.lr}")
    flush_print()

    flush_print("Architecture:")
    flush_print("  Particles (N,3) -> Encoder -> Scatter -> Hyperedge MLP -> Gather -> Decoder -> (N,3)")
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

    # History
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    # Header
    flush_print(f"{'Epoch':>6} | {'Train MSE':>12} | {'Val MSE':>12} | {'x MAE':>8} | {'y MAE':>8} | {'q MAE':>8} | {'LR':>10} | Status")
    flush_print("-" * 95)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_dataset, optimizer, device)
        val_metrics = validate(model, val_dataset, device)
        val_loss = val_metrics['mse']

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

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
                'config': {
                    'particle_dim': 3,
                    'hidden_dim': args.hidden_dim,
                    'n_layers': args.n_layers,
                }
            }, f"{args.checkpoint_dir}/phase1_best.pt")

        flush_print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
                   f"{val_metrics['x_mae']:>8.5f} | {val_metrics['y_mae']:>8.5f} | "
                   f"{val_metrics['q_mae']:>8.5f} | {lr:>10.2e} | {status}")

        # Save plot every 10 epochs
        if epoch % 10 == 0 or epoch == args.epochs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # Loss curve
            ax = axes[0]
            ax.plot(train_losses, 'b-', label='Train', linewidth=2)
            ax.plot(val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Phase 1: Reconstruction Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            # Log scale zoom
            ax = axes[1]
            ax.plot(train_losses, 'b-', label='Train', linewidth=2)
            ax.plot(val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Phase 1: Loss (Linear Scale)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            plt.close()
            flush_print(f"       | Plot saved: {args.save_plot}")

        # Checkpoint every 50 epochs
        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, f"{args.checkpoint_dir}/phase1_epoch{epoch}.pt")

    flush_print("-" * 95)
    flush_print(f"Phase 1 complete! Best validation MSE: {best_val_loss:.6f}")
    flush_print(f"Checkpoint saved: {args.checkpoint_dir}/phase1_best.pt")
    flush_print(f"Plot saved: {args.save_plot}")
    flush_print()
    flush_print("Next: Run Phase 2 to add the next hierarchy level")


if __name__ == '__main__':
    main()
