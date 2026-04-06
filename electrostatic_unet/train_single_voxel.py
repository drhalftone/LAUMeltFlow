"""
Single-voxel autoencoder training.

One voxel containing 1-10 particles. Tests scatter/gather with real aggregation.

Usage:
    python -m electrostatic_unet.train_single_voxel --epochs 100
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


class SingleVoxelDataset:
    """
    Dataset with 1-10 particles per sample, all in one voxel.

    1000 samples for each particle count (1, 2, 3, ..., 10).
    """

    def __init__(self, samples_per_count: int = 1000, max_particles: int = 10, seed: int = 42):
        torch.manual_seed(seed)

        self.samples: List[torch.Tensor] = []

        for n_particles in range(1, max_particles + 1):
            for _ in range(samples_per_count):
                # Random positions in [0, 1]^2
                x = torch.rand(n_particles)
                y = torch.rand(n_particles)
                # Random charges in [-1, 1]
                q = torch.rand(n_particles) * 2 - 1

                particles = torch.stack([x, y, q], dim=-1)  # (N, 3)
                self.samples.append(particles)

        # Shuffle
        perm = torch.randperm(len(self.samples))
        self.samples = [self.samples[i] for i in perm]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.samples[idx]


class SingleVoxelAutoencoder(nn.Module):
    """
    Single voxel autoencoder.

    Architecture:
        Particles (N, 3) -> Encoder MLP -> h_particles (N, hidden)
        Scatter (sum) to single voxel -> h_voxel (1, hidden)
        Voxel MLP -> h_voxel' (1, hidden)
        Gather (broadcast) to particles -> h_gathered (N, hidden)
        [h_gathered, particles] -> Decoder MLP -> reconstructed (N, 3)
    """

    def __init__(self, particle_dim: int = 3, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()

        self.particle_dim = particle_dim
        self.hidden_dim = hidden_dim

        # Encoder: particle features -> hidden
        self.encoder = self._make_mlp(particle_dim, hidden_dim, hidden_dim, n_layers)

        # Voxel MLP: process aggregated features
        self.voxel_mlp = self._make_mlp(hidden_dim, hidden_dim, hidden_dim, n_layers)

        # Decoder: (gathered + particle features) -> reconstructed
        self.decoder = self._make_mlp(hidden_dim + particle_dim, hidden_dim, particle_dim, n_layers)

    def _make_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            particles: (N, 3) particle features

        Returns:
            reconstructed: (N, 3)
        """
        N = particles.shape[0]

        # Encode
        h_particles = self.encoder(particles)  # (N, hidden)

        # Scatter: sum all particles into one voxel
        h_voxel = h_particles.sum(dim=0, keepdim=True)  # (1, hidden)

        # Voxel MLP
        h_voxel = self.voxel_mlp(h_voxel)  # (1, hidden)

        # Gather: broadcast voxel state to all particles
        h_gathered = h_voxel.expand(N, -1)  # (N, hidden)

        # Decode with skip connection
        h_combined = torch.cat([h_gathered, particles], dim=-1)  # (N, hidden + 3)
        reconstructed = self.decoder(h_combined)  # (N, 3)

        return reconstructed


def train_epoch(model, dataset, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for i in range(len(dataset)):
        particles = dataset[i].to(device)

        optimizer.zero_grad()
        reconstructed = model(particles)

        loss = criterion(reconstructed, particles)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset)


def validate(model, dataset, device):
    """Validate and return metrics, also per-particle-count."""
    model.eval()
    total_loss = 0.0

    # Track loss by particle count
    loss_by_count = {n: [] for n in range(1, 11)}

    criterion = nn.MSELoss(reduction='none')

    with torch.no_grad():
        for i in range(len(dataset)):
            particles = dataset[i].to(device)
            n = particles.shape[0]

            reconstructed = model(particles)

            loss = criterion(reconstructed, particles).mean().item()
            total_loss += loss

            if n <= 10:
                loss_by_count[n].append(loss)

    # Average per count
    avg_by_count = {n: sum(v)/len(v) if v else 0 for n, v in loss_by_count.items()}

    return {
        'mse': total_loss / len(dataset),
        'by_count': avg_by_count
    }


def main():
    parser = argparse.ArgumentParser(description='Single-voxel autoencoder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--samples-per-count', type=int, default=1000)
    parser.add_argument('--max-particles', type=int, default=10)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-plot', type=str, default='single_voxel_curve.png')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flush_print("=" * 70)
    flush_print("Single-Voxel Autoencoder")
    flush_print("=" * 70)
    flush_print(f"Device: {device}")
    flush_print(f"Epochs: {args.epochs}")
    flush_print(f"Samples per particle count: {args.samples_per_count}")
    flush_print(f"Max particles: {args.max_particles}")
    flush_print(f"Total samples: {args.samples_per_count * args.max_particles}")
    flush_print(f"Hidden dim: {args.hidden_dim}")
    flush_print(f"Learning rate: {args.lr}")
    flush_print()

    flush_print("Architecture:")
    flush_print("  Particles (N,3) -> Encoder -> Scatter (sum) -> Voxel MLP -> Gather -> Decoder -> (N,3)")
    flush_print()

    # Generate datasets
    flush_print("Generating datasets...")
    train_dataset = SingleVoxelDataset(
        samples_per_count=args.samples_per_count,
        max_particles=args.max_particles,
        seed=42
    )
    val_dataset = SingleVoxelDataset(
        samples_per_count=args.samples_per_count // 10,
        max_particles=args.max_particles,
        seed=123
    )
    flush_print(f"  Training: {len(train_dataset)} samples")
    flush_print(f"  Validation: {len(val_dataset)} samples")
    flush_print()

    # Create model
    model = SingleVoxelAutoencoder(
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

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    flush_print(f"{'Epoch':>6} | {'Train MSE':>12} | {'Val MSE':>12} | {'LR':>10} | {'Time':>6} | Status")
    flush_print("-" * 70)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_dataset, optimizer, device)
        val_metrics = validate(model, val_dataset, device)
        val_loss = val_metrics['mse']

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
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'particle_dim': 3,
                    'hidden_dim': args.hidden_dim,
                    'n_layers': args.n_layers,
                }
            }, f"{args.checkpoint_dir}/single_voxel_best.pt")

        flush_print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {lr:>10.2e} | {elapsed:>5.1f}s | {status}")

        # Show per-count breakdown every 20 epochs
        if epoch % 20 == 0:
            by_count = val_metrics['by_count']
            counts_str = " ".join([f"{n}:{by_count[n]:.4f}" for n in [1, 5, 10]])
            flush_print(f"       | Loss by N particles: {counts_str}")

        if epoch % 10 == 0 or epoch == args.epochs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            ax = axes[0]
            ax.plot(train_losses, 'b-', label='Train', linewidth=2)
            ax.plot(val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Single-Voxel Autoencoder (Log)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            ax = axes[1]
            by_count = val_metrics['by_count']
            counts = list(by_count.keys())
            losses = list(by_count.values())
            ax.bar(counts, losses, color='steelblue')
            ax.set_xlabel('Number of Particles')
            ax.set_ylabel('MSE Loss')
            ax.set_title(f'Loss by Particle Count (Epoch {epoch})')
            ax.grid(True, alpha=0.3, axis='y')

            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            plt.close()
            flush_print(f"       | Plot saved: {args.save_plot}")

    flush_print("-" * 70)
    flush_print(f"Training complete! Best validation MSE: {best_val_loss:.6f}")

    # Final breakdown
    flush_print("\nFinal loss by particle count:")
    val_metrics = validate(model, val_dataset, device)
    for n in range(1, 11):
        flush_print(f"  N={n:2d}: MSE={val_metrics['by_count'][n]:.6f}")


if __name__ == '__main__':
    main()
