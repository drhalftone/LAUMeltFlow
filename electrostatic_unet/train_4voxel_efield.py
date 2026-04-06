"""
4-voxel hierarchy for E-field prediction.

Layout: 2x2 leaf voxels -> 1 parent voxel
Tests if hierarchy can learn to predict electric field from particle interactions.

Usage:
    python -m electrostatic_unet.train_4voxel_efield --epochs 100
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
    print(*args, **kwargs)
    sys.stdout.flush()


def compute_coulomb_field_2d(positions: torch.Tensor, charges: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Compute 2D electric field at each particle from all others.
    E_i = sum_{j != i} q_j * (r_i - r_j) / |r_i - r_j|^2
    """
    N = positions.shape[0]
    device = positions.device

    r = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)
    r_mag = torch.norm(r, dim=-1)  # (N, N)

    mask = torch.eye(N, device=device, dtype=torch.bool)
    r_mag = r_mag.masked_fill(mask, 1.0)

    E_contributions = charges.unsqueeze(0).unsqueeze(-1) * r / (r_mag.unsqueeze(-1) ** 2 + eps)
    E_contributions = E_contributions.masked_fill(mask.unsqueeze(-1), 0.0)
    E = E_contributions.sum(dim=1)

    return E


class FourVoxelDataset:
    """
    Dataset with particles in a 2x2 voxel grid.

    Each sample has 4-20 particles distributed across the domain.
    Ground truth is the Coulomb E-field at each particle.
    """

    def __init__(self, n_samples: int = 1000, n_particles_range: Tuple[int, int] = (4, 20), seed: int = 42):
        torch.manual_seed(seed)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for _ in range(n_samples):
            n = torch.randint(n_particles_range[0], n_particles_range[1] + 1, (1,)).item()

            # Random positions in [0, 1]^2
            positions = torch.rand(n, 2)
            charges = torch.rand(n) * 2 - 1

            # Compute ground truth E-field
            E = compute_coulomb_field_2d(positions, charges)

            particles = torch.cat([positions, charges.unsqueeze(-1)], dim=-1)  # (N, 3)
            self.samples.append((particles, E))

        perm = torch.randperm(len(self.samples))
        self.samples = [self.samples[i] for i in perm]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class FourVoxelModel(nn.Module):
    """
    4-voxel hierarchy for E-field prediction.

    Architecture:
        ENCODER:
            Particles (N, 3) -> Encoder MLP -> h_particles (N, hidden)
            Assign to leaf voxel (2x2 grid based on position)
            Scatter (sum) to 4 leaf voxels -> h_leaves (4, hidden)
            Leaf encoder MLP -> h_leaves_enc (4, hidden)
            Scatter (sum) to parent -> h_parent (1, hidden)
            Parent MLP -> h_parent' (1, hidden)

        DECODER:
            h_parent' -> Gather to leaves (broadcast)
            [h_from_parent, h_leaves_enc] -> Leaf decoder MLP -> h_leaves_dec (4, hidden)
            Gather to particles -> h_gathered (N, hidden)
            [h_gathered, particles] -> Output MLP -> E_pred (N, 2)
    """

    def __init__(self, particle_dim: int = 3, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()

        self.hidden_dim = hidden_dim

        # Particle encoder
        self.particle_encoder = self._make_mlp(particle_dim, hidden_dim, hidden_dim, n_layers)

        # Leaf level (4 voxels)
        self.leaf_encoder = self._make_mlp(hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.leaf_decoder = self._make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, n_layers)  # skip concat

        # Parent level (1 voxel)
        self.parent_encoder = self._make_mlp(hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.parent_decoder = self._make_mlp(hidden_dim, hidden_dim, hidden_dim, n_layers)

        # Output: predict E-field (2D)
        self.output_mlp = self._make_mlp(hidden_dim + particle_dim, hidden_dim, 2, n_layers)

    def _make_mlp(self, in_dim, hidden_dim, out_dim, n_layers):
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def _assign_to_voxel(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Assign particles to 2x2 voxel grid.

        Voxel layout:
            [0] x<0.5, y<0.5    [1] x>=0.5, y<0.5
            [2] x<0.5, y>=0.5   [3] x>=0.5, y>=0.5
        """
        x_idx = (positions[:, 0] >= 0.5).long()
        y_idx = (positions[:, 1] >= 0.5).long()
        voxel_idx = y_idx * 2 + x_idx
        return voxel_idx

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            particles: (N, 3) particle features (x, y, q)

        Returns:
            E_pred: (N, 2) predicted electric field
        """
        device = particles.device
        N = particles.shape[0]
        positions = particles[:, :2]

        # Assign particles to voxels
        voxel_idx = self._assign_to_voxel(positions)  # (N,)

        # ============ ENCODER ============

        # Particle encoding
        h_particles = self.particle_encoder(particles)  # (N, hidden)

        # Scatter to 4 leaf voxels
        h_leaves = torch.zeros(4, self.hidden_dim, device=device)
        h_leaves.scatter_add_(0, voxel_idx.unsqueeze(1).expand(-1, self.hidden_dim), h_particles)

        # Leaf encoder
        h_leaves_enc = self.leaf_encoder(h_leaves)  # (4, hidden) - save for skip

        # Scatter to parent (sum all 4 leaves)
        h_parent = h_leaves_enc.sum(dim=0, keepdim=True)  # (1, hidden)

        # Parent encoder
        h_parent = self.parent_encoder(h_parent)  # (1, hidden)

        # ============ DECODER ============

        # Parent decoder
        h_parent_dec = self.parent_decoder(h_parent)  # (1, hidden)

        # Gather to leaves (broadcast)
        h_from_parent = h_parent_dec.expand(4, -1)  # (4, hidden)

        # Leaf decoder with skip
        h_leaves_combined = torch.cat([h_from_parent, h_leaves_enc], dim=-1)  # (4, hidden*2)
        h_leaves_dec = self.leaf_decoder(h_leaves_combined)  # (4, hidden)

        # Gather to particles
        h_gathered = h_leaves_dec[voxel_idx]  # (N, hidden)

        # Output with skip to particle features
        h_output = torch.cat([h_gathered, particles], dim=-1)  # (N, hidden + 3)
        E_pred = self.output_mlp(h_output)  # (N, 2)

        return E_pred


def train_epoch(model, dataset, optimizer, device):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for i in range(len(dataset)):
        particles, E_true = dataset[i]
        particles = particles.to(device)
        E_true = E_true.to(device)

        optimizer.zero_grad()
        E_pred = model(particles)
        loss = criterion(E_pred, E_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset)


def validate(model, dataset, device):
    model.eval()
    total_loss = 0.0
    total_rel_err = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for i in range(len(dataset)):
            particles, E_true = dataset[i]
            particles = particles.to(device)
            E_true = E_true.to(device)

            E_pred = model(particles)
            loss = criterion(E_pred, E_true)
            total_loss += loss.item()

            # Relative error
            E_mag = torch.norm(E_true, dim=-1)
            err = torch.norm(E_pred - E_true, dim=-1)
            rel_err = (err / (E_mag + 1e-6)).mean()
            total_rel_err += rel_err.item()

    return {
        'mse': total_loss / len(dataset),
        'rel_err': total_rel_err / len(dataset)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-train', type=int, default=5000)
    parser.add_argument('--n-val', type=int, default=500)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-plot', type=str, default='4voxel_efield_curve.png')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flush_print("=" * 70)
    flush_print("4-Voxel Hierarchy for E-Field Prediction")
    flush_print("=" * 70)
    flush_print(f"Device: {device}")
    flush_print(f"Epochs: {args.epochs}")
    flush_print(f"Training samples: {args.n_train}")
    flush_print(f"Hidden dim: {args.hidden_dim}")
    flush_print()

    flush_print("Architecture:")
    flush_print("  Encoder: Particles -> 4 leaves (2x2) -> 1 parent")
    flush_print("  Decoder: Parent -> 4 leaves (+ skip) -> Particles (+ skip) -> E-field")
    flush_print()

    flush_print("Generating datasets...")
    train_dataset = FourVoxelDataset(n_samples=args.n_train, n_particles_range=(4, 20), seed=42)
    val_dataset = FourVoxelDataset(n_samples=args.n_val, n_particles_range=(4, 20), seed=123)
    flush_print(f"  Training: {len(train_dataset)} samples")
    flush_print(f"  Validation: {len(val_dataset)} samples")
    flush_print()

    model = FourVoxelModel(hidden_dim=args.hidden_dim, n_layers=args.n_layers).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    flush_print(f"Model parameters: {n_params:,}")
    flush_print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    train_losses = []
    val_losses = []
    val_rel_errs = []
    best_val_loss = float('inf')

    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    flush_print(f"{'Epoch':>6} | {'Train MSE':>12} | {'Val MSE':>12} | {'Rel Err':>10} | {'LR':>10} | Status")
    flush_print("-" * 75)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dataset, optimizer, device)
        val_metrics = validate(model, val_dataset, device)
        val_loss = val_metrics['mse']
        rel_err = val_metrics['rel_err']

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        val_rel_errs.append(rel_err)

        lr = optimizer.param_groups[0]['lr']

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            status = "* best"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, f"{args.checkpoint_dir}/4voxel_efield_best.pt")

        flush_print(f"{epoch:>6} | {train_loss:>12.4f} | {val_loss:>12.4f} | {rel_err:>10.4f} | {lr:>10.2e} | {status}")

        if epoch % 20 == 0 or epoch == args.epochs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            ax = axes[0]
            ax.plot(train_losses, 'b-', label='Train', linewidth=2)
            ax.plot(val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('E-Field Prediction Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            ax = axes[1]
            ax.plot(val_rel_errs, 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Relative Error')
            ax.set_title('E-Field Relative Error')
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            plt.close()
            flush_print(f"       | Plot saved: {args.save_plot}")

    flush_print("-" * 75)
    flush_print(f"Training complete! Best validation MSE: {best_val_loss:.4f}")
    flush_print(f"Final relative error: {val_rel_errs[-1]:.4f} ({val_rel_errs[-1]*100:.1f}%)")


if __name__ == '__main__':
    main()
