"""
Baseline MLP for E-field prediction.

Tests if E-field prediction is learnable at all by using a simple MLP
that sees ALL particles directly (no hierarchy, no aggregation loss).

Usage:
    python -m electrostatic_unet.train_efield_baseline --epochs 100
"""

import argparse
import sys
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


class EFieldDataset:
    """Dataset for E-field prediction with fixed number of particles."""

    def __init__(self, n_samples: int = 1000, n_particles: int = 8, seed: int = 42):
        torch.manual_seed(seed)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.n_particles = n_particles

        # Track E-field statistics for normalization
        all_E_mags = []

        for _ in range(n_samples):
            positions = torch.rand(n_particles, 2)
            charges = torch.rand(n_particles) * 2 - 1
            E = compute_coulomb_field_2d(positions, charges)

            particles = torch.cat([positions, charges.unsqueeze(-1)], dim=-1)
            self.samples.append((particles, E))

            all_E_mags.append(torch.norm(E, dim=-1))

        # Compute normalization stats
        all_E_mags = torch.cat(all_E_mags)
        self.E_mean = all_E_mags.mean().item()
        self.E_std = all_E_mags.std().item()

        perm = torch.randperm(len(self.samples))
        self.samples = [self.samples[i] for i in perm]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class BaselineMLP(nn.Module):
    """
    Baseline MLP that sees ALL particles.

    Each particle's E-field is predicted using info from ALL particles.
    Uses self-attention style: for each particle, compute features relative to all others.
    """

    def __init__(self, n_particles: int = 8, hidden_dim: int = 128, n_layers: int = 3):
        super().__init__()

        self.n_particles = n_particles

        # Pair encoder: for particle i, encode (particle_i, particle_j) pairs
        # Input: (x_i, y_i, q_i, x_j, y_j, q_j, dx, dy, dist)
        self.pair_encoder = self._make_mlp(9, hidden_dim, hidden_dim, n_layers)

        # Aggregator: combine all pair features for particle i
        self.aggregator = self._make_mlp(hidden_dim, hidden_dim, hidden_dim, n_layers)

        # Output: predict E-field
        self.output = self._make_mlp(hidden_dim + 3, hidden_dim, 2, n_layers)

    def _make_mlp(self, in_dim, hidden_dim, out_dim, n_layers):
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            particles: (N, 3) - (x, y, q) for each particle

        Returns:
            E_pred: (N, 2) - predicted E-field at each particle
        """
        N = particles.shape[0]
        device = particles.device

        positions = particles[:, :2]  # (N, 2)

        # For each particle i, compute features from all other particles j
        # Expand to (N, N, features)
        p_i = particles.unsqueeze(1).expand(-1, N, -1)  # (N, N, 3)
        p_j = particles.unsqueeze(0).expand(N, -1, -1)  # (N, N, 3)

        # Relative position
        dx = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)
        dist = torch.norm(dx, dim=-1, keepdim=True) + 1e-6  # (N, N, 1)

        # Pair features: (x_i, y_i, q_i, x_j, y_j, q_j, dx, dy, dist)
        pair_features = torch.cat([p_i, p_j, dx, dist], dim=-1)  # (N, N, 9)

        # Encode pairs
        h_pairs = self.pair_encoder(pair_features)  # (N, N, hidden)

        # Mask self-interactions
        mask = torch.eye(N, device=device, dtype=torch.bool)
        h_pairs = h_pairs.masked_fill(mask.unsqueeze(-1), 0.0)

        # Aggregate: sum over j for each i
        h_agg = h_pairs.sum(dim=1)  # (N, hidden)

        # Process aggregated features
        h_agg = self.aggregator(h_agg)  # (N, hidden)

        # Output with skip to particle features
        h_out = torch.cat([h_agg, particles], dim=-1)  # (N, hidden + 3)
        E_pred = self.output(h_out)  # (N, 2)

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
    parser.add_argument('--n-particles', type=int, default=8)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-plot', type=str, default='efield_baseline_curve.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flush_print("=" * 70)
    flush_print("Baseline MLP for E-Field Prediction")
    flush_print("=" * 70)
    flush_print(f"Device: {device}")
    flush_print(f"Epochs: {args.epochs}")
    flush_print(f"Particles per sample: {args.n_particles}")
    flush_print(f"Training samples: {args.n_train}")
    flush_print(f"Hidden dim: {args.hidden_dim}")
    flush_print()

    flush_print("Architecture:")
    flush_print("  For each particle i, encode all (i,j) pairs")
    flush_print("  Sum pair features -> MLP -> E-field prediction")
    flush_print("  This mimics Coulomb's law structure directly")
    flush_print()

    flush_print("Generating datasets...")
    train_dataset = EFieldDataset(n_samples=args.n_train, n_particles=args.n_particles, seed=42)
    val_dataset = EFieldDataset(n_samples=args.n_val, n_particles=args.n_particles, seed=123)
    flush_print(f"  Training: {len(train_dataset)} samples")
    flush_print(f"  Validation: {len(val_dataset)} samples")
    flush_print(f"  E-field magnitude: mean={train_dataset.E_mean:.2f}, std={train_dataset.E_std:.2f}")
    flush_print()

    model = BaselineMLP(
        n_particles=args.n_particles,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    flush_print(f"Model parameters: {n_params:,}")
    flush_print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=10)

    train_losses = []
    val_losses = []
    val_rel_errs = []
    best_rel_err = float('inf')

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
        if rel_err < best_rel_err:
            best_rel_err = rel_err
            status = "* best"

        flush_print(f"{epoch:>6} | {train_loss:>12.4f} | {val_loss:>12.4f} | {rel_err:>10.4f} | {lr:>10.2e} | {status}")

        if epoch % 20 == 0 or epoch == args.epochs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            ax = axes[0]
            ax.plot(train_losses, 'b-', label='Train', linewidth=2)
            ax.plot(val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('E-Field Prediction Loss (Baseline)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            ax = axes[1]
            ax.plot(val_rel_errs, 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Relative Error')
            ax.set_title('E-Field Relative Error')
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='10% target')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            plt.close()
            flush_print(f"       | Plot saved: {args.save_plot}")

    flush_print("-" * 75)
    flush_print(f"Training complete!")
    flush_print(f"Best relative error: {best_rel_err:.4f} ({best_rel_err*100:.1f}%)")
    flush_print(f"Final relative error: {val_rel_errs[-1]:.4f} ({val_rel_errs[-1]*100:.1f}%)")

    if best_rel_err < 0.2:
        flush_print("\nE-field prediction IS learnable with pair-wise features!")
        flush_print("The 4-voxel model likely loses critical pairwise information in aggregation.")
    else:
        flush_print("\nE-field prediction is challenging even for baseline MLP.")


if __name__ == '__main__':
    main()
