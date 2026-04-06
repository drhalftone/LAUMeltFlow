"""
Physics-informed MLP for E-field prediction.

Key improvements over baseline:
1. Explicit 1/r^2 features (what Coulomb's law needs)
2. Normalized E-field targets
3. Softened singularity with larger epsilon

Usage:
    python -m electrostatic_unet.train_efield_physics --epochs 100
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


def compute_coulomb_field_2d(positions: torch.Tensor, charges: torch.Tensor, eps: float = 0.1) -> torch.Tensor:
    """
    Compute 2D electric field with softened singularity.
    Uses larger epsilon (0.1) to avoid extreme values near particles.
    """
    N = positions.shape[0]
    device = positions.device

    r = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)
    r_sq = (r ** 2).sum(dim=-1)  # (N, N)

    mask = torch.eye(N, device=device, dtype=torch.bool)
    r_sq = r_sq.masked_fill(mask, 1.0)  # avoid division by zero

    # Softened Coulomb: q / (r^2 + eps^2)
    inv_r_sq = 1.0 / (r_sq + eps ** 2)

    # E = q * r / |r|^2  (but we use r / (r^2 + eps^2) for softening)
    r_mag = torch.sqrt(r_sq + eps ** 2)
    r_hat = r / r_mag.unsqueeze(-1)  # unit direction

    E_contributions = charges.unsqueeze(0).unsqueeze(-1) * r_hat * inv_r_sq.unsqueeze(-1)
    E_contributions = E_contributions.masked_fill(mask.unsqueeze(-1), 0.0)
    E = E_contributions.sum(dim=1)

    return E


class EFieldDataset:
    """Dataset with normalized E-field targets."""

    def __init__(self, n_samples: int = 1000, n_particles: int = 8, seed: int = 42):
        torch.manual_seed(seed)
        self.samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.n_particles = n_particles

        all_E = []
        all_particles = []

        for _ in range(n_samples):
            positions = torch.rand(n_particles, 2)
            charges = torch.rand(n_particles) * 2 - 1
            E = compute_coulomb_field_2d(positions, charges)

            particles = torch.cat([positions, charges.unsqueeze(-1)], dim=-1)
            all_particles.append(particles)
            all_E.append(E)

        # Compute normalization stats across all samples
        all_E_cat = torch.cat(all_E, dim=0)  # (n_samples * n_particles, 2)
        self.E_mean = all_E_cat.mean(dim=0)  # (2,)
        self.E_std = all_E_cat.std(dim=0)  # (2,)
        self.E_std = torch.clamp(self.E_std, min=1e-6)

        flush_print(f"    E-field stats: mean=({self.E_mean[0]:.3f}, {self.E_mean[1]:.3f}), "
                   f"std=({self.E_std[0]:.3f}, {self.E_std[1]:.3f})")

        # Normalize E-field
        for particles, E in zip(all_particles, all_E):
            E_norm = (E - self.E_mean) / self.E_std
            self.samples.append((particles, E_norm))

        perm = torch.randperm(len(self.samples))
        self.samples = [self.samples[i] for i in perm]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def denormalize_E(self, E_norm: torch.Tensor) -> torch.Tensor:
        """Convert normalized E-field back to original scale."""
        device = E_norm.device
        return E_norm * self.E_std.to(device) + self.E_mean.to(device)


class PhysicsInformedMLP(nn.Module):
    """
    MLP that explicitly receives physics-based features.

    For each particle i and source j, we compute:
    - q_j: source charge
    - dx, dy: displacement from j to i
    - r: distance
    - 1/r^2: inverse square (what Coulomb needs)
    - dx/r, dy/r: unit direction

    The network learns to combine these into E-field contributions.
    """

    def __init__(self, hidden_dim: int = 64, n_layers: int = 3):
        super().__init__()

        # Input features per pair: q_j, dx, dy, r, 1/r^2, dx/r, dy/r = 7
        self.pair_mlp = self._make_mlp(7, hidden_dim, 2, n_layers)  # directly output 2D contribution

        # Optional: refine after aggregation
        self.refine_mlp = self._make_mlp(2 + 3, hidden_dim, 2, n_layers)  # (agg E, particle) -> E

    def _make_mlp(self, in_dim, hidden_dim, out_dim, n_layers):
        layers = [nn.Linear(in_dim, hidden_dim), nn.GELU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        """
        Args:
            particles: (N, 3) - (x, y, q)

        Returns:
            E_pred: (N, 2) - normalized E-field prediction
        """
        N = particles.shape[0]
        device = particles.device
        eps = 0.1

        positions = particles[:, :2]  # (N, 2)
        charges = particles[:, 2]  # (N,)

        # Compute pairwise features
        r = positions.unsqueeze(0) - positions.unsqueeze(1)  # (N, N, 2) - vector from j to i
        r_sq = (r ** 2).sum(dim=-1)  # (N, N)

        mask = torch.eye(N, device=device, dtype=torch.bool)
        r_sq_safe = r_sq.masked_fill(mask, 1.0)

        r_mag = torch.sqrt(r_sq_safe + eps ** 2)  # (N, N)
        inv_r_sq = 1.0 / (r_sq_safe + eps ** 2)  # (N, N)
        r_hat = r / r_mag.unsqueeze(-1)  # (N, N, 2)

        # Build pair features: for particle i, features from particle j
        # q_j, dx, dy, r, 1/r^2, dx/r, dy/r
        q_j = charges.unsqueeze(0).expand(N, -1)  # (N, N)

        pair_features = torch.stack([
            q_j,                    # source charge
            r[:, :, 0],             # dx
            r[:, :, 1],             # dy
            r_mag,                  # distance
            inv_r_sq,               # 1/r^2 (key physics!)
            r_hat[:, :, 0],         # unit dx
            r_hat[:, :, 1],         # unit dy
        ], dim=-1)  # (N, N, 7)

        # Predict contribution from each source
        E_contrib = self.pair_mlp(pair_features)  # (N, N, 2)

        # Mask self-interactions
        E_contrib = E_contrib.masked_fill(mask.unsqueeze(-1), 0.0)

        # Sum contributions
        E_agg = E_contrib.sum(dim=1)  # (N, 2)

        # Refine with particle info
        h = torch.cat([E_agg, particles], dim=-1)  # (N, 5)
        E_pred = self.refine_mlp(h)  # (N, 2)

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

            # Denormalize for relative error
            E_true_orig = dataset.denormalize_E(E_true)
            E_pred_orig = dataset.denormalize_E(E_pred)

            E_mag = torch.norm(E_true_orig, dim=-1)
            err = torch.norm(E_pred_orig - E_true_orig, dim=-1)
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
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-plot', type=str, default='efield_physics_curve.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flush_print("=" * 70)
    flush_print("Physics-Informed MLP for E-Field Prediction")
    flush_print("=" * 70)
    flush_print(f"Device: {device}")
    flush_print(f"Epochs: {args.epochs}")
    flush_print(f"Particles per sample: {args.n_particles}")
    flush_print(f"Training samples: {args.n_train}")
    flush_print()

    flush_print("Key improvements:")
    flush_print("  1. Explicit 1/r^2 feature (physics prior)")
    flush_print("  2. Normalized E-field targets")
    flush_print("  3. Softened singularity (eps=0.1)")
    flush_print()

    flush_print("Generating datasets...")
    train_dataset = EFieldDataset(n_samples=args.n_train, n_particles=args.n_particles, seed=42)
    val_dataset = EFieldDataset(n_samples=args.n_val, n_particles=args.n_particles, seed=123)
    flush_print(f"  Training: {len(train_dataset)} samples")
    flush_print(f"  Validation: {len(val_dataset)} samples")
    flush_print()

    model = PhysicsInformedMLP(
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    flush_print(f"Model parameters: {n_params:,}")
    flush_print()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=15)

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

        flush_print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {rel_err:>10.4f} | {lr:>10.2e} | {status}")

        if epoch % 20 == 0 or epoch == args.epochs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            ax = axes[0]
            ax.plot(train_losses, 'b-', label='Train', linewidth=2)
            ax.plot(val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss (normalized)')
            ax.set_title('E-Field Prediction Loss (Physics-Informed)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            ax = axes[1]
            ax.plot(val_rel_errs, 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Relative Error')
            ax.set_title('E-Field Relative Error')
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='10% target')
            ax.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='20% target')
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
        flush_print("\nSUCCESS: Physics-informed MLP can learn E-field prediction!")
    elif best_rel_err < 0.5:
        flush_print("\nPARTIAL SUCCESS: Model is learning but needs more capacity/training.")
    else:
        flush_print("\nModel struggling - may need architectural changes.")


if __name__ == '__main__':
    main()
