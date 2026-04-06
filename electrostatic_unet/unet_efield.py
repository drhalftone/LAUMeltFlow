"""
Traditional U-Net for E-field prediction from quantized particle positions.

Input: Charge "image" (H × W × 1) - charge values at voxel positions
Output: E-field "image" (H × W × 2) - E_x, E_y at each voxel

Usage:
    python -m electrostatic_unet.unet_efield --epochs 100
"""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Tuple


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class ConvBlock(nn.Module):
    """Two convolutions with BatchNorm and ReLU."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """
    Standard U-Net architecture.

    Encoder: Progressively downsample, increase channels
    Decoder: Progressively upsample, decrease channels
    Skip connections: Concatenate encoder features to decoder
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 2, base_features: int = 64):
        super().__init__()

        # Encoder (downsampling path)
        self.enc1 = ConvBlock(in_channels, base_features)
        self.enc2 = ConvBlock(base_features, base_features * 2)
        self.enc3 = ConvBlock(base_features * 2, base_features * 4)
        self.enc4 = ConvBlock(base_features * 4, base_features * 8)

        # Bottleneck
        self.bottleneck = ConvBlock(base_features * 8, base_features * 16)

        # Decoder (upsampling path)
        self.up4 = nn.ConvTranspose2d(base_features * 16, base_features * 8, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(base_features * 16, base_features * 8)  # *16 because of skip concat

        self.up3 = nn.ConvTranspose2d(base_features * 8, base_features * 4, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(base_features * 8, base_features * 4)

        self.up2 = nn.ConvTranspose2d(base_features * 4, base_features * 2, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(base_features * 4, base_features * 2)

        self.up1 = nn.ConvTranspose2d(base_features * 2, base_features, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(base_features * 2, base_features)

        # Output layer
        self.out_conv = nn.Conv2d(base_features, out_channels, kernel_size=1)

        # Pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        # Bottleneck
        b = self.bottleneck(self.pool(e4))

        # Decoder with skip connections
        d4 = self.up4(b)
        d4 = torch.cat([d4, e4], dim=1)  # Skip connection
        d4 = self.dec4(d4)

        d3 = self.up3(d4)
        d3 = torch.cat([d3, e3], dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.out_conv(d1)


def compute_efield_grid(charge_grid: torch.Tensor, eps: float = 0.5) -> torch.Tensor:
    """
    Compute E-field at every grid point from charge distribution.

    Args:
        charge_grid: (H, W) charge values at each grid point
        eps: softening parameter to avoid singularity

    Returns:
        E_field: (H, W, 2) E_x, E_y at each grid point
    """
    H, W = charge_grid.shape
    device = charge_grid.device

    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )

    # For each grid point, compute E-field from all charges
    # E_i = sum_j q_j * (r_i - r_j) / |r_i - r_j|^2

    # Flatten for broadcasting
    x_flat = x_coords.reshape(-1)  # (H*W,)
    y_flat = y_coords.reshape(-1)
    q_flat = charge_grid.reshape(-1)

    # Compute all pairwise displacements: (H*W, H*W)
    dx = x_flat.unsqueeze(0) - x_flat.unsqueeze(1)  # r_i - r_j
    dy = y_flat.unsqueeze(0) - y_flat.unsqueeze(1)

    r_sq = dx**2 + dy**2
    r_sq_soft = r_sq + eps**2  # Softened to avoid singularity

    # E contribution: q_j * (r_i - r_j) / |r_i - r_j|^2
    # Note: we want direction FROM j TO i, which is (r_i - r_j)
    E_x_contrib = q_flat.unsqueeze(0) * dx / r_sq_soft  # (H*W, H*W)
    E_y_contrib = q_flat.unsqueeze(0) * dy / r_sq_soft

    # Mask self-interactions
    mask = torch.eye(H * W, device=device, dtype=torch.bool)
    E_x_contrib = E_x_contrib.masked_fill(mask, 0.0)
    E_y_contrib = E_y_contrib.masked_fill(mask, 0.0)

    # Sum over all source charges
    E_x = E_x_contrib.sum(dim=1).reshape(H, W)
    E_y = E_y_contrib.sum(dim=1).reshape(H, W)

    return torch.stack([E_x, E_y], dim=-1)


class EFieldGridDataset:
    """Dataset of charge grids and corresponding E-fields."""

    def __init__(self, n_samples: int, grid_size: int = 32, n_particles_range: Tuple[int, int] = (5, 20), seed: int = 42):
        torch.manual_seed(seed)
        self.samples = []
        self.grid_size = grid_size

        flush_print(f"    Generating {n_samples} samples...")

        for i in range(n_samples):
            n_particles = torch.randint(n_particles_range[0], n_particles_range[1] + 1, (1,)).item()

            # Random particle positions (quantized to grid)
            x_pos = torch.randint(0, grid_size, (n_particles,))
            y_pos = torch.randint(0, grid_size, (n_particles,))

            # Random charges
            charges = torch.rand(n_particles) * 2 - 1  # [-1, 1]

            # Create charge grid (accumulate if multiple particles in same cell)
            charge_grid = torch.zeros(grid_size, grid_size)
            for j in range(n_particles):
                charge_grid[y_pos[j], x_pos[j]] += charges[j]

            # Compute ground truth E-field
            E_field = compute_efield_grid(charge_grid)

            self.samples.append((charge_grid, E_field))

            if (i + 1) % 500 == 0:
                flush_print(f"    Generated {i + 1}/{n_samples}")

        # Compute normalization statistics
        all_E = torch.stack([s[1] for s in self.samples])  # (N, H, W, 2)
        self.E_mean = all_E.mean(dim=(0, 1, 2))  # (2,)
        self.E_std = all_E.std(dim=(0, 1, 2))
        self.E_std = torch.clamp(self.E_std, min=1e-6)

        flush_print(f"    E-field stats: mean=({self.E_mean[0]:.3f}, {self.E_mean[1]:.3f}), "
                   f"std=({self.E_std[0]:.3f}, {self.E_std[1]:.3f})")

        # Normalize E-fields
        self.samples = [
            (charge, (E - self.E_mean) / self.E_std)
            for charge, E in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        charge_grid, E_field = self.samples[idx]
        # Add channel dimension: (H, W) -> (1, H, W)
        # E_field: (H, W, 2) -> (2, H, W)
        return charge_grid.unsqueeze(0), E_field.permute(2, 0, 1)


def train_epoch(model, dataset, optimizer, device, batch_size: int = 32):
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    indices = torch.randperm(len(dataset))

    for start in range(0, len(dataset), batch_size):
        batch_idx = indices[start:start + batch_size]

        charges = torch.stack([dataset[i][0] for i in batch_idx]).to(device)
        E_true = torch.stack([dataset[i][1] for i in batch_idx]).to(device)

        optimizer.zero_grad()
        E_pred = model(charges)
        loss = criterion(E_pred, E_true)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(batch_idx)

    return total_loss / len(dataset)


def validate(model, dataset, device, batch_size: int = 32):
    model.eval()
    total_loss = 0.0
    total_rel_err = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            end = min(start + batch_size, len(dataset))

            charges = torch.stack([dataset[i][0] for i in range(start, end)]).to(device)
            E_true = torch.stack([dataset[i][1] for i in range(start, end)]).to(device)

            E_pred = model(charges)
            loss = criterion(E_pred, E_true)
            total_loss += loss.item() * (end - start)

            # Denormalize for relative error
            E_true_orig = E_true * dataset.E_std.to(device).view(1, 2, 1, 1) + dataset.E_mean.to(device).view(1, 2, 1, 1)
            E_pred_orig = E_pred * dataset.E_std.to(device).view(1, 2, 1, 1) + dataset.E_mean.to(device).view(1, 2, 1, 1)

            E_mag = torch.norm(E_true_orig, dim=1)  # (B, H, W)
            err = torch.norm(E_pred_orig - E_true_orig, dim=1)
            rel_err = (err / (E_mag + 1e-6)).mean()
            total_rel_err += rel_err.item() * (end - start)

    return {
        'mse': total_loss / len(dataset),
        'rel_err': total_rel_err / len(dataset)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-train', type=int, default=2000)
    parser.add_argument('--n-val', type=int, default=200)
    parser.add_argument('--grid-size', type=int, default=32)
    parser.add_argument('--base-features', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save-plot', type=str, default='unet_efield_curve.png')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flush_print("=" * 70)
    flush_print("U-Net for E-Field Prediction")
    flush_print("=" * 70)
    flush_print(f"Device: {device}")
    flush_print(f"Grid size: {args.grid_size} x {args.grid_size}")
    flush_print(f"Base features: {args.base_features}")
    flush_print(f"Batch size: {args.batch_size}")
    flush_print()

    flush_print("Architecture:")
    flush_print("  Input:  Charge grid (1, H, W)")
    flush_print("  Output: E-field grid (2, H, W)")
    flush_print("  Standard U-Net with skip connections")
    flush_print()

    flush_print("Generating datasets...")
    train_dataset = EFieldGridDataset(
        n_samples=args.n_train,
        grid_size=args.grid_size,
        n_particles_range=(5, 20),
        seed=42
    )
    val_dataset = EFieldGridDataset(
        n_samples=args.n_val,
        grid_size=args.grid_size,
        n_particles_range=(5, 20),
        seed=123
    )
    flush_print(f"  Training: {len(train_dataset)} samples")
    flush_print(f"  Validation: {len(val_dataset)} samples")
    flush_print()

    model = UNet(
        in_channels=1,
        out_channels=2,
        base_features=args.base_features
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
        train_loss = train_epoch(model, train_dataset, optimizer, device, args.batch_size)
        val_metrics = validate(model, val_dataset, device, args.batch_size)
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
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_rel_err': rel_err,
            }, 'checkpoints/unet_efield_best.pt')

        flush_print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | {rel_err:>10.4f} | {lr:>10.2e} | {status}")

        if epoch % 20 == 0 or epoch == args.epochs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            ax = axes[0]
            ax.plot(train_losses, 'b-', label='Train', linewidth=2)
            ax.plot(val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('U-Net E-Field Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            ax = axes[1]
            ax.plot(val_rel_errs, 'g-', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Relative Error')
            ax.set_title('E-Field Relative Error')
            ax.axhline(y=0.1, color='r', linestyle='--', alpha=0.5, label='10%')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            plt.close()
            flush_print(f"       | Plot saved: {args.save_plot}")

    flush_print("-" * 75)
    flush_print(f"Training complete!")
    flush_print(f"Best relative error: {best_rel_err:.4f} ({best_rel_err*100:.1f}%)")


if __name__ == '__main__':
    main()
