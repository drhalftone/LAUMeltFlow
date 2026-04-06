"""
Single-voxel autoencoder WITHOUT skip connection.

Tests if the voxel path alone can reconstruct particles.
Spoiler: It can't - same input to decoder produces same output for all particles.

Usage:
    python -m electrostatic_unet.train_single_voxel_no_skip --epochs 50
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List


def flush_print(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


class SingleVoxelDataset:
    """Dataset with 1-10 particles per sample."""

    def __init__(self, samples_per_count: int = 1000, max_particles: int = 10, seed: int = 42):
        torch.manual_seed(seed)
        self.samples: List[torch.Tensor] = []

        for n_particles in range(1, max_particles + 1):
            for _ in range(samples_per_count):
                x = torch.rand(n_particles)
                y = torch.rand(n_particles)
                q = torch.rand(n_particles) * 2 - 1
                particles = torch.stack([x, y, q], dim=-1)
                self.samples.append(particles)

        perm = torch.randperm(len(self.samples))
        self.samples = [self.samples[i] for i in perm]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


class SingleVoxelNoSkip(nn.Module):
    """
    Single voxel autoencoder WITHOUT skip connection.

    The decoder only receives h_gathered (same for all particles).
    This should FAIL because same input -> same output.
    """

    def __init__(self, particle_dim: int = 3, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()

        self.particle_dim = particle_dim
        self.hidden_dim = hidden_dim

        self.encoder = self._make_mlp(particle_dim, hidden_dim, hidden_dim, n_layers)
        self.voxel_mlp = self._make_mlp(hidden_dim, hidden_dim, hidden_dim, n_layers)
        # NO skip: decoder only gets hidden_dim, not hidden_dim + particle_dim
        self.decoder = self._make_mlp(hidden_dim, hidden_dim, particle_dim, n_layers)

    def _make_mlp(self, in_dim, hidden_dim, out_dim, n_layers):
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, particles: torch.Tensor) -> torch.Tensor:
        N = particles.shape[0]

        # Encode
        h_particles = self.encoder(particles)  # (N, hidden)

        # Scatter: sum into one voxel
        h_voxel = h_particles.sum(dim=0, keepdim=True)  # (1, hidden)

        # Voxel MLP
        h_voxel = self.voxel_mlp(h_voxel)  # (1, hidden)

        # Gather: broadcast to all particles
        h_gathered = h_voxel.expand(N, -1)  # (N, hidden) - SAME for all!

        # Decode - NO skip connection
        reconstructed = self.decoder(h_gathered)  # (N, 3) - will be SAME for all!

        return reconstructed


def train_epoch(model, dataset, optimizer, device):
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
    model.eval()
    total_loss = 0.0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for i in range(len(dataset)):
            particles = dataset[i].to(device)
            reconstructed = model(particles)
            loss = criterion(reconstructed, particles)
            total_loss += loss.item()

    return total_loss / len(dataset)


def check_output_variance(model, dataset, device):
    """Check if outputs for particles in same sample are identical."""
    model.eval()

    with torch.no_grad():
        # Get a sample with multiple particles
        for i in range(len(dataset)):
            particles = dataset[i].to(device)
            if particles.shape[0] >= 5:
                reconstructed = model(particles)

                # Check variance of outputs
                var = reconstructed.var(dim=0).mean().item()

                flush_print(f"\nSample with {particles.shape[0]} particles:")
                flush_print(f"  Input particles (first 3):")
                for j in range(min(3, particles.shape[0])):
                    flush_print(f"    [{particles[j, 0]:.4f}, {particles[j, 1]:.4f}, {particles[j, 2]:.4f}]")
                flush_print(f"  Output (first 3):")
                for j in range(min(3, reconstructed.shape[0])):
                    flush_print(f"    [{reconstructed[j, 0]:.4f}, {reconstructed[j, 1]:.4f}, {reconstructed[j, 2]:.4f}]")
                flush_print(f"  Output variance across particles: {var:.6f}")

                return var
    return 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--samples-per-count', type=int, default=100)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flush_print("=" * 70)
    flush_print("Single-Voxel Autoencoder WITHOUT Skip Connection")
    flush_print("=" * 70)
    flush_print(f"Device: {device}")
    flush_print(f"This should FAIL: same h_gathered -> same output for all particles")
    flush_print()

    train_dataset = SingleVoxelDataset(samples_per_count=args.samples_per_count, seed=42)
    val_dataset = SingleVoxelDataset(samples_per_count=args.samples_per_count // 10, seed=123)

    model = SingleVoxelNoSkip(hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    flush_print(f"{'Epoch':>6} | {'Train MSE':>12} | {'Val MSE':>12}")
    flush_print("-" * 40)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_dataset, optimizer, device)
        val_loss = validate(model, val_dataset, device)
        flush_print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f}")

    flush_print("-" * 40)
    flush_print("\nChecking if outputs are identical for particles in same sample:")
    check_output_variance(model, val_dataset, device)


if __name__ == '__main__':
    main()
