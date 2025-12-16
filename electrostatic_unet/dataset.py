"""
Dataset generation for 2D electrostatic particle simulations.

Generates random particle configurations and computes ground truth
electric fields using Coulomb's law.
"""

import torch
from torch.utils.data import Dataset
from typing import Tuple, Optional
import random


def compute_coulomb_field_2d(
    positions: torch.Tensor,
    charges: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute 2D electric field at each particle from all others.

    In 2D, the electric field from a point charge falls off as 1/r:
        E = q * r_hat / |r|
        E = q * r / |r|^2

    Args:
        positions: (N, 2) particle positions
        charges: (N,) particle charges

    Returns:
        E: (N, 2) electric field at each particle
    """
    N = positions.shape[0]
    device = positions.device
    dtype = positions.dtype

    # Compute pairwise displacement vectors: r_ij = r_i - r_j
    # (N, 1, 2) - (1, N, 2) = (N, N, 2)
    r = positions.unsqueeze(1) - positions.unsqueeze(0)  # (N, N, 2)

    # Compute distances: |r_ij|
    r_mag = torch.norm(r, dim=-1)  # (N, N)

    # Avoid self-interaction (set diagonal to large value)
    mask = torch.eye(N, device=device, dtype=torch.bool)
    r_mag = r_mag.masked_fill(mask, 1.0)  # Avoid division by zero

    # Compute E contribution from each pair
    # E_i = sum_{j != i} q_j * (r_i - r_j) / |r_i - r_j|^2
    # Shape: (N, N, 2)
    E_contributions = charges.unsqueeze(0).unsqueeze(-1) * r / (r_mag.unsqueeze(-1) ** 2 + eps)

    # Zero out self-contributions
    E_contributions = E_contributions.masked_fill(mask.unsqueeze(-1), 0.0)

    # Sum over j
    E = E_contributions.sum(dim=1)  # (N, 2)

    return E


def compute_coulomb_field_2d_slow(
    positions: torch.Tensor,
    charges: torch.Tensor,
    eps: float = 1e-6
) -> torch.Tensor:
    """
    Compute 2D electric field using explicit loops (for reference/debugging).
    """
    N = len(positions)
    E = torch.zeros(N, 2, device=positions.device, dtype=positions.dtype)

    for i in range(N):
        for j in range(N):
            if i != j:
                r_ij = positions[i] - positions[j]
                r_mag = torch.norm(r_ij)
                E[i] += charges[j] * r_ij / (r_mag ** 2 + eps)

    return E


def generate_sample(
    n_particles: int,
    domain: Tuple[float, float] = (0.0, 1.0),
    margin: float = 0.05,
    device: torch.device = torch.device('cpu')
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate one training sample with random particles.

    Args:
        n_particles: number of particles
        domain: (min, max) for x and y coordinates
        margin: margin from domain boundaries
        device: torch device

    Returns:
        particles: (N, 3) tensor of (x, y, q)
        E: (N, 2) ground truth electric field
    """
    dmin, dmax = domain
    actual_min = dmin + margin * (dmax - dmin)
    actual_max = dmax - margin * (dmax - dmin)

    # Random positions in domain (with margin)
    positions = torch.rand(n_particles, 2, device=device)
    positions = positions * (actual_max - actual_min) + actual_min

    # Random charges in [-1, 1]
    charges = torch.rand(n_particles, device=device) * 2 - 1

    # Compute ground truth E field
    E = compute_coulomb_field_2d(positions, charges)

    # Combine into particle features: (x, y, q)
    particles = torch.cat([positions, charges.unsqueeze(-1)], dim=-1)

    return particles, E


class ElectrostaticDataset(Dataset):
    """
    Dataset of particle configurations with ground truth E fields.

    Samples have variable numbers of particles (within specified range).
    """

    def __init__(
        self,
        n_samples: int,
        n_particles_range: Tuple[int, int] = (10, 100),
        domain: Tuple[float, float] = (0.0, 1.0),
        margin: float = 0.05,
        seed: Optional[int] = None,
        precompute: bool = True,
        device: torch.device = torch.device('cpu')
    ):
        """
        Initialize dataset.

        Args:
            n_samples: number of samples to generate
            n_particles_range: (min, max) particles per sample
            domain: (min, max) for coordinates
            margin: margin from boundaries
            seed: random seed for reproducibility
            precompute: if True, generate all samples upfront
            device: torch device for tensors
        """
        self.n_samples = n_samples
        self.n_particles_range = n_particles_range
        self.domain = domain
        self.margin = margin
        self.device = device

        if seed is not None:
            torch.manual_seed(seed)
            random.seed(seed)

        if precompute:
            self.samples = []
            for _ in range(n_samples):
                n = random.randint(n_particles_range[0], n_particles_range[1])
                self.samples.append(generate_sample(n, domain, margin, device))
        else:
            self.samples = None

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.samples is not None:
            return self.samples[idx]
        else:
            n = random.randint(self.n_particles_range[0], self.n_particles_range[1])
            return generate_sample(n, self.domain, self.margin, self.device)


def test_dataset():
    """Test dataset generation and E field computation."""
    torch.manual_seed(42)

    # Test vectorized vs loop computation
    n = 20
    positions = torch.rand(n, 2)
    charges = torch.rand(n) * 2 - 1

    E_fast = compute_coulomb_field_2d(positions, charges)
    E_slow = compute_coulomb_field_2d_slow(positions, charges)

    error = (E_fast - E_slow).abs().max().item()
    print(f"Vectorized vs loop E field error: {error:.2e}")
    assert error < 1e-5, "E field computation mismatch!"

    # Test dataset
    dataset = ElectrostaticDataset(
        n_samples=10,
        n_particles_range=(10, 50),
        seed=42
    )

    print(f"\nDataset with {len(dataset)} samples")

    for i in range(3):
        particles, E = dataset[i]
        n_p = particles.shape[0]
        E_mag = torch.norm(E, dim=-1)
        print(f"  Sample {i}: {n_p} particles, "
              f"E magnitude range [{E_mag.min():.2f}, {E_mag.max():.2f}]")

    # Verify physics: E should point away from positive charges
    particles, E = dataset[0]
    pos = particles[:, :2]
    charges = particles[:, 2]

    # For a single positive charge, E at other particles should point away from it
    print("\nPhysics check:")
    idx_pos = (charges > 0.5).nonzero(as_tuple=True)[0]
    if len(idx_pos) > 0:
        i = idx_pos[0].item()
        print(f"  Particle {i}: charge = {charges[i]:.2f}")
        for j in range(min(3, len(particles))):
            if j != i:
                r = pos[j] - pos[i]
                E_j = E[j]
                # E should point in same direction as r (away from positive charge)
                dot = (r * E_j).sum()
                r_mag = torch.norm(r)
                E_mag = torch.norm(E_j)
                cos_angle = dot / (r_mag * E_mag + 1e-8)
                print(f"    E at particle {j}: direction cosine with r = {cos_angle:.3f}")

    print("\nDataset test passed!")


if __name__ == "__main__":
    test_dataset()
