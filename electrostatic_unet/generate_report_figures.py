"""
Generate figures for the electrostatic U-Net report.

Saves figures to reports/electrostatic_unet/figures/
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from electrostatic_unet.electrostatic_2d import (
    run_simulation, Particles, compute_potential
)


def create_output_dir(base_path: str) -> str:
    """Create output directory if it doesn't exist."""
    figures_dir = os.path.join(base_path, 'reports', 'electrostatic_unet', 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    return figures_dir


def generate_main_figure(result: dict, save_path: str):
    """
    Generate main 3-panel figure: charge density, potential, electric field.
    """
    particles = result['particles']
    x, y = result['x'], result['y']
    rho = result['rho']
    phi = result['phi']
    X, Y = result['X'], result['Y']
    Ex, Ey = result['Ex'], result['Ey']
    E_mag = result['E_mag']

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    # Particle masks
    pos_idx = particles.q > 0
    neg_idx = particles.q < 0

    # Plot 1: Charge density
    ax = axes[0]
    vmax = max(abs(rho.min()), abs(rho.max()))
    if vmax > 0:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = None
    im1 = ax.imshow(rho, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r', norm=norm)
    ax.scatter(particles.x[pos_idx], particles.y[pos_idx],
               s=50*np.abs(particles.q[pos_idx]), c='red', edgecolors='k', linewidths=0.5)
    ax.scatter(particles.x[neg_idx], particles.y[neg_idx],
               s=50*np.abs(particles.q[neg_idx]), c='blue', edgecolors='k', linewidths=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'(a) Charge Density $\rho$')
    ax.set_aspect('equal')
    plt.colorbar(im1, ax=ax, shrink=0.8)

    # Plot 2: Electrostatic potential
    ax = axes[1]
    vmax = max(abs(phi.min()), abs(phi.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = ax.imshow(phi, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r', norm=norm)
    ax.contour(X, Y, phi, levels=20, colors='k', linewidths=0.5, alpha=0.5)
    ax.scatter(particles.x[pos_idx], particles.y[pos_idx],
               s=50*np.abs(particles.q[pos_idx]), c='red', edgecolors='k', linewidths=0.5)
    ax.scatter(particles.x[neg_idx], particles.y[neg_idx],
               s=50*np.abs(particles.q[neg_idx]), c='blue', edgecolors='k', linewidths=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'(b) Electrostatic Potential $\phi$')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Plot 3: Electric field magnitude with streamlines
    ax = axes[2]
    im3 = ax.imshow(np.log10(E_mag + 1), extent=[0, 1, 0, 1], origin='lower', cmap='hot')

    # Streamlines
    ax.streamplot(x, y, Ex, Ey, color='gray', linewidth=0.5, density=1.5)

    ax.scatter(particles.x[pos_idx], particles.y[pos_idx],
               s=50*np.abs(particles.q[pos_idx]), c='red', edgecolors='k', linewidths=0.5)
    ax.scatter(particles.x[neg_idx], particles.y[neg_idx],
               s=50*np.abs(particles.q[neg_idx]), c='blue', edgecolors='k', linewidths=0.5)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'(c) Electric Field $|\mathbf{E}|$ (log scale)')
    ax.set_aspect('equal')
    plt.colorbar(im3, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_verification_figure(result: dict, save_path: str):
    """
    Generate Poisson equation verification figure.
    """
    laplacian = result['laplacian']
    rho = result['rho']

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Plot 1: Numerical Laplacian
    ax = axes[0]
    vmax = max(abs(laplacian.min()), abs(laplacian.max()))
    if vmax > 0:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = None
    im1 = ax.imshow(laplacian, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r', norm=norm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'(a) $\nabla^2 \phi$ (numerical)')
    ax.set_aspect('equal')
    plt.colorbar(im1, ax=ax, shrink=0.8)

    # Plot 2: Negative charge density
    ax = axes[1]
    vmax = max(abs(rho.min()), abs(rho.max()))
    if vmax > 0:
        norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    else:
        norm = None
    im2 = ax.imshow(-rho, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r', norm=norm)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'(b) $-\rho$ (charge density)')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def generate_superposition_figure(result: dict, save_path: str):
    """
    Generate superposition principle verification figure.
    """
    particles = result['particles']
    X, Y = result['X'], result['Y']
    phi = result['phi']
    epsilon = 2 / result['nx']

    n = len(particles.x)
    n_A = n // 2

    # Split particles
    particles_A = Particles(
        x=particles.x[:n_A],
        y=particles.y[:n_A],
        q=particles.q[:n_A]
    )
    particles_B = Particles(
        x=particles.x[n_A:],
        y=particles.y[n_A:],
        q=particles.q[n_A:]
    )

    # Compute individual potentials
    phi_A = compute_potential(particles_A, X, Y, epsilon)
    phi_B = compute_potential(particles_B, X, Y, epsilon)
    phi_sum = phi_A + phi_B
    error = phi - phi_sum

    fig, axes = plt.subplots(2, 2, figsize=(10, 9))

    # Common colormap settings
    vmax_phi = max(abs(phi.min()), abs(phi.max()))
    norm_phi = TwoSlopeNorm(vmin=-vmax_phi, vcenter=0, vmax=vmax_phi)

    # Plot 1: phi_A
    ax = axes[0, 0]
    im = ax.imshow(phi_A, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r', norm=norm_phi)
    pos_idx = particles_A.q > 0
    neg_idx = particles_A.q < 0
    ax.scatter(particles_A.x[pos_idx], particles_A.y[pos_idx],
               s=50*np.abs(particles_A.q[pos_idx]), c='red', edgecolors='k', linewidths=0.5)
    ax.scatter(particles_A.x[neg_idx], particles_A.y[neg_idx],
               s=50*np.abs(particles_A.q[neg_idx]), c='blue', edgecolors='k', linewidths=0.5)
    ax.set_title(r'(a) $\phi_A$ (first 10 charges)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Plot 2: phi_B
    ax = axes[0, 1]
    im = ax.imshow(phi_B, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r', norm=norm_phi)
    pos_idx = particles_B.q > 0
    neg_idx = particles_B.q < 0
    ax.scatter(particles_B.x[pos_idx], particles_B.y[pos_idx],
               s=50*np.abs(particles_B.q[pos_idx]), c='red', edgecolors='k', linewidths=0.5)
    ax.scatter(particles_B.x[neg_idx], particles_B.y[neg_idx],
               s=50*np.abs(particles_B.q[neg_idx]), c='blue', edgecolors='k', linewidths=0.5)
    ax.set_title(r'(b) $\phi_B$ (last 10 charges)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Plot 3: phi_A + phi_B
    ax = axes[1, 0]
    im = ax.imshow(phi_sum, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r', norm=norm_phi)
    pos_idx = particles.q > 0
    neg_idx = particles.q < 0
    ax.scatter(particles.x[pos_idx], particles.y[pos_idx],
               s=50*np.abs(particles.q[pos_idx]), c='red', edgecolors='k', linewidths=0.5)
    ax.scatter(particles.x[neg_idx], particles.y[neg_idx],
               s=50*np.abs(particles.q[neg_idx]), c='blue', edgecolors='k', linewidths=0.5)
    ax.set_title(r'(c) $\phi_A + \phi_B$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Plot 4: Error
    ax = axes[1, 1]
    vmax_err = max(abs(error.min()), abs(error.max())) if np.any(error != 0) else 1e-15
    im = ax.imshow(error, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r',
                   vmin=-vmax_err, vmax=vmax_err)
    ax.set_title(r'(d) Error: $\phi_{total} - (\phi_A + \phi_B)$')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.formatter.set_powerlimits((0, 0))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {save_path}")
    plt.close()


def main():
    # Get project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    # Create output directory
    figures_dir = create_output_dir(project_root)

    print("=" * 60)
    print("Generating Electrostatic Report Figures")
    print("=" * 60)
    print(f"Output directory: {figures_dir}")

    # Run simulation
    print("\nRunning simulation...")
    result = run_simulation(nx=128, ny=128, n_particles=20, seed=42, verbose=True)

    # Generate figures
    print("\nGenerating figures...")

    # Main 3-panel figure
    generate_main_figure(
        result,
        os.path.join(figures_dir, 'electrostatic_fields.png')
    )

    # Poisson verification
    generate_verification_figure(
        result,
        os.path.join(figures_dir, 'poisson_verification.png')
    )

    # Superposition test
    generate_superposition_figure(
        result,
        os.path.join(figures_dir, 'superposition_test.png')
    )

    print("\n" + "=" * 60)
    print("Done! Generated 3 figures:")
    print("  - electrostatic_fields.png")
    print("  - poisson_verification.png")
    print("  - superposition_test.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
