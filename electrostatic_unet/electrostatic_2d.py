"""
2D Electrostatic Potential from Point Charges.

Computes and visualizes the electrostatic potential field from
randomly placed point charges using the analytical 2D Coulomb formula.

Physics: phi(r) = -sum_i q_i * ln(|r - r_i|)
This satisfies Poisson's equation: nabla^2 phi = -rho/epsilon_0
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class Particles:
    """Point charge particles."""
    x: np.ndarray  # x positions
    y: np.ndarray  # y positions
    q: np.ndarray  # charges


def generate_particles(
    n_particles: int = 20,
    margin: float = 0.1,
    seed: Optional[int] = 42
) -> Particles:
    """
    Generate random point charges in [0,1]^2 domain.

    Parameters
    ----------
    n_particles : int
        Number of point charges
    margin : float
        Margin from domain boundaries
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    Particles
        Dataclass with x, y positions and charges q
    """
    if seed is not None:
        np.random.seed(seed)

    # Particle positions (avoid edges)
    x = margin + (1 - 2 * margin) * np.random.rand(n_particles)
    y = margin + (1 - 2 * margin) * np.random.rand(n_particles)

    # Random charges in [-1, 1]
    q = 2 * np.random.rand(n_particles) - 1

    return Particles(x=x, y=y, q=q)


def create_grid(
    nx: int = 128,
    ny: int = 128,
    domain: Tuple[float, float, float, float] = (0, 1, 0, 1)
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Create computational grid.

    Parameters
    ----------
    nx, ny : int
        Grid resolution
    domain : tuple
        (xmin, xmax, ymin, ymax)

    Returns
    -------
    X, Y : np.ndarray
        Meshgrid arrays
    x, y : np.ndarray
        1D coordinate arrays
    dx, dy : float
        Grid spacing
    """
    xmin, xmax, ymin, ymax = domain
    dx = (xmax - xmin) / nx
    dy = (ymax - ymin) / ny

    x = np.linspace(xmin + dx/2, xmax - dx/2, nx)
    y = np.linspace(ymin + dy/2, ymax - dy/2, ny)
    X, Y = np.meshgrid(x, y)

    return X, Y, x, y, dx, dy


def scatter_charges(
    particles: Particles,
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    domain: Tuple[float, float, float, float] = (0, 1, 0, 1)
) -> np.ndarray:
    """
    Scatter particle charges to grid (nearest-cell assignment).

    Parameters
    ----------
    particles : Particles
        Point charges
    nx, ny : int
        Grid resolution
    dx, dy : float
        Grid spacing
    domain : tuple
        (xmin, xmax, ymin, ymax)

    Returns
    -------
    np.ndarray
        Charge density on grid, shape (ny, nx)
    """
    xmin, _, ymin, _ = domain
    rho = np.zeros((ny, nx))

    for p in range(len(particles.x)):
        # Find nearest grid cell
        ix = int(round((particles.x[p] - xmin) / dx))
        iy = int(round((particles.y[p] - ymin) / dy))

        # Clamp to valid range
        ix = max(0, min(nx - 1, ix))
        iy = max(0, min(ny - 1, iy))

        # Deposit charge (nearest-cell assignment)
        rho[iy, ix] += particles.q[p] / (dx * dy)

    return rho


def compute_potential(
    particles: Particles,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float
) -> np.ndarray:
    """
    Compute analytical electrostatic potential.

    phi(r) = -sum_i q_i * ln(|r - r_i|)

    Parameters
    ----------
    particles : Particles
        Point charges
    X, Y : np.ndarray
        Meshgrid arrays
    epsilon : float
        Regularization distance to avoid singularity

    Returns
    -------
    np.ndarray
        Potential field, shape (ny, nx)
    """
    phi = np.zeros_like(X)

    for p in range(len(particles.x)):
        # Distance from this particle to all grid points
        r = np.sqrt((X - particles.x[p])**2 + (Y - particles.y[p])**2)

        # Regularize to avoid singularity
        r = np.maximum(r, epsilon)

        # Add contribution to potential
        phi -= particles.q[p] * np.log(r)

    return phi


def compute_electric_field(
    phi: np.ndarray,
    dx: float,
    dy: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute electric field from potential.

    E = -grad(phi)

    Parameters
    ----------
    phi : np.ndarray
        Potential field
    dx, dy : float
        Grid spacing

    Returns
    -------
    Ex, Ey : np.ndarray
        Electric field components
    E_mag : np.ndarray
        Electric field magnitude
    """
    # numpy gradient returns [dy, dx] for 2D array
    grad_y, grad_x = np.gradient(phi, dy, dx)
    Ex = -grad_x
    Ey = -grad_y
    E_mag = np.sqrt(Ex**2 + Ey**2)

    return Ex, Ey, E_mag


def compute_laplacian(phi: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Compute numerical Laplacian using finite differences.

    Parameters
    ----------
    phi : np.ndarray
        Potential field
    dx, dy : float
        Grid spacing

    Returns
    -------
    np.ndarray
        Laplacian of phi
    """
    # Second derivative in x
    d2phi_dx2 = np.zeros_like(phi)
    d2phi_dx2[:, 1:-1] = (phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]) / dx**2

    # Second derivative in y
    d2phi_dy2 = np.zeros_like(phi)
    d2phi_dy2[1:-1, :] = (phi[2:, :] - 2*phi[1:-1, :] + phi[:-2, :]) / dy**2

    return d2phi_dx2 + d2phi_dy2


def test_superposition(
    particles: Particles,
    X: np.ndarray,
    Y: np.ndarray,
    epsilon: float,
    phi_total: np.ndarray
) -> Dict:
    """
    Test superposition principle: phi(A+B) = phi(A) + phi(B).

    Parameters
    ----------
    particles : Particles
        All point charges
    X, Y : np.ndarray
        Meshgrid arrays
    epsilon : float
        Regularization distance
    phi_total : np.ndarray
        Pre-computed total potential

    Returns
    -------
    dict
        Test results with error metrics
    """
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

    # Check superposition
    phi_sum = phi_A + phi_B
    superposition_error = np.max(np.abs(phi_total - phi_sum))
    relative_error = superposition_error / np.max(np.abs(phi_total))

    return {
        'max_error': superposition_error,
        'relative_error': relative_error,
        'passed': relative_error < 1e-10
    }


def run_simulation(
    nx: int = 128,
    ny: int = 128,
    n_particles: int = 20,
    seed: Optional[int] = 42,
    verbose: bool = True
) -> Dict:
    """
    Run full electrostatic simulation.

    Parameters
    ----------
    nx, ny : int
        Grid resolution
    n_particles : int
        Number of point charges
    seed : int, optional
        Random seed
    verbose : bool
        Print diagnostic info

    Returns
    -------
    dict
        All computed fields and metadata
    """
    domain = (0, 1, 0, 1)

    # Generate particles
    particles = generate_particles(n_particles, seed=seed)

    if verbose:
        print(f"Generated {n_particles} particles")
        print(f"  Total charge: {np.sum(particles.q):.3f}")
        print(f"  Positive charges: {np.sum(particles.q > 0)}")
        print(f"  Negative charges: {np.sum(particles.q < 0)}")

    # Create grid
    X, Y, x, y, dx, dy = create_grid(nx, ny, domain)
    epsilon = 2 / nx  # Regularization distance

    # Scatter charges to grid
    rho = scatter_charges(particles, nx, ny, dx, dy, domain)

    # Compute analytical potential
    phi = compute_potential(particles, X, Y, epsilon)

    # Compute electric field
    Ex, Ey, E_mag = compute_electric_field(phi, dx, dy)

    # Compute Laplacian for verification
    laplacian = compute_laplacian(phi, dx, dy)

    # Test superposition
    superposition = test_superposition(particles, X, Y, epsilon, phi)

    if verbose:
        print(f"\n--- Superposition Test ---")
        print(f"Max error: {superposition['max_error']:.2e}")
        print(f"Relative error: {superposition['relative_error']*100:.2e}%")
        if superposition['passed']:
            print("Superposition test PASSED (machine precision)")
        else:
            print("Superposition test FAILED")

    return {
        'particles': particles,
        'X': X, 'Y': Y,
        'x': x, 'y': y,
        'dx': dx, 'dy': dy,
        'rho': rho,
        'phi': phi,
        'Ex': Ex, 'Ey': Ey, 'E_mag': E_mag,
        'laplacian': laplacian,
        'superposition': superposition,
        'nx': nx, 'ny': ny,
        'n_particles': n_particles
    }


def plot_fields(
    result: Dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Create visualization of charge density, potential, and electric field.

    Parameters
    ----------
    result : dict
        Output from run_simulation()
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    plt.Figure
        The figure object
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
               s=50*np.abs(particles.q[pos_idx]), c='red', edgecolors='k', label='+')
    ax.scatter(particles.x[neg_idx], particles.y[neg_idx],
               s=50*np.abs(particles.q[neg_idx]), c='blue', edgecolors='k', label='-')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'Charge Density $\rho$')
    ax.set_aspect('equal')
    plt.colorbar(im1, ax=ax, shrink=0.8)

    # Plot 2: Electrostatic potential
    ax = axes[1]
    vmax = max(abs(phi.min()), abs(phi.max()))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    im2 = ax.imshow(phi, extent=[0, 1, 0, 1], origin='lower', cmap='RdBu_r', norm=norm)
    ax.contour(X, Y, phi, levels=20, colors='k', linewidths=0.5, alpha=0.5)
    ax.scatter(particles.x[pos_idx], particles.y[pos_idx],
               s=50*np.abs(particles.q[pos_idx]), c='red', edgecolors='k')
    ax.scatter(particles.x[neg_idx], particles.y[neg_idx],
               s=50*np.abs(particles.q[neg_idx]), c='blue', edgecolors='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'Electrostatic Potential $\phi$')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    # Plot 3: Electric field magnitude with streamlines
    ax = axes[2]
    im3 = ax.imshow(np.log10(E_mag + 1), extent=[0, 1, 0, 1], origin='lower', cmap='hot')

    # Streamlines
    stream_density = 8
    startx = np.linspace(0, 1, stream_density)
    starty = np.linspace(0, 1, stream_density)
    SX, SY = np.meshgrid(startx, starty)
    ax.streamplot(x, y, Ex, Ey, color='gray', linewidth=0.5, density=1.5,
                  start_points=np.column_stack([SX.ravel(), SY.ravel()]))

    ax.scatter(particles.x[pos_idx], particles.y[pos_idx],
               s=50*np.abs(particles.q[pos_idx]), c='red', edgecolors='k')
    ax.scatter(particles.x[neg_idx], particles.y[neg_idx],
               s=50*np.abs(particles.q[neg_idx]), c='blue', edgecolors='k')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(r'Electric Field $|\mathbf{E}|$ (log scale)')
    ax.set_aspect('equal')
    plt.colorbar(im3, ax=ax, shrink=0.8, label=r'$\log_{10}(|E|+1)$')

    plt.suptitle(f'2D Electrostatic Field from {result["n_particles"]} Point Charges', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return fig


def plot_verification(
    result: Dict,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Plot verification: Laplacian vs charge density.

    Parameters
    ----------
    result : dict
        Output from run_simulation()
    save_path : str, optional
        Path to save figure
    show : bool
        Whether to display figure

    Returns
    -------
    plt.Figure
        The figure object
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
    ax.set_title(r'$\nabla^2 \phi$ (numerical)')
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
    ax.set_title(r'$-\rho$ (charge density)')
    ax.set_aspect('equal')
    plt.colorbar(im2, ax=ax, shrink=0.8)

    plt.suptitle(r'Verification: $\nabla^2 \phi \approx -\rho$', fontsize=12)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    if show:
        plt.show()

    return fig


if __name__ == "__main__":
    print("=" * 60)
    print("2D Electrostatic Potential Simulation")
    print("=" * 60)

    # Run simulation
    result = run_simulation(nx=128, ny=128, n_particles=20, seed=42)

    # Create visualizations
    plot_fields(result)
    plot_verification(result)
