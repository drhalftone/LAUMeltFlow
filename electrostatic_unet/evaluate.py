"""
Evaluation and visualization for H2GNN electrostatic field predictor.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import argparse
import os
from typing import Optional, Tuple

from .quadtree import Quadtree
from .h2gnn import H2GNN
from .dataset import ElectrostaticDataset, generate_sample


def load_model(
    checkpoint_path: str,
    device: torch.device = torch.device('cpu')
) -> H2GNN:
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint['config']
    model = H2GNN(
        max_depth=config['max_depth'],
        hidden_dim=config['hidden_dim'],
        n_mlp_layers=config['n_mlp_layers']
    ).to(device)

    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model


def evaluate_model(
    model: H2GNN,
    dataset: ElectrostaticDataset,
    max_depth: int = 4,
    device: torch.device = torch.device('cpu')
) -> dict:
    """
    Evaluate model on dataset.

    Returns:
        dict with metrics: mse, mae, relative_error, etc.
    """
    model.eval()

    total_mse = 0.0
    total_mae = 0.0
    total_relative_error = 0.0
    total_samples = 0
    total_particles = 0

    all_errors = []

    with torch.no_grad():
        for particles, E_true in dataset:
            particles = particles.to(device)
            E_true = E_true.to(device)

            # Build quadtree and predict
            positions = particles[:, :2]
            quadtree = Quadtree(positions, max_depth=max_depth)
            E_pred = model(particles, quadtree)

            # Compute errors
            error = E_pred - E_true
            mse = (error ** 2).mean().item()
            mae = error.abs().mean().item()

            E_mag = torch.norm(E_true, dim=-1)
            error_mag = torch.norm(error, dim=-1)
            relative_error = (error_mag / (E_mag + 1e-8)).mean().item()

            total_mse += mse
            total_mae += mae
            total_relative_error += relative_error
            total_samples += 1
            total_particles += len(particles)

            all_errors.append(error_mag.cpu().numpy())

    all_errors = np.concatenate(all_errors)

    return {
        'mse': total_mse / total_samples,
        'rmse': np.sqrt(total_mse / total_samples),
        'mae': total_mae / total_samples,
        'relative_error': total_relative_error / total_samples,
        'total_particles': total_particles,
        'error_percentiles': {
            '50': np.percentile(all_errors, 50),
            '90': np.percentile(all_errors, 90),
            '99': np.percentile(all_errors, 99),
        }
    }


def visualize_prediction(
    particles: torch.Tensor,
    E_true: torch.Tensor,
    E_pred: torch.Tensor,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    Visualize ground truth vs predicted E field.

    Args:
        particles: (N, 3) tensor of (x, y, q)
        E_true: (N, 2) ground truth E field
        E_pred: (N, 2) predicted E field
        title: figure title
        save_path: path to save figure
        show: whether to display figure

    Returns:
        matplotlib Figure
    """
    pos = particles[:, :2].cpu().numpy()
    charges = particles[:, 2].cpu().numpy()
    E_true_np = E_true.cpu().numpy()
    E_pred_np = E_pred.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Normalize arrow lengths for visualization
    E_scale = max(
        np.linalg.norm(E_true_np, axis=1).max(),
        np.linalg.norm(E_pred_np, axis=1).max()
    )
    if E_scale < 1e-8:
        E_scale = 1.0

    # Color normalization for charges
    vmax = max(abs(charges.min()), abs(charges.max()))
    if vmax < 1e-8:
        vmax = 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)

    # Plot 1: Ground truth
    ax = axes[0]
    scatter = ax.scatter(pos[:, 0], pos[:, 1], c=charges, cmap='RdBu_r',
                         s=60, edgecolors='k', linewidths=0.5, norm=norm)
    ax.quiver(pos[:, 0], pos[:, 1],
              E_true_np[:, 0] / E_scale, E_true_np[:, 1] / E_scale,
              scale=10, alpha=0.7, color='green')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Ground Truth E Field')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot 2: Predicted
    ax = axes[1]
    ax.scatter(pos[:, 0], pos[:, 1], c=charges, cmap='RdBu_r',
               s=60, edgecolors='k', linewidths=0.5, norm=norm)
    ax.quiver(pos[:, 0], pos[:, 1],
              E_pred_np[:, 0] / E_scale, E_pred_np[:, 1] / E_scale,
              scale=10, alpha=0.7, color='blue')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title('Predicted E Field')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Plot 3: Error magnitude
    ax = axes[2]
    error_mag = np.linalg.norm(E_pred_np - E_true_np, axis=1)
    scatter_err = ax.scatter(pos[:, 0], pos[:, 1], c=error_mag, cmap='hot',
                             s=60, edgecolors='k', linewidths=0.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f'Error |E_pred - E_true|\nMean: {error_mag.mean():.4f}')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter_err, ax=ax, shrink=0.8)

    if title:
        fig.suptitle(title, fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def visualize_quadtree(
    quadtree: Quadtree,
    particles: torch.Tensor,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Visualize quadtree structure with particles."""
    fig, ax = plt.subplots(figsize=(8, 8))

    pos = particles[:, :2].cpu().numpy()
    charges = particles[:, 2].cpu().numpy()

    # Plot particles
    vmax = max(abs(charges.min()), abs(charges.max()), 0.1)
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    ax.scatter(pos[:, 0], pos[:, 1], c=charges, cmap='RdBu_r',
               s=80, edgecolors='k', linewidths=1, norm=norm, zorder=10)

    # Plot voxel boundaries at each level
    colors = plt.cm.viridis(np.linspace(0, 1, quadtree.max_depth + 1))

    for level in range(quadtree.max_depth + 1):
        n_cells = 2 ** level
        cell_size = 1.0 / n_cells

        active = quadtree.get_active_voxels(level).cpu().numpy()

        for idx in active:
            iy = idx // n_cells
            ix = idx % n_cells

            x0 = ix * cell_size
            y0 = iy * cell_size

            # Draw rectangle
            rect = plt.Rectangle(
                (x0, y0), cell_size, cell_size,
                fill=False, edgecolor=colors[level],
                linewidth=2 - level * 0.3, alpha=0.7
            )
            ax.add_patch(rect)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect('equal')
    ax.set_title(f'Quadtree Structure (depth={quadtree.max_depth})')
    ax.set_xlabel('x')
    ax.set_ylabel('y')

    # Legend
    for level in range(quadtree.max_depth + 1):
        n_active = quadtree.num_active(level)
        ax.plot([], [], color=colors[level], linewidth=2,
                label=f'Level {level}: {n_active} voxels')
    ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def plot_training_curves(
    train_losses: list,
    val_losses: list,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """Plot training and validation loss curves."""
    fig, ax = plt.subplots(figsize=(10, 6))

    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax.plot(epochs, val_losses, 'r-', label='Validation Loss')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('H2GNN Training Curves')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    if show:
        plt.show()

    return fig


def main():
    parser = argparse.ArgumentParser(description='Evaluate H2GNN')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--n-test', type=int, default=100,
                        help='Number of test samples')
    parser.add_argument('--min-particles', type=int, default=10)
    parser.add_argument('--max-particles', type=int, default=100)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--output-dir', type=str, default='evaluation')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=999)

    args = parser.parse_args()

    # Set device
    if args.device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Evaluating on {device}")

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model
    print(f"Loading model from {args.checkpoint}")
    model = load_model(args.checkpoint, device)

    # Create test dataset
    print(f"Generating {args.n_test} test samples...")
    test_dataset = ElectrostaticDataset(
        n_samples=args.n_test,
        n_particles_range=(args.min_particles, args.max_particles),
        seed=args.seed,
        device=device
    )

    # Evaluate
    print("Evaluating...")
    metrics = evaluate_model(model, test_dataset, max_depth=args.depth, device=device)

    print("\nResults:")
    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE: {metrics['mae']:.6f}")
    print(f"  Relative Error: {metrics['relative_error']:.4f}")
    print(f"  Error Percentiles:")
    print(f"    50th: {metrics['error_percentiles']['50']:.4f}")
    print(f"    90th: {metrics['error_percentiles']['90']:.4f}")
    print(f"    99th: {metrics['error_percentiles']['99']:.4f}")

    # Visualize some examples
    print("\nGenerating visualizations...")

    for i in range(min(3, len(test_dataset))):
        particles, E_true = test_dataset[i]
        particles = particles.to(device)
        E_true = E_true.to(device)

        with torch.no_grad():
            positions = particles[:, :2]
            quadtree = Quadtree(positions, max_depth=args.depth)
            E_pred = model(particles, quadtree)

        save_path = os.path.join(args.output_dir, f'prediction_{i}.png')
        visualize_prediction(
            particles, E_true, E_pred,
            title=f'Sample {i} ({len(particles)} particles)',
            save_path=save_path,
            show=False
        )

        if i == 0:
            # Also visualize quadtree for first sample
            qt_path = os.path.join(args.output_dir, f'quadtree_{i}.png')
            visualize_quadtree(quadtree, particles, save_path=qt_path, show=False)

    print(f"\nVisualization saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
