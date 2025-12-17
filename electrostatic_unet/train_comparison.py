"""
Training script to compare H2GNN vs H2GNN-Attention.

Trains both models on identical data and compares their performance.
"""

import torch
import torch.nn.functional as F
import argparse
import os
import time
import json
from typing import Optional, Dict, List, Tuple

from .quadtree import Quadtree
from .h2gnn import H2GNN
from .h2gnn_attention import H2GNNAttention
from .dataset import ElectrostaticDataset


def train_model(
    model: torch.nn.Module,
    model_name: str,
    train_dataset: ElectrostaticDataset,
    val_dataset: ElectrostaticDataset,
    n_epochs: int,
    max_depth: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    checkpoint_dir: str,
    verbose: bool = True
) -> Tuple[List[float], List[float], float]:
    """
    Train a single model.

    Returns:
        train_losses, val_losses, best_val_loss
    """
    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"\n{'='*60}")
        print(f"Training {model_name}")
        print(f"{'='*60}")
        print(f"Parameters: {n_params:,}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=lr * 0.01
    )

    best_val_loss = float('inf')
    train_losses = []
    val_losses = []

    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_count = 0

        for particles, E_true in train_dataset:
            positions = particles[:, :2]
            quadtree = Quadtree(positions, max_depth=max_depth)

            E_pred = model(particles, quadtree)
            loss = F.mse_loss(E_pred, E_true)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        train_loss /= train_count

        # Validation
        model.eval()
        val_loss = 0.0
        val_count = 0

        with torch.no_grad():
            for particles, E_true in val_dataset:
                positions = particles[:, :2]
                quadtree = Quadtree(positions, max_depth=max_depth)
                E_pred = model(particles, quadtree)
                val_loss += F.mse_loss(E_pred, E_true).item()
                val_count += 1

        val_loss /= val_count

        scheduler.step()

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start

        if verbose and (epoch + 1) % 10 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:3d}/{n_epochs}: "
                  f"train={train_loss:.6f}, val={val_loss:.6f}, "
                  f"lr={current_lr:.2e}, time={epoch_time:.2f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(checkpoint_dir, f"{model_name}_best.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
            }, best_path)

    if verbose:
        print(f"  Best validation loss: {best_val_loss:.6f}")

    return train_losses, val_losses, best_val_loss


def compare_models(
    n_epochs: int = 200,
    n_train: int = 500,
    n_val: int = 100,
    n_particles_range: Tuple[int, int] = (20, 80),
    max_depth: int = 4,
    hidden_dim: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    k_per_voxel: int = 16,
    n_heads: int = 4,
    n_attn_layers: int = 2,
    checkpoint_dir: str = "checkpoints/comparison",
    device: Optional[str] = None,
    seed: int = 42,
    verbose: bool = True
) -> Dict:
    """
    Train and compare H2GNN vs H2GNN-Attention.
    """
    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    device = torch.device(device)

    if verbose:
        print(f"Comparing H2GNN vs H2GNN-Attention")
        print(f"Device: {device}")
        print(f"Epochs: {n_epochs}")
        print(f"Training samples: {n_train}")
        print(f"Validation samples: {n_val}")
        print(f"Particles per sample: {n_particles_range}")
        print(f"Quadtree depth: {max_depth}")
        print(f"Hidden dim: {hidden_dim}")
        print(f"Learning rate: {lr}")

    # Create datasets (same for both models)
    if verbose:
        print("\nGenerating datasets...")

    torch.manual_seed(seed)
    train_dataset = ElectrostaticDataset(
        n_samples=n_train,
        n_particles_range=n_particles_range,
        seed=seed,
        device=device
    )
    val_dataset = ElectrostaticDataset(
        n_samples=n_val,
        n_particles_range=n_particles_range,
        seed=seed + 1000,
        device=device
    )

    if verbose:
        print(f"  Training: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")

    os.makedirs(checkpoint_dir, exist_ok=True)

    results = {}

    # Train original H2GNN
    torch.manual_seed(seed)
    model_h2gnn = H2GNN(
        max_depth=max_depth,
        hidden_dim=hidden_dim,
        n_mlp_layers=2
    ).to(device)

    train_losses_h2gnn, val_losses_h2gnn, best_h2gnn = train_model(
        model_h2gnn, "H2GNN", train_dataset, val_dataset,
        n_epochs, max_depth, lr, weight_decay, device, checkpoint_dir, verbose
    )

    results['H2GNN'] = {
        'train_losses': train_losses_h2gnn,
        'val_losses': val_losses_h2gnn,
        'best_val_loss': best_h2gnn,
        'n_params': sum(p.numel() for p in model_h2gnn.parameters())
    }

    # Train H2GNN-Attention
    torch.manual_seed(seed)
    model_attention = H2GNNAttention(
        max_depth=max_depth,
        hidden_dim=hidden_dim,
        n_heads=n_heads,
        n_attn_layers=n_attn_layers,
        k_per_voxel=k_per_voxel
    ).to(device)

    train_losses_attn, val_losses_attn, best_attn = train_model(
        model_attention, "H2GNN-Attention", train_dataset, val_dataset,
        n_epochs, max_depth, lr, weight_decay, device, checkpoint_dir, verbose
    )

    results['H2GNN-Attention'] = {
        'train_losses': train_losses_attn,
        'val_losses': val_losses_attn,
        'best_val_loss': best_attn,
        'n_params': sum(p.numel() for p in model_attention.parameters())
    }

    # Summary
    if verbose:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")
        print(f"{'Model':<20} {'Parameters':>12} {'Best Val Loss':>15}")
        print(f"{'-'*47}")
        for name, res in results.items():
            print(f"{name:<20} {res['n_params']:>12,} {res['best_val_loss']:>15.6f}")

        improvement = (best_h2gnn - best_attn) / best_h2gnn * 100
        if improvement > 0:
            print(f"\nH2GNN-Attention is {improvement:.1f}% better than H2GNN")
        else:
            print(f"\nH2GNN is {-improvement:.1f}% better than H2GNN-Attention")

    # Save results
    results_path = os.path.join(checkpoint_dir, "comparison_results.json")
    with open(results_path, 'w') as f:
        # Convert to JSON-serializable format
        json_results = {}
        for name, res in results.items():
            json_results[name] = {
                'train_losses': res['train_losses'],
                'val_losses': res['val_losses'],
                'best_val_loss': res['best_val_loss'],
                'n_params': res['n_params']
            }
        json.dump(json_results, f, indent=2)

    # Plot results
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Training loss
        ax = axes[0]
        ax.semilogy(results['H2GNN']['train_losses'], label='H2GNN', alpha=0.8)
        ax.semilogy(results['H2GNN-Attention']['train_losses'], label='H2GNN-Attention', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Training Loss (log)')
        ax.set_title('Training Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Validation loss
        ax = axes[1]
        ax.semilogy(results['H2GNN']['val_losses'], label='H2GNN', alpha=0.8)
        ax.semilogy(results['H2GNN-Attention']['val_losses'], label='H2GNN-Attention', alpha=0.8)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss (log)')
        ax.set_title('Validation Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.suptitle('H2GNN vs H2GNN-Attention Comparison', fontsize=14)
        plt.tight_layout()

        plot_path = os.path.join(checkpoint_dir, "comparison_plot.png")
        plt.savefig(plot_path, dpi=150)
        if verbose:
            print(f"\nPlot saved to: {plot_path}")
        plt.close()

    except ImportError:
        if verbose:
            print("\nMatplotlib not available, skipping plot")

    return results


def main():
    parser = argparse.ArgumentParser(description='Compare H2GNN vs H2GNN-Attention')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--n-train', type=int, default=500)
    parser.add_argument('--n-val', type=int, default=100)
    parser.add_argument('--min-particles', type=int, default=20)
    parser.add_argument('--max-particles', type=int, default=80)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--k-per-voxel', type=int, default=16)
    parser.add_argument('--n-heads', type=int, default=4)
    parser.add_argument('--n-attn-layers', type=int, default=2)
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints/comparison')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()

    compare_models(
        n_epochs=args.epochs,
        n_train=args.n_train,
        n_val=args.n_val,
        n_particles_range=(args.min_particles, args.max_particles),
        max_depth=args.depth,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        weight_decay=args.weight_decay,
        k_per_voxel=args.k_per_voxel,
        n_heads=args.n_heads,
        n_attn_layers=args.n_attn_layers,
        checkpoint_dir=args.checkpoint_dir,
        device=args.device,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
