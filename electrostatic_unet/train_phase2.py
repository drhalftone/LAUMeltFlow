"""
Phase 2: Add Level 3 (parent of leaves) to the autoencoder.

Loads Phase 1 weights and adds the next hierarchy level.
Particles -> Leaves -> Level 3 -> Leaves -> Particles

Usage:
    python -m electrostatic_unet.train_phase2 --epochs 100 --n-train 10000
"""

import argparse
import sys
import time
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

from electrostatic_unet.dataset import ElectrostaticDataset
from electrostatic_unet.quadtree import Quadtree


def flush_print(*args, **kwargs):
    """Print with immediate flush."""
    print(*args, **kwargs)
    sys.stdout.flush()


class TwoLevelAutoencoder(nn.Module):
    """
    Phase 2: Two-level autoencoder (Leaves + Level 3).

    Architecture:
        ENCODER (bottom-up):
            Particles -> Encoder MLP -> h_particles
            Scatter to leaves -> Leaf Encoder MLP -> h_leaves_enc [save for skip]
            Scatter to Level 3 -> Level 3 Encoder MLP -> h_level3

        DECODER (top-down):
            h_level3 -> Gather to leaves
            Concat skip [h_from_parent, h_leaves_enc] -> Leaf Decoder MLP -> h_leaves_dec
            Gather to particles
            Concat skip [h_gathered, particles] -> Decoder MLP -> reconstructed
    """

    def __init__(self, particle_dim: int = 3, hidden_dim: int = 64, n_layers: int = 2):
        super().__init__()

        self.particle_dim = particle_dim
        self.hidden_dim = hidden_dim

        # Particle encoder: (x, y, q) -> hidden
        self.particle_encoder = self._make_mlp(particle_dim, hidden_dim, hidden_dim, n_layers)

        # Leaf level (Level 4)
        self.leaf_encoder = self._make_mlp(hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.leaf_decoder = self._make_mlp(hidden_dim * 2, hidden_dim, hidden_dim, n_layers)  # concat skip

        # Level 3 (parent of leaves)
        self.level3_encoder = self._make_mlp(hidden_dim, hidden_dim, hidden_dim, n_layers)
        self.level3_decoder = self._make_mlp(hidden_dim, hidden_dim, hidden_dim, n_layers)

        # Particle decoder: (hidden + particle features) -> reconstructed
        self.particle_decoder = self._make_mlp(hidden_dim + particle_dim, hidden_dim, particle_dim, n_layers)

    def _make_mlp(self, in_dim: int, hidden_dim: int, out_dim: int, n_layers: int) -> nn.Sequential:
        """Build MLP with GELU activations."""
        layers = []
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.GELU())
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, out_dim))
        return nn.Sequential(*layers)

    def forward(self, particles: torch.Tensor, quadtree: Quadtree) -> torch.Tensor:
        """Forward pass through two-level hierarchy."""
        device = particles.device
        N = particles.shape[0]
        max_depth = quadtree.max_depth  # 4

        # Get mappings
        particle_to_leaf = quadtree.particle_to_leaf.to(device)
        n_leaves = len(quadtree.active_voxels[max_depth])
        n_level3 = len(quadtree.active_voxels[max_depth - 1])

        # Child-to-parent mapping for leaves -> level 3
        leaf_to_level3 = quadtree.child_to_parent[max_depth].to(device)

        # ============ ENCODER (bottom-up) ============

        # Particles -> hidden
        h_particles = self.particle_encoder(particles)  # (N, hidden)

        # Scatter to leaves
        h_leaves = torch.zeros(n_leaves, self.hidden_dim, device=device)
        h_leaves.scatter_add_(
            0,
            particle_to_leaf.unsqueeze(1).expand(-1, self.hidden_dim),
            h_particles
        )

        # Leaf encoder
        h_leaves_enc = self.leaf_encoder(h_leaves)  # (n_leaves, hidden) - save for skip

        # Scatter to Level 3
        h_level3 = torch.zeros(n_level3, self.hidden_dim, device=device)
        h_level3.scatter_add_(
            0,
            leaf_to_level3.unsqueeze(1).expand(-1, self.hidden_dim),
            h_leaves_enc
        )

        # Level 3 encoder (bottleneck for this phase)
        h_level3 = self.level3_encoder(h_level3)  # (n_level3, hidden)

        # ============ DECODER (top-down) ============

        # Level 3 decoder
        h_level3_dec = self.level3_decoder(h_level3)  # (n_level3, hidden)

        # Gather to leaves (broadcast parent state)
        h_from_parent = h_level3_dec[leaf_to_level3]  # (n_leaves, hidden)

        # Leaf decoder with skip connection
        h_leaves_combined = torch.cat([h_from_parent, h_leaves_enc], dim=-1)  # (n_leaves, hidden*2)
        h_leaves_dec = self.leaf_decoder(h_leaves_combined)  # (n_leaves, hidden)

        # Gather to particles
        h_gathered = h_leaves_dec[particle_to_leaf]  # (N, hidden)

        # Particle decoder with skip connection
        h_combined = torch.cat([h_gathered, particles], dim=-1)  # (N, hidden + 3)
        reconstructed = self.particle_decoder(h_combined)  # (N, 3)

        return reconstructed

    def load_phase1_weights(self, checkpoint_path: str):
        """Load weights from Phase 1 checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        phase1_state = checkpoint['model_state_dict']

        # Map Phase 1 weights to Phase 2
        # Phase 1 encoder -> Phase 2 particle_encoder
        # Phase 1 hyperedge_mlp -> Phase 2 leaf_encoder
        # Phase 1 decoder -> Phase 2 particle_decoder

        new_state = {}
        for key, value in phase1_state.items():
            if key.startswith('encoder.'):
                new_key = key.replace('encoder.', 'particle_encoder.')
                new_state[new_key] = value
            elif key.startswith('hyperedge_mlp.'):
                new_key = key.replace('hyperedge_mlp.', 'leaf_encoder.')
                new_state[new_key] = value
            elif key.startswith('decoder.'):
                new_key = key.replace('decoder.', 'particle_decoder.')
                new_state[new_key] = value

        # Load partial state dict
        current_state = self.state_dict()
        current_state.update(new_state)
        self.load_state_dict(current_state)

        return len(new_state)


def train_epoch(model, dataset, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for i in range(len(dataset)):
        particles, _ = dataset[i]
        particles = particles.to(device)

        positions = particles[:, :2]
        quadtree = Quadtree(positions, max_depth=4)

        optimizer.zero_grad()
        reconstructed = model(particles, quadtree)

        loss = criterion(reconstructed, particles)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataset)


def validate(model, dataset, device):
    """Validate and return metrics."""
    model.eval()
    total_loss = 0.0
    total_x_err = 0.0
    total_y_err = 0.0
    total_q_err = 0.0
    n_particles = 0
    criterion = nn.MSELoss()

    with torch.no_grad():
        for i in range(len(dataset)):
            particles, _ = dataset[i]
            particles = particles.to(device)

            positions = particles[:, :2]
            quadtree = Quadtree(positions, max_depth=4)
            reconstructed = model(particles, quadtree)

            loss = criterion(reconstructed, particles)
            total_loss += loss.item()

            err = (reconstructed - particles).abs()
            total_x_err += err[:, 0].sum().item()
            total_y_err += err[:, 1].sum().item()
            total_q_err += err[:, 2].sum().item()
            n_particles += particles.shape[0]

    return {
        'mse': total_loss / len(dataset),
        'x_mae': total_x_err / n_particles,
        'y_mae': total_y_err / n_particles,
        'q_mae': total_q_err / n_particles,
    }


def main():
    parser = argparse.ArgumentParser(description='Phase 2: Two-level autoencoder')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--n-train', type=int, default=10000)
    parser.add_argument('--n-val', type=int, default=500)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--n-layers', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--phase1-checkpoint', type=str, default='checkpoints/phase1_best.pt')
    parser.add_argument('--freeze-phase1', action='store_true', help='Freeze Phase 1 weights')
    parser.add_argument('--save-plot', type=str, default='phase2_curve.png')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    flush_print("=" * 70)
    flush_print("Phase 2: Two-Level Autoencoder (Leaves + Level 3)")
    flush_print("=" * 70)
    flush_print(f"Device: {device}")
    flush_print(f"Epochs: {args.epochs}")
    flush_print(f"Training samples: {args.n_train}")
    flush_print(f"Hidden dim: {args.hidden_dim}")
    flush_print(f"Learning rate: {args.lr}")
    flush_print(f"Phase 1 checkpoint: {args.phase1_checkpoint}")
    flush_print(f"Freeze Phase 1 weights: {args.freeze_phase1}")
    flush_print()

    flush_print("Architecture:")
    flush_print("  Encoder: Particles -> Leaves -> Level 3")
    flush_print("  Decoder: Level 3 -> Leaves (+ skip) -> Particles (+ skip)")
    flush_print()

    # Generate datasets
    flush_print("Generating datasets...")
    train_dataset = ElectrostaticDataset(n_samples=args.n_train, n_particles_range=(10, 100))
    val_dataset = ElectrostaticDataset(n_samples=args.n_val, n_particles_range=(10, 100))
    flush_print(f"  Training: {len(train_dataset)} samples")
    flush_print(f"  Validation: {len(val_dataset)} samples")
    flush_print()

    # Create model
    model = TwoLevelAutoencoder(
        particle_dim=3,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    flush_print(f"Model parameters: {n_params:,}")

    # Load Phase 1 weights
    if Path(args.phase1_checkpoint).exists():
        n_loaded = model.load_phase1_weights(args.phase1_checkpoint)
        flush_print(f"Loaded {n_loaded} weight tensors from Phase 1")

        if args.freeze_phase1:
            # Freeze particle_encoder, leaf_encoder, particle_decoder
            for name, param in model.named_parameters():
                if any(x in name for x in ['particle_encoder', 'leaf_encoder', 'particle_decoder']):
                    param.requires_grad = False
            n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            flush_print(f"Frozen Phase 1 weights. Trainable parameters: {n_trainable:,}")
    else:
        flush_print(f"WARNING: Phase 1 checkpoint not found at {args.phase1_checkpoint}")
        flush_print("Training from scratch")

    flush_print()

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    Path(args.checkpoint_dir).mkdir(exist_ok=True)

    flush_print(f"{'Epoch':>6} | {'Train MSE':>12} | {'Val MSE':>12} | {'x MAE':>8} | {'y MAE':>8} | {'q MAE':>8} | {'LR':>10} | Status")
    flush_print("-" * 95)

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()

        train_loss = train_epoch(model, train_dataset, optimizer, device)
        val_metrics = validate(model, val_dataset, device)
        val_loss = val_metrics['mse']

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        lr = optimizer.param_groups[0]['lr']

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            status = "* best"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': {
                    'particle_dim': 3,
                    'hidden_dim': args.hidden_dim,
                    'n_layers': args.n_layers,
                }
            }, f"{args.checkpoint_dir}/phase2_best.pt")

        flush_print(f"{epoch:>6} | {train_loss:>12.6f} | {val_loss:>12.6f} | "
                   f"{val_metrics['x_mae']:>8.5f} | {val_metrics['y_mae']:>8.5f} | "
                   f"{val_metrics['q_mae']:>8.5f} | {lr:>10.2e} | {status}")

        if epoch % 10 == 0 or epoch == args.epochs:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            ax = axes[0]
            ax.plot(train_losses, 'b-', label='Train', linewidth=2)
            ax.plot(val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Phase 2: Reconstruction Loss (Log)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_yscale('log')

            ax = axes[1]
            ax.plot(train_losses, 'b-', label='Train', linewidth=2)
            ax.plot(val_losses, 'r-', label='Val', linewidth=2)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('MSE Loss')
            ax.set_title('Phase 2: Reconstruction Loss (Linear)')
            ax.legend()
            ax.grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(args.save_plot, dpi=150)
            plt.close()
            flush_print(f"       | Plot saved: {args.save_plot}")

        if epoch % 50 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'train_losses': train_losses,
                'val_losses': val_losses,
            }, f"{args.checkpoint_dir}/phase2_epoch{epoch}.pt")

    flush_print("-" * 95)
    flush_print(f"Phase 2 complete! Best validation MSE: {best_val_loss:.6f}")
    flush_print(f"Checkpoint saved: {args.checkpoint_dir}/phase2_best.pt")
    flush_print()
    flush_print("Next: Run Phase 3 to add Level 2")


if __name__ == '__main__':
    main()
