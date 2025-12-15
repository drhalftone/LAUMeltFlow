"""
Training script for the flux MLP using uniform grid-sampled data.

PyTorch CUDA implementation for fast GPU training.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, List
import time
import argparse


class FluxMLP(nn.Module):
    """
    MLP to learn the Roe flux function.

    Input: [rho_L, u_L, p_L, rho_R, u_R, p_R] (6 features)
    Output: [flux_rho, flux_rhou, flux_E] (3 features)
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 3,
        hidden_dims: List[int] = [256, 256, 256, 256, 256],
        activation: str = 'relu',
        use_residual: bool = False
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.use_residual = use_residual

        # Activation function
        if activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'silu':
            act_fn = nn.SiLU()
        else:
            act_fn = nn.ReLU()

        # Build layers
        layers = []
        dims = [input_dim] + hidden_dims

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(act_fn)

        # Output layer (no activation)
        layers.append(nn.Linear(hidden_dims[-1], output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights (He initialization for ReLU)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class FluxMLPResidual(nn.Module):
    """
    MLP with residual connections for learning the Roe flux.
    """

    def __init__(
        self,
        input_dim: int = 6,
        output_dim: int = 3,
        hidden_dim: int = 256,
        n_blocks: int = 5,
        activation: str = 'gelu'
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

        # Activation
        if activation == 'gelu':
            self.act = nn.GELU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        else:
            self.act = nn.ReLU()

        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Residual blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_blocks):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                self.act,
                nn.Linear(hidden_dim, hidden_dim),
            ))

        # Output projection
        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.input_proj(x))

        for block in self.blocks:
            h = h + block(h)  # Residual connection

        return self.output_proj(h)


def load_data(
    data_path: str,
    val_split: float = 0.1,
    normalize: bool = True,
    device: str = 'cuda'
) -> Tuple[DataLoader, DataLoader, Dict]:
    """Load and prepare data for PyTorch training."""

    print(f"Loading data from {data_path}...")
    data = np.load(data_path, allow_pickle=True)
    X = data['inputs'].astype(np.float32)
    Y = data['outputs'].astype(np.float32)

    # Check if gamma is included as input feature
    input_dim = X.shape[1]
    has_gamma = input_dim == 7
    if has_gamma:
        print(f"  Loaded {len(X):,} samples (includes gamma as 7th input)")
    else:
        print(f"  Loaded {len(X):,} samples")

    # Compute normalization stats
    stats = {}
    if normalize:
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0)
        X_std[X_std < 1e-8] = 1.0

        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0)
        Y_std[Y_std < 1e-8] = 1.0

        X = (X - X_mean) / X_std
        Y = (Y - Y_mean) / Y_std

        stats = {
            'X_mean': X_mean,
            'X_std': X_std,
            'Y_mean': Y_mean,
            'Y_std': Y_std
        }

        print(f"  Normalized: X mean={X.mean():.4f}, std={X.std():.4f}")
        print(f"              Y mean={Y.mean():.4f}, std={Y.std():.4f}")

    # Store input/output dimensions in stats
    stats['input_dim'] = input_dim
    stats['output_dim'] = Y.shape[1]
    stats['has_gamma'] = has_gamma

    # Shuffle and split
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    n_val = int(n_samples * val_split)

    val_idx = indices[:n_val]
    train_idx = indices[n_val:]

    X_train, Y_train = X[train_idx], Y[train_idx]
    X_val, Y_val = X[val_idx], Y[val_idx]

    print(f"  Train: {len(X_train):,}, Val: {len(X_val):,}")

    # Convert to tensors
    X_train_t = torch.from_numpy(X_train)
    Y_train_t = torch.from_numpy(Y_train)
    X_val_t = torch.from_numpy(X_val)
    Y_val_t = torch.from_numpy(Y_val)

    return X_train_t, Y_train_t, X_val_t, Y_val_t, stats


def train(
    model: nn.Module,
    X_train: torch.Tensor,
    Y_train: torch.Tensor,
    X_val: torch.Tensor,
    Y_val: torch.Tensor,
    n_epochs: int = 500,
    batch_size: int = 8192,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    scheduler_type: str = 'cosine',
    warmup_epochs: int = 10,
    device: str = 'cuda',
    verbose: bool = True,
    print_every: int = 10
) -> Dict:
    """Train the model with PyTorch - all data on GPU for speed."""

    model = model.to(device)

    # Move ALL data to GPU upfront (1M samples = ~24MB, easily fits)
    X_train = X_train.to(device)
    Y_train = Y_train.to(device)
    X_val = X_val.to(device)
    Y_val = Y_val.to(device)

    n_train = X_train.shape[0]
    n_batches = (n_train + batch_size - 1) // batch_size

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler
    if scheduler_type == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=n_epochs, eta_min=lr * 0.01
        )
    elif scheduler_type == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=20, verbose=True
        )
    else:
        scheduler = None

    # Loss function
    criterion = nn.MSELoss()

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
        'epoch_time': []
    }

    best_val_loss = float('inf')
    best_state_dict = None

    print(f"\nTraining on {device} (all data on GPU)...")
    print(f"  Samples: {n_train:,}")
    print(f"  Batch size: {batch_size}")
    print(f"  Batches/epoch: {n_batches}")
    print(f"  Learning rate: {lr}")
    print(f"  Scheduler: {scheduler_type}")
    print()

    for epoch in range(n_epochs):
        epoch_start = time.time()
        model.train()

        # Shuffle indices on GPU
        perm = torch.randperm(n_train, device=device)

        train_loss = 0.0
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, n_train)
            idx = perm[start_idx:end_idx]

            X_batch = X_train[idx]
            Y_batch = Y_train[idx]

            optimizer.zero_grad()
            Y_pred = model(X_batch)
            loss = criterion(Y_pred, Y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= n_batches

        # Validation
        model.eval()
        with torch.no_grad():
            Y_val_pred = model(X_val)
            val_loss = criterion(Y_val_pred, Y_val).item()

        # Update scheduler
        current_lr = optimizer.param_groups[0]['lr']
        if scheduler is not None:
            if scheduler_type == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()

        epoch_time = time.time() - epoch_start

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Print progress
        if verbose and ((epoch + 1) % print_every == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:4d}/{n_epochs}: "
                  f"train={train_loss:.6f}, val={val_loss:.6f}, "
                  f"lr={current_lr:.2e}, time={epoch_time:.2f}s")

    # Restore best model
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        print(f"\nRestored best model with val_loss={best_val_loss:.6f}")

    return history


def evaluate(
    model: nn.Module,
    X: torch.Tensor,
    Y: torch.Tensor,
    stats: Dict,
    device: str = 'cuda'
) -> Dict:
    """Evaluate model and return metrics in original scale."""

    model = model.to(device)
    model.eval()

    X = X.to(device)
    Y = Y.to(device)

    with torch.no_grad():
        Y_pred = model(X)

    # Convert to numpy and denormalize
    Y_pred_np = Y_pred.cpu().numpy()
    Y_np = Y.cpu().numpy()

    Y_pred_orig = Y_pred_np * stats['Y_std'] + stats['Y_mean']
    Y_orig = Y_np * stats['Y_std'] + stats['Y_mean']

    mse = np.mean((Y_pred_orig - Y_orig) ** 2)
    mae = np.mean(np.abs(Y_pred_orig - Y_orig))

    # Per-variable metrics
    var_names = ['flux_rho', 'flux_rhou', 'flux_E']
    per_var_mae = {}
    per_var_rel = {}

    for i, name in enumerate(var_names):
        mae_i = np.mean(np.abs(Y_pred_orig[:, i] - Y_orig[:, i]))
        std_i = np.std(Y_orig[:, i])
        per_var_mae[name] = mae_i
        per_var_rel[name] = mae_i / std_i * 100  # Relative error %

    return {
        'mse': mse,
        'mae': mae,
        'per_var_mae': per_var_mae,
        'per_var_rel': per_var_rel
    }


def save_model(
    model: nn.Module,
    path: str,
    stats: Dict,
    config: Dict,
    history: Dict = None
):
    """Save model in PyTorch format."""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'stats': stats,
        'config': config,
    }
    if history:
        save_dict['history'] = history

    torch.save(save_dict, path)
    print(f"Saved model to {path}")


def save_numpy_compatible(
    model: nn.Module,
    path: str,
    stats: Dict,
    config: Dict,
    history: Dict = None
):
    """Save model in NumPy-compatible format for simulate_with_mlp.py."""

    save_dict = {}

    # Extract weights and biases
    state_dict = model.state_dict()
    layer_idx = 0

    for name, param in state_dict.items():
        if 'weight' in name:
            # PyTorch stores weights as (out_features, in_features), transpose for NumPy
            save_dict[f'W{layer_idx}'] = param.cpu().numpy().T
        elif 'bias' in name:
            save_dict[f'b{layer_idx}'] = param.cpu().numpy()
            layer_idx += 1

    save_dict['n_layers'] = layer_idx

    # Save stats
    for key, val in stats.items():
        save_dict[f'stats_{key}'] = val

    # Save config
    save_dict['config_input_dim'] = config.get('input_dim', 6)
    save_dict['config_output_dim'] = config.get('output_dim', 3)
    save_dict['config_hidden_dims'] = np.array(config.get('hidden_dims', [256]*5))
    save_dict['config_activation'] = config.get('activation', 'relu')

    # Save history
    if history:
        save_dict['history_train_loss'] = np.array(history.get('train_loss', []))
        save_dict['history_val_loss'] = np.array(history.get('val_loss', []))
        save_dict['history_lr'] = np.array(history.get('lr', []))
        save_dict['history_epoch_time'] = np.array(history.get('epoch_time', []))

    np.savez(path, **save_dict)
    print(f"Saved NumPy-compatible model to {path}")


def main():
    parser = argparse.ArgumentParser(description='Train flux MLP (PyTorch CUDA)')
    parser.add_argument('--data', type=str, default='data/uniform_flux_data.npz',
                        help='Path to training data')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8192,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5,
                        help='Weight decay')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='Hidden dimension')
    parser.add_argument('--n-layers', type=int, default=5,
                        help='Number of hidden layers')
    parser.add_argument('--activation', type=str, default='relu',
                        choices=['relu', 'gelu', 'silu'], help='Activation function')
    parser.add_argument('--residual', action='store_true',
                        help='Use residual connections')
    parser.add_argument('--scheduler', type=str, default='cosine',
                        choices=['cosine', 'plateau', 'none'], help='LR scheduler')
    parser.add_argument('--val-split', type=float, default=0.1,
                        help='Validation split ratio')
    parser.add_argument('--print-every', type=int, default=1,
                        help='Print progress every N epochs')
    parser.add_argument('--output', type=str, default='flux_model_cuda.pt',
                        help='Output model path (.pt for PyTorch)')
    parser.add_argument('--output-npz', type=str, default='flux_model_cuda.npz',
                        help='Output NumPy-compatible model path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device (cuda or cpu)')

    args = parser.parse_args()

    # Check CUDA
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 60)
    print("Flux MLP Training (PyTorch CUDA)")
    print("=" * 60)

    if args.device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA Version: {torch.version.cuda}")

    # Set random seeds
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Load data
    print("\n1. Loading data...")
    X_train, Y_train, X_val, Y_val, stats = load_data(
        args.data, val_split=args.val_split, normalize=True, device=args.device
    )

    # Create model
    print("\n2. Creating model...")
    hidden_dims = [args.hidden_dim] * args.n_layers
    input_dim = stats['input_dim']
    output_dim = stats['output_dim']

    if args.residual:
        model = FluxMLPResidual(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=args.hidden_dim,
            n_blocks=args.n_layers,
            activation=args.activation
        )
        print(f"  Architecture: FluxMLPResidual ({input_dim} -> {args.hidden_dim}x{args.n_layers} residual blocks -> {output_dim})")
    else:
        model = FluxMLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=args.activation
        )
        print(f"  Architecture: {input_dim} -> {' -> '.join(map(str, hidden_dims))} -> {output_dim}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  Activation: {args.activation}")

    # Train
    print("\n3. Training...")
    start_time = time.time()

    history = train(
        model=model,
        X_train=X_train,
        Y_train=Y_train,
        X_val=X_val,
        Y_val=Y_val,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler_type=args.scheduler,
        device=args.device,
        verbose=True,
        print_every=args.print_every
    )

    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"Average epoch time: {np.mean(history['epoch_time']):.2f}s")

    # Evaluate
    print("\n4. Evaluating...")
    metrics = evaluate(model, X_val, Y_val, stats, device=args.device)

    print(f"  MSE: {metrics['mse']:.6f}")
    print(f"  MAE: {metrics['mae']:.4f}")
    print(f"  Per-variable MAE (relative error):")
    for name in ['flux_rho', 'flux_rhou', 'flux_E']:
        mae = metrics['per_var_mae'][name]
        rel = metrics['per_var_rel'][name]
        print(f"    {name}: {mae:.4f} ({rel:.2f}%)")

    # Save model
    print("\n5. Saving model...")
    config = {
        'input_dim': input_dim,
        'output_dim': output_dim,
        'hidden_dims': hidden_dims,
        'activation': args.activation,
        'residual': args.residual,
        'has_gamma': stats.get('has_gamma', False)
    }

    # Save PyTorch format
    save_model(model, args.output, stats, config, history)

    # Save NumPy-compatible format (for simulate_with_mlp.py)
    if not args.residual:  # Only works for simple MLP
        save_numpy_compatible(model, args.output_npz, stats, config, history)

    print("\nDone!")

    return model, history, metrics, stats


if __name__ == '__main__':
    main()
