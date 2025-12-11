"""
Training script for the state-prediction GNN.

Trains the GNN to predict U^{n+1} from U^n directly.

Supports GPU acceleration via DirectML (AMD GPUs on Windows).
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import time
import copy
import sys
import os

# Check for CUDA (NVIDIA GPU) support
CUDA_AVAILABLE = torch.cuda.is_available()

# Check for DirectML (AMD GPU) support
try:
    import torch_directml
    DML_AVAILABLE = True
except ImportError:
    DML_AVAILABLE = False

# Determine which GPU is available (prefer CUDA over DirectML)
GPU_AVAILABLE = CUDA_AVAILABLE or DML_AVAILABLE

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from meltflow.functions.parameters import create_parameters, apply_defaults
from meltflow.functions.grid import grid_setup
from meltflow.functions.state_var import state_var
from meltflow.functions.ghost_fluid import ghost_GFM, real_GFM, extrp_vel
from meltflow.functions.level_set import advc
from meltflow.functions.solver import run_slvr
from meltflow.input.configs import load_config

from .model_state import FullStateGNN
from .graph import create_1d_graph


class StateDataNormalizer:
    """Normalize node features for state prediction."""

    def __init__(self):
        self.node_mean = None
        self.node_std = None
        self.delta_mean = None
        self.delta_std = None

    def fit(self, graphs: List[Tuple[Data, torch.Tensor]]):
        """Compute normalization statistics."""
        all_nodes = []
        all_deltas = []

        for g, delta in graphs:
            all_nodes.append(g.x.numpy())
            all_deltas.append(delta.numpy())

        all_nodes = np.concatenate(all_nodes, axis=0)
        all_deltas = np.concatenate(all_deltas, axis=0)

        self.node_mean = torch.tensor(all_nodes.mean(axis=0), dtype=torch.float32)
        self.node_std = torch.tensor(all_nodes.std(axis=0) + 1e-8, dtype=torch.float32)
        self.delta_mean = torch.tensor(all_deltas.mean(axis=0), dtype=torch.float32)
        self.delta_std = torch.tensor(all_deltas.std(axis=0) + 1e-8, dtype=torch.float32)

        print(f"   Node stats: mean std range [{self.node_std.min():.2e}, {self.node_std.max():.2e}]")
        print(f"   Delta stats: mean={self.delta_mean.numpy()}, std={self.delta_std.numpy()}")

    def transform(self, graphs: List[Tuple[Data, torch.Tensor]]) -> List[Tuple[Data, torch.Tensor]]:
        """Normalize graphs and targets."""
        normalized = []
        for g, delta in graphs:
            g_new = copy.copy(g)
            g_new.x = (g.x - self.node_mean) / self.node_std
            delta_norm = (delta - self.delta_mean) / self.delta_std
            normalized.append((g_new, delta_norm))
        return normalized

    def inverse_transform_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """Convert normalized delta back to original scale."""
        return delta * self.delta_std + self.delta_mean


def generate_state_pairs(config_name: str = 'in_1Dsod1fl') -> List[Tuple[Data, torch.Tensor]]:
    """
    Generate (current_state, next_state_delta) pairs from simulation.

    Returns list of (graph, delta) where delta = U^{n+1} - U^n
    """
    print(f"   Generating state pairs from {config_name}...")

    # Load configuration
    config, init_func = load_config(config_name)

    # Grid setup
    n_var, n, x, dx, U, phi = grid_setup(config)
    config['n_var'] = n_var
    config['n'] = n
    config['dx'] = dx
    config = apply_defaults(config, config['n_dim'])
    prm = create_parameters(config)
    prm.n = n
    prm.dx = dx
    prm.n_var = n_var

    # Apply initial conditions
    U, phi = init_func(x, U, phi)

    # Fixed timestep
    dt_fixed = 0.1336323e-05

    pairs = []
    t = 0.0
    it = 0

    while t < prm.t_f:
        dt = dt_fixed
        if t + dt > prm.t_f:
            dt = prm.t_f - t

        # Save current state
        U_current = U.copy()
        phi_current = phi.copy()

        # Create graph for current state
        graph = create_1d_graph(x, U_current, phi_current)

        # Run one solver step
        t += dt
        W = state_var(prm, "cons", n_var, phi, U)
        UU, WW = ghost_GFM(prm, x, phi, U)
        WW = run_slvr(prm, dt, x, phi, UU, WW)
        V = extrp_vel(prm, x, phi, WW)
        phi = advc(prm, dt, V, phi)
        W = real_GFM(prm, phi, WW)
        U = state_var(prm, "prim", n_var, phi, W)

        # Compute delta = U^{n+1} - U^n
        delta = torch.tensor((U - U_current).T, dtype=torch.float32)  # Shape (n, n_var)

        pairs.append((graph, delta))
        it += 1

    print(f"   Generated {len(pairs)} state pairs")
    return pairs


def get_device(use_gpu: bool = True):
    """Get the best available device for training.

    Priority: CUDA (NVIDIA) > DirectML (AMD) > CPU
    """
    if use_gpu:
        if CUDA_AVAILABLE:
            device = torch.device('cuda')
            print(f"   Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            return device
        elif DML_AVAILABLE:
            device = torch_directml.device()
            print(f"   Using DirectML GPU: {torch_directml.device_name(0)}")
            return device
    print("   Using CPU")
    return torch.device('cpu')


def move_graph_to_device(graph: Data, device) -> Data:
    """Move graph data to the specified device."""
    graph_new = Data(
        x=graph.x.to(device),
        edge_index=graph.edge_index.to(device),
        edge_attr=graph.edge_attr.to(device)
    )
    return graph_new


class GraphDataset(torch.utils.data.Dataset):
    """Dataset wrapper for graph-target pairs."""

    def __init__(self, data: List[Tuple[Data, torch.Tensor]]):
        self.graphs = [d[0] for d in data]
        self.targets = [d[1] for d in data]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        # Add target to graph data for batching
        g = self.graphs[idx].clone()
        g.target = self.targets[idx]
        return g


def train_state_model(
    train_data: List[Tuple[Data, torch.Tensor]],
    val_data: Optional[List[Tuple[Data, torch.Tensor]]] = None,
    n_var: int = 3,
    hidden_dim: int = 256,
    n_layers: int = 4,
    n_message_passing: int = 2,
    n_epochs: int = 500,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    verbose: bool = True,
    use_gpu: bool = True,
    num_workers: int = 0
) -> Tuple[FullStateGNN, dict]:
    """Train the state prediction GNN with proper batching.

    Parameters
    ----------
    use_gpu : bool
        If True and DirectML is available, use AMD GPU for training.
    num_workers : int
        Number of worker processes for data loading.
    """
    # Get device
    device = get_device(use_gpu)

    model = FullStateGNN(
        n_var=n_var,
        n_node_features=5,
        n_edge_features=2,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        n_message_passing=n_message_passing
    )
    model = model.to(device)

    # Create data loaders with proper batching
    train_dataset = GraphDataset(train_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    val_loader = None
    if val_data:
        val_dataset = GraphDataset(val_data)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    history = {'train_loss': [], 'val_loss': [], 'epoch_time': []}

    for epoch in range(n_epochs):
        epoch_start = time.time()

        model.train()
        train_loss = 0.0
        n_samples = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Forward pass on batched graphs
            pred_delta = model(batch)

            # Loss against batched targets
            loss = criterion(pred_delta, batch.target)

            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs
            n_samples += batch.num_graphs

        train_loss /= n_samples
        history['train_loss'].append(train_loss)

        # Validation
        val_loss = 0.0
        if val_loader:
            model.eval()
            n_val_samples = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred_delta = model(batch)
                    loss = criterion(pred_delta, batch.target)
                    val_loss += loss.item() * batch.num_graphs
                    n_val_samples += batch.num_graphs
            val_loss /= n_val_samples
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        epoch_time = time.time() - epoch_start
        history['epoch_time'].append(epoch_time)

        if verbose and (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.6f}"
            if val_loader:
                msg += f", val_loss={val_loss:.6f}"
            msg += f", time={epoch_time:.2f}s"
            print(msg)

    # Move model back to CPU for saving/inference
    model = model.to('cpu')
    return model, history


def main():
    """Main training script for state prediction model."""
    print("=" * 60)
    print("MeltFlow State-Prediction GNN Training")
    print("=" * 60)

    # Set up CPU multi-threading for performance
    n_threads = os.cpu_count() or 4
    torch.set_num_threads(n_threads)
    print(f"\nCPU Threads: {n_threads}")

    # Check GPU availability
    if CUDA_AVAILABLE:
        print(f"GPU Acceleration: CUDA with {torch.cuda.get_device_name(0)}")
    elif DML_AVAILABLE:
        print(f"GPU Acceleration: DirectML with {torch_directml.device_name(0)}")
    else:
        print("GPU Acceleration: Not available (CPU only)")

    # Generate training data
    print("\n1. Generating training data...")
    all_pairs = generate_state_pairs('in_1Dsod1fl')

    # Split into train/val (sequential - keeps temporal coherence)
    n_train = int(0.8 * len(all_pairs))
    train_pairs_raw = all_pairs[:n_train]
    val_pairs_raw = all_pairs[n_train:]
    print(f"   Train samples: {len(train_pairs_raw)}")
    print(f"   Val samples: {len(val_pairs_raw)}")

    # Normalize data
    print("\n   Normalizing data...")
    normalizer = StateDataNormalizer()
    normalizer.fit(train_pairs_raw)
    train_pairs = normalizer.transform(train_pairs_raw)
    val_pairs = normalizer.transform(val_pairs_raw)

    # Train model - use larger batch size with GPU for better utilization
    batch_size = 64 if GPU_AVAILABLE else 32
    print("\n2. Training model...")
    model, history = train_state_model(
        train_data=train_pairs,
        val_data=val_pairs,
        n_var=3,
        hidden_dim=256,
        n_layers=4,
        n_message_passing=2,
        n_epochs=500,
        batch_size=batch_size,
        learning_rate=1e-3,
        verbose=True,
        use_gpu=GPU_AVAILABLE
    )

    # Evaluate
    print("\n3. Evaluating model...")
    model.eval()
    all_pred = []
    all_target = []
    with torch.no_grad():
        for graph, target_delta in val_pairs:
            pred_delta = model(graph)
            all_pred.append(pred_delta.numpy())
            all_target.append(target_delta.numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    mse = np.mean((all_pred - all_target) ** 2)
    mae = np.mean(np.abs(all_pred - all_target))
    print(f"   Normalized MSE: {mse:.6f}")
    print(f"   Normalized MAE: {mae:.6f}")

    # Per-variable metrics in original scale
    pred_orig = normalizer.inverse_transform_delta(torch.tensor(all_pred)).numpy()
    target_orig = normalizer.inverse_transform_delta(torch.tensor(all_target)).numpy()

    var_names = ['Density', 'Velocity', 'Pressure']
    print(f"\n   Per-variable MAE (original scale):")
    for i, name in enumerate(var_names):
        var_mae = np.mean(np.abs(pred_orig[:, i] - target_orig[:, i]))
        var_std = np.std(target_orig[:, i])
        print(f"      {name}: MAE={var_mae:.2e}, Target std={var_std:.2e}, Ratio={var_mae/(var_std+1e-10):.4f}")

    # Plot training history
    print("\n4. Plotting results...")
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.semilogy(history['train_loss'], label='Train')
    if history['val_loss']:
        ax.semilogy(history['val_loss'], label='Validation')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('State Prediction GNN Training')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('state_model_training.png', dpi=150)
    plt.show()

    # Save model and normalizer
    torch.save({
        'model_state_dict': model.state_dict(),
        'normalizer': {
            'node_mean': normalizer.node_mean,
            'node_std': normalizer.node_std,
            'delta_mean': normalizer.delta_mean,
            'delta_std': normalizer.delta_std
        }
    }, 'state_model.pt')
    print("\nModel saved to state_model.pt")

    return model, history, normalizer


if __name__ == '__main__':
    main()
