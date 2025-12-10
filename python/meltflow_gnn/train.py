"""
Training script for the GNN flux learner.

Trains the GNN to predict numerical flux from left/right states.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import time

from .model import EulerGNN, FluxMLP
from .data_generator import generate_training_dataset, FluxDataset
from .graph import create_1d_graph


def train_flux_model(
    train_graphs: List[Data],
    val_graphs: Optional[List[Data]] = None,
    n_var: int = 3,
    hidden_dim: int = 64,
    n_layers: int = 3,
    n_epochs: int = 100,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str = 'cpu',
    verbose: bool = True
) -> Tuple[EulerGNN, dict]:
    """
    Train the GNN flux model.

    Parameters
    ----------
    train_graphs : List[Data]
        Training data (graphs with flux targets)
    val_graphs : List[Data], optional
        Validation data
    n_var : int
        Number of conserved variables
    hidden_dim : int
        Hidden layer dimension
    n_layers : int
        Number of hidden layers
    n_epochs : int
        Number of training epochs
    batch_size : int
        Batch size
    learning_rate : float
        Learning rate
    device : str
        Device to train on ('cpu' or 'cuda')
    verbose : bool
        Print training progress

    Returns
    -------
    Tuple[EulerGNN, dict]
        Trained model and training history
    """
    # Create model
    model = EulerGNN(
        n_var=n_var,
        n_edge_features=2,
        hidden_dim=hidden_dim,
        n_layers=n_layers
    ).to(device)

    # Create data loaders
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    if val_graphs:
        val_loader = DataLoader(val_graphs, batch_size=batch_size)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'epoch_time': []
    }

    # Training loop
    for epoch in range(n_epochs):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        n_batches = 0

        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()

            # Compute predicted flux
            pred_flux = model.compute_flux(batch)

            # Get target flux (need to handle batched data)
            # For batched graphs, flux is concatenated
            target_flux = batch.flux

            # Only use edges going in positive direction (normal = 1)
            # to avoid double counting
            mask = batch.edge_attr[:, 1] > 0
            pred_flux = pred_flux[mask]
            target_flux = target_flux

            # Compute loss
            loss = criterion(pred_flux, target_flux)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            n_batches += 1

        train_loss /= n_batches
        history['train_loss'].append(train_loss)

        # Validation
        val_loss = 0.0
        if val_graphs:
            model.eval()
            n_val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    pred_flux = model.compute_flux(batch)
                    target_flux = batch.flux
                    mask = batch.edge_attr[:, 1] > 0
                    pred_flux = pred_flux[mask]
                    loss = criterion(pred_flux, target_flux)
                    val_loss += loss.item()
                    n_val_batches += 1
            val_loss /= n_val_batches
            history['val_loss'].append(val_loss)
            scheduler.step(val_loss)
        else:
            scheduler.step(train_loss)

        epoch_time = time.time() - epoch_start
        history['epoch_time'].append(epoch_time)

        if verbose and (epoch + 1) % 10 == 0:
            msg = f"Epoch {epoch+1}/{n_epochs}: train_loss={train_loss:.6f}"
            if val_graphs:
                msg += f", val_loss={val_loss:.6f}"
            msg += f", time={epoch_time:.2f}s"
            print(msg)

    return model, history


def evaluate_model(
    model: EulerGNN,
    test_graphs: List[Data],
    device: str = 'cpu'
) -> dict:
    """
    Evaluate the trained model on test data.

    Parameters
    ----------
    model : EulerGNN
        Trained model
    test_graphs : List[Data]
        Test data
    device : str
        Device

    Returns
    -------
    dict
        Evaluation metrics
    """
    model.eval()
    model.to(device)

    all_pred = []
    all_target = []

    with torch.no_grad():
        for graph in test_graphs:
            graph = graph.to(device)
            pred_flux = model.compute_flux(graph)
            target_flux = graph.flux

            # Only positive direction edges
            mask = graph.edge_attr[:, 1] > 0
            pred_flux = pred_flux[mask]

            all_pred.append(pred_flux.cpu().numpy())
            all_target.append(target_flux.cpu().numpy())

    all_pred = np.concatenate(all_pred, axis=0)
    all_target = np.concatenate(all_target, axis=0)

    # Compute metrics
    mse = np.mean((all_pred - all_target) ** 2)
    mae = np.mean(np.abs(all_pred - all_target))
    rel_error = np.mean(np.abs(all_pred - all_target) / (np.abs(all_target) + 1e-8))

    return {
        'mse': mse,
        'mae': mae,
        'relative_error': rel_error,
        'predictions': all_pred,
        'targets': all_target
    }


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """
    Plot training history.

    Parameters
    ----------
    history : dict
        Training history from train_flux_model
    save_path : str, optional
        Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss curves
    axes[0].semilogy(history['train_loss'], label='Train')
    if 'val_loss' in history and history['val_loss']:
        axes[0].semilogy(history['val_loss'], label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Epoch time
    axes[1].plot(history['epoch_time'])
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Time (s)')
    axes[1].set_title('Epoch Time')
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_flux_comparison(
    metrics: dict,
    var_names: List[str] = ['Mass', 'Momentum', 'Energy'],
    save_path: Optional[str] = None
):
    """
    Plot predicted vs target flux.

    Parameters
    ----------
    metrics : dict
        Output from evaluate_model
    var_names : List[str]
        Names of conserved variables
    save_path : str, optional
        Path to save figure
    """
    pred = metrics['predictions']
    target = metrics['targets']
    n_var = pred.shape[1]

    fig, axes = plt.subplots(1, n_var, figsize=(4*n_var, 4))
    if n_var == 1:
        axes = [axes]

    for i, (ax, name) in enumerate(zip(axes, var_names[:n_var])):
        ax.scatter(target[:, i], pred[:, i], alpha=0.3, s=1)
        lims = [
            min(target[:, i].min(), pred[:, i].min()),
            max(target[:, i].max(), pred[:, i].max())
        ]
        ax.plot(lims, lims, 'r--', label='Perfect prediction')
        ax.set_xlabel(f'Target {name} Flux')
        ax.set_ylabel(f'Predicted {name} Flux')
        ax.set_title(f'{name} Flux')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
    plt.show()


def main():
    """Main training script."""
    print("=" * 60)
    print("MeltFlow GNN Training")
    print("=" * 60)

    # Generate training data
    print("\n1. Generating training data...")
    train_graphs = generate_training_dataset(
        config_names=['in_1Dsod1fl'],
        n_trajectories_per_config=1,
        save_interval=5
    )

    # Split into train/val
    n_train = int(0.8 * len(train_graphs))
    val_graphs = train_graphs[n_train:]
    train_graphs = train_graphs[:n_train]
    print(f"   Train samples: {len(train_graphs)}")
    print(f"   Val samples: {len(val_graphs)}")

    # Train model
    print("\n2. Training model...")
    model, history = train_flux_model(
        train_graphs=train_graphs,
        val_graphs=val_graphs,
        n_var=3,
        hidden_dim=64,
        n_layers=3,
        n_epochs=100,
        batch_size=8,
        learning_rate=1e-3,
        verbose=True
    )

    # Evaluate
    print("\n3. Evaluating model...")
    metrics = evaluate_model(model, val_graphs)
    print(f"   MSE: {metrics['mse']:.6f}")
    print(f"   MAE: {metrics['mae']:.6f}")
    print(f"   Relative Error: {metrics['relative_error']:.4%}")

    # Plot results
    print("\n4. Plotting results...")
    plot_training_history(history)
    plot_flux_comparison(metrics)

    # Save model
    torch.save(model.state_dict(), 'flux_model.pt')
    print("\nModel saved to flux_model.pt")

    return model, history, metrics


if __name__ == '__main__':
    main()
