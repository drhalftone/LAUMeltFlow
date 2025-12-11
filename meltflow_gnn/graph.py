"""
Graph representation for 1D finite volume grids.

Maps the finite volume grid to a graph structure suitable for GNN:
- Nodes: Grid cells (elements)
- Edges: Cell interfaces (faces)
- Node features: Conserved/primitive variables at cell centers
- Edge features: Interface properties (normal, area, distance)
"""

import torch
import numpy as np
from torch_geometric.data import Data
from typing import Tuple, Optional


def create_1d_graph(
    x: np.ndarray,
    U: np.ndarray,
    phi: np.ndarray,
    include_boundaries: bool = True
) -> Data:
    """
    Create a PyTorch Geometric graph from 1D simulation data.

    In 1D, the grid looks like:

        |--0--|--1--|--2--|--..--|--n-1--|

    Each cell i has interfaces with cells i-1 and i+1.

    Parameters
    ----------
    x : np.ndarray
        Cell center coordinates, shape (n,)
    U : np.ndarray
        Primitive variables [rho, u, p], shape (n_var, n)
    phi : np.ndarray
        Level set function, shape (n,)
    include_boundaries : bool
        Whether to include boundary nodes

    Returns
    -------
    Data
        PyTorch Geometric Data object with:
        - x: Node features [rho, u, p, phi, x_coord], shape (n, 5)
        - edge_index: Edge connectivity, shape (2, n_edges)
        - edge_attr: Edge features [dx, normal], shape (n_edges, 2)
    """
    n = len(x)
    n_var = U.shape[0]

    # Node features: [rho, u, p, phi, x_coord]
    # For 1D: n_var = 3 (rho, u, p)
    node_features = np.zeros((n, n_var + 2))
    node_features[:, :n_var] = U.T  # Transpose to (n, n_var)
    node_features[:, n_var] = phi
    node_features[:, n_var + 1] = x

    # Edge connectivity: each interior interface connects two cells
    # For cell i, edges connect to i-1 (left) and i+1 (right)
    # We create directed edges in both directions

    edges_src = []
    edges_dst = []
    edge_attrs = []

    for i in range(n - 1):
        # Edge from cell i to cell i+1 (right interface)
        edges_src.append(i)
        edges_dst.append(i + 1)
        dx = x[i + 1] - x[i]
        normal = 1.0  # Pointing right
        edge_attrs.append([dx, normal])

        # Edge from cell i+1 to cell i (left interface, reverse direction)
        edges_src.append(i + 1)
        edges_dst.append(i)
        normal = -1.0  # Pointing left
        edge_attrs.append([dx, normal])

    edge_index = torch.tensor([edges_src, edges_dst], dtype=torch.long)
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)
    node_features = torch.tensor(node_features, dtype=torch.float32)

    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_attr)


def create_flux_training_data(
    x: np.ndarray,
    U: np.ndarray,
    phi: np.ndarray,
    flux: np.ndarray
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create training data for learning the numerical flux.

    For each interface, we want to learn:
        F* = f(U_left, U_right, normal)

    Parameters
    ----------
    x : np.ndarray
        Cell center coordinates, shape (n,)
    U : np.ndarray
        Primitive variables [rho, u, p], shape (n_var, n)
    phi : np.ndarray
        Level set function, shape (n,)
    flux : np.ndarray
        Numerical flux at interfaces, shape (n_var, n-1)

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        - inputs: [U_left, U_right, phi_left, phi_right, normal], shape (n-1, 2*n_var+3)
        - targets: Flux values, shape (n-1, n_var)
        - dx: Interface spacing, shape (n-1,)
    """
    n = len(x)
    n_var = U.shape[0]
    n_interfaces = n - 1

    # Input features for each interface
    # [rho_L, u_L, p_L, rho_R, u_R, p_R, phi_L, phi_R, normal]
    inputs = np.zeros((n_interfaces, 2 * n_var + 3))

    for i in range(n_interfaces):
        # Left state (cell i)
        inputs[i, :n_var] = U[:, i]
        # Right state (cell i+1)
        inputs[i, n_var:2*n_var] = U[:, i + 1]
        # Level set values
        inputs[i, 2*n_var] = phi[i]
        inputs[i, 2*n_var + 1] = phi[i + 1]
        # Normal (always +1 for rightward flux in 1D)
        inputs[i, 2*n_var + 2] = 1.0

    # Targets: numerical flux
    targets = flux.T  # Shape (n-1, n_var)

    # Interface spacing
    dx = np.diff(x)

    return (
        torch.tensor(inputs, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.float32),
        torch.tensor(dx, dtype=torch.float32)
    )


def graph_to_arrays(data: Data) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert PyTorch Geometric Data back to numpy arrays.

    Parameters
    ----------
    data : Data
        PyTorch Geometric Data object

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - x: Cell center coordinates
        - U: Primitive variables
        - phi: Level set function
    """
    node_features = data.x.numpy()
    n_var = node_features.shape[1] - 2  # Subtract phi and x_coord

    U = node_features[:, :n_var].T
    phi = node_features[:, n_var]
    x = node_features[:, n_var + 1]

    return x, U, phi
