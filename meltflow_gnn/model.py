"""
GNN model for learning numerical flux.

Architecture follows the message-passing framework:
1. Edge function: Compute flux from left/right states
2. Aggregation: Sum fluxes at each node
3. Node update: Update conserved variables

The key learning task is the edge function (numerical flux).
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from typing import Optional, Tuple


class FluxMLP(nn.Module):
    """
    MLP to learn the numerical flux function with antisymmetric constraint.

    Approximates: F* = f(U_left, U_right, edge_features)

    Enforces antisymmetry: F_ij = -F_ji
    This guarantees conservation: flux leaving cell i equals flux entering cell j.

    The antisymmetry is achieved by:
    1. Computing symmetric features: (x_i + x_j) and |x_i - x_j|
    2. Computing antisymmetric features: (x_i - x_j)
    3. The MLP outputs a "base flux" from symmetric features
    4. The final flux is scaled by a learned function of antisymmetric features

    Input: Concatenation of left state, right state, and edge features
    Output: Numerical flux vector (antisymmetric w.r.t. i,j swap)
    """

    def __init__(
        self,
        n_var: int = 3,
        n_edge_features: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
        antisymmetric: bool = True
    ):
        """
        Parameters
        ----------
        n_var : int
            Number of conserved variables (3 for 1D Euler: rho, rho*u, E)
        n_edge_features : int
            Number of edge features (dx, normal)
        hidden_dim : int
            Hidden layer dimension
        n_layers : int
            Number of hidden layers
        antisymmetric : bool
            If True, enforce F_ij = -F_ji constraint
        """
        super().__init__()

        self.n_var = n_var
        self.antisymmetric = antisymmetric
        self.n_node_features = n_var + 2  # +2 for phi and x_coord

        if antisymmetric:
            # For exact antisymmetry, we use: F(i,j) = g(i,j) - g(j,i)
            # g takes symmetric features + one-sided state
            # Symmetric features: (x_i + x_j)/2, |x_i - x_j|, dx
            # One-sided state: x_i (or x_j)
            # Total input: 2 * n_node_features + 1 + n_node_features = 3 * n_node_features + 1
            g_input_dim = 3 * self.n_node_features + 1

            # Network g: learns flux contribution from one side
            sym_layers = []
            sym_layers.append(nn.Linear(g_input_dim, hidden_dim))
            sym_layers.append(nn.ReLU())
            for _ in range(n_layers - 1):
                sym_layers.append(nn.Linear(hidden_dim, hidden_dim))
                sym_layers.append(nn.ReLU())
            sym_layers.append(nn.Linear(hidden_dim, n_var))
            self.sym_mlp = nn.Sequential(*sym_layers)
        else:
            # Original non-antisymmetric version
            input_dim = 2 * (n_var + 2) + n_edge_features

            layers = []
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())

            for _ in range(n_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(nn.ReLU())

            layers.append(nn.Linear(hidden_dim, n_var))
            self.mlp = nn.Sequential(*layers)

    def forward(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flux from neighboring states.

        Parameters
        ----------
        x_i : torch.Tensor
            Features of source nodes, shape (n_edges, n_features)
        x_j : torch.Tensor
            Features of target nodes, shape (n_edges, n_features)
        edge_attr : torch.Tensor
            Edge attributes, shape (n_edges, n_edge_features)

        Returns
        -------
        torch.Tensor
            Numerical flux, shape (n_edges, n_var)
            If antisymmetric=True, guarantees F(i,j) = -F(j,i)
        """
        if self.antisymmetric:
            # For exact antisymmetry: F(i,j) = g(i,j) - g(j,i)
            # where g is any function. This guarantees F(j,i) = g(j,i) - g(i,j) = -F(i,j)

            # Symmetric features (invariant to i,j swap)
            avg_state = (x_i + x_j) / 2  # Average
            abs_diff = torch.abs(x_i - x_j)  # Absolute difference
            dx = edge_attr[:, 0:1]  # Grid spacing (symmetric)

            sym_input = torch.cat([avg_state, abs_diff, dx], dim=-1)

            # Compute g(i,j): flux contribution from i's perspective
            # Uses x_i as the "source" state
            g_ij_input = torch.cat([sym_input, x_i], dim=-1)
            g_ij = self.sym_mlp(g_ij_input)

            # Compute g(j,i): flux contribution from j's perspective
            # Uses x_j as the "source" state
            g_ji_input = torch.cat([sym_input, x_j], dim=-1)
            g_ji = self.sym_mlp(g_ji_input)

            # Antisymmetric flux: F(i,j) = g(i,j) - g(j,i)
            flux = g_ij - g_ji

            return flux
        else:
            # Original implementation
            inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
            return self.mlp(inputs)


class FluxGNN(MessagePassing):
    """
    GNN layer that mimics one timestep of the DG/FV method.

    Message passing:
    1. message(): Compute numerical flux at each interface
    2. aggregate(): Sum incoming fluxes (with sign from normal)
    3. update(): Compute dU/dt from flux divergence
    """

    def __init__(
        self,
        n_var: int = 3,
        n_edge_features: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
        antisymmetric: bool = True
    ):
        super().__init__(aggr='add')  # Sum aggregation

        self.n_var = n_var
        self.flux_net = FluxMLP(n_var, n_edge_features, hidden_dim, n_layers, antisymmetric)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute flux divergence for each node.

        Parameters
        ----------
        x : torch.Tensor
            Node features, shape (n_nodes, n_features)
        edge_index : torch.Tensor
            Edge connectivity, shape (2, n_edges)
        edge_attr : torch.Tensor
            Edge attributes, shape (n_edges, n_edge_features)

        Returns
        -------
        torch.Tensor
            Flux divergence for each node, shape (n_nodes, n_var)
        """
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute message (numerical flux) for each edge.

        Parameters
        ----------
        x_i : torch.Tensor
            Features of source nodes
        x_j : torch.Tensor
            Features of target nodes
        edge_attr : torch.Tensor
            Edge attributes [dx, normal]

        Returns
        -------
        torch.Tensor
            Flux contribution, shape (n_edges, n_var)
        """
        # Compute numerical flux
        flux = self.flux_net(x_i, x_j, edge_attr)

        # Scale by normal direction (for proper flux differencing)
        # normal is the second edge attribute
        normal = edge_attr[:, 1:2]
        dx = edge_attr[:, 0:1]

        # Return flux * normal / dx (contribution to flux divergence)
        return flux * normal / dx


class EulerGNN(nn.Module):
    """
    Full GNN model for solving 1D Euler equations.

    Predicts the state at the next timestep:
        U^{n+1} = U^n - dt * div(F)

    where div(F) is computed by the FluxGNN layer.

    With antisymmetric=True, the flux function guarantees F_ij = -F_ji,
    which ensures exact conservation of mass, momentum, and energy.
    """

    def __init__(
        self,
        n_var: int = 3,
        n_edge_features: int = 2,
        hidden_dim: int = 64,
        n_layers: int = 3,
        n_message_passing: int = 1,
        antisymmetric: bool = True
    ):
        """
        Parameters
        ----------
        n_var : int
            Number of conserved variables
        n_edge_features : int
            Number of edge features
        hidden_dim : int
            Hidden layer dimension
        n_layers : int
            Number of hidden layers in flux MLP
        n_message_passing : int
            Number of message passing iterations
        antisymmetric : bool
            If True, enforce F_ij = -F_ji for conservation
        """
        super().__init__()

        self.n_var = n_var
        self.n_message_passing = n_message_passing
        self.antisymmetric = antisymmetric

        # Flux computation layers
        self.flux_layers = nn.ModuleList([
            FluxGNN(n_var, n_edge_features, hidden_dim, n_layers, antisymmetric)
            for _ in range(n_message_passing)
        ])

    def forward(
        self,
        data: Data,
        dt: float = 1.0
    ) -> torch.Tensor:
        """
        Predict state at next timestep.

        Parameters
        ----------
        data : Data
            PyTorch Geometric Data object with node/edge features
        dt : float
            Timestep size

        Returns
        -------
        torch.Tensor
            Updated node features (only conserved variables), shape (n_nodes, n_var)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Extract current conserved variables
        U = x[:, :self.n_var]

        # Compute flux divergence through message passing
        for layer in self.flux_layers:
            div_flux = layer(x, edge_index, edge_attr)
            # Update conserved variables
            U = U - dt * div_flux

        return U

    def compute_flux(
        self,
        data: Data
    ) -> torch.Tensor:
        """
        Compute numerical flux at interfaces (for training/analysis).

        Parameters
        ----------
        data : Data
            PyTorch Geometric Data object

        Returns
        -------
        torch.Tensor
            Flux at each edge, shape (n_edges, n_var)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Get source and target node features
        x_i = x[edge_index[0]]
        x_j = x[edge_index[1]]

        # Compute flux using the first layer's flux network
        return self.flux_layers[0].flux_net(x_i, x_j, edge_attr)
