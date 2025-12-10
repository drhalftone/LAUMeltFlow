"""
GNN model that predicts the full state update.

Instead of learning just the flux, this model learns:
    U^{n+1} = GNN(U^n, graph_structure)

This directly captures the full solver behavior including boundary conditions.
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.data import Data
from typing import Optional, Tuple


class StateUpdateMLP(nn.Module):
    """
    MLP that computes state update contribution from neighboring cells.

    For each edge, computes how the neighbor influences the target cell.
    """

    def __init__(
        self,
        n_input: int = 5,  # Node features: rho, u, p, phi, x
        n_edge: int = 2,   # Edge features: dx, normal
        n_output: int = 3,  # State variables: rho, u, p
        hidden_dim: int = 256,
        n_layers: int = 4
    ):
        super().__init__()

        # Input: [node_i features, node_j features, edge_attr]
        input_dim = 2 * n_input + n_edge

        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.SiLU())  # Smooth activation

        for _ in range(n_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())

        layers.append(nn.Linear(hidden_dim, n_output))

        self.mlp = nn.Sequential(*layers)

    def forward(self, x_i, x_j, edge_attr):
        inputs = torch.cat([x_i, x_j, edge_attr], dim=-1)
        return self.mlp(inputs)


class StateGNN(MessagePassing):
    """
    GNN layer that computes state updates via message passing.

    Each node aggregates messages from neighbors to compute its update.
    """

    def __init__(
        self,
        n_input: int = 5,
        n_edge: int = 2,
        n_output: int = 3,
        hidden_dim: int = 256,
        n_layers: int = 4
    ):
        super().__init__(aggr='add')

        self.n_output = n_output
        self.message_net = StateUpdateMLP(n_input, n_edge, n_output, hidden_dim, n_layers)

        # Node update network (combines self features with aggregated messages)
        self.update_net = nn.Sequential(
            nn.Linear(n_input + n_output, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_output)
        )

    def forward(self, x, edge_index, edge_attr):
        # Aggregate messages from neighbors
        aggr_msg = self.propagate(edge_index, x=x, edge_attr=edge_attr)

        # Combine with self features to get update
        combined = torch.cat([x, aggr_msg], dim=-1)
        delta = self.update_net(combined)

        return delta

    def message(self, x_i, x_j, edge_attr):
        return self.message_net(x_i, x_j, edge_attr)


class FullStateGNN(nn.Module):
    """
    Full GNN model that predicts next state from current state.

    Predicts: U^{n+1} = U^n + GNN(U^n, graph)

    Uses residual connection for stability.
    """

    def __init__(
        self,
        n_var: int = 3,
        n_node_features: int = 5,
        n_edge_features: int = 2,
        hidden_dim: int = 256,
        n_layers: int = 4,
        n_message_passing: int = 2
    ):
        super().__init__()

        self.n_var = n_var
        self.n_message_passing = n_message_passing

        # Multiple message passing layers
        self.gnn_layers = nn.ModuleList([
            StateGNN(n_node_features, n_edge_features, n_var, hidden_dim, n_layers)
            for _ in range(n_message_passing)
        ])

        # Final output layer
        self.output_net = nn.Sequential(
            nn.Linear(n_var * n_message_passing, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_var)
        )

    def forward(self, data: Data) -> torch.Tensor:
        """
        Predict state update.

        Parameters
        ----------
        data : Data
            Graph with node features [rho, u, p, phi, x]

        Returns
        -------
        torch.Tensor
            Predicted state update delta_U, shape (n_nodes, n_var)
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = data.edge_attr

        # Collect outputs from each message passing layer
        deltas = []
        for layer in self.gnn_layers:
            delta = layer(x, edge_index, edge_attr)
            deltas.append(delta)

        # Combine all layer outputs
        combined = torch.cat(deltas, dim=-1)
        final_delta = self.output_net(combined)

        return final_delta

    def predict_next_state(self, data: Data) -> torch.Tensor:
        """
        Predict next state U^{n+1} = U^n + delta.

        Parameters
        ----------
        data : Data
            Current state as graph

        Returns
        -------
        torch.Tensor
            Next state [rho, u, p], shape (n_nodes, n_var)
        """
        current_state = data.x[:, :self.n_var]  # First 3 features are rho, u, p
        delta = self.forward(data)
        return current_state + delta
