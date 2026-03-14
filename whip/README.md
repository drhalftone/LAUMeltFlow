# Whip: Bead Chain Simulation for GNN Surrogate Modeling

Simulates a chain of mass nodes connected by spring-rod elements, intended as a testbed for training graph neural network (GNN) surrogates on mesh-based physics.

## Components

| Component | Description |
|-----------|-------------|
| `simulation.py` | NumPy FEA-style simulation (global force assembly, symplectic Euler) |
| `visualize.py` | Matplotlib animation of simulation trajectories |
| `main.py` | Entry point for running simulations and generating training data |
| `qt/` | Interactive Qt/C++ version with real-time visualization |

## Building the Qt App

Requires Qt 6.x and MSVC 2022. From the `qt/` directory:

```bash
# Build
build.bat

# Run (sets up Qt DLL paths)
run.bat
```

## Current Model

The simulation models a **uniform spring-mass chain** using:
- Hooke's law springs between adjacent beads
- Viscous damping along rod axes
- Global velocity drag
- Symplectic Euler integration (first-order, energy-conserving)

Node 0 is a fixed anchor; all other nodes are free to move under gravity and spring forces.

## Known Limitations

### 1. Uniform mass distribution
All beads have equal mass (`total_mass / n_nodes`). A real whip tapers — mass per unit length decreases toward the tip. This taper is what causes velocity amplification: as a wave travels down the chain, energy is conserved but the moving mass decreases, so velocity increases. With uniform mass, the simulation behaves like a bead necklace, not a whip.

### 2. Springs instead of rigid constraints
Real whip segments are essentially inextensible. This model uses finite-stiffness Hooke's law springs, which allow stretching. High stiffness approximates rigidity but requires very small timesteps to remain stable. A constraint-based method (position-based dynamics or Lagrange multipliers) would be more physically appropriate for inextensible chains.

### 3. No driving input
A whip crack requires a specific handle motion — a quick forward-then-back flick that launches a loop down the chain. Currently the anchor is fixed in space. Prescribing a time-dependent anchor trajectory is needed to study whip-crack dynamics.

### 4. Single message-passing hop per timestep
In the Qt/C++ version, each bead only sees its immediate neighbors per timestep (mirroring a 1-layer GNN). The end bead's mass affects the anchor bead's forces only after the signal propagates through every intermediate link — one hop per step. The Python version computes forces globally but the same propagation delay exists in the physics.

## Next Steps: Energy-Conserving GNN

The current simulation is suitable for training a basic GNN surrogate. The next goal is to study systems where **conservation of energy across the chain** is a hard physical constraint — for example, a tapered whip where kinetic energy transfers from heavy base segments to light tip segments without loss.

Key questions for the GNN architecture:
- How to enforce global conservation laws (energy, momentum) in a local message-passing framework
- Whether Hamiltonian or Lagrangian neural network formulations can guarantee conservation by construction
- How multi-hop message passing or global readout layers can capture long-range energy transfer that single-hop architectures miss
