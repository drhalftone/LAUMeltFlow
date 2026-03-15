# Whip: Bead Chain Simulation for GNN Surrogate Modeling

Simulates a chain of mass nodes connected by spring-rod elements, intended as a testbed for training graph neural network (GNN) surrogates on mesh-based physics.

## Components

| Component | Description |
|-----------|-------------|
| `simulation.py` | NumPy FEA-style simulation (global force assembly, symplectic Euler) |
| `visualize.py` | Matplotlib animation of simulation trajectories |
| `run_sim.py` | Live simulation with real-time matplotlib visualization |
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

The simulation models a **tapered spring-mass chain** with:
- Hooke's law springs between adjacent beads
- Viscous damping along rod axes
- Global velocity drag
- Symplectic Euler integration (first-order, energy-conserving)
- SHAKE constraint projection for rod inextensibility
- Tapered mass distribution (configurable base-to-tip ratio)

The chain uses **16 beads** (power of 2) to support the binary-tree GNN architecture described below. Node 0 is a fixed anchor; nodes 1-15 are free to move under gravity and spring forces.

## Simulation Architecture

### Qt/C++ (signal/slot message passing)

The Qt simulation mirrors a GNN's computation in three phases per timestep:

1. **Force accumulation** (local): each bead receives neighbor state via `onNeighborState()` and accumulates spring/damping forces — analogous to GNN message passing + aggregation
2. **Integration** (local): each bead updates its own velocity/position from accumulated forces — analogous to GNN node update
3. **Constraint projection** (global): iterative SHAKE sweep over all rods to enforce inextensibility — no GNN analogue in a flat architecture

### Python (global assembly)

Identical physics but computed as a single vectorized pass over all edges. Both produce the same results.

## GNN Surrogate Plan: Binary-Tree Message Passing

### The problem with flat GNNs

A flat GNN with K message-passing layers has a receptive field of K hops. For a 16-bead chain, the tip bead needs 15 hops to influence the anchor — requiring 15 GNN layers for full propagation. The SHAKE constraint projection is a global operation that a local architecture cannot represent.

### Solution: hierarchical graph via binary tree

Instead of adding more message-passing layers, we add more **nodes** to the graph. A binary tree is built on top of the 16 bead nodes, creating interior nodes that provide shortcuts for long-range communication:

```
                              (30)                          Level 4 (root)
                            /      \
                       (28)          (29)                   Level 3
                      /    \        /    \
                  (24)     (25)  (26)    (27)               Level 2
                  / \      / \    / \     / \
               (16)(17) (18)(19)(20)(21)(22)(23)            Level 1
               /\  /\   /\  /\  /\  /\  /\  /\
              0 1  2 3  4 5 6 7 8 9 ...  12 13 14 15       Level 0 (beads)
```

- **16 bead nodes** (Level 0): physical beads with position, velocity, mass
- **15 interior nodes** (Levels 1-4): learned feature nodes, no physical state
- **31 total nodes**, all in one graph with one edge list

### Message passing on the tree

Every edge in the tree is a standard message-passing edge — the same `onNeighborState`-style operation. Messages flow in both directions:

- **Up-pass** (beads → root): children send messages to parents, aggregating local state into regional and then global summaries
- **Down-pass** (root → beads): parents send corrections back to children, communicating global constraints (energy conservation, inextensibility) to individual beads

The bead-to-bead chain edges (Level 0) still exist for local spring/damping forces. The tree edges are additional connections layered on top.

### Why binary tree

- **Full coverage in log2(N) hops**: for 16 beads, any bead-to-bead path through the tree is at most 8 hops (up 4, down 4), vs 15 hops along the chain
- **Global node at root**: node 30 has a path to every bead — global constraints like energy conservation can be enforced here
- **Power-of-2 bead count**: 16 beads gives a balanced binary tree with exactly 15 interior nodes
- **Same mechanism everywhere**: every edge uses the same message function, no special pooling or unpooling operations

### Relation to constraint projection

The SHAKE constraint solver iterates over all rods correcting pairs — essentially a Gauss-Seidel relaxation. This is slow because each correction disturbs neighbors. The binary tree provides a multigrid-like structure where:

- Level 0 edges handle local rod corrections
- Level 1 nodes coordinate pairs of rods
- Higher levels coordinate larger segments
- The root coordinates the entire chain

A trained GNN on this graph should learn to approximate the iterative SHAKE projection in a single forward pass through the tree.
