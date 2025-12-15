# 2D Electrostatic Potential U-Net

## Overview

This project implements a U-Net neural network that learns to compute electrostatic potential from point charge distributions in 2D. This serves as a foundational step toward more complex particle-field simulations, eventually including magnetic fields derived from moving charged particles.

## Motivation

We are exploring the intersection of graph neural networks and particle simulations. The eventual goal is a system where:

1. Charged particles scatter their state onto a spatial hierarchy (octree in 3D)
2. A U-Net processes this representation to derive electromagnetic fields
3. Particles gather field values to compute forces and update trajectories

Before tackling the full 3D magnetic field problem, we start with the simplest nontrivial case: learning the 2D electrostatic potential from static point charges.

## Physics Background

The electrostatic potential in 2D from point charges is:

$$\phi(\mathbf{r}) = \sum_i \frac{q_i}{2\pi\epsilon_0} \ln\left(\frac{1}{|\mathbf{r} - \mathbf{r}_i|}\right)$$

For simplicity, we absorb constants and use:

$$\phi(\mathbf{r}) = -\sum_i q_i \ln(|\mathbf{r} - \mathbf{r}_i|)$$

This potential satisfies Poisson's equation in 2D:

$$\nabla^2 \phi = -\rho / \epsilon_0$$

The U-Net learns to approximate the solution operator: given a charge density field ρ, output the potential field φ.

## Architecture

### Input (Scatter Step)

- Domain: 2D square grid, e.g., 128×128 or 256×256
- Particles have position (x, y) and charge q
- Each particle deposits its charge onto the grid using bilinear interpolation (or nearest-cell assignment for simplicity)
- Input tensor: single-channel image representing charge density ρ(x, y)

### U-Net

Standard encoder-decoder architecture:

- **Encoder**: Sequence of conv → ReLU → conv → ReLU → downsample (max pool or strided conv), doubling channels at each level
- **Bottleneck**: Convolutional processing at coarsest resolution
- **Decoder**: Sequence of upsample → concat skip connection → conv → ReLU → conv → ReLU, halving channels at each level
- **Output**: Single-channel image representing potential φ(x, y)

Suggested depth: 4 levels (e.g., 128 → 64 → 32 → 16 → 8 at bottleneck for 128×128 input)

### Output (Gather Step)

For this initial version, we simply output the full potential field. Later, particles would query this field via bilinear interpolation to get potential (or its gradient for force).

## Data Generation

Training data is generated analytically:

1. Sample N particles uniformly in the domain (N ~ 5-50, varied per sample)
2. Assign random charges q ∈ [-1, 1] to each particle
3. Compute charge density grid via scatter operation
4. Compute ground truth potential at each grid point using the analytical formula
5. Apply regularization: clamp minimum distance to avoid singularities at particle centers (e.g., |r - r_i| → max(|r - r_i|, ε) where ε ~ 1-2 grid cells)

Generate ~10,000 training samples, ~1,000 validation samples.

## Training

- **Loss**: MSE between predicted and ground truth potential
- **Optimizer**: Adam, learning rate ~1e-4
- **Batch size**: 16-32
- **Epochs**: Until validation loss plateaus (~50-200 epochs depending on complexity)

Optional: normalize potential fields to zero mean or bounded range to aid training stability.

## Evaluation Metrics

1. **L2 error**: Mean squared error on held-out test configurations
2. **Superposition test**: Generate potential for charge set A, charge set B, and A∪B; verify φ(A∪B) ≈ φ(A) + φ(B)
3. **Visual inspection**: Plot predicted vs ground truth for qualitative assessment
4. **Generalization**: Test on configurations with more/fewer particles than training distribution

## Implementation Requirements

- **Framework**: PyTorch
- **Visualization**: Matplotlib for plotting charge distributions and potential fields
- **Structure**:
  - `dataset.py`: Data generation and PyTorch Dataset class
  - `model.py`: U-Net architecture
  - `train.py`: Training loop
  - `evaluate.py`: Testing and visualization
  - `utils.py`: Scatter/gather operations, potential computation

## Future Extensions

Once this baseline works:

1. Move to 3D with regular grid
2. Replace regular grid with octree for adaptive resolution
3. Add vector outputs (electric field E = -∇φ)
4. Introduce time-varying charges (currents) and learn magnetic field B
5. Couple back to particle dynamics for full simulation loop

## Success Criteria

The model successfully learns the mapping ρ → φ when:

- Test MSE is significantly lower than a naive baseline (e.g., predicting mean potential)
- Superposition property holds approximately
- Visual predictions show correct qualitative structure (potential wells at positive charges, peaks at negative charges, smooth interpolation between)
