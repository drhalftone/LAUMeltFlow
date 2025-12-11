# GNN Flux Debugging Guide

Since DG(p=0) is identical to FVM and DG(p=1) is very close, the numerical framework is solid. The issue is likely in the **GNN flux approximation**.

## Potential GNN Issues

### 1. Training Data Problems

| Issue | Symptom |
|-------|---------|
| Not enough shock cases | GNN smears discontinuities |
| Only trained on one IC | Fails to generalize |
| Missing edge cases | Blows up or produces NaN |
| Insufficient data near discontinuities | Poor shock capturing |

**Recommendation**: Train on diverse scenarios:
- Sod shock tube with varying initial conditions
- Different pressure/density ratios
- Various diaphragm positions
- Random perturbations

### 2. Conservation Violation

The Roe flux is designed to conserve mass, momentum, and energy. A GNN has no such guarantee.

```
Roe:  F* = ½(F_L + F_R) - ½|Â|(W_R - W_L)  ← mathematically conservative

GNN:  F* = MLP(W_L, W_R)  ← no conservation guarantee
```

**Test**: Check if mass/momentum/energy are conserved over time in your GNN results.

```python
# Total mass should be constant
total_mass_initial = np.sum(W[0, :]) * dx

# After simulation
total_mass_final = np.sum(W[0, :]) * dx

mass_error = abs(total_mass_final - total_mass_initial) / total_mass_initial
print(f"Mass conservation error: {mass_error:.6e}")
```

### 3. Input/Output Scaling

The Roe flux deals with very different scales:

| Variable | Typical Range | Scale |
|----------|---------------|-------|
| ρ (density) | 0.1 - 1.0 | O(1) |
| u (velocity) | 0 - 350 m/s | O(100) |
| p (pressure) | 10,000 - 100,000 Pa | O(10⁵) |
| E (energy) | 10,000 - 250,000 | O(10⁵) |

If your GNN inputs/outputs aren't normalized properly, it may struggle.

**Recommendation**: Normalize inputs and outputs:
```python
# Example normalization
W_normalized = (W - W_mean) / W_std
F_normalized = (F - F_mean) / F_std
```

### 4. Flux Direction (Upwinding)

The Roe flux is **upwind-biased** - it respects wave propagation direction via eigenvalue decomposition. A vanilla MLP may not learn this.

**Test**: Does your GNN output change appropriately when you swap W_L and W_R?

```python
F_LR = gnn_flux(W_L, W_R)
F_RL = gnn_flux(W_R, W_L)

# These should NOT be equal
# They should NOT simply be negatives of each other
# The relationship depends on wave speeds and directions
```

### 5. Architecture Too Simple

The Roe flux involves:
1. Computing Roe-averaged quantities (ρ, u, h, a)
2. Eigenvalue decomposition (wave speeds)
3. Wave strength calculation
4. Flux assembly from eigenvectors

A shallow MLP may not have capacity to approximate this nonlinear function.

**Recommendation**: Try:
- Deeper networks (4+ layers)
- Wider layers (128+ neurons)
- Residual connections
- Physics-informed architecture

### 6. Boundary Condition Handling

The FVM/DG solvers have specific boundary condition logic. Make sure your GNN handles boundaries consistently.

---

## Debugging Steps

### Step 1: Compare Flux Values Directly

```python
import numpy as np
from meltflow.functions.flux import roe_flux1D

def compare_fluxes(W_L, W_R, gnn_model, gam=1.4):
    """Compare Roe flux vs GNN flux."""
    n_dim = 1

    # Roe flux (ground truth)
    F_roe = roe_flux1D(n_dim, gam, W_L, W_R)

    # GNN flux
    F_gnn = gnn_model.predict(W_L, W_R)  # Adjust to your API

    # Error metrics
    abs_error = np.abs(F_roe - F_gnn)
    rel_error = abs_error / (np.abs(F_roe) + 1e-10)

    print(f"Roe flux:  {F_roe}")
    print(f"GNN flux:  {F_gnn}")
    print(f"Abs error: {abs_error}")
    print(f"Rel error: {rel_error}")

    return F_roe, F_gnn, abs_error
```

### Step 2: Check Conservation Over Time

```python
def check_conservation(W_history, dx):
    """Check mass, momentum, energy conservation over time."""
    n_steps = len(W_history)

    mass = np.array([np.sum(W[0, :]) * dx for W in W_history])
    momentum = np.array([np.sum(W[1, :]) * dx for W in W_history])
    energy = np.array([np.sum(W[2, :]) * dx for W in W_history])

    print(f"Mass drift:     {(mass[-1] - mass[0]) / mass[0]:.6e}")
    print(f"Momentum drift: {(momentum[-1] - momentum[0]) / (abs(momentum[0]) + 1e-10):.6e}")
    print(f"Energy drift:   {(energy[-1] - energy[0]) / energy[0]:.6e}")
```

### Step 3: Test Symmetry Properties

```python
def test_flux_symmetry(gnn_model, W_L, W_R):
    """Test if GNN respects physical symmetry."""
    F_LR = gnn_model.predict(W_L, W_R)
    F_RL = gnn_model.predict(W_R, W_L)

    # For Roe flux: F(W_L, W_R) ≠ F(W_R, W_L) in general
    # But F(W, W) = F_physical(W) (flux at constant state)

    print(f"F(L,R): {F_LR}")
    print(f"F(R,L): {F_RL}")
    print(f"Equal?: {np.allclose(F_LR, F_RL)}")  # Should be False
```

### Step 4: Visualize Flux Error Distribution

```python
import matplotlib.pyplot as plt

def plot_flux_errors(x, F_roe_all, F_gnn_all):
    """Plot flux errors across the domain."""
    errors = np.abs(F_roe_all - F_gnn_all)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    labels = ['Mass flux', 'Momentum flux', 'Energy flux']
    for i, (ax, label) in enumerate(zip(axes, labels)):
        ax.plot(x[:-1], errors[i, :], 'r-')
        ax.set_xlabel('x')
        ax.set_ylabel(f'|F_roe - F_gnn|')
        ax.set_title(label)
        ax.grid(True)

    plt.tight_layout()
    plt.savefig('flux_errors.png')
    plt.show()
```

---

## Common Fixes

### Fix 1: Add Conservation Loss Term

```python
def conservation_loss(F_left, F_right, W_new, W_old, dt, dx):
    """Penalize conservation violations."""
    # dW/dt + dF/dx = 0  =>  W_new = W_old - dt/dx * (F_right - F_left)
    W_expected = W_old - dt/dx * (F_right - F_left)
    return torch.mean((W_new - W_expected)**2)
```

### Fix 2: Physics-Informed Architecture

```python
class PhysicsInformedFlux(nn.Module):
    def __init__(self):
        super().__init__()
        # Separate networks for different components
        self.avg_net = MLP(6, 64, 3)   # Learns averaged quantities
        self.wave_net = MLP(6, 64, 3)  # Learns wave contributions

    def forward(self, W_L, W_R):
        # Mimic Roe structure: F = F_avg - wave_correction
        W_cat = torch.cat([W_L, W_R], dim=-1)
        F_avg = 0.5 * (self.physical_flux(W_L) + self.physical_flux(W_R))
        wave_correction = self.wave_net(W_cat)
        return F_avg - wave_correction
```

### Fix 3: Data Augmentation

```python
def augment_training_data(W_L, W_R, F):
    """Augment with Galilean invariance."""
    # Add constant velocity (Galilean boost)
    for v_boost in [-50, 0, 50, 100]:
        W_L_boosted = W_L.copy()
        W_R_boosted = W_R.copy()
        W_L_boosted[1] += W_L[0] * v_boost  # rho*u -> rho*(u + v)
        W_R_boosted[1] += W_R[0] * v_boost
        # ... compute corresponding flux
```

---

## Diagnostic Checklist

- [ ] Flux values match Roe at smooth regions
- [ ] Flux values reasonable near shocks (not NaN/Inf)
- [ ] Mass is conserved to < 1% over simulation
- [ ] Momentum is conserved to < 1% over simulation
- [ ] Energy is conserved to < 1% over simulation
- [ ] GNN respects input ordering (L/R swap changes output)
- [ ] GNN handles boundary states correctly
- [ ] Training data includes shock cases
- [ ] Inputs/outputs properly normalized

---

## References

- Sanchez-Gonzalez, A., et al. (2020). Learning to Simulate Complex Physics with Graph Networks. *ICML*.
- Bar-Sinai, Y., et al. (2019). Learning data-driven discretizations for partial differential equations. *PNAS*.
- Kochkov, D., et al. (2021). Machine learning–accelerated computational fluid dynamics. *PNAS*.
