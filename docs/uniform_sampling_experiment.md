# Uniform Grid Sampling Experiment for GNN Flux Learning

**Date:** December 11, 2025

## Problem Statement

The original GNN flux learner had training data imbalance issues (see `reports/gnn_analysis/gnn_report.pdf`):
- Only 1.8% of training samples had negative velocity
- Only 1.7% had low pressure (< 10,000 Pa)
- Largest errors occurred at the right boundary, not at the shock

## Solution: Uniform Grid Sampling

Instead of generating training data from simulation trajectories (which naturally underrepresent extreme states), we sample the state space uniformly.

### Approach

1. **Define parameter ranges** based on the Sod shock tube problem:
   - ρ (density): [0.05, 1.2] kg/m³
   - u (velocity): [-150, 350] m/s
   - p (pressure): [5000, 120000] Pa

2. **Create uniform grid** over the 6D input space:
   - Inputs: [ρ_L, u_L, p_L, ρ_R, u_R, p_R] (left and right states)
   - With 10 samples per dimension: 10^6 = 1,000,000 total samples

3. **Compute Roe flux** for each input combination:
   - Outputs: [flux_ρ, flux_ρu, flux_E] (mass, momentum, energy fluxes)

4. **Train MLP** on the uniformly sampled data

## Implementation

### Files Created

1. **`meltflow_gnn/grid_sampler.py`** - Generates uniform training data
   ```bash
   python3 meltflow_gnn/grid_sampler.py --n-samples 10 --output data/uniform_flux_data.npz
   ```

2. **`meltflow_gnn/train_uniform.py`** - Pure NumPy MLP training (no PyTorch required)
   ```bash
   python3 meltflow_gnn/train_uniform.py --data data/uniform_flux_data.npz --epochs 50 --batch-size 16384
   ```

3. **`meltflow_gnn/simulate_with_mlp.py`** - Run Euler simulation with trained MLP
   ```bash
   python3 meltflow_gnn/simulate_with_mlp.py --model flux_model_uniform.npz --compare
   ```

### Training Results (NumPy, CPU)

- **Architecture:** 6 → 256 → 256 → 256 → 256 → 256 → 3 (265,731 parameters)
- **Training samples:** 900,000 (90% of 1M)
- **Validation samples:** 100,000 (10%)
- **Epochs:** 50
- **Final validation loss:** 0.000968 (normalized MSE)
- **Training time:** ~17 minutes on MacBook Pro (CPU only)

**Per-variable relative errors (MAE/std):**
| Variable | MAE | Std | Relative Error |
|----------|-----|-----|----------------|
| flux_ρ | 2.83 | 112 | 2.53% |
| flux_ρu | 1,101 | 48,900 | 2.25% |
| flux_E | 1.19M | 48.2M | 2.48% |

### Simulation Results

The trained MLP was tested on the Sod shock tube problem:
- **Grid:** 100 cells
- **Final time:** 0.00075 s
- **102 timesteps** completed successfully (no blowup!)

**Comparison with analytical Roe flux:**
| Variable | MAE |
|----------|-----|
| Density | 0.020 kg/m³ |
| Velocity | 15.7 m/s |
| Pressure | 1,833 Pa |

**Observations:**
- MLP captures expansion fan, shock, and contact discontinuity correctly
- Some oscillations near discontinuities (MLP slightly less dissipative)
- Simulation is stable

## Next Steps (for CUDA machine)

### Phase 1: Baseline Training with PyTorch

1. **Set up PyTorch with CUDA**
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Train on full 1M dataset (no validation split)**
   - Use all 1M points for training
   - Train for 200-500 epochs
   - Save model

### Phase 2: Adaptive Mesh Refinement

The key idea: **iteratively refine sampling in high-error regions**

1. **Evaluate error on all training points**
   - Run trained model on all 1M inputs
   - Compute |predicted - actual| for each sample
   - Identify high-error regions in the 6D state space

2. **Generate refined samples in high-error regions**
   - Find the (ρ_L, u_L, p_L, ρ_R, u_R, p_R) ranges where error > threshold
   - Generate dense samples in those regions (e.g., 5x density)
   - This targets the nonlinear regions (strong shocks, sonic points)

3. **Combine and retrain**
   - Merge original 1M samples with new refined samples
   - Retrain model on combined dataset
   - Repeat until error converges

**Implementation plan:**
```python
# Pseudocode for adaptive refinement
for iteration in range(max_iterations):
    # Train on current dataset
    model = train(X_all, Y_all, epochs=200)

    # Evaluate error on all points
    Y_pred = model.predict(X_all)
    errors = |Y_pred - Y_all|

    # Find high-error samples (e.g., top 10%)
    threshold = np.percentile(errors, 90)
    high_error_idx = errors > threshold
    high_error_X = X_all[high_error_idx]

    # Define bounding boxes around high-error regions
    # Generate new dense samples in those regions
    X_refined, Y_refined = generate_refined_samples(high_error_X)

    # Combine datasets
    X_all = np.concatenate([X_all, X_refined])
    Y_all = np.concatenate([Y_all, Y_refined])

    # Check convergence
    if max_error < target:
        break
```

### Phase 3: Validation and Testing

1. **Run simulation comparison**
   - Compare MLP vs analytical Roe on Sod problem
   - Check for oscillation reduction

2. **Test on different initial conditions**
   - Different shock strengths
   - Different domain sizes

### Phase 4: Optional Improvements

- Try deeper networks or residual connections
- Try GELU activation instead of ReLU
- Experiment with larger initial grid (15^6 ≈ 11M samples)

## File Locations

- **Training data:** `data/uniform_flux_data.npz` (1M samples)
- **Trained model:** `flux_model_uniform.npz`
- **Simulation output:** `mlp_simulation.png`
- **Grid sampler:** `meltflow_gnn/grid_sampler.py`
- **Training script:** `meltflow_gnn/train_uniform.py`
- **Simulation script:** `meltflow_gnn/simulate_with_mlp.py`

## Key Insights

### 1. Data Imbalance was the Bottleneck

The uniform sampling approach addresses the core issue identified in the GNN report: **the bottleneck was data imbalance, not model capacity**. By ensuring every region of state space gets equal representation, the MLP learns a more accurate flux function across all conditions, including the previously problematic extreme states.

### 2. Path Graphs Don't Need Message Passing

Since the 1D Euler problem is a **path graph** (every interior node has exactly one left and one right neighbor), we can replace GNN message passing with simple **concatenation**:

**Why this works:**
- Flux at interface i+1/2 only depends on cells i (left) and i+1 (right)
- No need to aggregate over variable numbers of neighbors
- Left and right are always distinct and ordered

**GNN approach (complex):**
```
node features → message passing → scatter/gather → aggregation → flux
```

**Concatenation approach (simple):**
```
[ρ_L, u_L, p_L, ρ_R, u_R, p_R] → MLP → [flux_ρ, flux_ρu, flux_E]
```

**Advantages:**
| Aspect | GNN | Concatenation MLP |
|--------|-----|-------------------|
| Complexity | High (PyTorch Geometric) | Low (NumPy/PyTorch) |
| Speed | Slower (scatter ops) | Faster (dense matmul) |
| Directionality | Must encode in edge attr | Natural (left, right order) |
| Dependencies | torch_geometric, torch_scatter | NumPy only |

**This is what we implemented:** The `train_uniform.py` MLP takes 6 inputs (left + right states concatenated) and outputs 3 fluxes. No graph structure needed.

### 3. Future Extension: State Update MLP

Could extend this further to predict the full state update:
```
Input:  [U_{i-1}, U_i, U_{i+1}, dt, dx]  (11 features)
Output: [U_i^{n+1}]                       (3 features: ρ, ρu, E)
```

This would learn the entire finite volume update in one step, bypassing flux computation entirely. The concatenation approach makes this straightforward since the stencil is fixed.
