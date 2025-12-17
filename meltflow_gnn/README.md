# MeltFlow GNN: Neural Network Surrogate for Roe Flux

This module contains scripts for training and evaluating neural network models that learn the Roe numerical flux function for compressible Euler equations. The trained models can replace the analytical Roe solver in shock tube simulations.

## Quick Start

### 1D Sod Shock Tube (Recommended Starting Point)

```bash
# Generate training data (1M samples, fixed gamma=1.4)
python grid_sampler.py --n-samples 10 --output data/uniform_flux_data.npz

# Train the MLP (GPU)
python train_uniform_cuda.py --data data/uniform_flux_data.npz --epochs 500 \
    --output ../models/flux_model_cuda.pt --output-npz ../models/flux_model_cuda.npz

# Run simulation comparison
python simulate_with_mlp.py --model ../models/flux_model_cuda.npz
```

## Scripts Overview

### Data Generation

| Script | Description |
|--------|-------------|
| `grid_sampler.py` | Generate 1D training data via uniform grid sampling |
| `grid_sampler_multiphase.py` | Generate data with **expanded ranges** for two-fluid cases |
| `grid_sampler_2d.py` | Generate 2D training data (5-cell stencil) |
| `data_generator.py` | Generate data from MeltFlow simulation trajectories (legacy) |

### Training

| Script | Description |
|--------|-------------|
| `train_uniform_cuda.py` | **Main training script** - PyTorch GPU training |
| `train_uniform.py` | NumPy-only CPU training (slower, no dependencies) |
| `train_uniform_2d_cuda.py` | 2D model training |
| `train.py` | Original GNN training from trajectories (legacy) |

### Simulation & Comparison

| Script | Description |
|--------|-------------|
| `test_multiphase.py` | **Two-fluid simulation** with gamma-parameterized MLP |
| `simulate_with_mlp.py` | Run Sod shock tube with MLP flux (standalone) |
| `simulate_gnn.py` | Run simulation with GNN flux |
| `compare.py` | Side-by-side MeltFlow vs GNN comparison |
| `compare_gamma.py` | Test gamma-parameterized model on interpolated values |
| `compare_state.py` | State-by-state flux comparison utilities |
| `simulate_2d_comparison.py` | 2D shock tube comparison |

### Models

| Script | Description |
|--------|-------------|
| `model_simple.py` | SimpleFluxMLP - recommended for 1D (6 or 7 inputs) |
| `model.py` | Original EulerGNN with message passing (12 inputs) |
| `model_state.py` | State prediction model (experimental) |
| `graph.py` | Graph construction utilities for GNN |

---

## Detailed Workflows

### Workflow 0: Multiphase Two-Fluid Simulation (NEW)

Train and test on the two-fluid Sod shock tube (`in_1Dsod2fl`).

**Important:** The standard model (`flux_mlp_gamma.pt`) was trained on single-fluid ranges and will **not** work well on two-fluid cases because the initial conditions are outside the training range:

| Parameter | Single-Fluid Training | Two-Fluid Left State |
|-----------|----------------------|---------------------|
| Density ρ | [0.05, 1.2] kg/m³ | **5.36 kg/m³** |
| Pressure p | [5k, 120k] Pa | **303,975 Pa** |

**Solution: Retrain with expanded ranges:**

```bash
# Full pipeline: generate data, train, and test
./train_multiphase.sh
```

Or step-by-step:

```bash
# Step 1: Generate 2M samples with expanded ranges
python grid_sampler_multiphase.py \
    --n-samples 2000000 \
    --gamma-values 1.2 1.25 1.289 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.67 \
    --output data/multiphase_flux_data.npz

# Step 2: Train
python train_uniform_cuda.py \
    --data data/multiphase_flux_data.npz \
    --epochs 500 \
    --output ../models/flux_mlp_multiphase.pt \
    --output-npz ../models/flux_mlp_multiphase.npz

# Step 3: Test
python test_multiphase.py \
    --model ../models/flux_mlp_multiphase.pt \
    --config in_1Dsod2fl \
    --output multiphase_comparison.png
```

**Expanded parameter ranges for multiphase:**
- ρ: [0.05, 7.0] kg/m³
- u: [-200, 500] m/s
- p: [5k, 400k] Pa
- γ: [1.2, 1.67] (includes 1.289 and 1.4)

**Key differences from single-fluid:**
- Two fluids with different gamma: γ₁ = 1.289 (left), γ₂ = 1.4 (right)
- Finer grid: dx = 0.001 (10x finer)
- More conservative CFL: 0.3 (vs 0.9)
- Level set φ tracks interface: φ > 0 → fluid 1, φ ≤ 0 → fluid 2

---

### Workflow 1: Train a Fixed-Gamma Model (γ = 1.4)

Best for air/diatomic gases.

```bash
# Step 1: Generate 1M samples
python grid_sampler.py \
    --n-samples 10 \
    --gamma 1.4 \
    --output data/uniform_flux_data.npz

# Step 2: Train (500 epochs, ~80 seconds on RTX 4070)
python train_uniform_cuda.py \
    --data data/uniform_flux_data.npz \
    --epochs 500 \
    --hidden-dim 256 \
    --n-layers 5 \
    --activation gelu \
    --output ../models/flux_model_cuda.pt \
    --output-npz ../models/flux_model_cuda.npz

# Step 3: Test
python simulate_with_mlp.py --model ../models/flux_model_cuda.npz
```

**Expected Results:**
- Flux relative error: ~0.5%
- Density MAE: ~0.004 kg/m³
- Velocity MAE: ~2 m/s
- Pressure MAE: ~330 Pa

### Workflow 2: Train a Gamma-Parameterized Model

Single model works for γ ∈ [1.2, 1.7] (covers most gases).

```bash
# Step 1: Generate 10M samples (10 gamma values × 1M states)
python grid_sampler.py \
    --n-samples 10 \
    --gamma-values 1.2 1.256 1.311 1.367 1.422 1.478 1.533 1.589 1.644 1.7 \
    --output data/uniform_flux_data_gamma.npz

# Step 2: Train (100 epochs sufficient for 10M samples)
python train_uniform_cuda.py \
    --data data/uniform_flux_data_gamma.npz \
    --epochs 100 \
    --output ../models/flux_mlp_gamma.pt \
    --output-npz ../models/flux_mlp_gamma.npz

# Step 3: Test on interpolated gamma values
python compare_gamma.py \
    --model ../models/flux_mlp_gamma.pt \
    --output gamma_comparison.png
```

**Expected Results:**
- <0.2% relative error on unseen γ values
- Works for: He (1.67), air (1.4), CO₂ (1.3), combustion products (1.1-1.2)

### Workflow 3: 2D Shock Tube

```bash
# Step 1: Generate 2D data (5-cell stencil → 4 interface fluxes)
python grid_sampler_2d.py \
    --n-samples 1000000 \
    --output data/uniform_flux_data_2d.npz

# Step 2: Train 2D model
python train_uniform_2d_cuda.py \
    --data data/uniform_flux_data_2d.npz \
    --epochs 1000 \
    --output ../models/flux_model_2d_cuda.pt

# Step 3: Run 2D simulation
python simulate_2d_comparison.py --model ../models/flux_model_2d_cuda.pt
```

---

## Model Architecture

### SimpleFluxMLP (Recommended)

```
Input:  [ρ_L, u_L, p_L, ρ_R, u_R, p_R]        (6 features)
   or:  [ρ_L, u_L, p_L, ρ_R, u_R, p_R, γ]     (7 features with gamma)

Hidden: 256 × 5 layers, GELU activation

Output: [F_ρ, F_ρu, F_E]                       (3 flux components)

Parameters: ~266K
```

### 2D Model

```
Input:  [W, C, E, S, N] × [ρ, u, v, p]         (20 features)
Output: [F_w, F_e, G_s, G_n] × [ρ, ρu, ρv, E]  (16 flux components)
Parameters: ~273K
```

---

## Trained Models

Located in `../models/`:

| File | Input Dim | Description |
|------|-----------|-------------|
| `flux_model_cuda.pt` | 6 | 1D, γ=1.4 fixed (PyTorch) |
| `flux_model_cuda.npz` | 6 | 1D, γ=1.4 fixed (NumPy) |
| `flux_mlp_gamma.pt` | 7 | 1D, γ as input (PyTorch) |
| `flux_mlp_gamma.npz` | 7 | 1D, γ as input (NumPy) |
| `flux_model_2d_cuda.pt` | 20 | 2D (PyTorch) |

---

## Parameter Ranges

Training data covers these ranges (based on Sod shock tube):

| Variable | Range | Units |
|----------|-------|-------|
| Density ρ | [0.05, 1.2] | kg/m³ |
| Velocity u | [-150, 350] | m/s |
| Pressure p | [5,000, 120,000] | Pa |
| Gamma γ | [1.2, 1.7] | - |

---

## Command-Line Reference

### grid_sampler.py

```
--n-samples N       Samples per dimension (total = N^6)  [default: 10]
--gamma G           Fixed gamma value                    [default: 1.4]
--gamma-values G... Multiple gamma values (7-dim input)
--expansion F       Range expansion factor               [default: 1.0]
--output PATH       Output .npz file path
```

### train_uniform_cuda.py

```
--data PATH         Input data file
--epochs N          Training epochs                      [default: 500]
--batch-size N      Batch size                           [default: 8192]
--lr RATE           Learning rate                        [default: 1e-3]
--hidden-dim N      Hidden layer dimension               [default: 256]
--n-layers N        Number of hidden layers              [default: 5]
--activation TYPE   relu, gelu, or silu                  [default: relu]
--scheduler TYPE    cosine, plateau, or none             [default: cosine]
--output PATH       PyTorch model output (.pt)
--output-npz PATH   NumPy model output (.npz)
--device DEVICE     cuda or cpu                          [default: cuda]
```

### compare_gamma.py

```
--model PATH        Trained model file (.pt)
--nx N              Grid resolution                      [default: 100]
--t-final T         Simulation end time                  [default: 0.00075]
--cfl C             CFL number                           [default: 0.5]
--output PATH       Output plot file
```

---

## Key Findings

1. **Uniform grid sampling** is essential - trajectory-based training has severe data imbalance
2. **Conservation constraints hurt** - antisymmetric F_ij = -F_ji caused training to explode
3. **Simple MLP works** - no need for GNN message passing in 1D (path graph)
4. **Model size sweet spot** - 256×5 (266K params) is optimal; larger models overfit
5. **Including γ as input improves generalization** - even for fixed-γ applications

---

## References

- Full analysis: `../reports/gnn_analysis/gnn_report.pdf`
- Specification: `../docs/uniform_sampling_experiment.md`
- Simplified model comparison: `../docs/simplified_gnn.md`
