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
| `grid_sampler_droplet.py` | Generate data for air/water droplet (1000:1 density ratio) |
| `grid_sampler_multiphase.py` | Generate data with **expanded ranges** for two-fluid cases |
| `grid_sampler_2d.py` | Generate 2D training data (5-cell stencil) |

### Training

| Script | Description |
|--------|-------------|
| `train_uniform_cuda.py` | **Main training script** - PyTorch GPU training |
| `train_droplet.py` | Training script for droplet case (6x384 architecture) |
| `train_uniform.py` | NumPy-only CPU training (slower, no dependencies) |
| `train_uniform_2d_cuda.py` | 2D model training |

---

## Detailed Workflows

### Workflow: Air/Water Droplet (in_1Dcdrop) - READY FOR GPU TRAINING

**Status:** Training data generated, ready for GPU training on remote system.

This workflow trains a model for the challenging air/water droplet case with a 1000:1 density ratio.

#### Physical Setup
- Liquid water droplet (rho=1000 kg/m^3) moving at 100 m/s
- Surrounded by stationary air (rho=1.226 kg/m^3)
- Droplet centered at x=0.5, radius=0.1

#### Parameter Ranges (from simulation analysis)
| Parameter | Min | Max | Notes |
|-----------|-----|-----|-------|
| Density rho | 0.9 kg/m^3 | 1000 kg/m^3 | **1000:1 ratio** |
| Velocity u | 0 m/s | 100 m/s | |
| Pressure p | 65,000 Pa | 150,000 Pa | ~1 atm |

#### Step 1: Data Generation (COMPLETED - CPU)

Training data has been pre-generated:
- **File:** `data/droplet_flux_data.npz` (138 MB)
- **Samples:** 2,000,000
- **Sampling:** Log-uniform for density (better coverage of 1000:1 ratio)

To regenerate (if needed):
```bash
python meltflow_gnn/grid_sampler_droplet.py \
    --n-samples 2000000 \
    --output data/droplet_flux_data.npz
```

#### Step 2: Train on GPU System

Copy the repository to your GPU system, then run:

```bash
# Using the batch script (Windows)
train_droplet.bat

# Or manually (Linux/Windows):
python meltflow_gnn/train_droplet.py \
    --data data/droplet_flux_data.npz \
    --epochs 500 \
    --hidden-dim 384 \
    --n-layers 6 \
    --activation gelu \
    --batch-size 8192 \
    --lr 1e-3 \
    --output models/flux_mlp_droplet.pt \
    --output-npz models/flux_mlp_droplet.npz \
    --device cuda
```

**Architecture:** 6x384 (larger than standard 5x256 due to high density ratio)

**Expected training time:** ~2-3 minutes on RTX 4070/A100

#### Step 3: Test the Model (CPU or GPU)

```bash
python meltflow_gnn/test_multiphase.py \
    --model models/flux_mlp_droplet.pt \
    --config in_1Dcdrop \
    --output droplet_comparison.png
```

#### Key Files
| File | Description |
|------|-------------|
| `data/droplet_flux_data.npz` | Pre-generated training data (138 MB) |
| `meltflow_gnn/grid_sampler_droplet.py` | Data generation script |
| `meltflow_gnn/train_droplet.py` | Training script |
| `train_droplet.bat` | Full pipeline script (Windows) |
| `train_droplet.sh` | Full pipeline script (Linux) |

---

### Workflow 0: Multiphase Two-Fluid Simulation

Train and test on the two-fluid Sod shock tube (`in_1Dsod2fl`).

```bash
# Full pipeline: generate data, train, and test
./train_multiphase.sh
```

---

### Workflow 1: Train a Fixed-Gamma Model (gamma = 1.4)

Best for air/diatomic gases.

```bash
python grid_sampler.py --n-samples 10 --gamma 1.4 --output data/uniform_flux_data.npz
python train_uniform_cuda.py --data data/uniform_flux_data.npz --epochs 500 \
    --output ../models/flux_model_cuda.pt
python simulate_with_mlp.py --model ../models/flux_model_cuda.npz
```

---

## Model Architecture

### SimpleFluxMLP (Standard)

```
Input:  [rho_L, u_L, p_L, rho_R, u_R, p_R]  (6 features)
Hidden: 256 x 5 layers, GELU activation
Output: [F_rho, F_rhou, F_E]                (3 flux components)
Parameters: ~266K
```

### Droplet Model (6x384)

```
Input:  [rho_L, u_L, p_L, rho_R, u_R, p_R]  (6 features)
Hidden: 384 x 6 layers, GELU activation
Output: [F_rho, F_rhou, F_E]                (3 flux components)
Parameters: ~600K (larger for 1000:1 density ratio)
```

---

## Trained Models

Located in `../models/`:

| File | Input Dim | Description |
|------|-----------|-------------|
| `flux_model_cuda.pt` | 6 | 1D, gamma=1.4 fixed |
| `flux_mlp_gamma.pt` | 7 | 1D, gamma as input |
| `flux_mlp_droplet.pt` | 6 | 1D, droplet case 6x384 |
| `flux_model_2d_cuda.pt` | 20 | 2D |

---

## Parameter Ranges

For droplet case (in_1Dcdrop):

| Variable | Range | Units |
|----------|-------|-------|
| Density rho | [0.1, 1200] | kg/m^3 |
| Velocity u | [-20, 120] | m/s |
| Pressure p | [48,000, 167,000] | Pa |
| Gamma | 1.4 | - |

---

## Key Findings

1. **Uniform grid sampling** is essential - trajectory-based training has severe data imbalance
2. **Simple MLP works** - no need for GNN message passing in 1D
3. **Log-uniform sampling** for density helps with high density ratios (droplet case)
4. **Model size sweet spot** - 256x5 for standard, 384x6 for high density ratio
