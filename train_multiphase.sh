#!/bin/bash
# Train and test MLP for multiphase (two-fluid) simulations

set -e

echo "=========================================="
echo "Multiphase MLP Training Pipeline"
echo "=========================================="

# Activate venv
source venv/bin/activate

# Step 1: Generate training data with expanded ranges
echo ""
echo "Step 1: Generating training data (2M samples)..."
echo "  Expanded ranges for two-fluid case:"
echo "    rho: [0.05, 7.0] kg/m³"
echo "    u:   [-200, 500] m/s"
echo "    p:   [5k, 400k] Pa"
echo ""

python meltflow_gnn/grid_sampler_multiphase.py \
    --n-samples 2000000 \
    --gamma-values 1.2 1.25 1.289 1.3 1.35 1.4 1.45 1.5 1.55 1.6 1.67 \
    --output data/multiphase_flux_data.npz

# Step 2: Train the model
echo ""
echo "Step 2: Training MLP (500 epochs)..."
python meltflow_gnn/train_uniform_cuda.py \
    --data data/multiphase_flux_data.npz \
    --epochs 500 \
    --hidden-dim 256 \
    --n-layers 5 \
    --activation gelu \
    --batch-size 16384 \
    --lr 1e-3 \
    --output models/flux_mlp_multiphase.pt \
    --output-npz models/flux_mlp_multiphase.npz \
    --device cpu

# Step 3: Test on two-fluid case
echo ""
echo "Step 3: Testing on in_1Dsod2fl..."
python meltflow_gnn/test_multiphase.py \
    --model models/flux_mlp_multiphase.pt \
    --config in_1Dsod2fl \
    --output multiphase_comparison_retrained.png

echo ""
echo "=========================================="
echo "Done! Check multiphase_comparison_retrained.png"
echo "=========================================="
