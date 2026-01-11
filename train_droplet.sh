#!/bin/bash
# Train and test MLP for air/water droplet (high density ratio) simulations

set -e

echo "=========================================="
echo "Droplet MLP Training Pipeline"
echo "=========================================="
echo ""
echo "This trains a model for the in_1Dcdrop case:"
echo "  - Air/water with 1000:1 density ratio"
echo "  - Log-uniform density sampling"
echo "  - Deeper/wider network (6x384)"
echo ""

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo ""
    echo "Virtual environment not found. Creating venv..."
    echo "=========================================="

    # Create venv
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create virtual environment"
        echo "Make sure Python 3.10-3.12 is installed"
        exit 1
    fi

    # Activate venv
    source venv/bin/activate

    # Upgrade pip
    echo ""
    echo "Upgrading pip..."
    python -m pip install --upgrade pip

    # Install numpy <2 first (PyTorch compatibility)
    echo ""
    echo "Installing numpy <2 (required for PyTorch compatibility)..."
    pip install "numpy>=1.24,<2"

    # Install other dependencies
    echo ""
    echo "Installing scipy, matplotlib, and numba..."
    pip install scipy matplotlib numba

    # Detect platform and install appropriate PyTorch
    echo ""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS - check for Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            echo "Detected Apple Silicon Mac - installing PyTorch with MPS support..."
            pip install torch torchvision torchaudio
            DEVICE="mps"
        else
            echo "Detected Intel Mac - installing PyTorch (CPU only)..."
            pip install torch torchvision torchaudio
            DEVICE="cpu"
        fi
    else
        # Linux - check for NVIDIA GPU
        if command -v nvidia-smi &> /dev/null; then
            echo "Detected NVIDIA GPU - installing PyTorch with CUDA..."
            pip install torch --index-url https://download.pytorch.org/whl/cu124
            DEVICE="cuda"
        else
            echo "No NVIDIA GPU detected - installing PyTorch (CPU only)..."
            pip install torch --index-url https://download.pytorch.org/whl/cpu
            DEVICE="cpu"
        fi
    fi

    # Verify installation
    echo ""
    echo "=========================================="
    echo "Verifying installation..."
    echo "=========================================="
    python -c "import numpy; print(f'numpy: {numpy.__version__}')"
    python -c "import torch; print(f'torch: {torch.__version__}')"
    python -c "import scipy; print(f'scipy: {scipy.__version__}')"
    python -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')"

    echo ""
    echo "=========================================="
    echo "Virtual environment setup complete!"
    echo "=========================================="
else
    # Activate existing venv
    source venv/bin/activate
fi

# Detect device if not already set
if [ -z "$DEVICE" ]; then
    # Check what's available
    DEVICE=$(python -c "
import torch
if torch.cuda.is_available():
    print('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('mps')
else:
    print('cpu')
")
fi

echo ""
echo "Using device: $DEVICE"
echo ""

# Step 1: Generate training data
echo "Step 1: Generating training data (2M samples)..."
echo "  Parameter ranges (log-uniform density):"
echo "    rho: [0.9, 1000] kg/m³"
echo "    u:   [0, 100] m/s"
echo "    p:   [65k, 150k] Pa"
echo ""

python meltflow_gnn/grid_sampler_droplet.py \
    --n-samples 2000000 \
    --output data/droplet_flux_data.npz

# Step 2: Train the model
echo ""
echo "Step 2: Training MLP (500 epochs, 6x384 architecture)..."
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
    --device $DEVICE

# Step 3: Test on droplet case
echo ""
echo "Step 3: Testing on in_1Dcdrop..."
python meltflow_gnn/test_multiphase.py \
    --model models/flux_mlp_droplet.pt \
    --config in_1Dcdrop \
    --output droplet_comparison.png

echo ""
echo "=========================================="
echo "Done! Check droplet_comparison.png"
echo "=========================================="
