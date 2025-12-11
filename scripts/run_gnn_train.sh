#!/bin/bash
#
# MeltFlow GNN Training Script
# Creates virtual environment, installs dependencies with CUDA support, runs training
#
# Usage:
#   ./run_gnn_train.sh           - Run GNN flux training
#   ./run_gnn_train.sh --help    - Show help
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$PROJECT_DIR/.venv"

echo "============================================"
echo "MeltFlow GNN Training Setup"
echo "============================================"

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "Error: Python 3 is required but not found"
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Create virtual environment if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# Check if dependencies are installed
install_deps=false
python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null || install_deps=true
python -c "import torch_geometric" 2>/dev/null || install_deps=true
python -c "import matplotlib" 2>/dev/null || install_deps=true

if [ "$install_deps" = true ]; then
    echo ""
    echo "Installing dependencies..."
    echo ""

    echo "Upgrading pip..."
    pip install --upgrade pip --quiet

    echo "Installing PyTorch with CUDA 12.4 support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

    echo "Installing PyTorch Geometric..."
    pip install torch_geometric

    echo "Installing other dependencies..."
    pip install numpy matplotlib scipy

    echo ""
    echo "Dependencies installed successfully"
fi

echo ""
echo "Verifying CUDA setup..."
python -c "import torch; print(f'   PyTorch: {torch.__version__}'); print(f'   CUDA available: {torch.cuda.is_available()}'); print(f'   GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

if [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo ""
    echo "Usage:"
    echo "  ./run_gnn_train.sh           Run GNN flux model training"
    echo "  ./run_gnn_train.sh --help    Show this help"
    echo ""
    echo "This script will:"
    echo "  1. Create a virtual environment if needed"
    echo "  2. Install PyTorch with CUDA support"
    echo "  3. Install PyTorch Geometric and other dependencies"
    echo "  4. Run the GNN training script"
    echo ""
    echo "Output:"
    echo "  - Training progress displayed in console"
    echo "  - Model saved to flux_model.pt"
    echo "  - Training plots displayed after completion"
    exit 0
fi

# Run training
echo ""
echo "============================================"
echo "Starting GNN Training"
echo "============================================"
echo ""
cd "$PROJECT_DIR"
python -m meltflow_gnn.train "$@"

echo ""
echo "Done!"
