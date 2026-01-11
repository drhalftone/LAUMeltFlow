@echo off
REM Train and test MLP for air/water droplet (high density ratio) simulations

setlocal EnableDelayedExpansion

echo ==========================================
echo Droplet MLP Training Pipeline
echo ==========================================
echo.
echo This trains a model for the in_1Dcdrop case:
echo   - Air/water with 1000:1 density ratio
echo   - Log-uniform density sampling
echo   - Deeper/wider network (6x384)
echo.

REM Check if venv exists, create if not
if not exist venv\Scripts\activate.bat (
    echo.
    echo Virtual environment not found. Creating venv...
    echo ==========================================

    REM Create venv
    python -m venv venv
    if %ERRORLEVEL% neq 0 (
        echo Error: Failed to create virtual environment
        echo Make sure Python 3.10-3.12 is installed and on PATH
        exit /b %ERRORLEVEL%
    )

    REM Activate venv
    call venv\Scripts\activate

    REM Upgrade pip
    echo.
    echo Upgrading pip...
    python -m pip install --upgrade pip

    REM Install numpy ^<2 first (PyTorch compatibility)
    echo.
    echo Installing numpy ^<2 ^(required for PyTorch compatibility^)...
    pip install "numpy>=1.24,<2"

    REM Install other dependencies
    echo.
    echo Installing scipy, matplotlib, and numba...
    pip install scipy matplotlib numba

    REM Install PyTorch with CUDA support
    echo.
    echo Installing PyTorch with CUDA 12.4...
    pip install torch --index-url https://download.pytorch.org/whl/cu124

    REM Verify installation
    echo.
    echo ==========================================
    echo Verifying installation...
    echo ==========================================
    python -c "import numpy; print(f'numpy: {numpy.__version__}')"
    python -c "import torch; print(f'torch: {torch.__version__}')"
    python -c "import scipy; print(f'scipy: {scipy.__version__}')"
    python -c "import matplotlib; print(f'matplotlib: {matplotlib.__version__}')"

    echo.
    echo ==========================================
    echo Virtual environment setup complete!
    echo ==========================================
) else (
    REM Activate existing venv
    call venv\Scripts\activate
)

REM Step 1: Generate training data
echo.
echo Step 1: Generating training data (2M samples with log-uniform density)...
echo   Parameter ranges:
echo     rho: [0.9, 1000] kg/m^3 (log-uniform)
echo     u:   [0, 100] m/s
echo     p:   [65k, 150k] Pa
echo.

python meltflow_gnn\grid_sampler_droplet.py ^
    --n-samples 2000000 ^
    --output data\droplet_flux_data.npz
if %ERRORLEVEL% neq 0 (
    echo Error in Step 1: Data generation failed
    exit /b %ERRORLEVEL%
)

REM Step 2: Train the model
echo.
echo Step 2: Training MLP (500 epochs, 6x384 architecture on GPU)...
python meltflow_gnn\train_droplet.py ^
    --data data\droplet_flux_data.npz ^
    --epochs 500 ^
    --hidden-dim 384 ^
    --n-layers 6 ^
    --activation gelu ^
    --batch-size 8192 ^
    --lr 1e-3 ^
    --output models\flux_mlp_droplet.pt ^
    --output-npz models\flux_mlp_droplet.npz ^
    --device cuda
if %ERRORLEVEL% neq 0 (
    echo Error in Step 2: Training failed
    exit /b %ERRORLEVEL%
)

REM Step 3: Test on droplet case
echo.
echo Step 3: Testing on in_1Dcdrop...
python meltflow_gnn\test_multiphase.py ^
    --model models\flux_mlp_droplet.pt ^
    --config in_1Dcdrop ^
    --output droplet_comparison.png
if %ERRORLEVEL% neq 0 (
    echo Error in Step 3: Testing failed
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo Done! Check droplet_comparison.png
echo ==========================================
