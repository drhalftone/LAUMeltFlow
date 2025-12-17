@echo off
REM Compare H2GNN vs H2GNN-Attention for electrostatic field prediction

setlocal EnableDelayedExpansion

echo ==========================================
echo H2GNN vs H2GNN-Attention Comparison
echo ==========================================

REM Check if venv exists
if not exist venv\Scripts\activate.bat (
    echo.
    echo ERROR: Virtual environment not found.
    echo Please run train_multiphase.bat first to create the venv.
    exit /b 1
)

REM Activate venv
call venv\Scripts\activate

REM Run comparison with reasonable defaults for quick testing
REM Increase epochs/samples for more thorough comparison
echo.
echo Running comparison training...
echo   Epochs: 200
echo   Training samples: 500
echo   Validation samples: 100
echo   Particles per sample: 20-80
echo.

python -m electrostatic_unet.train_comparison ^
    --epochs 200 ^
    --n-train 500 ^
    --n-val 100 ^
    --min-particles 20 ^
    --max-particles 80 ^
    --depth 4 ^
    --hidden-dim 64 ^
    --lr 1e-3 ^
    --k-per-voxel 16 ^
    --n-heads 4 ^
    --n-attn-layers 2 ^
    --checkpoint-dir checkpoints/comparison ^
    --device cpu

if %ERRORLEVEL% neq 0 (
    echo.
    echo Error: Training failed
    exit /b %ERRORLEVEL%
)

echo.
echo ==========================================
echo Done! Check:
echo   - checkpoints/comparison/comparison_plot.png
echo   - checkpoints/comparison/comparison_results.json
echo ==========================================
