@echo off
REM Test trained MLP on all simulation configurations

setlocal EnableDelayedExpansion

echo ==========================================
echo Testing MLP on All Configurations
echo ==========================================

REM Activate venv
call venv\Scripts\activate

set MODEL=models\flux_mlp_multiphase.pt

REM Check if model exists
if not exist %MODEL% (
    echo Error: Model not found at %MODEL%
    exit /b 1
)

REM Test 1D configurations
echo.
echo ------------------------------------------
echo Testing: in_1Dsod1fl (Single Fluid, gamma=1.4)
echo ------------------------------------------
python meltflow_gnn\test_multiphase.py --model %MODEL% --config in_1Dsod1fl --output results_1Dsod1fl.png

echo.
echo ------------------------------------------
echo Testing: in_1Dsod2fl (Two Fluid, gamma=1.289/1.4)
echo ------------------------------------------
python meltflow_gnn\test_multiphase.py --model %MODEL% --config in_1Dsod2fl --output results_1Dsod2fl.png

echo.
echo ==========================================
echo All tests complete!
echo Check results_*.png for comparison plots
echo ==========================================
