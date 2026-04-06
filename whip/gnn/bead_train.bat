@echo off
title Bead GNN Training
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python bead_train.py %*
echo.
echo Training complete.
pause
