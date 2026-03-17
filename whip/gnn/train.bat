@echo off
title GNN Training
cd /d "%~dp0"
call .venv\Scripts\activate.bat
python train.py %*
echo.
echo Training complete.
pause
