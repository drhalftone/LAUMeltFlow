@echo off
title HGNN Training
cd /d "%~dp0"
call ..\gnn\.venv\Scripts\activate.bat
python train.py %*
echo.
echo Training complete.
pause
