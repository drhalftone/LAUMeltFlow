@echo off
title HGNN Data Generation
cd /d "%~dp0"
call ..\gnn\.venv\Scripts\activate.bat
python data.py %*
echo.
echo Data generation complete.
pause
