@echo off
chcp 65001 > nul
setlocal enabledelayedexpansion

echo ======================================================================
echo    WM-811K WAFER DEFECT CLASSIFICATION - AUTOMATED RUNNER
echo ======================================================================
echo.

:: 1. Check for Dataset
if not exist "datasets\LSWMD.pkl" (
    echo [!] ERROR: Dataset not found!
    echo.
    echo     Please download 'LSWMD.pkl' from Kaggle:
    echo     https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
    echo.
    echo     And place it in the 'datasets' folder:
    echo     %~dp0datasets\LSWMD.pkl
    echo.
    echo     Once done, run this script again.
    echo.
    pause
    exit /b 1
) else (
    echo [+] Dataset found.
)

:: 2. Check for Python
python --version > nul 2>&1
if %errorlevel% neq 0 (
    echo [!] ERROR: Python is not installed or not in your PATH.
    echo     Please install Python 3.8+ and try again.
    pause
    exit /b 1
)

:: 3. Install Requirements
echo.
echo [i] Installing dependencies...
pip install -r requirement.txt
if %errorlevel% neq 0 (
    echo [!] ERROR: Failed to install dependencies.
    pause
    exit /b 1
)

:: 4. Run Pipeline
echo.
echo [>] Starting Pipeline...
echo.
python ml_flow\main.py

echo.
echo ======================================================================
echo    EXECUTION COMPLETE
echo ======================================================================
pause
