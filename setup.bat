@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ╔════════════════════════════════════════════════════════════════════╗
echo ║   WM-811K WAFER DEFECT CLASSIFICATION - ONE-CLICK SETUP           ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.

:: ============================================================================
:: STEP 1: Check Python Installation
:: ============================================================================
echo [1/5] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [✗] ERROR: Python is not installed or not in your PATH.
    echo.
    echo     Please install Python 3.9 or higher from:
    echo     https://www.python.org/downloads/
    echo.
    echo     Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

:: Get Python version and check if it's 3.9+
for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo [✓] Python %PYTHON_VERSION% detected

:: Extract major and minor version
for /f "tokens=1,2 delims=." %%a in ("%PYTHON_VERSION%") do (
    set PYTHON_MAJOR=%%a
    set PYTHON_MINOR=%%b
)

if %PYTHON_MAJOR% LSS 3 (
    echo [✗] ERROR: Python 3.9+ is required. You have Python %PYTHON_VERSION%
    pause
    exit /b 1
)
if %PYTHON_MAJOR% EQU 3 if %PYTHON_MINOR% LSS 9 (
    echo [✗] ERROR: Python 3.9+ is required. You have Python %PYTHON_VERSION%
    pause
    exit /b 1
)
echo.

:: ============================================================================
:: STEP 2: Create Virtual Environment
:: ============================================================================
echo [2/5] Creating virtual environment...
if exist ".venv\" (
    echo [!] Virtual environment already exists. Skipping creation.
) else (
    python -m venv .venv
    if %errorlevel% neq 0 (
        echo [✗] ERROR: Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [✓] Virtual environment created successfully
)
echo.

:: ============================================================================
:: STEP 3: Activate Virtual Environment
:: ============================================================================
echo [3/5] Activating virtual environment...
call .venv\Scripts\activate.bat
if %errorlevel% neq 0 (
    echo [✗] ERROR: Failed to activate virtual environment.
    pause
    exit /b 1
)
echo [✓] Virtual environment activated
echo.

:: ============================================================================
:: STEP 4: Upgrade pip and Install Dependencies
:: ============================================================================
echo [4/5] Upgrading pip and installing dependencies...
echo     This may take several minutes. Please wait...
echo.

python -m pip install --upgrade pip setuptools wheel --quiet
if %errorlevel% neq 0 (
    echo [✗] ERROR: Failed to upgrade pip.
    pause
    exit /b 1
)

pip install -r requirement.txt
if %errorlevel% neq 0 (
    echo [✗] ERROR: Failed to install dependencies.
    echo.
    echo     Try manually running:
    echo     .venv\Scripts\activate
    echo     pip install -r requirement.txt
    echo.
    pause
    exit /b 1
)
echo [✓] All dependencies installed successfully
echo.

:: ============================================================================
:: STEP 5: Verify Dataset and Create Folder
:: ============================================================================
echo [5/5] Checking dataset...
if not exist "datasets\" (
    echo [!] Creating datasets folder...
    mkdir datasets
)

if not exist "datasets\LSWMD.pkl" (
    echo.
    echo ┌────────────────────────────────────────────────────────────────┐
    echo │  ⚠️  DATASET NOT FOUND                                          │
    echo └────────────────────────────────────────────────────────────────┘
    echo.
    echo    The wafer dataset (LSWMD.pkl) is required to run the pipeline.
    echo.
    echo    📥 Download from Kaggle:
    echo    https://www.kaggle.com/datasets/qingyi/wm811k-wafer-map
    echo.
    echo    📂 Place the file here:
    echo    %~dp0datasets\LSWMD.pkl
    echo.
    echo    Once downloaded, you can run the pipeline using:
    echo    run_pipeline.bat
    echo.
) else (
    echo [✓] Dataset found: datasets\LSWMD.pkl
    echo.
    echo ╔════════════════════════════════════════════════════════════════════╗
    echo ║   ✓ SETUP COMPLETE - READY TO RUN!                                ║
    echo ╚════════════════════════════════════════════════════════════════════╝
    echo.
    echo    Your environment is ready. To run the pipeline:
    echo.
    echo    1. Double-click: run_pipeline.bat
    echo       OR
    echo    2. Run manually:
    echo       .venv\Scripts\activate
    echo       python ml_flow\main.py
    echo.
)

echo ════════════════════════════════════════════════════════════════════
echo.
pause
