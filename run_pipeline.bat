@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ╔════════════════════════════════════════════════════════════════════╗
echo ║   WM-811K WAFER DEFECT CLASSIFICATION - PIPELINE EXECUTOR         ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.

:: ============================================================================
:: STEP 1: Check for Virtual Environment
:: ============================================================================
echo [1/4] Checking virtual environment...
if not exist ".venv\Scripts\python.exe" (
    echo [✗] ERROR: Virtual environment not found!
    echo.
    echo     Please run setup first:
    echo     1. Double-click: setup.bat
    echo        OR
    echo     2. Run: python setup.py
    echo.
    pause
    exit /b 1
)
echo [✓] Virtual environment found
echo.

:: ============================================================================
:: STEP 2: Check for Dataset
:: ============================================================================
echo [2/4] Checking dataset...
if not exist "datasets\LSWMD.pkl" (
    echo [✗] ERROR: Dataset not found!
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
)
echo [✓] Dataset found
echo.

:: ============================================================================
:: STEP 3: Activate Virtual Environment and Run Pipeline
:: ============================================================================
echo [3/4] Activating environment and starting pipeline...
echo.
echo ────────────────────────────────────────────────────────────────────
echo    PIPELINE EXECUTION LOG
echo ────────────────────────────────────────────────────────────────────
echo.

:: Record start time
echo Pipeline started at: %DATE% %TIME%
echo.

:: Activate and run
call .venv\Scripts\activate.bat
python ml_flow\main.py

:: Capture exit code
set PIPELINE_EXIT_CODE=%errorlevel%

echo.
echo ────────────────────────────────────────────────────────────────────
echo Pipeline finished at: %DATE% %TIME%
echo ────────────────────────────────────────────────────────────────────
echo.

:: ============================================================================
:: STEP 4: Display Results Summary
:: ============================================================================
echo [4/4] Results Summary...
echo.

if %PIPELINE_EXIT_CODE% equ 0 (
    echo ╔════════════════════════════════════════════════════════════════════╗
    echo ║   ✓ PIPELINE COMPLETED SUCCESSFULLY                               ║
    echo ╚════════════════════════════════════════════════════════════════════╝
    echo.
    echo    📊 Results are available in:
    echo.
    echo    • Stage 1 Output:   data_loader_results\
    echo    • Stage 2 Output:   Feature_engineering_results\
    echo    • Stage 3 Output:   preprocessing_results\
    echo    • Stage 4 Output:   feature_selection_results\
    echo    • Stage 5 Output:   model_artifacts\
    echo.
    echo    📈 Check the master leaderboard:
    echo    model_artifacts\master_model_comparison.csv
    echo.
) else (
    echo ╔════════════════════════════════════════════════════════════════════╗
    echo ║   ✗ PIPELINE FAILED WITH ERRORS                                   ║
    echo ╚════════════════════════════════════════════════════════════════════╝
    echo.
    echo    Exit Code: %PIPELINE_EXIT_CODE%
    echo.
    echo    💡 Troubleshooting:
    echo    • Check pipeline.log for error details
    echo    • Verify dataset integrity
    echo    • Ensure all dependencies are installed
    echo    • Try running individual stages for debugging
    echo.
)

echo ════════════════════════════════════════════════════════════════════
echo.
pause

exit /b %PIPELINE_EXIT_CODE%
