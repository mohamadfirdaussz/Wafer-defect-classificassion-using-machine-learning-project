@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

echo ╔════════════════════════════════════════════════════════════════════╗
echo ║   CREATE DISTRIBUTION PACKAGE FOR WAFER DEFECT CLASSIFICATION     ║
echo ╚════════════════════════════════════════════════════════════════════╝
echo.

:: ============================================================================
:: STEP 1: Check for Python
:: ============================================================================
echo [1/3] Checking Python installation...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [✗] ERROR: Python is not installed or not in your PATH.
    pause
    exit /b 1
)
echo [✓] Python detected
echo.

:: ============================================================================
:: STEP 2: Run Python Distribution Script
:: ============================================================================
echo [2/3] Running distribution packaging script...
echo.
python create_distribution.py

set SCRIPT_EXIT_CODE=%errorlevel%
echo.

:: ============================================================================
:: STEP 3: Display Results
:: ============================================================================
echo [3/3] Packaging complete
echo.

if %SCRIPT_EXIT_CODE% equ 0 (
    echo ╔════════════════════════════════════════════════════════════════════╗
    echo ║   ✓ DISTRIBUTION PACKAGE CREATED SUCCESSFULLY                     ║
    echo ╚════════════════════════════════════════════════════════════════════╝
    echo.
    echo    📦 Package ready for distribution!
    echo.
    echo    Next steps:
    echo    1. Share the generated ZIP file with others
    echo    2. Recipients should extract the ZIP
    echo    3. Recipients should run setup.bat (Windows) or setup.py (Linux/Mac)
    echo    4. Then run run_pipeline.bat or python ml_flow/main.py
    echo.
) else (
    echo ╔════════════════════════════════════════════════════════════════════╗
    echo ║   ✗ PACKAGING FAILED                                              ║
    echo ╚════════════════════════════════════════════════════════════════════╝
    echo.
)

echo ════════════════════════════════════════════════════════════════════
echo.
pause

exit /b %SCRIPT_EXIT_CODE%
