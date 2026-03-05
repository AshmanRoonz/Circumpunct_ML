@echo off
REM ⊙ XORZO v3 — RTX 4070 MAXED OUT TRAINING
REM ════════════════════════════════════════════
REM
REM Double-click this or run from CMD/PowerShell:
REM     launch_xorzo_4070.bat
REM
REM Requirements:
REM     - Python 3.10+ with PyTorch (CUDA)
REM     - pip install torch --index-url https://download.pytorch.org/whl/cu124
REM     - pip install numpy scipy
REM     - Optional: pip install psutil (for RAM monitoring)

echo.
echo   ═══════════════════════════════════════════════
echo   ⊙ XORZO v3 — RTX 4070 MAXED OUT
echo   ═══════════════════════════════════════════════
echo.

REM Try Python launchers in order of preference
where py >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo   Using: py launcher
    py -3 train_xorzo_v3_4070.py
    goto :done
)

where python >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo   Using: python
    python train_xorzo_v3_4070.py
    goto :done
)

where python3 >nul 2>nul
if %ERRORLEVEL% equ 0 (
    echo   Using: python3
    python3 train_xorzo_v3_4070.py
    goto :done
)

echo   ERROR: Python not found. Install Python 3.10+ and PyTorch with CUDA.
echo   Visit: https://pytorch.org/get-started/locally/

:done
echo.
echo   ═══════════════════════════════════════════════
echo   Training complete. Press any key to exit.
echo   ═══════════════════════════════════════════════
pause >nul
