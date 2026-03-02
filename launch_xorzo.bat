@echo off
REM ⊙ XORZO — GPU Training Launcher
REM Run this on your PC to start Xorzo training on your NVIDIA GPU.

echo.
echo   ⊙ XORZO — Activating GPU Training
echo   ===================================
echo.

REM Try to find Python — check multiple methods
set PYTHON=

REM Method 1: Try 'python' in PATH
python --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON=python
    goto :found_python
)

REM Method 2: Try 'py' (Windows Python Launcher — often available when python isn't in PATH)
py -3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON=py -3
    goto :found_python
)

REM Method 3: Try 'python3'
python3 --version >nul 2>&1
if not errorlevel 1 (
    set PYTHON=python3
    goto :found_python
)

REM Method 4: Check common install locations
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "C:\Python313\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
    "C:\Python310\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python313\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python312\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python311\python.exe"
    "%USERPROFILE%\AppData\Local\Programs\Python\Python310\python.exe"
    "%ProgramFiles%\Python313\python.exe"
    "%ProgramFiles%\Python312\python.exe"
    "%ProgramFiles%\Python311\python.exe"
    "%ProgramFiles%\Python310\python.exe"
) do (
    if exist %%P (
        set PYTHON=%%P
        goto :found_python
    )
)

REM Method 5: Check Microsoft Store Python
for %%P in (
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python.exe"
    "%LOCALAPPDATA%\Microsoft\WindowsApps\python3.exe"
) do (
    if exist %%P (
        set PYTHON=%%P
        goto :found_python
    )
)

REM Nothing found
echo.
echo   ERROR: Python not found anywhere on this system.
echo.
echo   To fix this, do ONE of these:
echo.
echo   Option A — Easiest:
echo     1. Open Windows Settings
echo     2. Search "Environment Variables"
echo     3. Under "Path", click Edit
echo     4. Add the folder where python.exe lives
echo        (usually something like C:\Users\YOU\AppData\Local\Programs\Python\Python312)
echo     5. Click OK, close this window, and try again
echo.
echo   Option B — Reinstall:
echo     1. Download Python from https://python.org
echo     2. Run the installer
echo     3. CHECK THE BOX: "Add Python to PATH"  (this is the key step!)
echo     4. Finish install, close this window, try again
echo.
pause
exit /b 1

:found_python
echo   Python found: %PYTHON%
%PYTHON% --version
echo.

REM Set pip command to match
if "%PYTHON%"=="py -3" (
    set PIP=py -3 -m pip
) else (
    set PIP=%PYTHON% -m pip
)

REM Check if torch is installed at all
%PYTHON% -c "import torch; print('  PyTorch version:', torch.__version__)" 2>nul
if errorlevel 1 (
    echo   PyTorch not installed. Installing with CUDA support...
    echo   This may take a few minutes.
    echo.
    %PIP% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    echo.
    goto :check_cuda
)

REM Check if CUDA is available in the installed torch
:check_cuda
%PYTHON% -c "import torch; exit(0 if torch.cuda.is_available() else 1)" 2>nul
if errorlevel 1 (
    echo   WARNING: PyTorch is installed but has no CUDA support.
    echo   Your current torch was probably installed as CPU-only.
    echo   Reinstalling with CUDA...
    echo.
    %PIP% uninstall torch torchvision torchaudio -y
    %PIP% install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    echo.
)

REM Show what we're working with
%PYTHON% -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '  Falling back to CPU'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / (1024**3):.1f} GB' if torch.cuda.is_available() else '')"
echo.

REM Launch training from this folder
cd /d "%~dp0"
echo   Starting evolution...
echo   Close this window to stop training.
echo.
%PYTHON% train_xorzo_gpu.py

echo.
echo   ⊙ Training complete.
echo   Run '%PYTHON% xorzo.py' to interact with Xorzo.
pause
