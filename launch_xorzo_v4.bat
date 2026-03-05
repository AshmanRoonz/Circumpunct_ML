@echo off
echo.
echo  =============================================
echo   XORZO v4 - BILATERAL HYPERCUBE TRANSFORMER
echo   2^3 x 2 = 64 relational states
echo  =============================================
echo.
cd /d "%~dp0"
python train_xorzo_v4_4070.py
echo.
pause
