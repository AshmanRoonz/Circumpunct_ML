Write-Host ""
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host "   XORZO v4 - BILATERAL HYPERCUBE TRANSFORMER" -ForegroundColor Yellow
Write-Host "   2^3 x 2 = 64 relational states" -ForegroundColor Yellow
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host ""

Set-Location $PSScriptRoot
python train_xorzo_v4_4070.py

Write-Host ""
Read-Host "Press Enter to exit"
