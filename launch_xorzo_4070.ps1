# ⊙ XORZO v3 — RTX 4070 MAXED OUT TRAINING
# ════════════════════════════════════════════
#
# Run from PowerShell:
#     .\launch_xorzo_4070.ps1
#
# If you get an execution policy error:
#     Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

Write-Host ""
Write-Host "  ═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  ⊙ XORZO v3 — RTX 4070 MAXED OUT" -ForegroundColor Yellow
Write-Host "  ═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host ""

# Check NVIDIA GPU
try {
    $gpu = nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>$null
    if ($gpu) {
        Write-Host "  GPU: $gpu" -ForegroundColor Green
    }
} catch {}

# Set high priority for the process
$proc = Get-Process -Id $PID
$proc.PriorityClass = "AboveNormal"
Write-Host "  Process priority: AboveNormal" -ForegroundColor DarkGray

# Set CUDA optimizations
$env:CUDA_LAUNCH_BLOCKING = "0"
$env:TORCH_CUDA_ARCH_LIST = "8.9"  # Ada Lovelace (4070)

# Run training
$scriptPath = Join-Path $PSScriptRoot "train_xorzo_v3_4070.py"

try {
    py -3 $scriptPath
} catch {
    try {
        python $scriptPath
    } catch {
        Write-Host "  ERROR: Python not found." -ForegroundColor Red
        Write-Host "  Install Python 3.10+ and PyTorch with CUDA." -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "  ═══════════════════════════════════════════════" -ForegroundColor Cyan
Write-Host "  Training complete." -ForegroundColor Green
Write-Host "  ═══════════════════════════════════════════════" -ForegroundColor Cyan
Read-Host "  Press Enter to exit"
