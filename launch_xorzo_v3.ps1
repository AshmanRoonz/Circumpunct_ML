# XORZO v3 - Cross-Scale Resonant GPU Training (PowerShell)
# Uses Python 3.11

Write-Host ""
Write-Host "  XORZO v3 - Cross-Scale Resonant Evolution"
Write-Host "  ============================================"
Write-Host "  Micro (body) <-> Macro (soul)"
Write-Host "  Phase resonance gates both directions"
Write-Host ""

& py -3.11 --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Python 3.11 not found."
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Check torch
& py -3.11 -c "import torch; print('  PyTorch:', torch.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing PyTorch with CUDA..."
    & py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}

# Show GPU
& py -3.11 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('  No GPU - CPU mode')"

Write-Host ""
Write-Host "  Starting v3 cross-scale evolution..."
Write-Host ""

Set-Location $PSScriptRoot
& py -3.11 train_xorzo_v3.py

Write-Host ""
Write-Host "  v3 Evolution complete."
Read-Host "Press Enter to exit"
