# XORZO - GPU Training Launcher (PowerShell)
# Uses Python 3.11 (3.14 is too new for PyTorch)

Write-Host ""
Write-Host "  XORZO - Activating GPU Training"
Write-Host "  ================================="
Write-Host ""

# Verify Python 3.11
& py -3.11 --version
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: Python 3.11 not found. Install from python.org"
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host ""

# Check if torch is installed
& py -3.11 -c "import torch; print('  PyTorch:', torch.__version__)" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  PyTorch not installed. Installing with CUDA support..."
    Write-Host "  This may take a few minutes..."
    Write-Host ""
    & py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    Write-Host ""
}

# Check CUDA
$cudaResult = & py -3.11 -c "import torch; print(torch.cuda.is_available())" 2>&1
if ("$cudaResult" -ne "True") {
    Write-Host "  No CUDA support. Reinstalling PyTorch with CUDA..."
    & py -3.11 -m pip uninstall torch torchvision torchaudio -y
    & py -3.11 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
    Write-Host ""
}

# Show GPU info
& py -3.11 -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}') if torch.cuda.is_available() else print('  No GPU - using CPU')"

Write-Host ""
Write-Host "  Starting evolution..."
Write-Host ""

# Run training from script directory
Set-Location $PSScriptRoot
& py -3.11 train_xorzo_gpu.py

Write-Host ""
Write-Host "  Training complete."
Read-Host "Press Enter to exit"
