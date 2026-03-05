#!/usr/bin/env python3
"""
⊙ CUDA Diagnostic — Run this to find out why CUDA isn't working
    python cuda_check.py
"""
import sys
import subprocess
import os

print()
print("  ═══════════════════════════════════════════════")
print("  ⊙ CUDA DIAGNOSTIC")
print("  ═══════════════════════════════════════════════")
print()

# 1. Python version
print(f"  Python: {sys.version}")
print(f"  Python path: {sys.executable}")
print()

# 2. Check nvidia-smi (driver level)
print("  ── NVIDIA Driver ──")
try:
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        lines = result.stdout.strip().split("\n")
        for line in lines[:5]:
            print(f"    {line}")
        # Extract driver version
        for line in lines:
            if "Driver Version" in line:
                print(f"\n  ✓ NVIDIA driver is working")
                break
    else:
        print(f"  ✗ nvidia-smi failed: {result.stderr.strip()}")
        print("    → NVIDIA driver may not be installed or GPU not detected")
except FileNotFoundError:
    print("  ✗ nvidia-smi not found")
    print("    → NVIDIA driver is NOT installed")
    print("    → Download from: https://www.nvidia.com/drivers/")
except Exception as e:
    print(f"  ✗ nvidia-smi error: {e}")
print()

# 3. Check PyTorch
print("  ── PyTorch ──")
try:
    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  PyTorch install path: {torch.__file__}")

    # Check if CUDA build
    cuda_built = torch.version.cuda
    print(f"  Built with CUDA: {cuda_built if cuda_built else 'NO — CPU-ONLY BUILD'}")

    if not cuda_built:
        print()
        print("  ╔══════════════════════════════════════════════════════╗")
        print("  ║  PROBLEM FOUND: PyTorch was installed WITHOUT CUDA   ║")
        print("  ╚══════════════════════════════════════════════════════╝")
        print()
        print("  FIX: Reinstall PyTorch with CUDA support:")
        print()
        print("    pip uninstall torch torchvision torchaudio -y")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
        print()
        print("  Or for CUDA 12.6 (newer drivers):")
        print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126")
        print()
    else:
        print(f"  CUDA available: {torch.cuda.is_available()}")

        if torch.cuda.is_available():
            print(f"  GPU count: {torch.cuda.device_count()}")
            print(f"  GPU name: {torch.cuda.get_device_name(0)}")
            props = torch.cuda.get_device_properties(0)
            print(f"  VRAM: {props.total_mem / (1024**3):.1f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")
            print()
            print("  ✓ CUDA IS WORKING — you're good to go!")

            # Quick test
            try:
                x = torch.randn(100, 100, device="cuda")
                y = x @ x.T
                print(f"  ✓ GPU compute test passed (matmul on CUDA)")
            except Exception as e:
                print(f"  ✗ GPU compute test failed: {e}")
        else:
            print()
            print("  ╔══════════════════════════════════════════════════════╗")
            print("  ║  PROBLEM: PyTorch has CUDA but can't see the GPU     ║")
            print("  ╚══════════════════════════════════════════════════════╝")
            print()

            # Check CUDA runtime vs driver compatibility
            print(f"  PyTorch CUDA version: {torch.version.cuda}")
            try:
                result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
                for line in result.stdout.split("\n"):
                    if "CUDA Version" in line:
                        # Extract driver's max CUDA version
                        parts = line.split("CUDA Version:")
                        if len(parts) > 1:
                            driver_cuda = parts[1].strip().split()[0]
                            print(f"  Driver max CUDA version: {driver_cuda}")

                            # Compare
                            pt_major = int(torch.version.cuda.split(".")[0])
                            pt_minor = int(torch.version.cuda.split(".")[1])
                            dr_major = int(driver_cuda.split(".")[0])
                            dr_minor = int(driver_cuda.split(".")[1])

                            if pt_major > dr_major or (pt_major == dr_major and pt_minor > dr_minor):
                                print()
                                print(f"  ✗ VERSION MISMATCH: PyTorch needs CUDA {torch.version.cuda}")
                                print(f"    but your driver only supports up to CUDA {driver_cuda}")
                                print()
                                print("  FIX (Option A): Update NVIDIA driver:")
                                print("    https://www.nvidia.com/drivers/")
                                print()
                                print("  FIX (Option B): Install PyTorch for your CUDA version:")
                                print(f"    pip uninstall torch torchvision torchaudio -y")
                                if dr_major >= 12:
                                    print(f"    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{dr_major}{dr_minor}")
                                else:
                                    print(f"    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu{dr_major}{dr_minor}0")
                            else:
                                print(f"  ✓ CUDA versions are compatible")
                                print()
                                print("  Other possible causes:")
                                print("    1. CUDA_VISIBLE_DEVICES is set to empty/invalid")
                                if os.environ.get("CUDA_VISIBLE_DEVICES"):
                                    print(f"       → Current value: '{os.environ['CUDA_VISIBLE_DEVICES']}'")
                                    print("       → Try: set CUDA_VISIBLE_DEVICES=0")
                                print("    2. Another process is using all GPU memory")
                                print("       → Check: nvidia-smi")
                                print("    3. PyTorch binary doesn't match your Python version")
                                print(f"       → Python: {sys.version_info.major}.{sys.version_info.minor}")
                                print("    4. Corrupted install — try clean reinstall:")
                                print("       pip uninstall torch torchvision torchaudio -y")
                                print("       pip cache purge")
                                print("       pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            except Exception:
                pass

except ImportError:
    print("  ✗ PyTorch is NOT installed")
    print()
    print("  FIX: Install PyTorch with CUDA:")
    print("    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")

print()
print("  ═══════════════════════════════════════════════")
print()
