<#
FrameGuard one-step environment setup for a fresh Windows laptop.

Run from the project folder in normal (not Administrator) PowerShell:

    powershell -ExecutionPolicy Bypass -File .\setup.ps1

What it does:
  1. Finds Python 3.13 (installs it for the current user with winget if missing).
  2. Creates .venv in this folder (an existing .venv is reused, never deleted).
  3. Installs PyTorch: a CUDA build if an NVIDIA GPU is detected, otherwise CPU.
  4. Installs every other dependency at the exact versions in requirements.lock.txt.
  5. Downloads the MediaPipe face detector and ImageNet EfficientNet-B0 weights
     and verifies their SHA-256 hashes.
  6. Checks that everything imports and loads, reports the GPU, and runs the unit tests.

Options:
  -Cpu              Force the CPU build of PyTorch even if an NVIDIA GPU exists.
  -CudaIndex cu130  Use a different PyTorch CUDA wheel index (default cu126,
                    which works with NVIDIA driver 528 or newer).
  -SkipTests        Skip the unit tests at the end.
#>
param(
    [switch]$Cpu,
    [string]$CudaIndex = "cu126",
    [switch]$SkipTests
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"   # makes Invoke-WebRequest much faster
Set-Location -LiteralPath $PSScriptRoot

# Versions. The CPU pair matches requirements.lock.txt exactly. The CUDA pair is the
# newest matched torch/torchvision release published for CUDA 12.6 on Windows.
$PythonVersion = "3.13"
$CpuTorch = @("torch==2.15.0", "torchvision==0.29.2")
$CudaTorch = @("torch==2.14.1", "torchvision==0.29.1")

$Assets = @(
    @{ Url = "https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite"
       Path = "models/blaze_face_short_range.tflite"
       Sha256 = "B4578F35940BF5A1A655214A1CCE5CAB13EBA73C1297CD78E1A04C2380B0152F" },
    @{ Url = "https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth"
       Path = ".cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth"
       Sha256 = "7F5810BC96DEF8F7552D5B7E68D53C4786F81167D28291B21C0D90E1FCA14934" }
)

function Write-Step($text) { Write-Host "`n==> $text" -ForegroundColor Cyan }
function Fail($text) { Write-Host "`nSETUP FAILED: $text" -ForegroundColor Red; exit 1 }

function Invoke-Checked {
    param([string]$File, [string[]]$Arguments)
    & $File @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "'$File $($Arguments -join ' ')' exited with code $LASTEXITCODE" }
}

function Find-Python {
    $launcher = Get-Command py -ErrorAction SilentlyContinue
    if ($launcher) {
        $path = & py "-$PythonVersion" -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $path) { return $path.Trim() }
    }
    foreach ($name in @("python", "python3")) {
        $command = Get-Command $name -ErrorAction SilentlyContinue
        if ($command -and $command.Source -notlike "*WindowsApps*") {
            $version = & $command.Source -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
            if ($version -eq $PythonVersion) { return $command.Source }
        }
    }
    $default = Join-Path $env:LOCALAPPDATA "Programs\Python\Python313\python.exe"
    if (Test-Path $default) { return $default }
    return $null
}

# ---------------------------------------------------------------- 1. Python
Write-Step "Looking for Python $PythonVersion"
$python = Find-Python
if (-not $python) {
    if (-not (Get-Command winget -ErrorAction SilentlyContinue)) {
        Fail "Python $PythonVersion not found and winget is unavailable. Install Python $PythonVersion (64-bit) from https://www.python.org/downloads/ and run this script again."
    }
    Write-Host "Python $PythonVersion not found. Installing it for the current user with winget..."
    & winget install --id Python.Python.3.13 --scope user --exact --silent --accept-package-agreements --accept-source-agreements
    $python = Find-Python
    if (-not $python) { Fail "Python $PythonVersion was installed but not found. Open a new PowerShell window and run this script again." }
}
Write-Host "Using $python"

# ---------------------------------------------------------------- 2. venv
Write-Step "Creating virtual environment in .venv"
$venvPython = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
if (Test-Path $venvPython) {
    $venvVersion = & $venvPython -c "import sys; print('%d.%d' % sys.version_info[:2])"
    if ($venvVersion -ne $PythonVersion) {
        Fail ".venv exists but uses Python $venvVersion. Rename or remove .venv yourself, then run this script again."
    }
    Write-Host ".venv already exists; reusing it."
} else {
    Invoke-Checked $python @("-m", "venv", ".venv")
}
Invoke-Checked $venvPython @("-m", "pip", "install", "--upgrade", "pip", "--quiet")

# ---------------------------------------------------------------- 3. PyTorch
Write-Step "Installing PyTorch"
$gpuName = $null
if (-not $Cpu -and (Get-Command nvidia-smi -ErrorAction SilentlyContinue)) {
    $gpuName = (& nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>$null | Select-Object -First 1)
}
if ($gpuName) {
    Write-Host "NVIDIA GPU detected: $gpuName"
    Write-Host "Installing CUDA build ($CudaIndex): $($CudaTorch -join ', ')"
    Invoke-Checked $venvPython (@("-m", "pip", "install") + $CudaTorch + @("--index-url", "https://download.pytorch.org/whl/$CudaIndex"))
} else {
    if ($Cpu) { Write-Host "CPU build requested." } else { Write-Host "No NVIDIA GPU detected (nvidia-smi not found). Installing CPU build." }
    Invoke-Checked $venvPython (@("-m", "pip", "install") + $CpuTorch + @("--index-url", "https://download.pytorch.org/whl/cpu"))
}

# ---------------------------------------------------------------- 4. Other dependencies
Write-Step "Installing remaining dependencies at locked versions"
$constraints = Join-Path $env:TEMP "frameguard-constraints.txt"
Get-Content requirements.lock.txt | Where-Object { $_ -notmatch '^(torch|torchvision)==' } | Set-Content -Encoding ascii $constraints
Invoke-Checked $venvPython @("-m", "pip", "install", "-r", "requirements.txt", "-c", $constraints)
Remove-Item $constraints -ErrorAction SilentlyContinue
Invoke-Checked $venvPython @("-m", "pip", "check")

# ---------------------------------------------------------------- 5. Model assets
Write-Step "Downloading and verifying model assets"
foreach ($asset in $Assets) {
    $target = Join-Path $PSScriptRoot $asset.Path
    New-Item -ItemType Directory -Force -Path (Split-Path $target) | Out-Null
    if ((Test-Path $target) -and ((Get-FileHash $target -Algorithm SHA256).Hash -eq $asset.Sha256)) {
        Write-Host "OK (already present): $($asset.Path)"
        continue
    }
    $partial = "$target.partial"
    Invoke-WebRequest -Uri $asset.Url -OutFile $partial
    $hash = (Get-FileHash $partial -Algorithm SHA256).Hash
    if ($hash -ne $asset.Sha256) {
        Remove-Item $partial
        Fail "Hash mismatch for $($asset.Path). Expected $($asset.Sha256), got $hash."
    }
    Move-Item -Force $partial $target
    Write-Host "OK (downloaded, hash verified): $($asset.Path)"
}

# ---------------------------------------------------------------- 6. Verification
Write-Step "Verifying the environment"
$env:TORCH_HOME = Join-Path $PSScriptRoot ".cache\torch"
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot ".cache\matplotlib"
$check = @'
import torch, torchvision, cv2, mediapipe, numpy, sklearn
from pathlib import Path
from frameguard.media import FaceDetector
from frameguard.model import FrameGuard
print(f"torch {torch.__version__}, torchvision {torchvision.__version__}, opencv {cv2.__version__}, mediapipe {mediapipe.__version__}")
detector = FaceDetector(Path("models/blaze_face_short_range.tflite")); detector.close()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = FrameGuard(pretrained=True).to(device).eval()
with torch.no_grad():
    out = model(torch.zeros(2, 3, 224, 224, device=device))
assert out.shape[0] == 2 and torch.isfinite(out).all()
if device == "cuda":
    p = torch.cuda.get_device_properties(0)
    print(f"CUDA OK: {p.name}, {p.total_memory / 1024**3:.1f} GB VRAM, CUDA {torch.version.cuda}")
else:
    print("Running on CPU (no usable CUDA device).")
print("Face detector and EfficientNet-B0 loaded; forward pass OK.")
'@
$check | & $venvPython -
if ($LASTEXITCODE -ne 0) { Fail "Environment verification failed (see error above)." }
if ($gpuName) {
    & $venvPython -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "WARNING: an NVIDIA GPU was detected but PyTorch cannot use it. Update the NVIDIA driver, or rerun with -CudaIndex cu130 / cu132." -ForegroundColor Yellow
    }
}

if (-not $SkipTests) {
    Write-Step "Running unit tests"
    Invoke-Checked $venvPython @("-m", "unittest", "discover", "-s", "tests")
}

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "Next: put the two dataset ZIPs in data\ and run:"
Write-Host "    .\.venv\Scripts\python.exe scripts\run_experiment.py --device auto"
