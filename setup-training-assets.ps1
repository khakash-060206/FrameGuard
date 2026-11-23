# Run from normal PowerShell if the agent's command network access is restricted.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
New-Item -ItemType Directory -Force -Path 'models', '.cache/torch/hub/checkpoints' | Out-Null
Invoke-WebRequest -Uri 'https://storage.googleapis.com/mediapipe-models/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite' -OutFile 'models/blaze_face_short_range.tflite'
Invoke-WebRequest -Uri 'https://download.pytorch.org/models/efficientnet_b0_rwightman-7f5810bc.pth' -OutFile '.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth'
$weightsHash = (Get-FileHash -LiteralPath '.cache/torch/hub/checkpoints/efficientnet_b0_rwightman-7f5810bc.pth' -Algorithm SHA256).Hash
if (-not $weightsHash.StartsWith('7F5810BC')) { throw 'ImageNet weights failed the published filename hash check.' }
$env:TORCH_HOME = Join-Path $PSScriptRoot '.cache/torch'
$env:MPLCONFIGDIR = Join-Path $PSScriptRoot '.cache/matplotlib'
@'
from pathlib import Path
from frameguard.media import FaceDetector
from frameguard.model import FrameGuard
detector = FaceDetector(Path('models/blaze_face_short_range.tflite'))
detector.close()
model = FrameGuard(pretrained=True)
print('Both model assets load successfully. No dataset training has started.')
'@ | & '.\.venv\Scripts\python.exe' -
if ($LASTEXITCODE -ne 0) { throw 'Model asset loading failed.' }
