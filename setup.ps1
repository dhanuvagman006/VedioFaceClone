# One-time setup (already done on this PC). Re-run to rebuild the environment:
#   powershell -ExecutionPolicy Bypass -File setup.ps1
# Creates .venv (Python 3.11), installs CUDA PyTorch + both TTS engines, downloads models (~12 GB total).
$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $here

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    py -3.11 -m venv .venv
    if (-not $?) { throw "Python 3.11 is required (py -3.11). Install it from python.org." }
}
$py = Join-Path $here ".venv\Scripts\python.exe"
$cache = Join-Path $here ".pip-cache"

& $py -m pip install --upgrade pip setuptools wheel --cache-dir $cache
& $py -m pip install torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128 --cache-dir $cache
& $py -m pip install -r requirements.txt -c constraints.txt --cache-dir $cache

# MuseTalk 1.5 code for lip sync (not on PyPI), at the commit this project was built against
$mt = Join-Path $here "third_party\MuseTalk"
if (-not (Test-Path (Join-Path $mt "musetalk"))) {
    $sha = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"
    $zip = Join-Path $env:TEMP "musetalk-$sha.zip"
    $tmp = Join-Path $env:TEMP "musetalk-x"
    Invoke-WebRequest "https://github.com/TMElyralab/MuseTalk/archive/$sha.zip" -OutFile $zip -UseBasicParsing
    Expand-Archive $zip -DestinationPath $tmp -Force
    New-Item -ItemType Directory -Force (Join-Path $here "third_party") | Out-Null
    Move-Item (Join-Path $tmp "MuseTalk-$sha") $mt
    Set-Content (Join-Path $mt "COMMIT.txt") $sha
    Remove-Item $zip, $tmp -Recurse -Force
}

$env:PYTHONPATH = $here
& $py -c "import torch; print('CUDA available:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
& $py -m vclone.download
Write-Host "`nDone. Try:  .\speak.bat my_voice.wav `"Hello, this is my cloned voice.`""
