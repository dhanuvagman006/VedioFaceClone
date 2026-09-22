#!/usr/bin/env bash
# One-time setup on Linux or Google Colab: CUDA PyTorch, both TTS engines, lip sync (LatentSync 1.6 on
# 15 GB+ GPUs, MuseTalk 1.5 everywhere), and all models (~12-15 GB, ~20 GB with LatentSync).
#   bash setup.sh            creates .venv (a normal Linux PC)
#   bash setup.sh --system   installs into the current Python (Colab does this automatically)
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${1:-}" == "--system" || -n "${COLAB_RELEASE_TAG:-}" ]]; then
    PY=python3
else
    [[ -x .venv/bin/python ]] || python3 -m venv .venv
    PY=.venv/bin/python
fi

"$PY" -m pip install -q --upgrade pip
"$PY" -m pip install -q torch==2.8.0 torchaudio==2.8.0 torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cu128
"$PY" -m pip install -q -r requirements.txt -c constraints.txt

# MuseTalk 1.5 code for lip sync (not on PyPI), at the commit this project was built against
SHA=0a89dec45a0192b824e3cf4daf96c239440c5ed8
if [[ ! -d third_party/MuseTalk/musetalk ]]; then
    mkdir -p third_party
    curl -fsSL "https://github.com/TMElyralab/MuseTalk/archive/$SHA.tar.gz" | tar -xz -C third_party
    mv "third_party/MuseTalk-$SHA" third_party/MuseTalk
    echo "$SHA" > third_party/MuseTalk/COMMIT.txt
fi

# LatentSync 1.6, the realistic lip-sync engine, on GPUs with 15 GB+ (Colab's T4 and up). It pins an older
# stack (torch 2.5, numpy 1.26), so it gets its own Python 3.10 environment, built with uv.
LS=third_party/LatentSync
LS_SHA=a229c3948406bc2cf6eaf4873e662e70c6a04746
GPU_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i 0 2>/dev/null | head -n 1 | tr -dc '0-9' || true)
if [[ -f "$LS/.venv/vclone-ready" ]]; then
    echo "LatentSync environment ready"
elif (( ${GPU_MIB:-0} >= 14336 )); then
    echo "Setting up LatentSync 1.6 (its own Python environment, a few minutes)..."
    if [[ ! -d "$LS/latentsync" ]]; then
        rm -rf "$LS"
        mkdir -p third_party
        curl -fsSL "https://github.com/bytedance/LatentSync/archive/$LS_SHA.tar.gz" | tar -xz -C third_party
        mv "third_party/LatentSync-$LS_SHA" "$LS"
        echo "$LS_SHA" > "$LS/COMMIT.txt"
    fi
    "$PY" -m pip install -q uv
    rm -rf "$LS/.venv"
    # uv's own CPython 3.10 includes the headers that InsightFace's C++ extension needs to build
    "$PY" -m uv venv -q --python 3.10 --python-preference only-managed "$LS/.venv"
    LSPY="$LS/.venv/bin/python"
    # InsightFace builds from source against this environment's numpy 1.26 (no build isolation)
    "$PY" -m uv pip install -q --python "$LSPY" numpy==1.26.4 cython setuptools wheel
    "$PY" -m uv pip install -q --python "$LSPY" --index-strategy unsafe-best-match \
        --no-build-isolation-package insightface -r "$LS/requirements.txt"
    touch "$LS/.venv/vclone-ready"
else
    echo "note: no GPU with 15 GB+ here, so LatentSync is skipped; talking videos use MuseTalk"
fi

command -v ffmpeg > /dev/null || echo "note: ffmpeg not found; the bundled imageio-ffmpeg copy will be used"
"$PY" -c "import torch; print('CUDA available:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
PYTHONPATH="$PWD" "$PY" -m vclone.download
chmod +x speak.sh
echo "Done. Try:  ./speak.sh my_voice.wav \"Hello, this is my cloned voice.\""
