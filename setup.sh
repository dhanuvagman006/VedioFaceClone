#!/usr/bin/env bash
# One-time setup on Linux or Google Colab: CUDA PyTorch, both TTS engines, lip sync, and all models (~12 GB).
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

command -v ffmpeg > /dev/null || echo "note: ffmpeg not found; the bundled imageio-ffmpeg copy will be used"
"$PY" -c "import torch; print('CUDA available:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
PYTHONPATH="$PWD" "$PY" -m vclone.download
chmod +x speak.sh
echo "Done. Try:  ./speak.sh my_voice.wav \"Hello, this is my cloned voice.\""
