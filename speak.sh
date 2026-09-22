#!/usr/bin/env bash
# Read text aloud in a cloned voice (Linux, macOS, Google Colab).
#   ./speak.sh my_voice.wav "Text to read" [options]      ./speak.sh person.mp4 -f script.txt -o talking.mp4
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="$HERE/.venv/bin/python"
[ -x "$PY" ] || PY="python3"  # Colab: packages live in the system Python
PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m vclone "$@"
