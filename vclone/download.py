"""Fetch every model once so later runs work offline:  python -m vclone.download [--all]"""
import sys

from .models import (F5_REPO, GFPGAN_BYTES, GFPGAN_URL, LIPSYNC_FILES, MODELS_DIR, QWEN_BASE, SPEAKER_REPO,
                     VOCOS_REPO, WHISPER_REPO, auto_qwen_size, model_dir, setup_env, url_file)


def main() -> None:
    setup_env()
    # The voice model size this machine will use (1.7b on 10 GB+ GPUs like Colab's T4, else 0.6b).
    sizes = {"0.6b", "1.7b"} if "--all" in sys.argv else {auto_qwen_size()}
    for size in sorted(sizes):
        model_dir(QWEN_BASE[size])
    model_dir(WHISPER_REPO, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    model_dir(SPEAKER_REPO, allow_patterns=["*.json", "*.bin", "*.safetensors"])
    model_dir(F5_REPO, allow_patterns=["F5TTS_v1_Base/model_1250000.safetensors", "F5TTS_v1_Base/vocab.txt"])
    model_dir(VOCOS_REPO, allow_patterns=["config.yaml", "pytorch_model.bin"])
    for repo, files in LIPSYNC_FILES.items():  # talking-video models (MuseTalk 1.5)
        model_dir(repo, allow_patterns=files)
    url_file(GFPGAN_URL, "GFPGAN", GFPGAN_BYTES)  # sharpens the lip-synced mouth
    import face_alignment  # its face detector + 68-point landmark weights land in models\torch
    face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False,
                                 compile=False)  # torch.compile needs Triton, which Windows lacks
    total = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    print(f"All models ready in {MODELS_DIR} ({total / 2**30:.1f} GB)")


if __name__ == "__main__":
    main()
