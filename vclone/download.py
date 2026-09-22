"""Fetch every model once so later runs work offline:  python -m vclone.download [--all]"""
import sys

from .models import (F5_REPO, LIPSYNC_FILES, MODELS_DIR, QWEN_BASE, SPEAKER_REPO, VOCOS_REPO, WHISPER_REPO,
                     model_dir, setup_env)


def main() -> None:
    setup_env()
    model_dir(QWEN_BASE["0.6b"])
    model_dir(WHISPER_REPO, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    model_dir(SPEAKER_REPO, allow_patterns=["*.json", "*.bin", "*.safetensors"])
    model_dir(F5_REPO, allow_patterns=["F5TTS_v1_Base/model_1250000.safetensors", "F5TTS_v1_Base/vocab.txt"])
    model_dir(VOCOS_REPO, allow_patterns=["config.yaml", "pytorch_model.bin"])
    for repo, files in LIPSYNC_FILES.items():  # talking-video models (MuseTalk 1.5)
        model_dir(repo, allow_patterns=files)
    import face_alignment  # its face detector + 68-point landmark weights land in models\torch
    face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device="cpu", flip_input=False,
                                 compile=False)  # torch.compile needs Triton, which Windows lacks
    if "--all" in sys.argv:
        model_dir(QWEN_BASE["1.7b"])
    total = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    print(f"All models ready in {MODELS_DIR} ({total / 2**30:.1f} GB)")


if __name__ == "__main__":
    main()
