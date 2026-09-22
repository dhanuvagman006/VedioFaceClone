"""Model locations and one-time downloads (everything lives under D:\\GPU\\models)."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
COMMAND = "speak.bat" if os.name == "nt" else "./speak.sh"  # how users start the tool on this OS

QWEN_BASE = {"0.6b": "Qwen/Qwen3-TTS-12Hz-0.6B-Base", "1.7b": "Qwen/Qwen3-TTS-12Hz-1.7B-Base"}
F5_REPO = "SWivid/F5-TTS"
VOCOS_REPO = "charactr/vocos-mel-24khz"
WHISPER_REPO = "openai/whisper-large-v3-turbo"
SPEAKER_REPO = "microsoft/wavlm-base-plus-sv"

# Lip sync (MuseTalk 1.5). Code is vendored at a pinned commit in third_party\MuseTalk.
MUSETALK_CODE = ROOT / "third_party" / "MuseTalk"
MUSETALK_COMMIT = "0a89dec45a0192b824e3cf4daf96c239440c5ed8"
MUSETALK_REPO = "TMElyralab/MuseTalk"
SD_VAE_REPO = "stabilityai/sd-vae-ft-mse"
WHISPER_TINY_REPO = "openai/whisper-tiny"
FACE_PARSE_REPO = "ManyOtherFunctions/face-parse-bisent"  # the source MuseTalk's own download script uses

# GFPGAN v1.4 face restoration (Apache-2.0), official TencentARC release; sharpens the lip-synced mouth.
GFPGAN_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
GFPGAN_BYTES = 348_632_874

LIPSYNC_FILES = {
    MUSETALK_REPO: ["musetalkV15/unet.pth", "musetalkV15/musetalk.json"],
    SD_VAE_REPO: ["config.json", "diffusion_pytorch_model.safetensors"],
    WHISPER_TINY_REPO: ["config.json", "preprocessor_config.json", "model.safetensors"],
    FACE_PARSE_REPO: ["79999_iter.pth"],  # full BiSeNet incl. its ResNet backbone
}


def setup_env() -> None:
    """Keep every download inside the project folder and silence noisy libraries."""
    os.environ.setdefault("HF_HOME", str(MODELS_DIR / "hf-cache"))
    os.environ.setdefault("TORCH_HOME", str(MODELS_DIR / "torch"))  # face-alignment's detector weights
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("PYTHONWARNINGS", "ignore")
    import warnings
    warnings.filterwarnings("ignore")
    import transformers
    transformers.logging.set_verbosity_error()


def auto_qwen_size(device: str = "cuda:0") -> str:
    """The 1.7B voice model clones more closely but needs ~8 GB in fp32 (e.g. Colab's T4); 4 GB cards get 0.6B."""
    import torch
    if device.startswith("cuda") and torch.cuda.is_available():
        if torch.cuda.get_device_properties(torch.device(device)).total_memory >= 10 * 2**30:
            return "1.7b"
    return "0.6b"


def url_file(url: str, folder: str, expected_bytes: int | None = None) -> Path:
    """A single file downloaded from `url` into models\\<folder>\\ the first time (size-checked)."""
    target = MODELS_DIR / folder / url.rsplit("/", 1)[-1]
    if target.exists() and (expected_bytes is None or target.stat().st_size == expected_bytes):
        return target
    import torch
    target.parent.mkdir(parents=True, exist_ok=True)
    print(f"[models] downloading {url} -> {target} (first run only)", flush=True)
    partial = target.with_name(target.name + ".part")
    torch.hub.download_url_to_file(url, str(partial), progress=True)
    if expected_bytes is not None and partial.stat().st_size != expected_bytes:
        partial.unlink()
        raise RuntimeError(f"Download of {url} is incomplete; run again")
    partial.replace(target)
    return target


def model_dir(repo_id: str, allow_patterns: list[str] | None = None) -> Path:
    """Local folder holding `repo_id`, downloading it from Hugging Face the first time."""
    target = MODELS_DIR / repo_id.split("/")[-1]
    marker = target / ".download-complete"
    if marker.exists():
        return target
    from huggingface_hub import snapshot_download
    print(f"[models] downloading {repo_id} -> {target} (first run only)", flush=True)
    snapshot_download(repo_id, local_dir=str(target), allow_patterns=allow_patterns)
    marker.write_text("ok\n")
    return target
