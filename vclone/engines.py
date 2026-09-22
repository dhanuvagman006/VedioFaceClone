"""TTS back-ends. Both clone a voice zero-shot from a short reference clip.

qwen : Qwen3-TTS 12Hz Base (Alibaba, Jan 2026). Best accuracy, 10 languages, Apache-2.0.
f5   : F5-TTS v1 Base (flow matching). Very close timbre match, English + Chinese, CC-BY-NC.
"""
from __future__ import annotations

import contextlib
import gc
import io
import re
import sys
import types

import numpy as np
import torch

from .models import F5_REPO, QWEN_BASE, VOCOS_REPO, auto_qwen_size, model_dir
from .quality import spoken_chars

SAMPLE_RATE = 24000


def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@contextlib.contextmanager
def quiet():
    """Swallow the chatty prints some libraries emit while loading."""
    with contextlib.redirect_stdout(io.StringIO()):
        yield


class Engine:
    name = "base"
    sample_rate = SAMPLE_RATE
    batch_size = 4
    normalize_numbers = False  # expand digits to words before synthesis

    def load(self) -> None: ...
    def set_voice(self, audio: np.ndarray, sr: int, text: str, chars_per_sec: float) -> None: ...
    def max_chars(self) -> int: ...
    def generate(self, texts: list[str], seed: int) -> list[np.ndarray]: ...

    def unload(self) -> None:
        for attr in list(vars(self)):
            if attr in ("model", "vocoder", "prompt", "cond"):
                setattr(self, attr, None)
        free_gpu()


class QwenEngine(Engine):
    name = "qwen"
    batch_size = 6  # token-by-token generation is launch-bound on small GPUs: extra takes are nearly free

    def __init__(self, device: str, size: str = "auto", language: str = "auto",
                 xvector_only: bool = False, temperature: float | None = None, top_p: float | None = None):
        self.device, self.language = device, language
        self.size = auto_qwen_size(device) if size == "auto" else size
        self.xvector_only = xvector_only
        self.gen_kwargs = {k: v for k, v in dict(temperature=temperature, top_p=top_p).items() if v is not None}
        self.model = self.prompt = None
        self.cps = 14.0

    def load(self) -> None:
        # The `sox` binding is only used by the 25 Hz tokenizer, not by these 12 Hz models; a stub
        # keeps it from shelling out to a missing sox.exe at import time.
        sys.modules.setdefault("sox", types.ModuleType("sox"))
        with quiet():
            from qwen_tts import Qwen3TTSModel
        from .qwen_fast import patch_code_predictor
        path = model_dir(QWEN_BASE[self.size])
        if self.device.startswith("cuda") and torch.cuda.is_bf16_supported(including_emulation=False):
            dtype = torch.bfloat16
        else:  # e.g. Colab's T4 has no bfloat16, and Qwen can overflow in float16
            dtype = torch.float32
        with quiet():
            self.model = Qwen3TTSModel.from_pretrained(str(path), device_map=self.device, dtype=dtype,
                                                       attn_implementation="sdpa")
        self.accel = patch_code_predictor(self.model)
        # Decode takes one at a time: the codec decoder is the VRAM peak for a batch.
        tokenizer = self.model.model.speech_tokenizer
        batch_decode = tokenizer.decode

        def decode_each(encoded):
            if not isinstance(encoded, list) or len(encoded) < 2:
                return batch_decode(encoded)
            wavs, fs = [], None
            for item in encoded:
                w, fs = batch_decode([item])
                wavs.extend(w)
            return wavs, fs
        tokenizer.decode = decode_each

    def set_voice(self, audio, sr, text, chars_per_sec) -> None:
        self.cps = chars_per_sec
        self.prompt = self.model.create_voice_clone_prompt(
            ref_audio=(audio.astype(np.float32), sr), ref_text=text, x_vector_only_mode=self.xvector_only)

    def max_chars(self) -> int:
        return 220

    def generate(self, texts, seed):
        torch.manual_seed(seed)
        if self.language == "english":  # count digits as the words that will be spoken
            from .text import normalize_english
            texts_spoken = [normalize_english(t) for t in texts]
        else:
            texts_spoken = texts
        longest = max(spoken_chars(t) for t in texts_spoken)
        # 12.5 codec frames per second; allow twice your normal pace plus 4 s before cutting off.
        max_tokens = int(12.5 * (2.0 * longest / self.cps + 4.0))
        wavs, _ = self.model.generate_voice_clone(
            text=list(texts), language=self.language, voice_clone_prompt=self.prompt,
            max_new_tokens=max(64, max_tokens), **self.gen_kwargs)
        return [np.asarray(w, dtype=np.float32).reshape(-1) for w in wavs]


class F5Engine(Engine):
    name = "f5"
    batch_size = 3
    normalize_numbers = True

    def __init__(self, device: str, steps: int = 32, cfg_strength: float = 2.0,
                 sway: float = -1.0, speed: float = 1.0):
        self.device, self.steps, self.cfg_strength, self.sway, self.speed = device, steps, cfg_strength, sway, speed
        self.model = self.vocoder = self.cond = None

    def load(self) -> None:
        from importlib.resources import files

        from omegaconf import OmegaConf
        with quiet():
            from f5_tts.infer.utils_infer import load_model, load_vocoder
            from f5_tts.model import DiT
        ckpt = model_dir(F5_REPO, allow_patterns=["F5TTS_v1_Base/model_1250000.safetensors",
                                                  "F5TTS_v1_Base/vocab.txt"]) / "F5TTS_v1_Base"
        vocos = model_dir(VOCOS_REPO, allow_patterns=["config.yaml", "pytorch_model.bin"])
        cfg = OmegaConf.load(str(files("f5_tts").joinpath("configs/F5TTS_v1_Base.yaml")))
        with quiet():
            self.vocoder = load_vocoder("vocos", is_local=True, local_path=str(vocos), device=self.device)
            self.model = load_model(DiT, cfg.model.arch, str(ckpt / "model_1250000.safetensors"),
                                    mel_spec_type="vocos", vocab_file=str(ckpt / "vocab.txt"), device=self.device)

    def set_voice(self, audio, sr, text, chars_per_sec) -> None:
        assert sr == SAMPLE_RATE
        wav = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        rms = float(wav.pow(2).mean().sqrt())
        self.gain = min(1.0, rms / 0.1)  # F5 works at RMS 0.1; restore your level afterwards
        self.cond = (wav * (0.1 / max(rms, 1e-6)) if rms < 0.1 else wav).to(self.device)
        text = text.strip()
        self.ref_text = text + (" " if text.endswith((".", "!", "?", "。")) else ". ")
        self.ref_frames = self.cond.shape[-1] // 256
        self.ref_bytes = len(self.ref_text.encode("utf-8"))

    def max_chars(self) -> int:
        # F5 sees reference + new speech together in a window of about 22 s.
        ref_sec = self.cond.shape[-1] / SAMPLE_RATE
        room = max(4.0, 22.0 - ref_sec)
        return int(max(60, min(240, self.ref_bytes / ref_sec * room * self.speed)))

    def generate(self, texts, seed):
        from f5_tts.model.utils import convert_char_to_pinyin
        torch.manual_seed(seed)
        prompts = convert_char_to_pinyin([self.ref_text + t for t in texts])
        durations = []
        for t in texts:
            nbytes = len(t.encode("utf-8"))
            speed = 0.3 if nbytes < 10 else self.speed
            durations.append(self.ref_frames + int(self.ref_frames / self.ref_bytes * nbytes / speed))
        with torch.inference_mode():
            mel, _ = self.model.sample(
                cond=self.cond.expand(len(texts), -1), text=prompts,
                duration=torch.tensor(durations, device=self.device, dtype=torch.long),
                steps=self.steps, cfg_strength=self.cfg_strength, sway_sampling_coef=self.sway)
            out = []
            for i, dur in enumerate(durations):
                gen = mel[i:i + 1, self.ref_frames:dur, :].float().permute(0, 2, 1)
                wav = self.vocoder.decode(gen).squeeze().float().cpu().numpy()
                out.append((wav * self.gain).astype(np.float32))
        return out


QWEN_LANGUAGES = ("auto", "english", "chinese", "japanese", "korean", "german", "french", "russian",
                  "portuguese", "spanish", "italian")


def check_language(engine: str, language: str) -> None:
    """Fail fast, before any model is loaded."""
    if engine == "qwen" and language not in QWEN_LANGUAGES:
        raise ValueError(f"Qwen3-TTS can't speak '{language}'. Supported: {', '.join(QWEN_LANGUAGES)}")
    if engine == "f5" and language not in ("english", "chinese", "auto"):
        raise ValueError(f"F5-TTS only speaks English and Chinese; use --engine qwen for {language}.")


def make_engine(name: str, device: str, **opts) -> Engine:
    if name == "qwen":
        return QwenEngine(device, size=opts.get("qwen_size", "auto"), language=opts.get("language", "auto"),
                          xvector_only=opts.get("xvector_only", False),
                          temperature=opts.get("temperature"), top_p=opts.get("top_p"))
    if name == "f5":
        return F5Engine(device, steps=opts.get("steps", 32), speed=opts.get("speed", 1.0),
                        cfg_strength=opts.get("cfg_strength", 2.0))
    raise ValueError(f"Unknown engine '{name}'")


def language_name(code_or_name: str | None, text: str) -> str:
    """Map --language (or auto-detect from the text) to the name Qwen3-TTS expects."""
    names = {"en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean", "de": "german",
             "fr": "french", "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian"}
    if code_or_name and code_or_name.lower() != "auto":
        key = code_or_name.lower()
        return names.get(key, key)
    if re.search(r"[\u3040-\u30ff]", text):
        return "japanese"
    if re.search(r"[\uac00-\ud7af]", text):
        return "korean"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "chinese"
    if re.search(r"[\u0400-\u04ff]", text):
        return "russian"
    letters = [c for c in text if c.isalpha()]
    if letters and all(c.isascii() for c in letters):
        return "english"
    return "auto"
