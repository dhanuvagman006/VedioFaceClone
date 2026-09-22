"""Take selection: Whisper checks the words, WavLM checks that it sounds like you."""
from __future__ import annotations

import math
import re
import unicodedata
from dataclasses import dataclass

import numpy as np
import torch

from .audio import resample
from .models import SPEAKER_REPO, WHISPER_REPO, model_dir

WHISPER_LANG = {"english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko", "german": "de",
                "french": "fr", "russian": "ru", "portuguese": "pt", "spanish": "es", "italian": "it"}


class Transcriber:
    """Whisper large-v3-turbo in fp16 (about 1.6 GB of VRAM)."""

    def __init__(self, device: str):
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        path = model_dir(WHISPER_REPO, allow_patterns=["*.json", "*.safetensors", "*.txt"])
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.processor = WhisperProcessor.from_pretrained(path)
        self.model = WhisperForConditionalGeneration.from_pretrained(
            path, torch_dtype=self.dtype, low_cpu_mem_usage=True).to(device).eval()

    def _features(self, clips, sr):
        batch = [resample(c, sr, 16000) for c in clips]
        feats = self.processor.feature_extractor(batch, sampling_rate=16000, return_tensors="pt")
        return feats.input_features.to(self.device, self.dtype)

    @torch.inference_mode()
    def transcribe(self, clips, sr, language=None, beams=1, batch_size=8) -> list[str]:
        texts = []
        for i in range(0, len(clips), batch_size):
            ids = self.model.generate(self._features(clips[i:i + batch_size], sr), task="transcribe",
                                      language=language, num_beams=beams, max_new_tokens=256)
            texts += [t.strip() for t in self.processor.batch_decode(ids, skip_special_tokens=True)]
        return texts

    @torch.inference_mode()
    def transcribe_scored(self, clip, sr, language=None, beams=5) -> tuple[str, float]:
        """Transcript plus Whisper's mean log-probability per token (above -0.4 is confident)."""
        out = self.model.generate(self._features([clip], sr), task="transcribe", language=language,
                                  num_beams=beams, max_new_tokens=256, return_dict_in_generate=True,
                                  output_scores=True)
        text = self.processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
        scores = getattr(out, "sequences_scores", None)
        return text, float(scores[0]) if scores is not None else 0.0

    @torch.inference_mode()
    def language_probs(self, clip, sr) -> dict[str, float]:
        """Whisper's probability for every language code, e.g. {'en': 0.93, 'hi': 0.04, ...}."""
        sot = self.processor.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")
        logits = self.model(input_features=self._features([clip], sr),
                            decoder_input_ids=torch.tensor([[sot]], device=self.device)).logits[0, -1].float()
        lang_to_id = self.model.generation_config.lang_to_id
        probs = torch.softmax(logits[list(lang_to_id.values())], dim=-1).tolist()
        return {token.strip("<|>"): p for token, p in zip(lang_to_id, probs)}


class SpeakerEncoder:
    """WavLM x-vector speaker-verification model (about 0.4 GB)."""

    def __init__(self, device: str):
        from transformers import AutoFeatureExtractor, WavLMForXVector
        path = model_dir(SPEAKER_REPO, allow_patterns=["*.json", "*.bin", "*.safetensors"])
        self.device = device
        self.extractor = AutoFeatureExtractor.from_pretrained(path)
        self.model = WavLMForXVector.from_pretrained(path).to(device).eval()

    @torch.inference_mode()
    def embed(self, clip: np.ndarray, sr: int) -> torch.Tensor:
        wav = resample(clip, sr, 16000)
        if 0 < len(wav) < 32000:  # x-vector pooling needs ~2 s; loop very short words
            wav = np.tile(wav, int(np.ceil(32000 / len(wav))))
        inputs = self.extractor(wav, sampling_rate=16000, return_tensors="pt")
        emb = self.model(**inputs.to(self.device)).embeddings[0].float()
        return torch.nn.functional.normalize(emb, dim=-1).cpu()


# ----------------------------------------------------------------------------- metrics

def _plain(text: str, hyphen: str = " ") -> str:
    text = unicodedata.normalize("NFKC", text).lower().replace("-", hyphen)
    text = re.sub(r"['’]", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


_CANONICAL = {"dr": "doctor", "mr": "mister", "mrs": "missus", "ms": "miz", "prof": "professor",
              "vs": "versus", "ok": "okay"}


def _spelled(text: str, hyphen: str = " ") -> str:
    from .text import normalize_english
    return " ".join(_CANONICAL.get(w, w) for w in _plain(normalize_english(text), hyphen).split())


def _whisper_style(tokenizer, text: str, hyphen: str = " ") -> str:
    text = tokenizer.normalize(text.replace("-", hyphen))  # contractions, titles, spellings, numbers
    text = re.sub(r"(?<=\d)[.:,](?=\d)", " ", text)  # 6.30 / 6:30 / 3.14 -> digit groups
    text = re.sub(r"\b([12]\d)(\d\d)\b", r"\1 \2", text)  # 2026 and "twenty twenty-six" -> 20 26
    return re.sub(r"\s+", " ", text).strip()


def word_error(target: str, heard: str, english: bool, tokenizer=None) -> float:
    """Error rate of what Whisper heard vs the text. Both sides are normalised several equivalent
    ways and the kindest score wins, so '6:30' vs '6.30', 'Dr.' vs 'Doctor', '2026' vs '20, 26'
    or 'cleanup' vs 'clean-up' never count as mistakes."""
    scores = []
    for hyphen in (" ", ""):
        if not english:
            scores.append(error_rate(_plain(target, hyphen), _plain(heard, hyphen)))
            continue
        scores.append(error_rate(_spelled(target, hyphen), _spelled(heard, hyphen)))
        if tokenizer is not None:
            try:
                scores.append(error_rate(_whisper_style(tokenizer, target, hyphen),
                                         _whisper_style(tokenizer, heard, hyphen)))
            except Exception:
                pass
    return min(scores)


def _tokens(text: str) -> list[str]:
    if re.search(r"[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]", text):
        return [c for c in text if not c.isspace()]  # character error rate for CJK
    return text.split()


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def error_rate(target: str, heard: str) -> float:
    ref, hyp = _tokens(target), _tokens(heard)
    if not ref:
        return 0.0 if not hyp else 1.0
    return _edit_distance(ref, hyp) / len(ref)


def spoken_chars(text: str) -> int:
    return sum(1 for c in text if c.isalnum())


@dataclass
class Take:
    audio: np.ndarray
    seed: int
    heard: str = ""
    wer: float | None = None
    similarity: float | None = None
    pace: float | None = None  # actual / expected duration (1.0 = same pace as your recording)
    score: float = 0.0

    def compute_score(self, pace_weight: float = 2.0) -> float:
        """Voice match minus heavy penalties for misread words and unnatural pacing."""
        if self.similarity is None or not math.isfinite(self.similarity):
            self.similarity = 0.0
        pace_penalty = max(0.0, abs(math.log(self.pace)) - math.log(1.4)) if self.pace and self.pace > 0 else 1.0
        self.score = self.similarity - 3.0 * (self.wer or 0.0) - pace_weight * pace_penalty
        return self.score
