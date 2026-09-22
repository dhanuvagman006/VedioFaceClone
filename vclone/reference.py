"""Turn any recording of your voice into a clean reference clip plus transcript (cached)."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import audio as A
from .models import COMMAND, ROOT
from .quality import WHISPER_LANG, spoken_chars

SUPPORTED_CODES = set(WHISPER_LANG.values())  # the ten languages Qwen3-TTS speaks

SR = 24000
VOICES_DIR = ROOT / "voices"
PREP_VERSION = 2

# Read aloud once (about 10 s) so the transcript is exact: `speak.bat --script` / `./speak.sh --script` prints it.
ENROLL_SCRIPT = ("Hi, this is my real voice. I'm speaking naturally, the way I usually talk with my friends. "
                 "The quick brown fox jumps over the lazy dog.")


@dataclass
class Reference:
    audio: np.ndarray  # processed clip, 24 kHz mono
    sr: int
    text: str
    folder: Path
    language: str | None = None  # Whisper code of the language spoken in the clip

    @property
    def text_usable(self) -> bool:
        """Qwen can only follow a reference transcript in one of its own languages."""
        return self.language is None or self.language in SUPPORTED_CODES

    @property
    def duration(self) -> float:
        return len(self.audio) / self.sr

    @property
    def chars_per_sec(self) -> float:
        speech = max(1.0, len(A.trim_silence(self.audio, self.sr, pad=0.0)) / self.sr)
        return float(np.clip(spoken_chars(self.text) / speech, 6.0, 30.0))


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _pick_segment(audio: np.ndarray, sr: int, min_len: float, max_len: float) -> tuple[int, int]:
    """Choose the best min_len..max_len window that starts and ends in pauses."""
    target = min(max_len, max(min_len, 10.0))
    runs = A.silent_runs(audio, sr, min_len=0.12)
    cuts = sorted({0, len(audio), *[(s + e) // 2 for s, e in runs]})
    pause = {0: 0.6, len(audio): 0.6, **{(s + e) // 2: (e - s) / sr for s, e in runs}}
    db, hop, _ = A.frame_db(audio, sr)
    speech = A.speech_mask(db)
    best, best_score = None, -np.inf
    for i, start in enumerate(cuts):
        for end in cuts[i + 1:]:
            dur = (end - start) / sr
            if dur > max_len:
                break
            if dur < min_len:
                continue
            frames = slice(start // hop, max(start // hop + 1, end // hop))
            mask = speech[frames]
            levels = db[frames][mask]
            steadiness = float(np.std(levels)) if len(levels) > 10 else 20.0
            clipped = float(np.mean(np.abs(audio[start:end]) > 0.98))
            score = (float(mask.mean()) - abs(dur - target) / target - steadiness / 25.0 - 50.0 * clipped
                     + 0.5 * min(pause[end], 0.6) + 0.3 * min(pause[start], 0.6))
            if score > best_score:
                best, best_score = (start, end), score
    if best is None:  # no usable pauses: cut at the quietest moment before max_len
        lo, hi = int(min_len * sr) // hop, min(len(db), int(max_len * sr) // hop)
        end = (lo + int(np.argmin(db[lo:hi]))) * hop if hi > lo else int(max_len * sr)
        best = (0, min(len(audio), end))
    return best


def language_code(name: str | None) -> str | None:
    """'english' / 'English' / 'en' -> 'en' (None for auto)."""
    if not name or name.lower() == "auto":
        return None
    from transformers.models.whisper.tokenization_whisper import LANGUAGES, TO_LANGUAGE_CODE
    key = name.lower()
    if key in LANGUAGES:
        return key
    if key in TO_LANGUAGE_CODE:
        return TO_LANGUAGE_CODE[key]
    raise ValueError(f"Unknown language '{name}' for --ref-language")


def _language_label(code: str) -> str:
    from transformers.models.whisper.tokenization_whisper import LANGUAGES
    return LANGUAGES.get(code, code).title()


def _choose_language(probs: dict[str, float], log) -> str:
    """Accents and noise often fool Whisper's language guess (an Indian-English clip once came back
    89% 'Icelandic'), so prefer the most likely language the voice model can actually speak."""
    top = max(probs, key=probs.get)
    if top in SUPPORTED_CODES:
        return top
    best = max(SUPPORTED_CODES & probs.keys(), key=probs.get)
    if probs[top] >= 0.8 and probs[best] < 0.05:
        log(f"[voice] your clip sounds like {_language_label(top)} ({probs[top]:.0%}), which the voice model "
            "can't read; it will clone from your voice's timbre only")
        return top
    log(f"[voice] Whisper's first guess was {_language_label(top)} ({probs[top]:.0%}); using "
        f"{_language_label(best)} ({probs[best]:.0%}) instead. Force it with --ref-language if that's wrong.")
    return best


def _norm_word(word: str) -> str:
    return re.sub(r"[^\w]", "", word.lower())


def _match_script(script: str, heard: str) -> str | None:
    """The words of `script` that the clip covers, located through what Whisper heard.
    Returns None when the clip doesn't look like a reading of the script."""
    import difflib
    words, heard_words = script.split(), [_norm_word(w) for w in heard.split()]
    if not heard_words:
        return None
    matcher = difflib.SequenceMatcher(None, [_norm_word(w) for w in words], heard_words, autojunk=False)
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if sum(b.size for b in blocks) < 0.6 * len(heard_words):
        return None
    first, last = blocks[0], blocks[-1]
    start = max(0, first.a - first.b)  # extend over words Whisper misheard at either edge
    end = min(len(words), last.a + last.size + (len(heard_words) - last.b - last.size))
    return " ".join(words[start:end])


def prepare_reference(src: str | Path, *, transcriber_factory, ref_text: str | None = None,
                      language: str | None = None, max_len: float = 12.0, min_len: float = 5.0,
                      refresh: bool = False, log=print) -> Reference:
    """`language` forces the clip's language (Whisper code); `ref_text` is what was said."""
    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Reference audio not found: {src}")
    known = " ".join(ref_text.split()) if ref_text else None
    settings = f"{_file_hash(src)}|v{PREP_VERSION}|{max_len}|{known or ''}|{language or ''}"
    key = hashlib.sha1(settings.encode()).hexdigest()[:10]
    stem = re.sub(r"[^\w\-]+", "_", src.stem)[:40] or "voice"
    folder = VOICES_DIR / f"{stem}-{key}"
    wav_path, txt_path, info_path = folder / "ref.wav", folder / "ref.txt", folder / "info.json"

    if not refresh and wav_path.exists() and txt_path.exists():
        audio, _ = A.load_audio(wav_path, SR)
        info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}
        log(f"[voice] using cached reference {folder.name} ({len(audio) / SR:.1f}s)")
        return Reference(audio, SR, txt_path.read_text(encoding="utf-8").strip(), folder, info.get("language"))

    log(f"[voice] preparing reference from {src.name}")
    raw, _ = A.load_audio(src, SR)
    if len(raw) < SR:
        raise ValueError("Reference audio is shorter than 1 second; record 10-20 s of natural speech.")
    audio = A.highpass(raw, SR, 70.0)
    db, _, _ = A.frame_db(audio, SR)
    snr = float(np.percentile(db, 95) - np.percentile(db, 10))
    audio = A.trim_silence(audio, SR, pad=0.08)
    audio = A.shorten_pauses(audio, SR, max_pause=0.5, keep=0.3)
    total = len(audio) / SR

    segment = (0.0, total)
    whole = bool(known) and total <= 15.0  # a short recording with known words is used as is
    if not whole and total > max_len:
        s, e = _pick_segment(audio, SR, min_len, max_len)
        segment = (s / SR, e / SR)
        audio = A.trim_silence(audio[s:e], SR, pad=0.06)
        log(f"[voice] picked the cleanest {len(audio) / SR:.1f}s from {total:.1f}s of speech "
            f"({segment[0]:.1f}s-{segment[1]:.1f}s)")
    if len(audio) / SR < 3.0:
        log("[voice] warning: under 3 s of speech - similarity will be weaker; 8-12 s is ideal.")
    if snr < 30:
        log(f"[voice] note: background noise is audible (SNR ~{snr:.0f} dB). A quieter recording "
            "gives a cleaner clone.")

    audio = A.normalize_loudness(audio, SR, target_lufs=-20.0, ceiling_db=-1.5)
    audio = np.concatenate([np.zeros(int(0.1 * SR), np.float32), audio, np.zeros(int(0.25 * SR), np.float32)])

    log("[voice] transcribing your clip with Whisper (one time)...")
    transcriber = transcriber_factory()
    try:
        probs = transcriber.language_probs(audio, SR)
        code = language or _choose_language(probs, log)
        heard, confidence = transcriber.transcribe_scored(audio, SR, language=code)
    finally:
        del transcriber
    heard = " ".join(heard.split())

    script = _match_script(known or ENROLL_SCRIPT, heard)
    if known and whole:
        text, source = known, "--ref-text"
        if script is None:
            log(f'[voice] warning: Whisper heard "{heard}", which differs from --ref-text; double-check it.')
    elif script:
        text, source = script, "--ref-text" if known else "the enrollment script"
        log(f"[voice] matched {source}: using its exact words instead of Whisper's guess")
    else:
        text, source = heard, "whisper"
        if known:
            log("[voice] warning: couldn't find --ref-text in the chosen clip; using Whisper's transcript.")
        elif confidence < -0.25 or code != max(probs, key=probs.get):
            # Calibrated: a clean studio clip scores about -0.14; a noisy accented clip with wrong words -0.31.
            log("[voice] warning: Whisper wasn't sure about these words, and a wrong transcript makes the "
                f"clone sound unlike you. Check ref.txt, or record the script from `{COMMAND} --script`.")
    if spoken_chars(text) < 3:
        raise ValueError("Whisper heard no words in the reference clip. Is it speech?")

    folder.mkdir(parents=True, exist_ok=True)
    A.save_audio(wav_path, audio, SR)
    txt_path.write_text(text + "\n", encoding="utf-8")
    top3 = dict(sorted(probs.items(), key=lambda kv: -kv[1])[:3])
    info_path.write_text(json.dumps({
        "source": str(src), "segment_seconds": [round(segment[0], 2), round(segment[1], 2)],
        "duration": round(len(audio) / SR, 2), "language": code,
        "language_guess": {k: round(v, 3) for k, v in top3.items()},
        "transcript_source": source, "whisper_heard": heard, "whisper_confidence": round(confidence, 3),
        "snr_db": round(snr, 1), "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f'[voice] transcript ({_language_label(code)}): "{text}"')
    log(f"[voice] saved to {folder} (edit ref.txt if the transcript is wrong)")
    return Reference(audio, SR, text, folder, code)
