"""Audio helpers: decode any format, silence handling, loudness and export."""
from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf

_SNDFILE_EXTS = {".wav", ".flac", ".ogg", ".oga", ".mp3", ".aif", ".aiff"}


def find_ffmpeg() -> str | None:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def resample(audio: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return audio.astype(np.float32)
    import soxr
    return soxr.resample(audio.astype(np.float32), sr_in, sr_out, quality="VHQ").astype(np.float32)


def load_audio(path, sr: int | None = None) -> tuple[np.ndarray, int]:
    """Decode any audio/video file to mono float32, optionally resampled to `sr`."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")
    audio, file_sr = None, None
    if path.suffix.lower() in _SNDFILE_EXTS:
        try:
            data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
            audio = data.mean(axis=1)
        except Exception:
            audio = None
    if audio is None:  # m4a, aac, mp4, opus, webm, wma, amr, ... -> ffmpeg
        exe = find_ffmpeg()
        if exe is None:
            raise RuntimeError(f"Can't decode '{path.name}' without ffmpeg; convert it to WAV first.")
        file_sr = 48000
        proc = subprocess.run(
            [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-i", str(path),
             "-vn", "-ac", "1", "-ar", str(file_sr), "-f", "f32le", "-"],
            capture_output=True)
        if proc.returncode != 0 or not proc.stdout:
            detail = proc.stderr.decode(errors="ignore").strip()[-600:]
            raise RuntimeError(f"ffmpeg could not decode '{path}': {detail}")
        audio = np.frombuffer(proc.stdout, dtype=np.float32).copy()
    audio = np.nan_to_num(audio.astype(np.float32))
    if sr and sr != file_sr:
        return resample(audio, file_sr, sr), sr
    return audio, file_sr


# ----------------------------------------------------------------------------- silence

def frame_db(audio: np.ndarray, sr: int, hop_s: float = 0.01, win_s: float = 0.03):
    """Short-time RMS level in dBFS. Returns (levels, hop, win) in samples."""
    hop, win = max(1, int(sr * hop_s)), max(1, int(sr * win_s))
    if len(audio) < win:
        audio = np.pad(audio, (0, win - len(audio)))
    csum = np.concatenate([[0.0], np.cumsum(audio.astype(np.float64) ** 2)])
    starts = np.arange(0, len(audio) - win + 1, hop)
    energy = (csum[starts + win] - csum[starts]) / win
    return 10 * np.log10(energy + 1e-12), hop, win


def speech_mask(db: np.ndarray) -> np.ndarray:
    """True for frames that contain speech, with a threshold adapted to the noise floor."""
    loud, floor = np.percentile(db, 95), np.percentile(db, 10)
    threshold = min(max(floor + 10.0, loud - 45.0), loud - 20.0)
    return db > threshold


def silent_runs(audio: np.ndarray, sr: int, min_len: float = 0.2) -> list[tuple[int, int]]:
    """(start, end) sample ranges of pauses lasting at least `min_len` seconds."""
    db, hop, win = frame_db(audio, sr)
    mask = speech_mask(db).astype(np.int8)
    edges = np.diff(np.concatenate([[1], mask, [1]]))
    runs = []
    for s, e in zip(np.where(edges == -1)[0], np.where(edges == 1)[0]):
        if (e - s) * hop >= min_len * sr:
            runs.append((int(s * hop + win // 2) if s > 0 else 0,
                         int(min(len(audio), e * hop + win // 2)) if e < len(mask) else len(audio)))
    return runs


def trim_silence(audio: np.ndarray, sr: int, pad: float = 0.06) -> np.ndarray:
    db, hop, win = frame_db(audio, sr)
    idx = np.where(speech_mask(db))[0]
    if len(idx) == 0:
        return audio
    start = max(0, idx[0] * hop - int(pad * sr))
    end = min(len(audio), idx[-1] * hop + win + int(pad * sr))
    return audio[start:end]


def crossfade_join(pieces: list[np.ndarray], sr: int, fade_s: float = 0.01) -> np.ndarray:
    pieces = [p for p in pieces if len(p)]
    if not pieces:
        return np.zeros(0, np.float32)
    out = pieces[0].astype(np.float32)
    for p in pieces[1:]:
        n = min(int(fade_s * sr), len(out), len(p))
        if n == 0:
            out = np.concatenate([out, p])
            continue
        ramp = np.linspace(0.0, 1.0, n, dtype=np.float32)
        mixed = out[-n:] * (1 - ramp) + p[:n] * ramp
        out = np.concatenate([out[:-n], mixed, p[n:]])
    return out


def shorten_pauses(audio: np.ndarray, sr: int, max_pause: float = 0.5, keep: float = 0.3) -> np.ndarray:
    """Cut internal pauses longer than `max_pause` down to `keep` seconds."""
    runs = [(s, e) for s, e in silent_runs(audio, sr, min_len=max_pause) if s > 0 and e < len(audio)]
    if not runs:
        return audio
    half = int(keep * sr / 2)
    pieces, cursor = [], 0
    for s, e in runs:
        pieces.append(audio[cursor:s + half])
        cursor = e - half
    pieces.append(audio[cursor:])
    return crossfade_join(pieces, sr)


def fade_edges(audio: np.ndarray, sr: int, fade_s: float = 0.008) -> np.ndarray:
    n = min(len(audio) // 2, int(fade_s * sr))
    if n <= 0:
        return audio
    ramp = np.sin(np.linspace(0, np.pi / 2, n, dtype=np.float32)) ** 2
    audio = audio.astype(np.float32).copy()
    audio[:n] *= ramp
    audio[-n:] *= ramp[::-1]
    return audio


# ----------------------------------------------------------------------------- filters / loudness

def highpass(audio: np.ndarray, sr: int, cutoff: float = 70.0) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    sos = butter(4, cutoff, btype="highpass", fs=sr, output="sos")
    return sosfiltfilt(sos, audio).astype(np.float32)


def limit_peaks(audio: np.ndarray, sr: int, ceiling_db: float = -1.0) -> np.ndarray:
    """Smooth look-ahead peak limiter: output never exceeds the ceiling."""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    ceiling = 10 ** (ceiling_db / 20)
    if np.max(np.abs(audio), initial=0.0) <= ceiling:
        return audio
    need = np.minimum(1.0, ceiling / np.maximum(np.abs(audio), 1e-9))
    w = max(2, int(0.01 * sr))
    gain = minimum_filter1d(need, size=2 * w + 1, mode="nearest")
    gain = uniform_filter1d(gain, size=2 * (w // 2) + 1, mode="nearest")
    return np.clip(audio * gain, -ceiling, ceiling).astype(np.float32)


def normalize_loudness(audio: np.ndarray, sr: int, target_lufs: float = -18.0,
                       ceiling_db: float = -1.0) -> np.ndarray:
    if not len(audio) or np.max(np.abs(audio)) < 1e-6:
        return audio
    loudness = None
    if len(audio) >= int(0.5 * sr):
        try:
            import pyloudnorm as pyln
            loudness = pyln.Meter(sr).integrated_loudness(audio.astype(np.float64))
        except Exception:
            loudness = None
    if loudness is None or not np.isfinite(loudness):
        loudness = 20 * np.log10(np.sqrt(np.mean(audio.astype(np.float64) ** 2)) + 1e-9)
    gain = 10 ** ((target_lufs - loudness) / 20)
    return limit_peaks((audio * gain).astype(np.float32), sr, ceiling_db)


# ----------------------------------------------------------------------------- saving

_FFMPEG_CODECS = {
    ".mp3": ["-c:a", "libmp3lame", "-b:a", "256k"],
    ".m4a": ["-c:a", "aac", "-b:a", "256k"],
    ".aac": ["-c:a", "aac", "-b:a", "256k"],
    ".ogg": ["-c:a", "libvorbis", "-q:a", "8"],
    ".opus": ["-c:a", "libopus", "-b:a", "160k"],
}


def save_audio(path, audio: np.ndarray, sr: int) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.clip(audio, -1.0, 1.0).astype(np.float32)
    ext = path.suffix.lower()
    if ext in (".wav", ".flac"):
        sf.write(str(path), audio, sr, subtype="PCM_24" if ext == ".flac" else "PCM_16")
        return path
    exe = find_ffmpeg()
    if exe is None:
        raise RuntimeError(f"Writing {ext} needs ffmpeg; use a .wav output instead.")
    with tempfile.TemporaryDirectory() as tmp:
        wav = Path(tmp) / "out.wav"
        sf.write(str(wav), audio, sr, subtype="PCM_16")
        proc = subprocess.run(
            [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(wav),
             *_FFMPEG_CODECS.get(ext, []), str(path)],
            capture_output=True)
        if proc.returncode != 0:
            raise RuntimeError(f"ffmpeg failed to write {path}: {proc.stderr.decode(errors='ignore')[-600:]}")
    return path
