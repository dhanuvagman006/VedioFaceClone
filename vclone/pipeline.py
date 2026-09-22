"""End to end: reference clip -> text chunks -> takes on the GPU -> best take -> mastered file."""
from __future__ import annotations

import random
import re
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from . import audio as A
from . import latentsync
from .engines import SAMPLE_RATE, check_language, free_gpu, language_name, make_engine
from .lipsync import VIDEO_EXTS, is_video, prepare_avatar, render
from .models import COMMAND, ROOT
from .quality import WHISPER_LANG, SpeakerEncoder, Take, Transcriber, spoken_chars, word_error
from .reference import language_code, prepare_reference
from .text import clean_text, normalize_english, plan_chunks

QUALITY_PRESETS = {  # takes per chunk, F5 sampling steps
    "fast": (1, 32),
    "high": (3, 32),
    "max": (5, 64),
}
BAD_WER = 0.2  # a chosen take above this word error rate gets re-generated
GOOD_MATCH = 0.90  # WavLM voice match; good clones scored 0.95-0.98, a mis-transcribed reference ~0.75


def _default_out(text: str, ext: str) -> Path:
    words = "_".join(re.findall(r"\w+", text)[:5])[:40] or "speech"
    return ROOT / "outputs" / f"{time.strftime('%Y%m%d-%H%M%S')}_{words}{ext}"


def _face_source(args) -> Path | None:
    """The video (or photo) to lip-sync: --face, else the recording itself when it is a video.
    An audio file name in --out means audio only."""
    out_ext = Path(args.out).suffix.lower() if args.out else None
    face = args.face or (args.ref if is_video(args.ref) else None)
    if out_ext and out_ext not in VIDEO_EXTS:
        return None
    if face is None and out_ext in VIDEO_EXTS:
        raise ValueError("To make a video, use a video of the person as the recording, or add --face video.mp4")
    if face and not Path(face).is_file():
        raise FileNotFoundError(f"Face video not found: {face}")
    return Path(face) if face else None


def _generate(engine, chunks, todo, takes, n_takes, seed, log):
    # Similar lengths share a batch, so short takes don't idle while a long one finishes.
    order = sorted(todo, key=lambda ci: len(chunks[ci].speak))
    jobs = [(ci, k) for ci in order for k in range(n_takes)]
    done = 0
    for b in range(0, len(jobs), engine.batch_size):
        batch = jobs[b:b + engine.batch_size]
        batch_seed = seed + b
        wavs = engine.generate([chunks[ci].speak for ci, _ in batch], seed=batch_seed)
        for (ci, _), wav in zip(batch, wavs):
            takes[ci].append(Take(audio=wav, seed=batch_seed))
        done += len(batch)
        log(f"[tts] generated {done}/{len(jobs)} takes")


def _judge(chunks, takes, todo, ref, lang, device, log):
    fresh = [(ci, t) for ci in todo for t in takes[ci] if t.wer is None]
    log(f"[check] Whisper is listening to {len(fresh)} take(s)...")
    transcriber = Transcriber(device)
    heard = transcriber.transcribe([t.audio for _, t in fresh], SAMPLE_RATE, language=WHISPER_LANG.get(lang))
    english = lang == "english"
    for (ci, t), h in zip(fresh, heard):
        t.heard = h
        t.wer = word_error(chunks[ci].text, h, english, transcriber.processor.tokenizer)
    del transcriber
    free_gpu()

    encoder = SpeakerEncoder(device)
    ref_emb = encoder.embed(ref.audio, ref.sr)
    for ci, t in fresh:
        long_enough = len(t.audio) > 0.1 * SAMPLE_RATE
        t.similarity = float(encoder.embed(t.audio, SAMPLE_RATE) @ ref_emb) if long_enough else 0.0
        spoken = normalize_english(chunks[ci].text) if english else chunks[ci].text
        expected = spoken_chars(spoken) / ref.chars_per_sec
        actual = len(A.trim_silence(t.audio, SAMPLE_RATE, pad=0.0)) / SAMPLE_RATE
        t.pace = actual / max(0.3, expected)
        t.compute_score(pace_weight=2.0 if expected >= 1.5 else 0.0)  # pace means little for a word or two
    del encoder
    free_gpu()


def _lipsync_engine(args, face) -> str | None:
    """auto: LatentSync where setup.sh installed it (Linux/Colab) and the GPU has 15 GB+, else MuseTalk."""
    if face is None:
        return None
    if args.lipsync == "auto":
        return "latentsync" if latentsync.usable() else "musetalk"
    if args.lipsync == "latentsync" and not latentsync.installed():
        raise ValueError("LatentSync isn't set up here. It runs on Linux/Colab after `bash setup.sh`; "
                         "use --lipsync musetalk on this machine.")
    return args.lipsync


def _device(args, log) -> str:
    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    if device.startswith("cpu"):
        log("[warn] no CUDA GPU found - running on CPU will be slow")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    return device


def _voice_child(options: dict, out: str, wav: str) -> None:
    """Separate process (spawn) that makes the speech for LatentSync. When it exits, all the GPU memory
    the voice models used is truly free again; a live process can keep gigabytes reserved."""
    import argparse
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    from .models import setup_env
    setup_env()
    log = lambda msg: print(msg, flush=True)  # noqa: E731
    try:
        args = argparse.Namespace(**options)
        text = clean_text(args.text)
        speech, _, gen_seconds = _voice(args, _device(args, log), text, language_name(args.language, text),
                                        Path(out), None, log)
        A.save_audio(Path(wav), speech, SAMPLE_RATE)
        log(f"[tts] speech ready: {len(speech) / SAMPLE_RATE:.1f}s, generated in {gen_seconds:.1f}s")
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)


def _voice_in_child(args, out: Path, wav: Path) -> None:
    import multiprocessing
    proc = multiprocessing.get_context("spawn").Process(target=_voice_child, args=(dict(vars(args)), str(out), str(wav)))
    proc.start()
    proc.join()
    if proc.exitcode != 0 or not wav.exists():
        raise RuntimeError("the voice step failed (see the message above)")


def run(args, log=print) -> Path:
    t_start = time.time()
    text = clean_text(args.text)
    if not spoken_chars(text):
        raise ValueError("There is no text to speak.")
    lang = language_name(args.language, text)
    check_language(args.engine, lang)
    face = _face_source(args)
    out = Path(args.out) if args.out else _default_out(args.text, ".mp4" if face else ".wav")
    lipsync = _lipsync_engine(args, face)

    if lipsync == "latentsync":
        log("[face] lip sync: LatentSync 1.6")
        cycle = latentsync.prepare_face(face, refresh=args.refresh_face, log=log)
        # The voice runs in a process of its own and exits before LatentSync starts, so LatentSync
        # (which needs most of a 15 GB T4) gets the whole GPU. This process never touches the GPU.
        with tempfile.TemporaryDirectory(prefix="vclone_") as tmp:
            wav = Path(tmp) / "speech.wav"
            _voice_in_child(args, out, wav)
            latentsync.render(cycle, wav, out, steps=args.lipsync_steps,
                              seed=args.seed if args.seed is not None else 1247, log=log)
            duration = sf.info(str(wav)).duration
        log(f"[out] {out}  ({duration:.1f}s video, total {time.time() - t_start:.0f}s)")
        return out
    if lipsync == "musetalk":
        why = ""
        if args.lipsync == "auto":
            why = (" (LatentSync, the more realistic engine, needs a 15 GB+ GPU)" if latentsync.installed() else
                   " (LatentSync, the more realistic engine, is set up by setup.sh on Linux/Colab)")
        log(f"[face] lip sync: MuseTalk 1.5{why}")

    device = _device(args, log)
    speech, avatar, gen_seconds = _voice(args, device, text, lang, out, face, log)
    out_sr = SAMPLE_RATE
    if avatar:
        with tempfile.TemporaryDirectory() as tmp:
            wav = A.save_audio(Path(tmp) / "speech.wav", speech, SAMPLE_RATE)
            render(avatar, wav, out, device, restore=0.8 if args.restore is None else args.restore, log=log)
    else:
        out_sr = args.sample_rate or SAMPLE_RATE
        if out_sr != SAMPLE_RATE:
            speech = A.resample(speech, SAMPLE_RATE, out_sr)
        A.save_audio(out, speech, out_sr)
    log(f"[out] {out}  ({len(speech) / out_sr:.1f}s of audio, generated in {gen_seconds:.1f}s, "
        f"total {time.time() - t_start:.1f}s)")
    return out


def _voice(args, device: str, text: str, lang: str, out: Path, face: Path | None, log):
    """Reference clip -> TTS takes -> Whisper and voice checks -> mastered 24 kHz speech.
    With `face` (MuseTalk) the face video is analysed right after the voice clip, so a video without a
    usable face fails before the slow voice work. Returns (speech, avatar or None, generation seconds)."""
    n_takes, f5_steps = QUALITY_PRESETS[args.quality]
    n_takes = max(1, args.takes or n_takes)
    judging = n_takes > 1 or args.check

    ref = prepare_reference(args.ref, transcriber_factory=lambda: Transcriber(device),
                            ref_text=args.ref_text, language=language_code(args.ref_language),
                            refresh=args.refresh_voice, log=log)
    free_gpu()
    avatar = prepare_avatar(face, device, refresh=args.refresh_face, log=log) if face else None
    xvector_only = args.xvector_only
    if not ref.text_usable:
        if args.engine == "f5":
            raise ValueError("F5-TTS needs a reference recording in English or Chinese; use --engine qwen.")
        xvector_only = True  # Qwen can't follow a transcript in this language: clone the timbre only

    engine = make_engine(args.engine, device, language=lang, qwen_size=args.qwen_size,
                         xvector_only=xvector_only, temperature=args.temperature, top_p=args.top_p,
                         steps=args.steps or f5_steps, speed=args.speed)
    spell_out = engine.normalize_numbers and lang == "english"
    model = f"qwen {engine.size}" if args.engine == "qwen" else args.engine
    log(f"[tts] loading {model} on {device} (language: {lang})")
    engine.load()
    if args.verbose and getattr(engine, "accel", None):
        log(f"[tts] code predictor speed-up: {engine.accel}")
    engine.set_voice(ref.audio, ref.sr, ref.text, ref.chars_per_sec)
    max_chars = int(engine.max_chars() * (0.85 if spell_out else 1.0))  # spelled-out numbers are longer
    chunks = plan_chunks(text, max_chars, sentence_pause=args.pause, paragraph_pause=args.paragraph_pause)
    if spell_out:
        for chunk in chunks:
            chunk.speak = normalize_english(chunk.text)
    seed = args.seed if args.seed is not None else random.randint(0, 2**31 - 1)
    log(f"[tts] {len(chunks)} chunk(s) x {n_takes} take(s), seed {seed}")

    takes: list[list[Take]] = [[] for _ in chunks]
    todo = list(range(len(chunks)))
    t_gen = time.time()
    _generate(engine, chunks, todo, takes, n_takes, seed, log)
    gen_seconds = time.time() - t_gen

    for attempt in range(args.retries + 1 if judging else 0):
        engine.unload()
        _judge(chunks, takes, todo, ref, lang, device, log)
        todo = [ci for ci in todo if max(takes[ci], key=lambda t: t.score).wer > BAD_WER]
        if not todo or attempt == args.retries:
            break
        log(f"[check] {len(todo)} chunk(s) still had misread words - generating more takes")
        engine.load()
        engine.set_voice(ref.audio, ref.sr, ref.text, ref.chars_per_sec)
        _generate(engine, chunks, todo, takes, n_takes, seed + 7919 * (attempt + 1), log)
    engine.unload()
    del engine
    free_gpu()

    chosen = [max(ts, key=lambda t: t.score) if judging else ts[0] for ts in takes]
    pieces = []
    for chunk, take in zip(chunks, chosen):
        pieces.append(A.fade_edges(A.trim_silence(take.audio, SAMPLE_RATE, pad=0.05), SAMPLE_RATE))
        if chunk.pause_after > 0:
            pieces.append(np.zeros(int(chunk.pause_after * SAMPLE_RATE), np.float32))
    speech = np.concatenate(pieces)
    speech = A.normalize_loudness(speech, SAMPLE_RATE, target_lufs=args.loudness, ceiling_db=-1.0)

    if args.save_takes:
        folder = out.with_name(out.stem + "_takes")
        for ci, ts in enumerate(takes):
            for k, t in enumerate(ts):
                tag = "_BEST" if t is chosen[ci] else ""
                A.save_audio(folder / f"chunk{ci + 1:03d}_take{k + 1}{tag}.wav", t.audio, SAMPLE_RATE)
        log(f"[out] all takes saved in {folder}")

    if judging and args.verbose:
        for ci, ts in enumerate(takes):
            for k, t in enumerate(ts, 1):
                mark = "*" if t is chosen[ci] else " "
                log(f"  {mark} chunk {ci + 1} take {k}: score {t.score:+.3f}  word error {t.wer:.0%}  "
                    f"voice match {t.similarity:.3f}  pace {t.pace:.2f}x  heard: {t.heard}")
    if judging:
        for ci, (chunk, t) in enumerate(zip(chunks, chosen), 1):
            flag = "  <-- check this part" if t.wer > BAD_WER else ""
            if len(chunks) <= 8 or flag:
                log(f"[check] chunk {ci}: word error {t.wer:.0%}, voice match {t.similarity:.3f}, "
                    f"pace {t.pace:.2f}x{flag}" + (f'\n        "{chunk.text}"' if flag else ""))
        if len(chunks) > 8:
            log(f"[check] {len(chunks)} chunks: average word error {np.mean([t.wer for t in chosen]):.0%}, "
                f"average voice match {np.mean([t.similarity for t in chosen]):.3f}")
        match = float(np.mean([t.similarity for t in chosen]))
        if match < GOOD_MATCH:
            log(f"[check] voice match {match:.2f} is weak (your own voice scores ~0.96 against itself). For a "
                f"closer clone, record 10-20 s in a quiet room reading `{COMMAND} --script`, or use --quality max.")
    return speech, avatar, gen_seconds
