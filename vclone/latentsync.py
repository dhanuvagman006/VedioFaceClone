"""Talking video with LatentSync 1.6 (ByteDance): the most realistic open lip-sync model we can run.

It generates the lower face at 512x512 with a 16-frame temporal model (no frame-to-frame jitter) and was
trained against a lip-sync expert network, so it is sharper, steadier and more accurate than MuseTalk.
It needs a Colab-class GPU and pins an older stack (torch 2.5, diffusers 0.32, numpy 1.26, InsightFace),
so it lives in its own Python 3.10 environment in third_party/LatentSync/.venv (setup.sh builds it on
Linux/Colab) and runs as a subprocess (latentsync_runner.py). Its face detector, InsightFace, ships
models licensed for non-commercial use only."""
from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import soundfile as sf

from . import audio as A
from .lipsync import AI_TAG, AVATARS_DIR, FPS, _file_hash
from .models import LATENTSYNC_CODE, LATENTSYNC_FILES, LATENTSYNC_REPO, LIPSYNC_FILES, SD_VAE_REPO, model_dir

MIN_GPU_GIB = 14      # auto mode needs Colab's T4 (15 GB) or bigger; smaller cards use MuseTalk
MAX_SIDE = 1280       # 720p: the face comes out near the model's 512 px, so the new mouth is as sharp as the frame
MAX_SECONDS = 60      # face footage used; shorter videos play forward then backward
SEGMENT_SECONDS = 30  # LatentSync keeps every frame in RAM, so longer scripts are rendered in 30 s pieces
RUNNER = Path(__file__).with_name("latentsync_runner.py")
READY = LATENTSYNC_CODE / ".venv" / "vclone-ready"  # setup.sh writes it once the environment is complete
# The face video is cached near-losslessly, and temporary copies even more so: LatentSync re-encodes once
# more (crf 13), and every lossy generation would soften the real footage around the new mouth.
CACHE_X264 = ["-c:v", "libx264", "-preset", "fast", "-crf", "12", "-pix_fmt", "yuv420p"]
TEMP_X264 = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "10", "-pix_fmt", "yuv420p"]


def python() -> Path:
    return LATENTSYNC_CODE / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def installed() -> bool:
    return (LATENTSYNC_CODE / "latentsync").is_dir() and python().exists() and READY.exists()


def gpu_gib() -> float | None:
    """Memory of the first GPU. Read with nvidia-smi, so this process never starts CUDA: LatentSync, in its
    own process, gets the whole GPU."""
    try:
        proc = subprocess.run(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits", "-i", "0"],
                              capture_output=True, text=True, timeout=60)
        return float(proc.stdout.split()[0]) / 1024
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


def usable() -> bool:
    """Whether auto mode picks LatentSync: installed, with a GPU big enough for it."""
    return installed() and (gpu_gib() or 0) >= MIN_GPU_GIB


def _ffmpeg(*args, what: str) -> None:
    exe = A.find_ffmpeg()
    if exe is None:
        raise RuntimeError("ffmpeg is needed for talking videos")
    proc = subprocess.run([exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", *map(str, args)],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg could not {what}: {proc.stderr.decode(errors='ignore')[-500:]}")


def prepare_face(src: str | Path, *, refresh: bool = False, log=print) -> Path:
    """The face video as upright 25 fps 720p, played forward and then backward, so looping it to any
    length never jumps. Cached in avatars/; runs before the voice so a bad video fails early."""
    src = Path(src).expanduser().resolve()
    key = hashlib.sha1(f"{_file_hash(src)}|latentsync2|{MAX_SIDE}|{MAX_SECONDS}".encode()).hexdigest()[:10]
    stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in src.stem)[:40]
    folder = AVATARS_DIR / f"{stem}-{key}"
    cycle = folder / "latentsync_cycle.mp4"
    if cycle.exists() and not refresh:
        log(f"[face] using cached face video {folder.name}")
        return cycle
    log(f"[face] preparing {src.name} (one time; cached in {folder.name})")
    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "latentsync_cycle.tmp.mp4"
    graph = (f"[0:v]fps={FPS},scale='min({MAX_SIDE},iw)':'min({MAX_SIDE},ih)':force_original_aspect_ratio=decrease,"
             "scale=trunc(iw/2)*2:trunc(ih/2)*2,format=yuv420p,split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0[v]")
    _ffmpeg("-t", MAX_SECONDS, "-i", src, "-filter_complex", graph, "-map", "[v]", "-an", *CACHE_X264, tmp,
            what=f"read the face video {src.name}")
    tmp.replace(cycle)
    return cycle


def _link_checkpoints(weights: Path) -> None:
    """LatentSync reads checkpoints/ relative to its own folder; point that at models/LatentSync-1.6.
    (InsightFace then also downloads its face models there, under auxiliary/.)"""
    link = LATENTSYNC_CODE / "checkpoints"
    if (link / "latentsync_unet.pt").exists():
        return
    if link.is_symlink():
        link.unlink()
    elif link.is_dir():
        if any(p.name != ".gitkeep" for p in link.iterdir()):
            raise RuntimeError(f"{link} exists without the LatentSync weights; remove it and run again")
        shutil.rmtree(link)
    link.symlink_to(weights.resolve(), target_is_directory=True)


def _run(video: Path, audio: Path, out: Path, *, vae: Path, steps: int, seed: int, frames: int, work: Path) -> None:
    """One LatentSync pass in its own environment. Its progress bars print straight to this console."""
    runner = LATENTSYNC_CODE / "vclone_runner.py"  # inside LatentSync's folder, so its imports resolve there
    shutil.copyfile(RUNNER, runner)
    # Keep this environment's settings out of it: vclone's paths, and a notebook's matplotlib backend (Colab
    # exports its inline backend, which LatentSync's environment lacks, so matplotlib refuses to import).
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "MPLBACKEND")}
    env.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg", NO_ALBUMENTATIONS_UPDATE="1")  # stay offline, headless
    cmd = [str(python()), runner.name, "--video", video, "--audio", audio, "--out", out, "--vae", vae,
           "--steps", steps, "--seed", seed, "--frames", frames, "--temp", work]
    proc = subprocess.run([str(c) for c in cmd], cwd=str(LATENTSYNC_CODE), env=env)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError("LatentSync failed (see its messages above)")


def render(cycle: Path, audio: Path, out: Path, *, steps: int = 20, seed: int = 1247, log=print) -> Path:
    """Write `out` (.mp4): the face video from prepare_face() lip-synced to `audio` by LatentSync 1.6."""
    if not installed():
        raise RuntimeError("LatentSync isn't set up here: run `bash setup.sh` (Linux/Colab) or use --lipsync musetalk")
    started = time.time()
    _link_checkpoints(model_dir(LATENTSYNC_REPO, allow_patterns=LATENTSYNC_FILES))
    vae = model_dir(SD_VAE_REPO, allow_patterns=LIPSYNC_FILES[SD_VAE_REPO])
    speech, sr = sf.read(str(audio), dtype="float32")
    duration = len(speech) / sr
    out = Path(out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    # LatentSync's own ffmpeg calls go through a shell unquoted, so every path it sees is in this temp folder.
    with tempfile.TemporaryDirectory(prefix="vclone_ls_") as tmp_dir:
        tmp = Path(tmp_dir)
        looped = tmp / "face.mp4"  # footage for the whole speech (+1 s), so LatentSync never loops on its own
        _ffmpeg("-stream_loop", "-1", "-i", cycle, "-t", f"{duration + 1:.3f}", "-an", *TEMP_X264, looped,
                what="loop the face video")
        parts = max(1, math.ceil(duration / SEGMENT_SECONDS - 1e-6))
        synced = []
        for k in range(parts):
            start = k * SEGMENT_SECONDS
            length = min(SEGMENT_SECONDS, duration - start)
            frames = math.ceil(length * FPS - 1e-6)  # a 30 s part is exactly 750 frames, so parts join in sync
            wav, video, result = tmp / f"speech_{k}.wav", looped, tmp / f"synced_{k}.mp4"
            sf.write(str(wav), speech[round(start * sr):round((start + length) * sr)], sr)
            if parts > 1:
                video = tmp / f"face_{k}.mp4"
                # half a frame early, so rounding can't skip the part's first frame
                _ffmpeg("-ss", f"{max(0.0, start - 0.5 / FPS):.3f}", "-i", looped, "-t", f"{length + 0.5:.3f}",
                        "-an", *TEMP_X264, video, what="cut the face video")
                log(f"[face] LatentSync part {k + 1}/{parts} ({length:.0f}s of video)")
            else:
                t4 = f"{max(1, round(length / 30 * 20))}-{max(2, round(length / 30 * 40))}"  # ~20-40 min per 30 s
                log(f"[face] LatentSync is lip-syncing {length:.1f}s of video (roughly {t4} min on a T4, "
                    "much less on an L4/A100)")
            _run(video, wav, result, vae=vae, steps=steps, seed=seed, frames=frames, work=tmp / f"work_{k}")
            synced.append(result)
        video = synced[0]
        if len(synced) > 1:
            listing = tmp / "parts.txt"
            listing.write_text("".join(f"file '{p.as_posix()}'\n" for p in synced), encoding="utf-8")
            video = tmp / "joined.mp4"
            _ffmpeg("-f", "concat", "-safe", "0", "-i", listing, "-an", "-c:v", "copy", video, what="join the parts")
        # Put the full-quality voice under the new video (LatentSync's own copy has 16 kHz audio), plus the AI tag.
        # The video already has exactly the audio's length (to the frame); -shortest would drop its last frames.
        partial = out.with_name(out.stem + ".partial.mp4")  # never leave a broken file under the real name
        try:
            _ffmpeg("-i", video, "-i", audio, "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-movflags", "+faststart",
                    "-metadata", f"comment={AI_TAG}", partial, what="write the video")
            partial.replace(out)
        finally:
            partial.unlink(missing_ok=True)
    log(f"[face] LatentSync took {time.time() - started:.0f}s")
    return out
