"""Talking video: lip-sync a real video of the person to the new speech with MuseTalk 1.5.

The face video is analysed once (frames, face boxes, VAE latents, blend masks) and cached in
avatars\\<name>-<hash>\\. Each new script then only runs MuseTalk's single-step UNet on the mouth
region and blends it back into the person's real footage, so their head motion, blinks and
expressions stay their own. Everything runs in fp16 and fits a 4 GB GPU (one model at a time)."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch

from . import audio as A
from .engines import free_gpu, quiet
from .models import (FACE_PARSE_REPO, LIPSYNC_FILES, MUSETALK_CODE, MUSETALK_REPO, ROOT, SD_VAE_REPO,
                     WHISPER_TINY_REPO, model_dir)

FPS = 25            # MuseTalk 1.5 was trained on 25 fps video
FACE = 256          # face crop resolution the model works at
MAX_SECONDS = 60    # longer face videos are trimmed; shorter ones play forward then backward
MAX_SIDE = 1920     # 4K phone video is scaled down to 1080p
TURN = 0.15         # nose off the eyes' centre line by this many eye distances more than usual: a ~20-25 deg turn
SIDE = 0.35         # ... and by this much from dead centre: the face is seen half from the side (~40-45 deg)
TURN_MARGIN = 6     # frames also left out next to a turn: its first frames already move
AVATARS_DIR = ROOT / "avatars"
PREP_VERSION = 2  # v2 adds points.npy (face restoration)
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".wmv"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
AI_TAG = "AI-generated: synthetic voice and lip sync (vclone)"


def is_video(path) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTS


def _import_musetalk() -> None:
    """MuseTalk isn't on PyPI; its code is vendored at a pinned commit in third_party\\MuseTalk."""
    if not (MUSETALK_CODE / "musetalk").is_dir():
        raise RuntimeError(f"MuseTalk code is missing from {MUSETALK_CODE}; run setup.ps1")
    if str(MUSETALK_CODE) not in sys.path:
        sys.path.insert(0, str(MUSETALK_CODE))


def _weights(repo: str) -> Path:
    return model_dir(repo, allow_patterns=LIPSYNC_FILES[repo])


# ----------------------------------------------------------------------------- avatar (cached)

@dataclass
class Avatar:
    folder: Path
    frames: list[Path]
    boxes: np.ndarray       # (N, 4) face boxes x1, y1, x2, y2 the model works on
    crop_boxes: np.ndarray  # (N, 4) larger boxes the blend masks cover
    latents: torch.Tensor   # (N, 8, 32, 32) fp16: masked + reference VAE latents per frame
    size: tuple[int, int]   # frame width, height
    points: np.ndarray      # (N, 5, 2) eye centres, nose tip, mouth corners: aligns the face restoration
    found: np.ndarray       # (N,) the face was detected (other frames borrow boxes from their neighbours)

    def mask(self, i: int) -> np.ndarray:
        return cv2.imread(str(self.folder / "masks" / f"{i:06d}.png"), cv2.IMREAD_GRAYSCALE)


def _file_hash(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _extract_frames(src: Path, folder: Path, max_seconds: float) -> list[Path]:
    folder.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in IMAGE_EXTS:
        img = cv2.imdecode(np.fromfile(str(src), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Can't read the face photo {src}")
        scale = min(1.0, MAX_SIDE / max(img.shape[:2]))
        if scale < 1:
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        img = img[: img.shape[0] // 2 * 2, : img.shape[1] // 2 * 2]
        cv2.imwrite(str(folder / "000000.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 97])
        return [folder / "000000.jpg"]
    exe = A.find_ffmpeg()
    if exe is None:
        raise RuntimeError("ffmpeg is needed to read the face video")
    # Constant 25 fps, phone rotation applied, 4K scaled to 1080p, even dimensions for H.264.
    vf = (f"fps={FPS},scale='min({MAX_SIDE},iw)':'min({MAX_SIDE},ih)':force_original_aspect_ratio=decrease,"
          "scale=trunc(iw/2)*2:trunc(ih/2)*2")
    proc = subprocess.run([exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y", "-i", str(src),
                           "-t", str(max_seconds), "-an", "-vf", vf, "-q:v", "2", "-start_number", "0",
                           str(folder / "%06d.jpg")], capture_output=True)
    frames = sorted(folder.glob("*.jpg"))
    if proc.returncode != 0 or not frames:
        raise RuntimeError(f"ffmpeg could not read frames from {src}: {proc.stderr.decode(errors='ignore')[-500:]}")
    return frames


def _smooth(values: np.ndarray, window: int = 5) -> np.ndarray:
    """Centred moving average along time: steadies the face crop between frames (less jitter)."""
    if len(values) < 3:
        return values
    window = min(window, len(values) // 2 * 2 + 1)
    pad = window // 2
    padded = np.pad(values, ((pad, pad), (0, 0)), mode="edge")
    kernel = np.ones(window) / window
    return np.stack([np.convolve(padded[:, c], kernel, mode="valid") for c in range(values.shape[1])], axis=1)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) of every stretch of True values."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2])]


def facing_camera(points: np.ndarray, found: np.ndarray) -> tuple[int, int, list[tuple[int, int]]]:
    """The footage a talking video replays: [start, end) of the longest stretch where the face is found and
    not turned away (more than ~20-25 degrees from the person's usual pose), and the frame ranges where it
    is turned away or missing. Played forward and back, that stretch never shows the person looking away.
    A head turn moves the nose tip sideways off the line between the eyes; measured in eye distances, that
    doesn't depend on the face's size, position or tilt."""
    n = len(points)
    if n == 0 or not found.any():
        raise ValueError("No face was found in the video. Use a video where the face is visible and front-on.")
    if n == 1:  # a photo
        return 0, 1, []
    left, right, nose = points[:, 0], points[:, 1], points[:, 2]
    eyes = right - left
    turn = np.einsum("ij,ij->i", nose - (left + right) / 2, eyes) / np.maximum(np.einsum("ij,ij->i", eyes, eyes), 1e-6)
    turn = np.where(found & np.isfinite(turn), turn, np.inf)
    turn = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(turn, 2, mode="edge"), 5), axis=1)  # jitter
    ok = found & np.isfinite(turn)  # after smoothing too: LatentSync stops at any frame without a face
    if ok.any():
        # The person's usual pose, from frames not seen half from the side, so a video that mostly looks away
        # doesn't make looking away the norm. (A camera off to one side just shifts it; that's fine.)
        front = ok & (np.abs(turn) <= SIDE)
        ok &= np.abs(turn - np.median(turn[front if front.any() else ok])) <= TURN
    runs = _runs(ok)
    start, end = max(runs, key=lambda r: r[1] - r[0]) if runs else (0, 0)
    longest = end - start
    start += TURN_MARGIN if start > 0 else 0
    end -= TURN_MARGIN if end < n else 0
    if end - start < min(FPS, n):
        raise ValueError(f"The face is turned away or hidden in most of the video: the longest stretch facing "
                         f"the camera is {longest / FPS:.1f}s. Use a video where the person faces the camera for "
                         "at least a few seconds.")
    return start, end, [(a, b) for a, b in _runs(~ok) if b - a >= 3]


def footage_note(n: int, start: int, end: int, away: list[tuple[int, int]]) -> str | None:
    """The progress line saying which part of the face video is used, when it isn't all of it."""
    if start == 0 and end == n:
        return None
    where = ", ".join(f"{a / FPS:.1f}-{b / FPS:.1f}s" for a, b in away) or "briefly"
    return (f"[face] using {start / FPS:.1f}-{end / FPS:.1f}s of the video, the longest part facing the camera "
            f"(turned away or hidden at {where})")


def _detect_face(fa, rgb: np.ndarray) -> np.ndarray | None:
    """Largest face box in full-resolution coordinates, detected on a small copy (the detector's
    post-processing is slow at full size)."""
    scale = min(1.0, 480 / max(rgb.shape[:2]))
    small = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else rgb
    detections = fa.face_detector.detect_from_image(small)
    if not len(detections):
        return None
    det = max(detections, key=lambda d: (d[2] - d[0]) * (d[3] - d[1]))
    return np.array(det[:4], dtype=np.float64) / scale


def _face_boxes(frames: list[Path], device: str, log,
                detect_every: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MuseTalk's face box from 68 facial landmarks: from the chin up to the same distance above
    the nose, and landmark-wide. (MuseTalk gets the landmarks from DWPose, which needs mmcv; FAN
    gives the same 68-point layout without it.) The detector runs every few frames; in between,
    its box follows the landmarks, which is far faster and just as accurate for a talking head.
    Also returns 5 points per frame (eyes, nose, mouth corners) for aligning the face restoration and
    spotting head turns, and which frames had a face."""
    import face_alignment

    from .restore import five_points
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device=device, flip_input=False,
                                      compile=False,
                                      dtype=torch.float16 if device.startswith("cuda") else torch.float32)
    torch.backends.cudnn.benchmark = False  # face_alignment switches it on globally
    boxes = np.full((len(frames), 4), np.nan)
    points = np.full((len(frames), 5, 2), np.nan)
    det, anchor = None, None
    for i, path in enumerate(frames):
        rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        if det is None or i % detect_every == 0:
            found = _detect_face(fa, rgb)
            if found is not None:
                det, anchor = found, None
        if det is not None:
            with quiet():
                marks, scores, _ = fa.get_landmarks_from_image(rgb, detected_faces=[det], return_landmark_score=True)
            if marks and float(np.mean(scores[0])) > 0.2:
                lm = marks[0]
                points[i] = five_points(lm)
                centre = lm.mean(axis=0)
                if anchor is not None:  # follow the head until the next detection
                    det = det + np.tile(centre - anchor, 2)
                anchor = centre
                y2 = lm[:, 1].max()
                nose_y = lm[29, 1]
                box = np.array([lm[:, 0].min(), max(0.0, nose_y - (y2 - nose_y)), lm[:, 0].max(), y2])
                if box[2] - box[0] <= 0 or box[3] - box[1] <= 0 or box[0] < 0:
                    box = det.copy()  # landmark box unusable: fall back to the detector box, as MuseTalk does
                box[3] += 0.04 * (box[3] - box[1])  # v1.5 keeps a little margin below the chin
                boxes[i] = np.clip(box, 0, [w, h, w, h])
            else:
                det = None  # lost the face: detect again on the next frame
        if (i + 1) % 100 == 0 or i + 1 == len(frames):
            log(f"[face] located the face in {i + 1}/{len(frames)} frames")
    del fa
    free_gpu()
    found = ~np.isnan(boxes[:, 0])
    if not found.any():
        raise ValueError("No face was found in the video. Use a video where the face is visible, front-on and "
                         "well lit.")
    idx = np.arange(len(frames))
    flat = points.reshape(len(frames), 10)
    for c in range(4):  # frames where the face was missed borrow from their neighbours (render skips them)
        boxes[:, c] = np.interp(idx, idx[found], boxes[found, c])
    for c in range(10):
        flat[:, c] = np.interp(idx, idx[found], flat[found, c])
    return np.round(_smooth(boxes)).astype(int), _smooth(flat).reshape(-1, 5, 2).astype(np.float32), found


@torch.inference_mode()
def _encode_latents(frames: list[Path], boxes: np.ndarray, device: str, batch: int = 16) -> torch.Tensor:
    """Per frame: [VAE latents of the face with its lower half blanked, latents of the full face]."""
    from diffusers import AutoencoderKL
    vae = AutoencoderKL.from_pretrained(_weights(SD_VAE_REPO), torch_dtype=torch.float16).to(device).eval()
    vae.enable_slicing()  # one image at a time keeps VRAM low on 4 GB cards
    scale = vae.config.scaling_factor
    out, pending = [], []

    def flush():
        x = torch.from_numpy(np.stack(pending)).to(device).permute(0, 3, 1, 2).half() / 127.5 - 1
        masked = x.clone()
        masked[:, :, FACE // 2:, :] = -1
        lat = [vae.encode(v).latent_dist.mode() * scale for v in (masked, x)]
        out.append(torch.cat(lat, dim=1).cpu())
        pending.clear()

    for path, (x1, y1, x2, y2) in zip(frames, boxes):
        crop = cv2.imread(str(path))[y1:y2, x1:x2]
        pending.append(cv2.cvtColor(cv2.resize(crop, (FACE, FACE), interpolation=cv2.INTER_LANCZOS4),
                                    cv2.COLOR_BGR2RGB))
        if len(pending) == batch:
            flush()
    if pending:
        flush()
    del vae
    free_gpu()
    return torch.cat(out)


def _face_parser(device: str):
    _import_musetalk()
    from musetalk.utils.face_parsing import FaceParsing
    from musetalk.utils.face_parsing import model as bisenet
    checkpoint = _weights(FACE_PARSE_REPO) / "79999_iter.pth"

    class Parser(FaceParsing):
        def model_init(self, resnet_path=None, model_pth=None):
            # MuseTalk first loads ImageNet ResNet-18 weights into the backbone, but the full BiSeNet
            # checkpoint below overwrites every one of them, and that 2017 file is a legacy pickle
            # that can't be loaded safely (weights_only). So skip it.
            init_weight = bisenet.Resnet18.init_weight
            bisenet.Resnet18.init_weight = lambda *_: None
            try:
                net = bisenet.BiSeNet(None)
            finally:
                bisenet.Resnet18.init_weight = init_weight
            net.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            return net.to(device).eval()

    return Parser(left_cheek_width=90, right_cheek_width=90)  # MuseTalk 1.5 defaults


def _blend_masks(frames: list[Path], boxes: np.ndarray, folder: Path, device: str, log) -> np.ndarray:
    """Jaw-shaped masks (from face parsing of the original frames) that decide where the new mouth goes."""
    _import_musetalk()
    from musetalk.utils.blending import get_image_prepare_material
    parser = _face_parser(device)
    folder.mkdir(parents=True, exist_ok=True)
    crop_boxes = []
    for i, (path, box) in enumerate(zip(frames, boxes)):
        mask, crop_box = get_image_prepare_material(cv2.imread(str(path)), [int(v) for v in box], fp=parser,
                                                    mode="jaw")
        cv2.imwrite(str(folder / f"{i:06d}.png"), mask)
        crop_boxes.append(crop_box)
        if (i + 1) % 250 == 0 or i + 1 == len(frames):
            log(f"[face] blend masks {i + 1}/{len(frames)}")
    del parser
    free_gpu()
    return np.array(crop_boxes, dtype=int)


def prepare_avatar(src: str | Path, device: str, *, refresh: bool = False, max_seconds: float = MAX_SECONDS,
                   log=print) -> Avatar:
    """Analyse the face video once; later calls reuse the cache."""
    src = Path(src).expanduser().resolve()
    if not src.is_file():
        raise FileNotFoundError(f"Face video not found: {src}")
    key = hashlib.sha1(f"{_file_hash(src)}|v{PREP_VERSION}|{max_seconds}".encode()).hexdigest()[:10]
    folder = AVATARS_DIR / f"{''.join(c if c.isalnum() or c in '-_' else '_' for c in src.stem)[:40]}-{key}"
    info_path = folder / "info.json"
    if refresh or not info_path.exists():
        started = time.time()
        log(f"[face] analysing {src.name} (one time; cached in {folder.name})")
        info_path.unlink(missing_ok=True)
        for sub in ("frames", "masks"):  # leftovers of an interrupted run
            shutil.rmtree(folder / sub, ignore_errors=True)
        frames = _extract_frames(src, folder / "frames", max_seconds)
        size = cv2.imread(str(frames[0])).shape[1::-1]
        log(f"[face] {len(frames)} frames at {size[0]}x{size[1]}, {FPS} fps")
        boxes, points, found = _face_boxes(frames, device, log)
        facing_camera(points, found)  # an unusable video fails here, before the slow work
        latents = _encode_latents(frames, boxes, device)
        crop_boxes = _blend_masks(frames, boxes, folder / "masks", device, log)
        np.save(folder / "boxes.npy", boxes)
        np.save(folder / "points.npy", points)
        np.save(folder / "found.npy", found)
        np.save(folder / "crop_boxes.npy", crop_boxes)
        torch.save(latents, folder / "latents.pt")
        info_path.write_text(json.dumps({
            "source": str(src), "frames": len(frames), "size": list(size), "fps": FPS,
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, indent=2), encoding="utf-8")
        log(f"[face] ready in {time.time() - started:.0f}s")
    else:
        log(f"[face] using cached face analysis {folder.name}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    frames = sorted((folder / "frames").glob("*.jpg"))
    latents = torch.load(folder / "latents.pt", weights_only=True)
    if len(frames) != info["frames"] or len(latents) != info["frames"]:
        raise RuntimeError(f"The face cache in {folder} is incomplete; run again with --refresh-face")
    found_path = folder / "found.npy"  # caches from before it was saved: count every frame as found
    found = np.load(found_path) if found_path.exists() else np.ones(len(frames), bool)
    avatar = Avatar(folder, frames, np.load(folder / "boxes.npy"), np.load(folder / "crop_boxes.npy"), latents,
                    tuple(info["size"]), np.load(folder / "points.npy"), found)
    facing_camera(avatar.points, avatar.found)  # fail before the voice work, not after it
    return avatar


# ----------------------------------------------------------------------------- rendering

class _VideoWriter:
    """Streams frames straight into ffmpeg (H.264 + AAC MP4), no temporary images on disk."""

    def __init__(self, out: Path, size: tuple[int, int], audio: Path, crf: int = 17):
        exe = A.find_ffmpeg()
        out.parent.mkdir(parents=True, exist_ok=True)
        cmd = [exe, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
               "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{size[0]}x{size[1]}", "-r", str(FPS), "-i", "-",
               "-i", str(audio), "-map", "0:v:0", "-map", "1:a:0",
               "-c:v", "libx264", "-preset", "medium", "-crf", str(crf), "-pix_fmt", "yuv420p",
               "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-shortest", "-movflags", "+faststart",
               "-metadata", f"comment={AI_TAG}", str(out)]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, frame: np.ndarray) -> None:
        self.proc.stdin.write(np.ascontiguousarray(frame).tobytes())

    def close(self) -> None:
        self.proc.stdin.close()
        err = self.proc.stderr.read().decode(errors="ignore")
        if self.proc.wait() != 0:
            raise RuntimeError(f"ffmpeg failed to write the video: {err[-500:]}")


@torch.inference_mode()
def _audio_prompts(audio: Path, device: str) -> torch.Tensor:
    """Whisper-tiny encoder features, 10 steps (0.2 s) around each video frame: (frames, 50, 384)."""
    _import_musetalk()
    from musetalk.utils.audio_processor import AudioProcessor
    from transformers import WhisperModel
    folder = _weights(WHISPER_TINY_REPO)
    processor = AudioProcessor(feature_extractor_path=str(folder))
    whisper = WhisperModel.from_pretrained(folder, torch_dtype=torch.float16).to(device).eval()
    features, n_samples = processor.get_audio_feature(str(audio))
    with quiet():
        prompts = processor.get_whisper_chunk(features, device, torch.float16, whisper, n_samples, fps=FPS,
                                              audio_padding_length_left=2, audio_padding_length_right=2)
    del whisper
    free_gpu()
    return prompts


def _load_unet(device: str):
    """MuseTalk's UNet in fp16. The first load converts the 3.4 GB fp32 checkpoint and keeps a 1.7 GB
    fp16 copy next to it (unet_fp16.safetensors), which later runs load in a couple of seconds."""
    from accelerate import init_empty_weights
    from diffusers import UNet2DConditionModel
    from safetensors.torch import load_file, save_file
    folder = _weights(MUSETALK_REPO) / "musetalkV15"
    config = json.loads((folder / "musetalk.json").read_text(encoding="utf-8"))
    with init_empty_weights():  # no 3.4 GB random init; the weights are assigned below
        unet = UNet2DConditionModel(**{k: v for k, v in config.items() if not k.startswith("_")})
    fp16 = folder / "unet_fp16.safetensors"
    if fp16.exists():
        state = load_file(str(fp16))
    else:
        state = torch.load(folder / "unet.pth", map_location="cpu", weights_only=True, mmap=True)
        state = {k: v.half().contiguous() for k, v in state.items()}
        save_file(state, str(fp16.with_suffix(".tmp")))
        fp16.with_suffix(".tmp").replace(fp16)
    unet.load_state_dict(state, assign=True)
    return unet.to(device).eval()


def _blend(frame: np.ndarray, face: np.ndarray, box, mask: np.ndarray, crop_box) -> np.ndarray:
    """Paste the new face into the frame through the blend mask. Same result as MuseTalk's PIL
    get_image_blending, but only the face region is touched, which matters at 1080p."""
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop_box
    h, w = frame.shape[:2]
    ax1, ay1, ax2, ay2 = max(cx1, 0), max(cy1, 0), min(cx2, w), min(cy2, h)  # crop box inside the frame
    region = frame[ay1:ay2, ax1:ax2]
    patched = region.copy()
    patched[y1 - ay1:y2 - ay1, x1 - ax1:x2 - ax1] = face
    alpha = mask[ay1 - cy1:ay2 - cy1, ax1 - cx1:ax2 - cx1, None].astype(np.float32) / 255.0
    frame[ay1:ay2, ax1:ax2] = (patched * alpha + region * (1.0 - alpha) + 0.5).astype(np.uint8)
    return frame


def _restore_pass(avatar: Avatar, generated: list, composite_one, write, strength: float, device: str, log) -> None:
    """Second pass: blend each generated mouth into its frame, then sharpen it with GFPGAN."""
    from .restore import MouthRestorer
    try:
        restorer = MouthRestorer(device, strength)
    except Exception as exc:  # e.g. no network for the one-time download: still deliver the lip sync
        log(f"[face] mouth sharpening unavailable ({type(exc).__name__}: {exc}); writing the video without it")
        restorer = None
    started, chunk = time.time(), 16
    for start in range(0, len(generated), chunk):
        part = generated[start:start + chunk]
        frames = [composite_one(j, face) for j, face in part]
        if restorer is not None:
            idx = [j for j, _ in part]
            try:
                restorer.process(frames, [avatar.points[j] for j in idx], [avatar.mask(j) for j in idx],
                                 [avatar.crop_boxes[j] for j in idx])
            except Exception as exc:  # sharpening is an extra: never lose the whole video to it
                log(f"[face] mouth sharpening failed ({type(exc).__name__}: {exc}); finishing without it")
                restorer = None
                frames = [composite_one(j, face) for j, face in part]
        for frame in frames:
            write(frame)
        done = start + len(part)
        if done % 256 < chunk or done == len(generated):
            log(f"[face] sharpening the mouth {done}/{len(generated)} frames")
    if restorer is not None:
        log(f"[face] sharpening took {time.time() - started:.0f}s")
    del restorer
    free_gpu()


@torch.inference_mode()
def render(avatar: Avatar, audio: Path, out: Path, device: str, *, batch: int = 8, restore: float = 0.8,
           log=print) -> Path:
    """Write `out` (.mp4): the avatar's footage with the mouth re-generated to speak `audio`.
    `restore` > 0 sharpens the generated mouth with GFPGAN (0 = off, 1 = full strength)."""
    from concurrent.futures import ThreadPoolExecutor

    _import_musetalk()
    from diffusers import AutoencoderKL
    from musetalk.models.unet import PositionalEncoding

    started = time.time()
    prompts = _audio_prompts(audio, device)
    unet = _load_unet(device)
    vae = AutoencoderKL.from_pretrained(_weights(SD_VAE_REPO), torch_dtype=torch.float16).to(device).eval()
    # One face at a time: decoding 8 at once next to the UNet peaks at 3.8 GB, past what a 4 GB card
    # has free, and Windows then silently spills to system RAM (5x slower). Sliced it peaks at 2.1 GB.
    vae.enable_slicing()
    pos = PositionalEncoding(d_model=384).to(device).half()
    if device.startswith("cuda"):
        log(f"[face] lip-sync models ready in {time.time() - started:.0f}s "
            f"(GPU memory in use {torch.cuda.memory_reserved() / 2**30:.1f} GB)")
    timestep = torch.tensor([0], device=device)  # MuseTalk is a single-step model at t=0
    # Forward then backward through the footage, so a script longer than the video never jumps; only the
    # longest part facing the camera, so a moment where the person looks away never shows up.
    first, end, away = facing_camera(avatar.points, avatar.found)
    note = footage_note(len(avatar.frames), first, end, away)
    if note:
        log(note)
    order = list(range(first, end)) + list(range(end - 1, first - 1, -1))
    total = prompts.shape[0]
    log(f"[face] lip-syncing {total} frames ({total / FPS:.1f}s)")
    writer = _VideoWriter(out, avatar.size, audio)
    last: list[np.ndarray] = []
    restoring = restore > 0
    generated: list[tuple[int, np.ndarray]] = []  # (frame index, 256 px mouth) for the sharpening pass

    def composite_one(j: int, face: np.ndarray) -> np.ndarray:
        x1, y1, x2, y2 = (int(v) for v in avatar.boxes[j])
        face = cv2.resize(np.ascontiguousarray(face), (x2 - x1, y2 - y1), interpolation=cv2.INTER_LANCZOS4)
        return _blend(cv2.imread(str(avatar.frames[j])), face, (x1, y1, x2, y2), avatar.mask(j),
                      [int(v) for v in avatar.crop_boxes[j]])

    def write(frame: np.ndarray) -> None:
        writer.write(frame)
        last[:] = [frame]

    def composite(idx, faces):  # CPU side, overlapped with the next batch on the GPU
        for j, face in zip(idx, faces):
            write(composite_one(j, face))

    pending, finished = None, False
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            for start in range(0, total, batch):
                idx = [order[i % len(order)] for i in range(start, min(total, start + batch))]
                latents = avatar.latents[idx].to(device)
                audio_features = pos(prompts[start:start + len(idx)].half())
                pred = unet(latents, timestep, encoder_hidden_states=audio_features).sample
                faces = vae.decode(pred / vae.config.scaling_factor).sample
                faces = ((faces.float() / 2 + 0.5).clamp(0, 1) * 255).round().byte()
                faces = faces.permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]  # RGB -> BGR
                if restoring:
                    generated.extend((j, np.ascontiguousarray(face)) for j, face in zip(idx, faces))
                else:
                    if pending is not None:
                        pending.result()  # frames must reach ffmpeg in order
                    pending = pool.submit(composite, idx, faces)
                done = min(total, start + batch)
                if done % (batch * 25) < batch or done == total:
                    log(f"[face] {done}/{total} frames")
            if pending is not None:
                pending.result()
        unet = vae = pos = None  # the sharpening model gets the GPU to itself (one model at a time)
        free_gpu()
        if restoring:
            _restore_pass(avatar, generated, composite_one, write, restore, device, log)
        if last:
            writer.write(last[0])  # one spare frame so the video never ends before the audio
        finished = True
    finally:
        try:
            writer.close()
        except RuntimeError:
            if finished:
                raise
        if not finished:
            out.unlink(missing_ok=True)  # never leave a half-written, unplayable video behind
    free_gpu()
    log(f"[face] lip sync took {time.time() - started:.0f}s")
    return out
