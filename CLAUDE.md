# vclone: local voice cloning + lip-synced talking video

Takes a short recording of a person (audio, or a ~30 s video) plus some text. It writes an audio file of that voice reading the text, or an MP4 of the person saying it with the mouth lip-synced into their real footage. Runs fully offline on the user's NVIDIA GPU (Windows 11, no GUI), and on Linux/Google Colab. Lip sync: LatentSync 1.6 (realistic, slow) on 15 GB+ GPUs where `setup.sh` installed it, i.e. Colab, where the user actually makes videos; MuseTalk 1.5 (fast) elsewhere, incl. the 4 GB laptop. README.md is the user guide; this file is for working on the code.

## Running it
- Windows: use only the project venv `D:\GPU\.venv\Scripts\python.exe` (Python 3.11). Never the system Python, and never `pip install` outside `.venv`.
- CLI: `speak.bat <recording> "text" [options]` (Linux/Colab: `./speak.sh`). It sets `PYTHONPATH` to the repo and runs `python -m vclone`. `speak.bat -h` lists every option. A video recording (or `--face`) plus an `.mp4` (or no) `-o` gives a talking video.
- Python API: `from vclone import speak` with the repo on `PYTHONPATH`.
- Setup: `setup.ps1` (Windows) / `setup.sh` (Linux, Colab). `colab.ipynb` drives Colab. Both fetch MuseTalk's code into `third_party\MuseTalk` at the pinned commit and run `python -m vclone.download`. On a 15 GB+ GPU, `setup.sh` also fetches LatentSync's code into `third_party/LatentSync` and builds its own Python 3.10 env there (`.venv`, with uv; `.venv/vclone-ready` marks it complete).

## Code map (`vclone\`)
| Module | Role |
|---|---|
| `cli.py` | argparse front end, calls `pipeline.run` |
| `pipeline.py` | end to end: reference → face analysis → text chunks → takes on the GPU → best take → mastered audio → lip sync (`QUALITY_PRESETS`, `BAD_WER`, `GOOD_MATCH`). With LatentSync the voice runs in a spawned child process (`_voice_in_child`) so its GPU memory is truly freed; the parent never starts CUDA |
| `reference.py` | recording → cleaned 5–12 s clip + transcript (enrollment-script match or Whisper, language limited to Qwen's 10), cached in `voices\<name>-<hash>\` |
| `engines.py` | TTS back ends `QwenEngine` (default, 10 languages) and `F5Engine` (English/Chinese); `free_gpu()` |
| `qwen_fast.py` | CUDA-graph replay of Qwen3-TTS's code predictor (~2× faster) |
| `quality.py` | ranks takes: Whisper word error (several equivalent normalisations, kindest wins) + WavLM voice similarity |
| `lipsync.py` | MuseTalk 1.5: face video → cached avatar in `avatars\<name>-<hash>\` (frames, FAN-landmark boxes, 5-point face landmarks, VAE latents, blend masks) → `render()` streams lip-synced frames into ffmpeg; with `restore` > 0 a second pass sharpens the mouth |
| `restore.py`, `gfpgan_arch.py` | GFPGAN v1.4 mouth sharpening: align each frame's face to the 512 px FFHQ template (5 points), restore with fixed noise, paste back only through MuseTalk's jaw mask. `gfpgan_arch.py` is the vendored Apache-2.0 network (basicsr imports removed) |
| `latentsync.py` | LatentSync 1.6 orchestration (our env): `prepare_face()` caches the face video as 25 fps 720p forward+backward (`avatars\...\latentsync_cycle.mp4`), `render()` loops it to the speech, splits into 30 s parts of exact frame counts, runs the runner per part, joins, muxes the full-quality voice + AI tag. `usable()` checks GPU memory with nvidia-smi |
| `latentsync_runner.py` | runs inside LatentSync's own env (copied into its folder); must not import vclone. Mirrors upstream `scripts/inference.py`; its docstring lists every deviation (fp16 on T4, VAE slicing, guidance/DeepCache by GPU size, cast-then-mmap checkpoint load, no input re-encode, `--frames` trim) |
| `text.py` | clean-up, English number/abbreviation normalisation, sentence-aware chunking |
| `audio.py` | decode via ffmpeg, silence handling, loudness, peak limiter, export |
| `models.py`, `download.py` | model repos (`LIPSYNC_FILES`, `GFPGAN_URL`), `auto_qwen_size()` (1.7B on 10 GB+ GPUs, else 0.6B), `COMMAND` name per OS, one-time downloads into `models\` (`model_dir` for Hugging Face, `url_file` for size-checked URLs) |

## Hard constraints
- **4 GB GPU (RTX A2000 Laptop, ~3.2 GB usable).** Only one model sits on the GPU at a time (Whisper, then the TTS model, then the checkers, then MuseTalk), freed with `del` + `free_gpu()` between stages; peak is ~2.7 GB. Never load two models at once. If VRAM runs out, Windows silently spills to system RAM instead of failing: the VAE decoding 8 faces next to the UNet peaked at 3.8 GB and ran 5× slower. Keep `vae.enable_slicing()`, bf16 (fp32 where bf16 isn't native, e.g. T4), and the batch sizes unless you measure a change, and watch `torch.cuda.max_memory_reserved()`.
- **LatentSync only runs on Colab-class GPUs** (15 GB+; upstream wants 18 GB, our runner fits the T4), never on the 4 GB laptop. It pins its own stack (torch 2.5.1, diffusers 0.32.2, numpy 1.26.4, InsightFace 0.7.3 built from source) from its `requirements.txt`: install those only into `third_party/LatentSync/.venv`, never into the main env. Its shell-built ffmpeg commands don't quote paths, so it only ever sees paths in our temp folder.
- **`qwen_fast.py` was verified to produce identical logits (float32) to the original code.** Re-verify the same way after any change there. Likewise `lipsync._blend` is pixel-identical to MuseTalk's `get_image_blending`, tracked face boxes were within ~1 px of per-frame detection, and `restore._paste` maps a face back with mean error 0.07/255 (identity-restorer test on a rotated face).
- Mouth sharpening runs only after the MuseTalk UNet/VAE are freed (one model at a time); it keeps the 256 px mouths in RAM between the passes. GFPGAN runs in fp32 with `randomize_noise=False`: random noise per frame would make the texture shimmer. If it can't load, the video is still written without it. Avatar caches are versioned (`PREP_VERSION`); bump it when the cached data changes.
- **Pinned stack:** torch/torchaudio 2.8.0 + torchvision 0.23.0 (CUDA 12.8 wheels), qwen-tts 0.1.1, f5-tts 1.1.22, transformers 4.57.3, huggingface_hub 0.36.2 (must stay < 1.0 for transformers 4.57; newer diffusers want 1.x), diffusers 0.39.0, opencv-python 4.14, face-alignment 1.5.0 (construct with `compile=False`; Windows has no Triton). MuseTalk code is pinned to commit `0a89dec` (`models.MUSETALK_COMMIT`), LatentSync to `a229c39` (`models.LATENTSYNC_COMMIT`, also in `setup.sh`). Always install with `-c constraints.txt`, and dry-run first (`pip install --dry-run`): pip only warns about breaking already-installed packages after it has done it. Ask before upgrading any of these or downloading new models.
- Load checkpoints with `weights_only=True`. MuseTalk's ImageNet ResNet-18 file is a legacy pickle and is deliberately skipped, since the BiSeNet checkpoint contains the whole backbone.
- Keep CLI flags and the `speak()` API backward compatible, and update README.md when options or behaviour change.
- Licences: F5-TTS weights are CC-BY-NC (non-commercial); Qwen3-TTS and GFPGAN are Apache-2.0; MuseTalk code and weights are MIT/commercial-OK; LatentSync code Apache-2.0, weights OpenRAIL++; InsightFace's buffalo_l (LatentSync's face detector, auto-downloaded into `models\LatentSync-1.6\auxiliary`) is non-commercial. Outputs carry an `AI-generated` metadata tag; keep it.

## Data: handle with care
- `voices\` (the user's voice clips and transcripts) and `avatars\` (frames and face analysis of their videos) are private biometric data: never delete, upload or share them. `info.json` in each folder records the original file under `source`. `.gitignore` keeps them, all media files, `outputs\`, `models\` and `third_party\` out of git; keep it that way.
- `models\` (~12 GB locally, ~15 GB on Colab with the 1.7B voice model; incl. the generated `unet_fp16.safetensors` cache) and `outputs\` (generated audio/video): don't delete without asking.

## Checking a change
There is no test suite and no linter in the venv.
1. Quick: `speak.bat -h`, and an import check (`python -c "import vclone.pipeline"` with the venv Python and `PYTHONPATH=D:\GPU`).
2. `text.py`, `reference._match_script`/`_choose_language` and most of `audio.py` are plain Python/NumPy: exercise changed functions directly with small inputs, no GPU needed.
3. Voice end to end (GPU, ~15–30 s): reuse the `source` recording from a cached voice so its cache is hit, e.g.
   `speak.bat "<source from voices\...\info.json>" "Quick check, one two three." --quality fast --check -o outputs\check.wav`.
   The `[check]` line should show 0% word error; `[out]` reports timing.
4. Talking video (GPU): MuseTalk's own samples avoid touching the user's data, e.g.
   `speak.bat third_party\MuseTalk\data\audio\eng.wav "Quick check, one two three." --face third_party\MuseTalk\data\video\sun.mp4 --quality fast -o <scratch>\check.mp4`.
   The first run analyses the face (~2.5 min); `[face] lip sync took` should be ~3 s per second of video on the A2000. Delete the sample's `voices\eng-*` / `avatars\sun-*` caches afterwards. Compare speed work against the README timings.
5. Colab can't be run from here. Keep `setup.sh`/`speak.sh` LF-only (`.gitattributes`) and check them with `bash -n`.
   LatentSync's orchestration runs on CPU: replace `latentsync._run` with ffmpeg keeping the first `frames` frames of its input (the runner's contract), point `AVATARS_DIR`, `model_dir` and `installed` at scratch, and check with synthetic videos whose frames each have a distinct brightness that every frame lands in place across loops and 30 s parts (use part lengths whose cuts avoid the forward/backward turnaround, where frames repeat). Its GPU side only runs on Colab.
6. If you couldn't run the GPU check, say so plainly instead of claiming the change works.

## Style
Match the existing code: `from __future__ import annotations`, type hints, `pathlib`, small focused functions, comments that explain why. Progress messages use the `[stage] message` form (`[voice]`, `[tts]`, `[check]`, `[face]`, `[out]`).
