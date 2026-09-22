# vclone: a person's own voice and face, saying any text

Give it a **~30 second video** of a person talking to the camera plus a **text script**, and it makes a video of
that person saying the script **in their own cloned voice, lip-synced**. It keeps their real head movement,
blinks and expressions. Or give it just a voice recording and get audio.
Runs offline on a small NVIDIA GPU (built on an RTX A2000 Laptop, 4 GB), or on **Google Colab**. On a 15 GB+
GPU (Colab's free T4 and up) the mouth is made by **LatentSync 1.6**, the most realistic open lip-sync model;
smaller GPUs use the much faster MuseTalk 1.5.

```bat
:: Windows
speak.bat person.mp4 -f script.txt -o talking.mp4          (video in -> talking video out)
speak.bat my_voice.m4a "Hello! This is my cloned voice."    (voice in -> audio out)
```
```bash
# Linux / Google Colab: same options
./speak.sh person.mp4 -f script.txt -o talking.mp4
```

Results go to `outputs\` unless you pass `-o`. The first run with a new video analyses the voice and face once
(a few minutes). Later scripts with the same video reuse that and only generate the new speech and lip sync.

---

## 1. Record the input (this matters most for quality)

**The video** (for talking videos):
* 25–60 seconds of the person talking to the camera. Face visible and roughly front-on **in every frame**
  (LatentSync stops at a frame without a face), good light, steady camera.
* The output reuses their real footage, so record the head movement and expressions you want to see
  (e.g. friendly and relaxed). Only the mouth region is regenerated.
* One person, nothing covering the mouth. Portrait or landscape both work; the output is 720p with LatentSync
  (the face then comes out at the model's own 512 px) and up to 1080p with MuseTalk.

**The voice** (comes from the same video, or from any audio file):
* **Best: read the built-in script.** `speak.bat --script` prints a ~10 s text. If the person reads it in the
  recording, the tool recognizes it and learns the voice from the exact words. A wrong transcript makes
  the clone sound like someone else.
* Quiet room (no fan, TV or music), mic ~20 cm away, speaking naturally. Energy, pace and accent are copied.
* Any format works: `.mp4 .mov .webm` video, or `.wav .mp3 .m4a .ogg/.opus .flac` audio.

The voice clip (5–12 s) and its transcript are cached in `voices\<name>-<id>\`, and the face analysis in
`avatars\<name>-<id>\`. **Check the transcript it prints.** If Whisper wasn't sure or got a word wrong, fix
`voices\<name>-<id>\ref.txt` (or pass `--ref-text "exact words"`) and run again. Accents and noise can fool
the language guess; force it with `--ref-language english` (or `hindi`, `tamil`, ...).

**Reading the check line:** `voice match` compares each take with the recording. A voice scores about 0.96
against itself. Above 0.93 is a close clone; below 0.90 the tool tells you how to improve the recording.

## 2. Use it

```bat
:: talking video from a 30 s video + script file
speak.bat person.mp4 -f script.txt -o talking.mp4

:: voice from one file, face from another video (or a photo, which gives a static head)
speak.bat voice.m4a "Some text" --face person.mp4 -o talking.mp4

:: audio only (an audio file name as output)
speak.bat person.mp4 "Text to read" -o hello.wav

:: quick draft (1 take, no checking) vs. best quality (5 takes)
speak.bat person.mp4 "Quick test" --quality fast -o test.mp4
speak.bat person.mp4 -f script.txt --quality max -o final.mp4

:: other languages: their voice speaking French
speak.bat person.mp4 "Bonjour à tous, comment allez-vous ?" --language french -o fr.mp4
```

Works from any folder, e.g. `D:\GPU\speak.bat C:\rec\me.mp4 "Hi"`. Run `speak.bat -h` for all options.

| Option | What it does |
|---|---|
| `-o FILE` | `.mp4` = talking video; `.wav` (default without a face), `.flac`, `.mp3`, `.m4a`, `.ogg` = audio |
| `-f FILE` | Read the text from a `.txt` file |
| `-q fast/high/max` | 1, 3 (default) or 5 takes per sentence group; the best take wins |
| `--face FILE` | Video (or photo) to lip-sync; default: the recording itself when it is a video |
| `--engine qwen/f5` | Voice model: `qwen` (default) or `f5` (a second opinion, English/Chinese only) |
| `-l LANG` | Language of the text: `auto` (default), english, chinese, japanese, korean, german, french, russian, portuguese, spanish, italian |
| `--script` | Print the ~10 s text to read aloud in the recording |
| `--ref-text "..."` | Exact words spoken in the recording (under 15 s: used whole; longer: matched to the chosen clip) |
| `--ref-language LANG` | Language spoken in the recording, if the automatic guess is wrong |
| `--seed N` | Reproducible output (the seed of every run is printed) |
| `--pause 0.3` / `--paragraph-pause 0.7` | Silence between sentences / paragraphs (seconds) |
| `--loudness -18` | Output loudness in LUFS (-16 for podcasts) |
| `--lipsync auto` | Lip-sync engine: `latentsync` (most realistic, 15 GB+ GPU, slow), `musetalk` (fast, softer mouth); `auto` (default) = LatentSync where `setup.sh` installed it and the GPU is big enough |
| `--lipsync-steps 20` | LatentSync denoising steps; `12` is ~40% faster with slightly less detail |
| `--restore 0.8` | MuseTalk only: sharpen the mouth with GFPGAN: `0` off, `0.5` subtle, `1` strongest (default 0.8) |
| `--qwen-size auto` | Voice model: `auto` (default) picks 1.7B, the closer clone, on GPUs with 10 GB+ (Colab's T4) and 0.6B on smaller ones |
| `--refresh-voice` / `--refresh-face` | Re-analyse the recording / face video instead of using the cache |
| `-V`, `--save-takes` | Show every take's score / keep every take as a WAV |

## 3. What to expect

* **Voice:** cloned from the recording. Best of 3 takes, each checked by Whisper for misread words and
  by a speaker-verification model for how much it sounds like the person.
* **Lip sync:** the lower face is regenerated to match every syllable and blended into the real footage. Head
  motion, blinks and eyes are the person's own. If the script is longer than the video, the footage plays
  forward then backward, so there are no jumps.
  * **LatentSync 1.6** (15 GB+ GPUs, e.g. Colab): a latent diffusion model working on the face at 512×512,
    16 frames at a time, trained against a lip-sync expert network. Sharp, steady (no frame-to-frame jitter)
    and accurate: the realistic option. Slow: roughly 20–40 min per 30 s of video on a T4, a few minutes on
    an A100.
  * **MuseTalk 1.5** (any GPU, 4 GB is enough): one step at 256×256, ~3 s per second of video. The mouth is
    softer and can flicker slightly; good for previews.
* **Emotion:** the voice carries the tone of the text (punctuation, `!`, `?`) and of the recording. The face shows
  the expressions from the recording. It doesn't invent new ones per sentence (e.g. a sudden laugh); models
  that do need 24 GB+ GPUs.
* **Sharpness:** LatentSync's 512 px face matches a 720p frame, so the mouth is as sharp as the rest. MuseTalk
  generates the mouth at 256×256, which is soft on close-ups, so there each face is aligned to 512 px and
  sharpened with **GFPGAN** face restoration where the mouth was regenerated (`--restore`). Either way, filming
  at arm's length, head and shoulders in view, gives the most natural result.
* **Limits:** moustaches and exact lip colour can come out a little different. Side profiles don't work well.

Measured on the RTX A2000 Laptop (4 GB):

| Job | Time |
|---|---|
| First use of a new 30 s video (voice + face analysis, once) | ~3–4 min |
| 30 s script → talking video (voice best of 3 + lip sync) | ~2–3 min |
| One sentence, audio only, best of 3 | ~15–30 s |

On Colab's T4, a 30 s talking video with LatentSync is expected to take roughly 20–40 min (not yet measured).

Every video is tagged `AI-generated` in its MP4 metadata.

## 4. How it works

```
recording ─► voice clip: decode, trim, shorten pauses, pick best 5-12 s, level ─► transcript (script words
             or Whisper, language limited to what the voice model speaks) ─► cached in voices\
face video ─► 25 fps frames ─► face box per frame (68 landmarks, smoothed) ─► VAE latents + jaw-shaped blend
             masks (face parsing) ─► cached in avatars\
text ─► clean ─► sentence-aware chunks ─► Qwen3-TTS clones the voice: 3 takes per chunk (bf16, CUDA graphs)
check ─► Whisper word errors + WavLM voice match + pace ─► best take per chunk; misread chunks get another round
master ─► natural pauses, -18 LUFS, peak limiter
lip sync ─► Whisper-tiny audio features ─► MuseTalk UNet (one step, fp16) ─► VAE decode ─► blend mouth into
            the real frame ─► GFPGAN sharpens the mouth (aligned 512 px face, fixed noise) ─► H.264 + AAC MP4
LatentSync ─► face video as 25 fps 720p, forward + backward (cached) ─► voice made in a separate process that
            exits, freeing the GPU ─► LatentSync 1.6 in its own environment: InsightFace aligns each face to
            512 px ─► 20 DDIM steps per 16 frames ─► paste back ─► 30 s pieces joined to the frame ─► MP4
```

LatentSync pins an older stack (torch 2.5, diffusers 0.32, numpy 1.26), so `setup.sh` gives it its own Python 3.10
environment in `third_party\LatentSync\.venv`, and `vclone\latentsync_runner.py` runs there. Changes from its
own inference script so it fits Colab's 15 GB T4: fp16 on the T4, the VAE works one frame at a time, the model is
cast before its checkpoint is loaded (half the RAM), and 15 GB cards skip classifier-free guidance and use
DeepCache. 20 GB+ cards get the upstream settings (guidance 1.5, no DeepCache).

Only one model sits on the GPU at a time, so it all fits in 4 GB (peak ~2.7 GB). Details that keep it fast there:
* Qwen3-TTS's code predictor is replayed with **CUDA graphs** (`vclone\qwen_fast.py`, ~2× faster, identical
  logits in float32).
* The VAE decodes one face at a time, because batches spill into system RAM on a 4 GB card and run 5× slower.
* The face video is analysed once and cached.

| Voice engine | `qwen` (default) | `f5` |
|---|---|---|
| Model | Qwen3-TTS-12Hz-0.6B-Base | F5-TTS v1 Base |
| Languages | 10 (see `-l`) | English, Chinese |
| License | Apache-2.0 | CC-BY-NC-4.0 (non-commercial) |

## 5. Put it on GitHub and run it in Google Colab

**Push the code.** `.gitignore` keeps the models (~12 GB), the virtual environment, and all personal
data (`voices\`, `avatars\`, `outputs\`, audio and video files) out of the repository. Only the code goes up:

```bat
cd D:\GPU
git init
git add .
git status
git commit -m "vclone: voice cloning + lip-synced talking video"
git branch -M main
git remote add origin https://github.com/dhanuvagman006/VedioFaceClone.git
git config --global credential.helper manager
git push -u origin main
```

GitHub doesn't accept your account password for `git push`. The `credential.helper manager` line makes the push
open a browser window to sign in, and the login is then remembered.

Check that `git status` lists no `.mp4`, `.wav` or `.m4a` files before committing.

**Run it in Colab.**
1. Open **https://colab.research.google.com/github/dhanuvagman006/VedioFaceClone/blob/main/colab.ipynb**
2. **Runtime → Change runtime type → T4 GPU** (an L4 or A100 on Colab Pro is several times faster).
3. Run the cells from top to bottom. They install everything, download the models, ask you to upload the
   video, and give you the talking video to download.

Colab notes:
* A Colab machine is wiped when the session ends, so setup runs again in each new session (~10–15 min,
  including LatentSync's own environment). Set `KEEP_MODELS_ON_DRIVE = True` to keep the models in Google
  Drive instead (needs ~20 GB free).
* The T4 has 15 GB of VRAM, so the bigger 1.7B voice model (a closer clone) and LatentSync are used there
  automatically. For a quick preview set `LIPSYNC = "musetalk"` in the notebook (about 2 min).
* After pushing new code, re-run step 1 of the notebook: it pulls the latest version.
* For a **private** repo, the clone needs a GitHub token: `https://<token>@github.com/<you>/<repo>.git`.
  The repo holds no personal data, so a public one works too.
* On Linux or Colab use `./speak.sh` instead of `speak.bat`; `bash setup.sh` replaces `setup.ps1`.

## 6. Troubleshooting

* **Doesn't sound like the person**: have them read the `--script` text in the recording (in a quiet room);
  check the transcript it prints / `ref.txt`; try `--quality max`.
* **A word is mispronounced**: run again (new seed) or `--quality max`. Spell unusual names phonetically.
* **"A face was found in only N frames"** (MuseTalk) or **"Face not detected"** (LatentSync): use a video where
  the face stays visible and front-on in every frame.
* **LatentSync takes too long**: `--lipsync-steps 12`, a faster Colab GPU, or `--lipsync musetalk` for drafts.
* **Mouth looks soft** (MuseTalk on a 1080p close-up): expected at 256 px; use LatentSync, or film slightly
  further away.
* **GPU out of memory**: close other GPU apps (games, browsers with hardware video), use `--quality fast`.
* **The recording is in a language Qwen doesn't speak**: the voice is still cloned from its timbre
  (automatically, or force it with `--xvector-only`).

## 7. Files

```
vclone\            the pipeline: cli, pipeline, reference (voice clip), engines (TTS), quality (take checks),
                   lipsync (MuseTalk talking video), latentsync + latentsync_runner (LatentSync talking video),
                   restore + gfpgan_arch (mouth sharpening), text, audio, models, download
speak.bat/.sh      launchers (Windows / Linux + Colab)
setup.ps1/.sh      one-time setup (Windows / Linux + Colab): packages, MuseTalk + LatentSync code, models
colab.ipynb        Google Colab notebook
models\            downloaded models, ~12 GB (~20 GB with LatentSync)   (not in git)
third_party\       MuseTalk 1.5 and LatentSync 1.6 code at pinned commits, LatentSync's own .venv
                   (not in git; setup downloads it)
voices\ avatars\   cached voice clips and face analyses         (private, not in git)
outputs\           generated audio and video                    (private, not in git)
.venv\             Python 3.11, PyTorch 2.8 + CUDA 12.8 (Windows)
```

Python API (with this folder on `PYTHONPATH`):

```python
from vclone import speak
speak("person.mp4", "Hello from Python!", "hello.mp4")                  # talking video
speak("my_voice.m4a", "Hello!", "hello.wav", quality="fast", language="english")
```

## 8. Use it responsibly

Only clone people who agreed to it, and don't use clones to impersonate anyone or deceive viewers or listeners.
Outputs carry an `AI-generated` tag in their metadata. Licenses: MuseTalk (MIT; its weights allow commercial use),
Qwen3-TTS (Apache-2.0), GFPGAN (Apache-2.0) and Whisper (MIT) are permissive. LatentSync's code is Apache-2.0 and
its weights OpenRAIL++ (use restrictions, no misuse). Non-commercial only: the F5-TTS weights, and the InsightFace
face-detection models LatentSync uses. Check the other components' licenses before commercial use.
