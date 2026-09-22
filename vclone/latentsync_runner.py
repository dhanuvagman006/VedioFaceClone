"""LatentSync 1.6 inference, run inside LatentSync's own Python environment (third_party/LatentSync/.venv)
with LatentSync's folder as the working directory. vclone copies this file there and starts it; it does
not import vclone (different packages and versions).

It mirrors LatentSync's scripts/inference.py (Apache-2.0, ByteDance) with changes for Colab-class GPUs:
- fp16 on compute capability 7.x too (Colab's T4); upstream only enables fp16 above 7, which makes the
  T4 run in fp32: twice the memory and several times slower.
- The VAE encodes/decodes one frame at a time. Upstream does 16 frames at 512 px at once, which is most
  of its 18 GB requirement.
- Settings follow the GPU: 20 GB+ gets upstream's quality settings (guidance 1.5, no DeepCache); smaller
  cards (15 GB T4) skip classifier-free guidance, which halves the memory, and use DeepCache (~2x faster).
- The VAE is loaded from vclone's models folder instead of downloading it again.
- The UNet is cast to fp16 before its checkpoint is (memory-mapped and) loaded: ~5 GB less peak RAM.
- Fewer lossy re-encodes: the input (already 25 fps) is read as is, and the output is LatentSync's own
  frame file (crf 13), not its re-encoded copy with 16 kHz audio (vclone adds the full-quality voice).
- --frames keeps exactly the frames that belong to the audio (LatentSync adds ~2), so parts join without drift."""
import argparse
import os

import torch
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf

import latentsync.pipelines.lipsync_pipeline as lipsync_pipeline
from latentsync.models.unet import UNet3DConditionModel
from latentsync.pipelines.lipsync_pipeline import LipsyncPipeline
from latentsync.utils import util
from latentsync.whisper.audio2feature import Audio2Feature


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True, help="25 fps face video, at least as long as the audio")
    p.add_argument("--audio", required=True)
    p.add_argument("--out", required=True, help="lip-synced video without audio")
    p.add_argument("--vae", required=True, help="folder with sd-vae-ft-mse")
    p.add_argument("--config", default="configs/unet/stage2_512.yaml")
    p.add_argument("--ckpt", default="checkpoints/latentsync_unet.pt")
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--seed", type=int, default=1247)
    p.add_argument("--frames", type=int, help="keep this many frames (default: all)")
    p.add_argument("--temp", required=True, help="scratch folder (emptied first)")
    p.add_argument("--guidance", type=float, default=None, help="default: 1.5 on 20 GB+ GPUs, else 1.0")
    p.add_argument("--deepcache", choices=["auto", "on", "off"], default="auto")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("error: LatentSync needs an NVIDIA GPU")
    config = OmegaConf.load(args.config)
    props = torch.cuda.get_device_properties(0)
    big = props.total_memory >= 20 * 2**30
    dtype = torch.float16 if torch.cuda.get_device_capability()[0] >= 7 else torch.float32
    guidance = args.guidance if args.guidance is not None else (1.5 if big else 1.0)
    deepcache = args.deepcache == "on" or (args.deepcache == "auto" and not big)
    print(f"[face] LatentSync 1.6 on {props.name} ({props.total_memory / 2**30:.0f} GB): "
          f"{str(dtype).replace('torch.', '')}, guidance {guidance}, DeepCache {'on' if deepcache else 'off'}, "
          f"{args.steps} steps", flush=True)

    # The pipeline looks these up by name when it runs, so replacing them here changes only this run.
    lipsync_pipeline.read_video = lambda path, use_decord=False: util.read_video(
        path, change_fps=False, use_decord=use_decord)
    if args.frames:
        lipsync_pipeline.write_video = lambda path, frames, fps: util.write_video(path, frames[:args.frames], fps)

    scheduler = DDIMScheduler.from_pretrained("configs")
    whisper = "checkpoints/whisper/tiny.pt" if config.model.cross_attention_dim == 384 else "checkpoints/whisper/small.pt"
    audio_encoder = Audio2Feature(model_path=whisper, device="cuda", num_frames=config.data.num_frames,
                                  audio_feat_length=config.data.audio_feat_length)
    vae = AutoencoderKL.from_pretrained(args.vae, torch_dtype=dtype)
    vae.config.scaling_factor = 0.18215
    vae.config.shift_factor = 0
    vae.enable_slicing()
    # Upstream's from_pretrained holds the fp32 model and the whole 5 GB checkpoint in RAM together (~10 GB,
    # near the limit of Colab's 12.7 GB). Cast first, and memory-map the checkpoint: same weights, half the RAM.
    unet = UNet3DConditionModel.from_config(OmegaConf.to_container(config.model)).to(dtype=dtype)
    try:
        state = torch.load(args.ckpt, map_location="cpu", mmap=True, weights_only=True)
    except RuntimeError:  # only zip-format checkpoints can be memory-mapped
        state = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    unet.load_state_dict(state["state_dict"], strict=False)  # strict=False as upstream
    del state
    pipeline = LipsyncPipeline(vae=vae, audio_encoder=audio_encoder, unet=unet, scheduler=scheduler).to("cuda")
    if deepcache:
        from DeepCache import DeepCacheSDHelper
        helper = DeepCacheSDHelper(pipe=pipeline)
        helper.set_params(cache_interval=3, cache_branch_id=0)
        helper.enable()
    set_seed(args.seed)
    muxed = os.path.join(args.temp, "muxed.mp4")
    pipeline(
        video_path=args.video,
        audio_path=args.audio,
        video_out_path=muxed,
        num_frames=config.data.num_frames,
        num_inference_steps=args.steps,
        guidance_scale=guidance,
        weight_dtype=dtype,
        width=config.data.resolution,
        height=config.data.resolution,
        mask_image_path=config.data.mask_image_path,
        temp_dir=args.temp,
    )
    frames_file = os.path.join(args.temp, "video.mp4")  # written by the pipeline before it muxes
    os.replace(frames_file if os.path.exists(frames_file) else muxed, args.out)
    print(f"[face] LatentSync peak GPU memory: {torch.cuda.max_memory_reserved() / 2**30:.1f} of "
          f"{props.total_memory / 2**30:.0f} GB", flush=True)


if __name__ == "__main__":
    main()
