"""Sharpen the lip-synced mouth with GFPGAN v1.4 face restoration.

MuseTalk regenerates the lower face at 256x256, so on a close-up video (a face 600-900 px wide) the lips,
teeth and moustache come out soft. Here each frame's face is aligned to GFPGAN's 512x512 FFHQ template
using the cached eye/nose/mouth landmarks, restored, and pasted back only where MuseTalk changed the
frame (its jaw mask). Eyes, hair and background stay the untouched original footage."""
from __future__ import annotations

import cv2
import numpy as np
import torch

from .models import GFPGAN_BYTES, GFPGAN_URL, url_file

SIZE = 512
# Where FFHQ-aligned 512 px faces have: left eye, right eye, nose tip, left and right mouth corner (facexlib).
TEMPLATE = np.array([[192.98138, 239.94708], [318.90277, 240.1936], [256.63416, 314.01935],
                     [201.26117, 371.41043], [313.08905, 371.15118]], np.float32)


def five_points(lm68: np.ndarray) -> np.ndarray:
    """Eye centres, nose tip and mouth corners from the 68 iBUG face landmarks."""
    return np.stack([lm68[36:42].mean(0), lm68[42:48].mean(0), lm68[30], lm68[48], lm68[54]]).astype(np.float32)


def align_matrix(points: np.ndarray) -> np.ndarray:
    """Similarity transform taking a frame's 5 face points onto the 512 px template."""
    matrix, _ = cv2.estimateAffinePartial2D(points.astype(np.float32), TEMPLATE, method=cv2.LMEDS)
    return matrix


class MouthRestorer:
    def __init__(self, device: str, strength: float = 0.8, batch: int = 4):
        from .gfpgan_arch import GFPGANv1Clean
        net = GFPGANv1Clean(out_size=SIZE, num_style_feat=512, channel_multiplier=2, decoder_load_path=None,
                            fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1,
                            sft_half=True)  # the GFPGAN v1.3/v1.4 configuration
        state = torch.load(url_file(GFPGAN_URL, "GFPGAN", GFPGAN_BYTES), map_location="cpu", weights_only=True)
        net.load_state_dict(state["params_ema"] if "params_ema" in state else state["params"], strict=True)
        self.net = net.to(device).eval()
        self.device, self.strength, self.batch = device, float(np.clip(strength, 0.0, 1.0)), batch
        # Fade the restoration out towards the edges of the aligned crop so it never leaves a seam.
        edge = np.zeros((SIZE, SIZE), np.float32)
        edge[24:-24, 24:-24] = 1.0
        self.edge = cv2.GaussianBlur(edge, (0, 0), 12)

    @torch.inference_mode()
    def restore(self, faces: list[np.ndarray]) -> list[np.ndarray]:
        """Aligned 512x512 BGR uint8 faces in, restored faces out."""
        # contiguous(): the permuted (channels-last) layout would make GFPGAN's internal .view() fail
        x = torch.from_numpy(np.stack(faces)[..., ::-1].copy()).to(self.device).permute(0, 3, 1, 2)
        x = x.contiguous().float()
        # Fixed (stored) noise instead of fresh random noise per frame: no shimmering texture in video.
        out = self.net(x / 127.5 - 1, return_rgb=False, randomize_noise=False)[0]
        out = ((out.clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()[..., ::-1]
        return [np.ascontiguousarray(face) for face in out]

    def process(self, frames: list[np.ndarray], points, masks, crop_boxes) -> list[np.ndarray]:
        """Sharpen the mouth area of already lip-synced frames, in place."""
        matrices = [align_matrix(p) for p in points]
        aligned = [cv2.warpAffine(frame, m, (SIZE, SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                                  borderValue=(135, 133, 132)) for frame, m in zip(frames, matrices)]
        restored = []
        for i in range(0, len(aligned), self.batch):
            restored.extend(self.restore(aligned[i:i + self.batch]))
        for frame, face, matrix, mask, crop_box in zip(frames, restored, matrices, masks, crop_boxes):
            self._paste(frame, face, matrix, mask, crop_box)
        return frames

    def _paste(self, frame: np.ndarray, face: np.ndarray, matrix: np.ndarray, mask: np.ndarray, crop_box) -> None:
        """Warp the restored face back into the frame, through MuseTalk's feathered jaw mask."""
        h, w = frame.shape[:2]
        cx1, cy1, cx2, cy2 = (int(v) for v in crop_box)
        ax1, ay1, ax2, ay2 = max(cx1, 0), max(cy1, 0), min(cx2, w), min(cy2, h)  # mask region inside the frame
        back = cv2.invertAffineTransform(matrix)
        back[:, 2] -= (ax1, ay1)  # warp straight into that region
        size = (ax2 - ax1, ay2 - ay1)
        face = cv2.warpAffine(face, back, size, flags=cv2.INTER_LINEAR)
        edge = cv2.warpAffine(self.edge, back, size, flags=cv2.INTER_LINEAR)
        alpha = mask[ay1 - cy1:ay2 - cy1, ax1 - cx1:ax2 - cx1].astype(np.float32) / 255.0 * edge * self.strength
        region = frame[ay1:ay2, ax1:ax2]
        frame[ay1:ay2, ax1:ax2] = (face * alpha[..., None] + region * (1.0 - alpha[..., None]) + 0.5).astype(np.uint8)
