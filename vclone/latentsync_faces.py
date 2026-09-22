"""Face check for LatentSync, run inside its own environment (third_party/LatentSync/.venv) with LatentSync's
folder as the working directory. vclone copies this file there and starts it; it does not import vclone.

For every frame of a video: would LatentSync's face detector accept a face there, and where are its eyes,
nose tip and mouth corners. Same InsightFace model, input size, colour order and size/shape limits as
latentsync/utils/face_detector.py, a little stricter on confidence (re-encoding can move a borderline
score). vclone then replays only footage where the person faces the camera: LatentSync stops at any frame
without a face, and a moment where the person looks away would show up in every video."""
import argparse

import cv2
import numpy as np
from insightface.app import FaceAnalysis


def pick(faces, min_score: float):
    """The face LatentSync would use: the largest one within its size and shape limits."""
    best, best_area = None, 0
    for face in faces:
        x1, y1, x2, y2 = face.bbox.astype(int).tolist()
        w, h = x2 - x1, y2 - y1
        if w < 50 or h < 80 or not 0.2 <= w / h <= 1.5 or face.det_score < min_score:
            continue
        if w * h > best_area:
            best, best_area = face, w * h
    return best


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--out", required=True, help=".npz with found (N,) and points (N, 5, 2)")
    p.add_argument("--min-score", type=float, default=0.6, help="LatentSync accepts 0.5")
    args = p.parse_args()

    app = FaceAnalysis(allowed_modules=["detection"], root="checkpoints/auxiliary",
                       providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(512, 512))
    cap = cv2.VideoCapture(args.video)
    found, points = [], []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        face = pick(app.get(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), args.min_score)  # LatentSync passes RGB
        found.append(face is not None)
        points.append(face.kps if face is not None else np.full((5, 2), np.nan))
        if len(found) % 250 == 0:
            print(f"[face] checking the face: {len(found)} frames", flush=True)
    cap.release()
    np.savez(args.out, found=np.array(found, bool), points=np.array(points, np.float32).reshape(-1, 5, 2))
    print(f"[face] checked {len(found)} frames: a face in {sum(found)}", flush=True)


if __name__ == "__main__":
    main()
