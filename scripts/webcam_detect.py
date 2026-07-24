"""
webcam_detect.py — run the detector LIVE on the laptop webcam and write an
annotated output video. Headless-OpenCV friendly: no cv2.imshow / GUI window,
so it works with the pinned opencv-python-headless build.

  ./.venv/Scripts/python.exe scripts/webcam_detect.py \
      --weights dist/detector.safetensors --num-classes 10 \
      --seconds 15 --out webcam_detect.mp4 --save-frames frames

Weights load through the VERIFIED path (secure_io.load_model checks the Ed25519
signature first). Capture runs for --seconds, then the annotated .mp4 is written.

NOTE: this model was trained on VisDrone AERIAL imagery (top-down drone view), so a
normal indoor webcam scene is out-of-domain — expect few/no meaningful detections.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.model import CustomDetector
from src.model.postprocess import detections_from_outputs
from src.secure_io import load_model
from src.train import pick_device

# VisDrone 10-class names (configs/dataset.yaml only names 5 placeholders, so we
# label with the real classes the model was trained on).
VISDRONE = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
            5: "truck", 6: "tricycle", 7: "awn-tricycle", 8: "bus", 9: "motor"}


def draw(frame, boxes, scores, labels, imgsz):
    h, w = frame.shape[:2]
    sx, sy = w / imgsz, h / imgsz
    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        name = VISDRONE.get(int(c), str(int(c)))
        cv2.putText(frame, f"{name}:{s:.2f}", (p1[0], max(p1[1] - 4, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("dist/detector.safetensors"))
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--camera", type=int, default=0, help="webcam index (0 = default)")
    ap.add_argument("--seconds", type=float, default=15.0, help="how long to capture")
    ap.add_argument("--imgsz", type=int, default=768)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=Path("webcam_detect.mp4"))
    ap.add_argument("--save-frames", type=Path, default=None,
                    help="optional dir: also save annotated stills every ~1s")
    ap.add_argument("--verify-key", type=Path, default=Path("keys/model_ed25519.pub"))
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    model = CustomDetector(args.num_classes).to(device).eval()
    model.load_state_dict(load_model(args.weights, args.verify_key, device), strict=False)
    print(f"loaded weights (signature verified): {args.weights}")

    # CAP_DSHOW is the most reliable backend on Windows.
    cap = cv2.VideoCapture(args.camera, cv2.CAP_DSHOW)
    if not cap.isOpened():
        raise SystemExit(
            f"cannot open camera {args.camera}. Check it's not in use by another app, "
            "and that camera access is allowed for desktop apps in Windows Privacy settings.")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    print(f"camera opened: {w}x{h}, capturing for {args.seconds:.0f}s ...")

    if args.save_frames:
        args.save_frames.mkdir(parents=True, exist_ok=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    n_frames, total_dets, last_save = 0, 0, 0.0
    t0 = time.time()

    with torch.no_grad():
        while time.time() - t0 < args.seconds:
            ok, frame = cap.read()
            if not ok:
                print("frame grab failed; stopping."); break

            resized = cv2.resize(frame, (args.imgsz, args.imgsz))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model(t)
            boxes, scores, labels = detections_from_outputs(
                out, args.num_classes, args.imgsz, args.score)

            frame = draw(frame, boxes.cpu(), scores.cpu(), labels.cpu(), args.imgsz)
            elapsed = time.time() - t0
            fps = (n_frames + 1) / max(elapsed, 1e-6)
            cv2.putText(frame, f"{len(boxes)} det  {fps:.1f} FPS", (8, 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

            if writer is None:
                writer = cv2.VideoWriter(str(args.out), fourcc, 15.0, (frame.shape[1], frame.shape[0]))
            writer.write(frame)

            if args.save_frames and (elapsed - last_save) >= 1.0:
                cv2.imwrite(str(args.save_frames / f"frame_{n_frames:04d}.jpg"), frame)
                last_save = elapsed

            n_frames += 1
            total_dets += len(boxes)
            if n_frames % 10 == 0:
                print(f"  t={elapsed:4.1f}s  frames={n_frames}  fps={fps:.1f}  "
                      f"dets(last frame)={len(boxes)}")

    cap.release()
    if writer is not None:
        writer.release()
    dur = time.time() - t0
    print(f"\ndone: {n_frames} frames in {dur:.1f}s ({n_frames/max(dur,1e-6):.1f} FPS), "
          f"{total_dets} total detections")
    print(f"wrote {args.out}")
    if args.save_frames:
        print(f"sample stills in {args.save_frames}/")


if __name__ == "__main__":
    main()
