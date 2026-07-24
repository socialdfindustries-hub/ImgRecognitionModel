"""
video_detect.py — run the detector on a VIDEO FILE and write an annotated copy.
Headless-OpenCV friendly (no GUI window). Weights load through the VERIFIED path.

  ./.venv/Scripts/python.exe scripts/video_detect.py \
      --video path/to/drone.mp4 --num-classes 10 --imgsz 1024 \
      --score 0.3 --out dist/drone_detected.mp4

Options: --tile 0 (whole frame). --max-frames 0 (all). --stride 1 (every frame).
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

VISDRONE = {0: "pedestrian", 1: "people", 2: "bicycle", 3: "car", 4: "van",
            5: "truck", 6: "tricycle", 7: "awn-tricycle", 8: "bus", 9: "motor"}


def draw(frame, boxes, scores, labels, imgsz):
    h, w = frame.shape[:2]
    sx, sy = w / imgsz, h / imgsz
    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        p1 = (int(x1 * sx), int(y1 * sy)); p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        name = VISDRONE.get(int(c), str(int(c)))
        cv2.putText(frame, f"{name}:{s:.2f}", (p1[0], max(p1[1] - 4, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("dist/detector.safetensors"))
    ap.add_argument("--video", type=Path, required=True, help="input video file")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--stride", type=int, default=1, help="process every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="0 = whole video")
    ap.add_argument("--out", type=Path, default=Path("dist/video_detected.mp4"))
    ap.add_argument("--verify-key", type=Path, default=Path("keys/model_ed25519.pub"))
    args = ap.parse_args()

    if not args.video.exists():
        raise SystemExit(f"video not found: {args.video}")

    device = pick_device()
    print(f"device: {device}")
    model = CustomDetector(args.num_classes).to(device).eval()
    model.load_state_dict(load_model(args.weights, args.verify_key, device), strict=False)
    print(f"loaded weights (signature verified): {args.weights}")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {args.video}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1280
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 720
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    out_fps = max(src_fps / max(args.stride, 1), 1.0)
    print(f"video: {w}x{h}, {src_fps:.1f} FPS, {total} frames -> processing every {args.stride}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.out), cv2.VideoWriter_fourcc(*"mp4v"), out_fps, (w, h))

    idx, kept, total_dets, t0 = 0, 0, 0, time.time()
    with torch.no_grad():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % args.stride != 0:
                idx += 1; continue

            resized = cv2.resize(frame, (args.imgsz, args.imgsz))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            boxes, scores, labels = detections_from_outputs(
                model(t), args.num_classes, args.imgsz, args.score)

            frame = draw(frame, boxes.cpu(), scores.cpu(), labels.cpu(), args.imgsz)
            cv2.putText(frame, f"frame {idx}   {len(boxes)} det", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            writer.write(frame)
            kept += 1; total_dets += len(boxes); idx += 1
            if kept % 25 == 0:
                print(f"  {kept} frames processed  ({total_dets} dets, "
                      f"{kept/max(time.time()-t0,1e-6):.1f} FPS)")
            if args.max_frames and kept >= args.max_frames:
                break

    cap.release(); writer.release()
    dur = time.time() - t0
    print(f"\ndone: {kept} frames, {total_dets} dets (avg {total_dets/max(kept,1):.1f}/frame) "
          f"in {dur:.1f}s ({kept/max(dur,1e-6):.1f} FPS)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
