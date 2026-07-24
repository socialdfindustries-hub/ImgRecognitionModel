"""
stream_detect.py — play a folder of images through the detector and write ONE
annotated playback video (a "live detection" reel). Headless-OpenCV friendly.

  ./.venv/Scripts/python.exe scripts/stream_detect.py \
      --images data/train/images --num-classes 10 --imgsz 1024 \
      --limit 250 --out dist/stream_detect.mp4

Samples evenly across the whole folder (--stride auto) so the reel shows variety
rather than one contiguous drone clip. Weights load through the VERIFIED path.
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
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
        name = VISDRONE.get(int(c), str(int(c)))
        cv2.putText(frame, f"{name}:{s:.2f}", (p1[0], max(p1[1] - 4, 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
    return frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("dist/detector.safetensors"))
    ap.add_argument("--images", type=Path, required=True, help="folder of images")
    ap.add_argument("--num-classes", type=int, default=10)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--limit", type=int, default=250, help="max frames in the reel")
    ap.add_argument("--stride", type=int, default=0, help="0 = auto-spread across folder")
    ap.add_argument("--fps", type=float, default=8.0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--out", type=Path, default=Path("dist/stream_detect.mp4"))
    ap.add_argument("--verify-key", type=Path, default=Path("keys/model_ed25519.pub"))
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    model = CustomDetector(args.num_classes).to(device).eval()
    model.load_state_dict(load_model(args.weights, args.verify_key, device), strict=False)
    print(f"loaded weights (signature verified): {args.weights}")

    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    files = sorted(p for p in args.images.iterdir() if p.suffix.lower() in exts)
    if not files:
        raise SystemExit(f"no images in {args.images}")
    stride = args.stride or max(1, len(files) // args.limit)
    picked = files[::stride][:args.limit]
    print(f"{len(files)} images found; sampling {len(picked)} (stride {stride}) -> reel")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(args.out), fourcc, args.fps, (args.width, args.height))

    total_dets, t0 = 0, time.time()
    with torch.no_grad():
        for i, fp in enumerate(picked):
            bgr = cv2.imread(str(fp))
            if bgr is None:
                continue
            resized = cv2.resize(bgr, (args.imgsz, args.imgsz))
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
            out = model(t)
            boxes, scores, labels = detections_from_outputs(
                out, args.num_classes, args.imgsz, args.score)

            bgr = draw(bgr, boxes.cpu(), scores.cpu(), labels.cpu(), args.imgsz)
            frame = cv2.resize(bgr, (args.width, args.height))
            cv2.putText(frame, f"{fp.name}   {len(boxes)} det", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
            writer.write(frame)
            total_dets += len(boxes)
            if (i + 1) % 25 == 0:
                print(f"  {i+1}/{len(picked)} frames  ({total_dets} dets so far)")

    writer.release()
    dur = time.time() - t0
    print(f"\ndone: {len(picked)} frames, {total_dets} total detections "
          f"(avg {total_dets/max(len(picked),1):.1f}/frame) in {dur:.1f}s")
    print(f"wrote {args.out}  ({len(picked)/args.fps:.0f}s reel @ {args.fps:.0f} FPS)")


if __name__ == "__main__":
    main()
