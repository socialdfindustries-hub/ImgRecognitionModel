"""
predict.py — run the detector on ANY custom image and save an annotated copy.

  ./.venv/bin/python -m src.predict \
      --weights dist/detector.safetensors \
      --num-classes 10 --image /path/to/your.jpg --out out.jpg

Loads weights through the VERIFIED path (secure_io.load_model checks the Ed25519
signature first). Works with or without tiling (--tile for small aerial objects).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from .model import CustomDetector
from .model.postprocess import detections_from_outputs, tile_image
from .secure_io import load_model
from .train import pick_device


def load_image(path: Path, imgsz: int):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise FileNotFoundError(f"cannot read image: {path}")
    h0, w0 = bgr.shape[:2]
    resized = cv2.resize(bgr, (imgsz, imgsz))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    t = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()
    return t, bgr, (w0, h0)


def draw(bgr, boxes, scores, labels, imgsz, orig_wh):
    w0, h0 = orig_wh
    sx, sy = w0 / imgsz, h0 / imgsz
    for (x1, y1, x2, y2), s, c in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
        p1 = (int(x1 * sx), int(y1 * sy))
        p2 = (int(x2 * sx), int(y2 * sy))
        cv2.rectangle(bgr, p1, p2, (0, 255, 0), 2)
        cv2.putText(bgr, f"{c}:{s:.2f}", (p1[0], max(p1[1] - 4, 0)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return bgr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--image", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("prediction.jpg"))
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--score", type=float, default=0.3)
    ap.add_argument("--verify-key", type=Path, default=Path("keys/model_ed25519.pub"))
    ap.add_argument("--tile", type=int, default=0, help="tile size (0 = whole frame)")
    args = ap.parse_args()

    device = pick_device()
    model = CustomDetector(args.num_classes).to(device).eval()
    model.load_state_dict(load_model(args.weights, args.verify_key, device),
                          strict=False)

    img, bgr, orig_wh = load_image(args.image, args.imgsz)

    if args.tile and args.tile < args.imgsz:
        boxes, scores, labels = [], [], []
        for tile, xo, yo in tile_image(img, args.tile):
            pad = torch.zeros(3, args.tile, args.tile)
            pad[:, :tile.shape[1], :tile.shape[2]] = tile
            out = model(pad.unsqueeze(0).to(device))
            b, s, l = detections_from_outputs(out, args.num_classes, args.tile, args.score)
            b = b.clone(); b[:, 0::2] += xo; b[:, 1::2] += yo
            boxes.append(b); scores.append(s); labels.append(l)
        boxes = torch.cat(boxes) if boxes else torch.empty(0, 4)
        scores = torch.cat(scores) if scores else torch.empty(0)
        labels = torch.cat(labels) if labels else torch.empty(0, dtype=torch.int64)
    else:
        out = model(img.unsqueeze(0).to(device))
        boxes, scores, labels = detections_from_outputs(
            out, args.num_classes, args.imgsz, args.score)

    print(f"{len(boxes)} detections above score={args.score}")
    annotated = draw(bgr, boxes.cpu(), scores.cpu(), labels.cpu(), args.imgsz, orig_wh)
    cv2.imwrite(str(args.out), annotated)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
