"""
convert_visdrone.py — reformat raw VisDrone-DET annotations into the LABEL TEXT
format this project's own src/data/dataset.py reads: one .txt per image with
`cls cx cy w h` (all normalized 0..1, space-separated).

This is a pure text/label reformat. It does NOT use YOLO, Ultralytics, or any
third-party detector — it only rewrites the annotation files into the same plain
convention dataset.py already expects. After this, feed the result to
scripts/ingest_dataset.py (which validates + hash-pins it) like any other data.

Raw VisDrone annotation line (comma-separated):
    x, y, w, h, score, category, truncation, occlusion
  category: 1=pedestrian 2=people 3=bicycle 4=car 5=van 6=truck 7=tricycle
            8=awning-tricycle 9=bus 10=motor   (0=ignored-region, 11=other -> dropped)
  output class id = category - 1   -> 0..9

Usage:
    python scripts/convert_visdrone.py --src VisDrone2019-DET-train --out data_src/train
    python scripts/convert_visdrone.py --src VisDrone2019-DET-val   --out data_src/val
    # then:
    python scripts/ingest_dataset.py --src data_src/train --num-classes 10 ...
"""
import argparse
import os
from pathlib import Path

from PIL import Image

IMG_EXTS = (".jpg", ".jpeg", ".png")


def convert(src: Path, out: Path, link: bool) -> int:
    img_dir = src / "images"
    ann_dir = src / "annotations"
    if not img_dir.is_dir() or not ann_dir.is_dir():
        raise SystemExit(f"expected {img_dir} and {ann_dir} (raw VisDrone layout)")
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    n_img = n_box = 0
    for ann in sorted(ann_dir.glob("*.txt")):
        stem = ann.stem
        img = next((img_dir / (stem + e) for e in IMG_EXTS
                    if (img_dir / (stem + e)).exists()), None)
        if img is None:
            continue
        W, H = Image.open(img).size
        lines = []
        for row in ann.read_text().splitlines():
            parts = row.strip().rstrip(",").split(",")
            if len(parts) < 6:
                continue
            try:
                x, y, w, h = (float(v) for v in parts[:4])
                cat = int(float(parts[5]))
            except ValueError:
                continue
            if cat < 1 or cat > 10 or w <= 0 or h <= 0:
                continue                       # drop ignored(0)/other(11)/degenerate
            cls = cat - 1
            cx = (x + w / 2) / W
            cy = (y + h / 2) / H
            # clamp to [0,1] so a box hugging the border can't emit >1 coords
            cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
            lines.append(f"{cls} {cx:.6f} {cy:.6f} {w / W:.6f} {h / H:.6f}")
        if not lines:
            continue                            # skip images with no kept objects
        dst_img = out / "images" / img.name
        if not dst_img.exists():
            if link:
                os.symlink(img.resolve(), dst_img)
            else:
                dst_img.write_bytes(img.read_bytes())
        (out / "labels" / (stem + ".txt")).write_text("\n".join(lines))
        n_img += 1
        n_box += len(lines)
    print(f"converted {n_img} images, {n_box} boxes -> {out}")
    return n_img


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True,
                    help="raw VisDrone split dir (has images/ and annotations/)")
    ap.add_argument("--out", type=Path, required=True,
                    help="output dir; writes images/ + labels/ in project label format")
    ap.add_argument("--copy", action="store_true",
                    help="copy images instead of symlinking (use when the source "
                         "is a temp/Drive path that won't persist)")
    a = ap.parse_args()
    convert(a.src, a.out, link=not a.copy)
