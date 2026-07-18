"""
validate.py — the dataset doctor. Refuse to train on bad labels.

From-scratch training has no pretrained prior to fall back on, so it amplifies
label noise directly into lost accuracy. Every image/label pair is checked here
before it is allowed into a training manifest.

Checks per label file (YOLO format: `cls cx cy w h`, normalized 0..1):
  - class id is an integer in [0, num_classes)
  - exactly 5 fields per line
  - cx,cy,w,h in [0,1]; w,h > 0; box stays inside the frame
  - image is readable and non-empty
  - flags images with NO label (background) vs missing-label mistakes
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

try:
    import cv2
except Exception:
    cv2 = None


@dataclass
class Report:
    images: int = 0
    labeled: int = 0
    background: int = 0        # image with an empty/absent label file
    boxes: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    class_counts: dict[int, int] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors

    def summary(self) -> str:
        lines = [
            f"images={self.images} labeled={self.labeled} "
            f"background={self.background} boxes={self.boxes}",
            f"classes: {dict(sorted(self.class_counts.items()))}",
            f"errors={len(self.errors)} warnings={len(self.warnings)}",
        ]
        for e in self.errors[:20]:
            lines.append(f"  ERROR: {e}")
        for w in self.warnings[:10]:
            lines.append(f"  warn:  {w}")
        return "\n".join(lines)


def validate_split(images_dir: Path, labels_dir: Path, num_classes: int,
                   check_images: bool = True) -> Report:
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    exts = {".jpg", ".jpeg", ".png"}
    rep = Report()
    imgs = sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in exts)
    if not imgs:
        rep.errors.append(f"no images under {images_dir}")
        return rep

    for img in imgs:
        rep.images += 1
        if check_images and cv2 is not None:
            m = cv2.imread(str(img))
            if m is None or m.size == 0:
                rep.errors.append(f"unreadable/empty image: {img.name}")
                continue

        lbl = labels_dir / (img.stem + ".txt")
        if not lbl.exists() or not lbl.read_text().strip():
            rep.background += 1
            continue

        rep.labeled += 1
        for ln, line in enumerate(lbl.read_text().splitlines(), 1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                rep.errors.append(f"{lbl.name}:{ln} expected 5 fields, got {len(parts)}")
                continue
            try:
                c = int(float(parts[0]))
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                rep.errors.append(f"{lbl.name}:{ln} non-numeric field")
                continue
            if not (0 <= c < num_classes):
                rep.errors.append(f"{lbl.name}:{ln} class {c} out of range [0,{num_classes})")
            if not (0 < w <= 1 and 0 < h <= 1):
                rep.errors.append(f"{lbl.name}:{ln} bad w/h ({w:.3f},{h:.3f})")
            if not (0 <= cx <= 1 and 0 <= cy <= 1):
                rep.errors.append(f"{lbl.name}:{ln} center outside frame ({cx:.3f},{cy:.3f})")
            if cx - w / 2 < -1e-3 or cx + w / 2 > 1 + 1e-3 or \
               cy - h / 2 < -1e-3 or cy + h / 2 > 1 + 1e-3:
                rep.warnings.append(f"{lbl.name}:{ln} box extends past frame edge")
            rep.boxes += 1
            rep.class_counts[c] = rep.class_counts.get(c, 0) + 1

    if rep.labeled == 0:
        rep.errors.append("no labeled images found — nothing to learn from")
    if rep.background > rep.labeled and rep.labeled > 0:
        rep.warnings.append(
            f"more background ({rep.background}) than labeled ({rep.labeled}) "
            "images — confirm this is intentional")
    # class balance warning
    if rep.class_counts:
        mx = max(rep.class_counts.values())
        for c, n in rep.class_counts.items():
            if n * 20 < mx:
                rep.warnings.append(f"class {c} has only {n} boxes vs {mx} for the top "
                                    "class — likely too few to learn well")
    return rep
