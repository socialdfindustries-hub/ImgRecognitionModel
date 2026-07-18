"""
dataset.py — aerial detection dataset.

Reads images + YOLO-format labels (one .txt per image: `cls cx cy w h`, all
normalized 0..1). Before training, the split's bytes are verified against a
pinned hash manifest (integrity.py) so you always know exactly what you trained
on — a swapped/poisoned dataset is caught here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    import cv2
except Exception:  # keep import-safe on machines without opencv yet
    cv2 = None


class AerialDetectionDataset(Dataset):
    def __init__(self, images_dir: Path, labels_dir: Path, imgsz: int = 1280,
                 manifest: Path | None = None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.imgsz = imgsz
        exts = {".jpg", ".jpeg", ".png"}
        self.items = sorted(
            p for p in self.images_dir.rglob("*") if p.suffix.lower() in exts)
        if not self.items:
            raise FileNotFoundError(f"no images under {self.images_dir}")
        if manifest is not None:
            from .integrity import verify_manifest
            verify_manifest(self.images_dir, manifest)  # untrusted data gate

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        img_path = self.items[i]
        if cv2 is None:
            raise RuntimeError("opencv not installed; `pip install -r requirements.txt`")
        img = cv2.imread(str(img_path))              # BGR HxWxC
        if img is None:
            raise RuntimeError(f"unreadable image: {img_path}")
        h0, w0 = img.shape[:2]
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # C,H,W

        boxes, labels = self._load_labels(img_path)
        target = {
            "boxes": boxes,          # xyxy in pixels of the resized image
            "labels": labels,        # int64 class ids
            "image_path": str(img_path),
        }
        return img, target

    def _load_labels(self, img_path: Path):
        lbl = self.labels_dir / (img_path.stem + ".txt")
        boxes, labels = [], []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, w, h = map(float, parts)
                x1 = (cx - w / 2) * self.imgsz
                y1 = (cy - h / 2) * self.imgsz
                x2 = (cx + w / 2) * self.imgsz
                y2 = (cy + h / 2) * self.imgsz
                boxes.append([x1, y1, x2, y2])
                labels.append(int(c))
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(labels, dtype=torch.int64)
        return boxes, labels


def collate(batch):
    """Detection batches have variable #boxes per image -> keep targets as a list."""
    imgs = torch.stack([b[0] for b in batch], 0)
    targets = [b[1] for b in batch]
    return imgs, targets
