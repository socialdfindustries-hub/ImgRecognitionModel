"""
dataset.py — aerial detection dataset.

Reads images + YOLO-format labels (one .txt per image: `cls cx cy w h`, all
normalized 0..1). Before training, the split's bytes are verified against a
pinned hash manifest (integrity.py) so you always know exactly what you trained
on — a swapped/poisoned dataset is caught here.

Two sampling modes:
  * val / whole-frame  (train=False or tile=0): the full image is resized to
    imgsz. Boxes are scaled to match. This is the honest, deterministic view used
    for evaluation.
  * train tile-crop    (train=True and tile>0): a `tile`x`tile` window is cropped
    at FULL resolution and resized to imgsz. This is the small-object lever for
    altitude — objects keep their native pixel scale instead of being crushed by a
    single whole-frame downscale. A per-sample scale jitter widens/narrows the
    window (crop_scale) to simulate a range of altitudes/GSDs, and horizontal-flip
    + brightness/contrast jitter add appearance variety. Boxes are cropped,
    rescaled, clipped, and sub-`min_box_px` fragments dropped.

Return format is identical in both modes: (image C,H,W float32 in 0..1, target
dict with `boxes` xyxy in imgsz pixels and int64 `labels`) — so the training loop,
loss, and anchor matching are unchanged.
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
                 manifest: Path | None = None, *,
                 train: bool = False, tile: int = 0,
                 crop_scale: tuple[float, float] = (0.7, 1.8),
                 flip_p: float = 0.5, photometric: bool = True,
                 min_box_px: float = 2.0, seed: int | None = None):
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.imgsz = imgsz
        # augmentation config (only consulted when train=True)
        self.train = train
        self.tile = int(tile)
        self.crop_scale = crop_scale
        self.flip_p = flip_p
        self.photometric = photometric
        self.min_box_px = float(min_box_px)
        self._seed = seed
        self._gen: np.random.Generator | None = None
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

    def _rng(self) -> np.random.Generator:
        # Lazily built so each DataLoader worker process gets an independent
        # stream (workers are forked copies; a shared generator would replay the
        # same crops in every worker). Seed only for deterministic tests.
        if self._gen is None:
            self._gen = np.random.default_rng(self._seed)
        return self._gen

    def __getitem__(self, i: int):
        img_path = self.items[i]
        if cv2 is None:
            raise RuntimeError("opencv not installed; `pip install -r requirements.txt`")
        img = cv2.imread(str(img_path))              # BGR HxWxC, full resolution
        if img is None:
            raise RuntimeError(f"unreadable image: {img_path}")
        h0, w0 = img.shape[:2]

        boxes, labels = self._load_labels(img_path, w0, h0)  # xyxy px (full-res)

        if self.train and self.tile > 0:
            img, boxes, labels = self._tile_crop(img, boxes, labels)
        else:
            img, boxes = self._resize_whole(img, boxes, w0, h0)

        if self.train:
            if self._rng().random() < self.flip_p:
                img, boxes = self._hflip(img, boxes)
            if self.photometric:
                img = self._jitter(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # C,H,W
        target = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64).reshape(-1),
            "image_path": str(img_path),
        }
        return img, target

    # --- geometry helpers ---------------------------------------------------
    def _resize_whole(self, img, boxes, w0, h0):
        """Whole frame -> imgsz square. Deterministic; used for val."""
        img = cv2.resize(img, (self.imgsz, self.imgsz))
        if len(boxes):
            boxes = boxes.copy()
            boxes[:, [0, 2]] *= self.imgsz / w0
            boxes[:, [1, 3]] *= self.imgsz / h0
        return img, boxes

    def _tile_crop(self, img, boxes, labels):
        """Crop a scale-jittered `tile` window at full res, resize to imgsz.
        Objects keep native pixel scale (small-object lever). Bias the crop toward
        a ground-truth box so most tiles carry at least one positive."""
        H, W = img.shape[:2]
        lo, hi = self.crop_scale
        cs = int(round(self.tile * self._rng().uniform(lo, hi)))
        cs = max(16, min(cs, H, W))                  # square crop, can't exceed image
        if len(boxes) and self._rng().random() < 0.8:
            k = int(self._rng().integers(0, len(boxes)))
            bx = 0.5 * (boxes[k, 0] + boxes[k, 2])
            by = 0.5 * (boxes[k, 1] + boxes[k, 3])
            x0 = int(round(bx - cs / 2))
            y0 = int(round(by - cs / 2))
        else:
            x0 = int(self._rng().integers(0, W - cs + 1)) if W > cs else 0
            y0 = int(self._rng().integers(0, H - cs + 1)) if H > cs else 0
        x0 = min(max(x0, 0), max(W - cs, 0))
        y0 = min(max(y0, 0), max(H - cs, 0))

        crop = img[y0:y0 + cs, x0:x0 + cs]
        crop = cv2.resize(crop, (self.imgsz, self.imgsz))
        scale = self.imgsz / cs
        if len(boxes):
            b = boxes.copy()
            b[:, [0, 2]] = (b[:, [0, 2]] - x0) * scale
            b[:, [1, 3]] = (b[:, [1, 3]] - y0) * scale
            boxes, labels = self._clip_filter(b, labels)
        return crop, boxes, labels

    def _clip_filter(self, boxes, labels):
        """Clip boxes to the imgsz canvas and drop anything that fell outside or
        shrank below min_box_px (sub-pixel fragments = label noise)."""
        boxes = boxes.copy()
        # Basic slices (0::2 -> x cols, 1::2 -> y cols) so the assignment writes
        # back into `boxes`. Fancy indexing like boxes[:, [0, 2]] returns a COPY,
        # so np.clip(out=...) on it would silently no-op.
        boxes[:, 0::2] = np.clip(boxes[:, 0::2], 0, self.imgsz)
        boxes[:, 1::2] = np.clip(boxes[:, 1::2], 0, self.imgsz)
        w = boxes[:, 2] - boxes[:, 0]
        h = boxes[:, 3] - boxes[:, 1]
        keep = (w >= self.min_box_px) & (h >= self.min_box_px)
        return boxes[keep], np.asarray(labels)[keep]

    def _hflip(self, img, boxes):
        img = np.ascontiguousarray(img[:, ::-1, :])
        if len(boxes):
            boxes = boxes.copy()
            x1 = boxes[:, 0].copy()
            x2 = boxes[:, 2].copy()
            boxes[:, 0] = self.imgsz - x2
            boxes[:, 2] = self.imgsz - x1
        return img, boxes

    def _jitter(self, img):
        """Brightness/contrast jitter. Keeps uint8 so the downstream cvtColor is
        unchanged; robustness to exposure without touching geometry."""
        alpha = float(self._rng().uniform(0.8, 1.2))   # contrast
        beta = float(self._rng().uniform(-15, 15))     # brightness
        out = img.astype(np.float32) * alpha + beta
        return np.clip(out, 0, 255).astype(np.uint8)

    # --- labels -------------------------------------------------------------
    def _load_labels(self, img_path: Path, w0: int, h0: int):
        """Return (boxes xyxy in FULL-RES pixels, labels) as numpy arrays."""
        lbl = self.labels_dir / (img_path.stem + ".txt")
        boxes, labels = [], []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                c, cx, cy, w, h = map(float, parts)
                x1 = (cx - w / 2) * w0
                y1 = (cy - h / 2) * h0
                x2 = (cx + w / 2) * w0
                y2 = (cy + h / 2) * h0
                boxes.append([x1, y1, x2, y2])
                labels.append(int(c))
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        return boxes, labels


def collate(batch):
    """Detection batches have variable #boxes per image -> keep targets as a list."""
    imgs = torch.stack([b[0] for b in batch], 0)
    targets = [b[1] for b in batch]
    return imgs, targets
