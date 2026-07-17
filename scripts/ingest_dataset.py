"""
ingest_dataset.py — bring YOUR labeled data into the project, safely.

Accepts a source folder of images + YOLO-format .txt labels (e.g. a CVAT or
Roboflow "YOLO" export). It:
  1. VALIDATES every label (src/data/validate.py) and REFUSES to proceed on errors
  2. splits into train/val (deterministic, seeded by filename hash — no RNG)
  3. copies into data/train, data/val
  4. builds hash-pinned integrity manifests (data/manifests/*.json)

  ./.venv/bin/python scripts/ingest_dataset.py \
      --src /path/to/export --num-classes 6 --val-frac 0.2

Deployed model trains ONLY on data that passed this gate.
"""
import argparse
import hashlib
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data.validate import validate_split
from src.data.integrity import build_manifest

EXTS = {".jpg", ".jpeg", ".png"}


def find_pairs(src: Path):
    """Locate images and their sibling .txt labels (any subfolder layout)."""
    imgs = sorted(p for p in src.rglob("*") if p.suffix.lower() in EXTS)
    pairs = []
    for img in imgs:
        # look for label next to the image, or in a parallel 'labels' dir
        cands = [img.with_suffix(".txt"),
                 Path(str(img.parent).replace("images", "labels")) / (img.stem + ".txt")]
        lbl = next((c for c in cands if c.exists()), None)
        pairs.append((img, lbl))
    return pairs


def in_val(name: str, val_frac: float) -> bool:
    """Deterministic split: hash the filename -> stable bucket, no RNG needed."""
    d = int(hashlib.sha256(name.encode()).hexdigest(), 16) % 1000
    return d < val_frac * 1000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--dest", type=Path, default=Path("data"))
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"source not found: {args.src}")

    pairs = find_pairs(args.src)
    if not pairs:
        raise SystemExit(f"no images found under {args.src}")
    print(f"found {len(pairs)} images")

    # copy into a staging split first, then validate the staged result
    for split in ("train", "val"):
        (args.dest / split / "images").mkdir(parents=True, exist_ok=True)
        (args.dest / split / "labels").mkdir(parents=True, exist_ok=True)

    n_train = n_val = 0
    for img, lbl in pairs:
        split = "val" if in_val(img.name, args.val_frac) else "train"
        shutil.copy2(img, args.dest / split / "images" / img.name)
        dst_lbl = args.dest / split / "labels" / (img.stem + ".txt")
        if lbl and lbl.exists():
            shutil.copy2(lbl, dst_lbl)
        else:
            dst_lbl.write_text("")   # explicit background image
        n_train += split == "train"
        n_val += split == "val"
    print(f"split -> train={n_train} val={n_val}")

    # VALIDATION GATE
    failed = False
    for split in ("train", "val"):
        rep = validate_split(args.dest / split / "images",
                             args.dest / split / "labels", args.num_classes)
        print(f"\n[{split}]\n{rep.summary()}")
        failed = failed or not rep.ok
    if failed:
        raise SystemExit("\nVALIDATION FAILED — fix the label errors above before training.")

    # hash-pin
    (args.dest / "manifests").mkdir(exist_ok=True)
    for split in ("train", "val"):
        n = build_manifest(args.dest / split,
                           args.dest / "manifests" / f"{split}.json")
        print(f"pinned {n} files -> manifests/{split}.json")
    print("\nINGEST OK — data validated + pinned. Ready to train.")


if __name__ == "__main__":
    main()
