"""
evaluate.py — measure the detector's ACCURACY on the validation set (data the
model never trained on). Prints mAP@0.5 + per-class AP + overall precision/recall.

This is the number that tells you if training was worth anything — a low training
loss does NOT prove good detections; this does.

  ./.venv/bin/python -m src.evaluate \
      --weights dist/detector.safetensors --num-classes 10 \
      --images data/val/images --labels data/val/labels

Loads weights through the VERIFIED path (Ed25519 signature checked first).
mAP is VOC-style all-points AP at IoU 0.5. No external metric libraries.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torchvision.ops import box_iou

from .data.dataset import AerialDetectionDataset
from .model import CustomDetector
from .model.postprocess import detections_from_outputs
from .secure_io import load_model
from .train import pick_device


def voc_ap(rec: list[float], prec: list[float]) -> float:
    """Area under the precision-recall curve (VOC all-points)."""
    if not prec:
        return 0.0
    mrec = np.concatenate(([0.0], np.array(rec), [rec[-1]]))
    mpre = np.concatenate(([0.0], np.array(prec), [0.0]))
    for i in range(len(mpre) - 2, -1, -1):          # precision envelope
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


@torch.no_grad()
def evaluate(model, ds, num_classes, imgsz, device, score_thr=0.05, iou_thr=0.5):
    per_class = defaultdict(list)     # class -> list of (score, is_tp)
    n_gt = defaultdict(int)           # class -> #ground-truth boxes
    for idx in range(len(ds)):
        img, target = ds[idx]
        gt_boxes = target["boxes"]                 # xyxy pixels (imgsz), cpu
        gt_labels = target["labels"]
        for c in gt_labels.tolist():
            n_gt[c] += 1
        out = model(img.unsqueeze(0).to(device))
        pb, ps, pl = detections_from_outputs(out, num_classes, imgsz, score_thr)
        pb, ps, pl = pb.cpu(), ps.cpu(), pl.cpu()
        for c in range(num_classes):
            g = gt_boxes[gt_labels == c]
            p_mask = pl == c
            pboxes, pscores = pb[p_mask], ps[p_mask]
            order = pscores.argsort(descending=True)
            pboxes, pscores = pboxes[order], pscores[order]
            matched = torch.zeros(len(g), dtype=torch.bool)
            ious = box_iou(pboxes, g) if (len(g) and len(pboxes)) else None
            for j in range(len(pboxes)):
                is_tp = 0
                if ious is not None:
                    best = int(ious[j].argmax())
                    if ious[j, best] >= iou_thr and not matched[best]:
                        is_tp = 1
                        matched[best] = True
                per_class[c].append((float(pscores[j]), is_tp))
        if (idx + 1) % 100 == 0:
            print(f"  evaluated {idx + 1}/{len(ds)} images")

    aps, tp_tot, fp_tot = {}, 0, 0
    for c in range(num_classes):
        entries = sorted(per_class[c], key=lambda x: -x[0])
        tp = fp = 0
        precs, recs = [], []
        total = max(n_gt[c], 1)
        for score, is_tp in entries:
            tp += is_tp
            fp += 1 - is_tp
            precs.append(tp / (tp + fp))
            recs.append(tp / total)
        aps[c] = voc_ap(recs, precs)
        tp_tot += tp
        fp_tot += fp
    mAP = sum(aps.values()) / num_classes
    total_gt = sum(n_gt.values())
    precision = tp_tot / max(tp_tot + fp_tot, 1)
    recall = tp_tot / max(total_gt, 1)
    return mAP, aps, n_gt, precision, recall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--score", type=float, default=0.05)
    ap.add_argument("--verify-key", type=Path, default=Path("keys/model_ed25519.pub"))
    ap.add_argument("--names", type=Path, default=Path("configs/dataset.yaml"))
    args = ap.parse_args()

    device = pick_device()
    print(f"device: {device}")
    model = CustomDetector(args.num_classes).to(device).eval()
    model.load_state_dict(load_model(args.weights, args.verify_key, device), strict=False)

    ds = AerialDetectionDataset(args.images, args.labels, args.imgsz)
    print(f"evaluating on {len(ds)} images...")
    mAP, aps, n_gt, precision, recall = evaluate(
        model, ds, args.num_classes, args.imgsz, device, args.score)

    # optional class names
    names = {}
    try:
        import yaml
        names = yaml.safe_load(args.names.read_text()).get("names", {})
    except Exception:
        pass

    print("\n================  ACCURACY  ================")
    print(f"mAP@0.5:        {mAP:.4f}")
    print(f"precision:      {precision:.4f}")
    print(f"recall:         {recall:.4f}")
    print("--- per-class AP@0.5 ---")
    for c in range(args.num_classes):
        nm = names.get(c, f"class {c}")
        print(f"  {c} {nm:<14} AP={aps[c]:.4f}  (gt boxes: {n_gt.get(c, 0)})")
    print("============================================")


if __name__ == "__main__":
    main()
