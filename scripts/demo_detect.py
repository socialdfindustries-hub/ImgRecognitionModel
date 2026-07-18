"""
demo_detect.py — prove the from-scratch model actually LEARNS to detect.

Generates a tiny synthetic dataset (bright squares = class 0, circles = class 1
on a dark background), trains the REAL CustomDetector from scratch for a few
hundred steps, then runs detection on fresh unseen images and measures whether
the predicted boxes land on the true objects (IoU).

This is a sanity check of the whole pipeline (model -> loss -> train -> NMS ->
detect), NOT a real aerial model. It uses only synthetic shapes.

  ./.venv/bin/python scripts/demo_detect.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import cv2
import torch

from src.model import CustomDetector
from src.model.loss import DetectionLoss
from src.model.postprocess import detections_from_outputs
from src.train import pick_device

IMG = 256
NUM_CLASSES = 2
torch.manual_seed(0)
rng = np.random.RandomState(0)


def make_sample():
    """Return (image CxHxW float, boxes xyxy pixels, labels)."""
    img = np.full((IMG, IMG, 3), 60, np.uint8)
    boxes, labels = [], []
    for _ in range(rng.randint(1, 3)):
        s = rng.randint(40, 80)
        x = rng.randint(5, IMG - s - 5)
        y = rng.randint(5, IMG - s - 5)
        cls = rng.randint(0, 2)
        if cls == 0:  # square
            cv2.rectangle(img, (x, y), (x + s, y + s), (230, 230, 230), -1)
        else:         # circle
            cv2.circle(img, (x + s // 2, y + s // 2), s // 2, (200, 200, 200), -1)
        boxes.append([x, y, x + s, y + s])
        labels.append(cls)
    t = torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
    return t.permute(2, 0, 1).contiguous(), \
        torch.tensor(boxes, dtype=torch.float32), \
        torch.tensor(labels, dtype=torch.int64), img


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0


def main():
    device = pick_device()
    print(f"device: {device}  |  training from scratch on synthetic shapes")

    # fixed training set
    train = [make_sample() for _ in range(48)]
    model = CustomDetector(NUM_CLASSES).to(device)  # NO pretrained backbone
    crit = DetectionLoss(NUM_CLASSES)
    BASE_LR = 1e-4
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=1e-4)

    model.train()
    STEPS, BATCH, WARMUP = 600, 8, 100
    for step in range(STEPS):
        # linear LR warmup — critical for stable from-scratch detection training
        lr = BASE_LR * min(1.0, (step + 1) / WARMUP)
        for g in opt.param_groups:
            g["lr"] = lr
        idx = rng.choice(len(train), BATCH, replace=False)
        imgs = torch.stack([train[i][0] for i in idx]).to(device)
        targets = [{"boxes": train[i][1], "labels": train[i][2]} for i in idx]
        out = model(imgs)
        loss = crit(out, targets)
        opt.zero_grad(); loss["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == STEPS - 1:
            print(f"step {step:3d}  loss={float(loss['loss']):.3f}  "
                  f"cls={float(loss['cls']):.3f}  reg={float(loss['reg']):.3f}")

    # --- evaluate on FRESH unseen images ---
    model.eval()
    hits, total, shots = 0, 0, []
    for k in range(6):
        t, gt_boxes, gt_labels, vis = make_sample()
        out = model(t.unsqueeze(0).to(device))
        boxes, scores, labels = detections_from_outputs(
            out, NUM_CLASSES, IMG, score_thr=0.3, iou_thr=0.4)
        boxes = boxes.cpu()
        for gb, gl in zip(gt_boxes.tolist(), gt_labels.tolist()):
            total += 1
            best = max((iou(gb, pb.tolist()) for pb in boxes), default=0) if len(boxes) else 0
            if best >= 0.5:
                hits += 1
        # draw GT (blue) + predictions (green)
        for gb in gt_boxes.tolist():
            cv2.rectangle(vis, (int(gb[0]), int(gb[1])), (int(gb[2]), int(gb[3])), (255, 120, 0), 1)
        for pb, s, l in zip(boxes.tolist(), scores.tolist(), labels.tolist()):
            cv2.rectangle(vis, (int(pb[0]), int(pb[1])), (int(pb[2]), int(pb[3])), (0, 255, 0), 2)
            cv2.putText(vis, f"{'sq' if l==0 else 'ci'}:{s:.2f}", (int(pb[0]), max(int(pb[1])-3, 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        shots.append(vis)

    grid = cv2.hconcat([cv2.vconcat(shots[:3]), cv2.vconcat(shots[3:])])
    cv2.imwrite("scratch_detect_demo.jpg", grid)
    print(f"\nDETECTION on unseen images: {hits}/{total} objects localized (IoU>=0.5)")
    print("wrote scratch_detect_demo.jpg  (blue=ground truth, green=model prediction)")


if __name__ == "__main__":
    main()
