# Data Collection & Labeling Plan

You chose **train fully from scratch on your own data** — no pretrained weights,
no third-party datasets. That is the most secure and fully-owned path, and it
means the model's entire accuracy comes from the data you collect here. This plan
is how you get enough clean, correctly-labeled data to make it accurate.

---

## 1. Define your classes (do this first)
Edit `configs/dataset.yaml`. Rules that protect accuracy:
- **Keep it tight.** 3–8 classes to start. Every class you add multiplies the
  labeling work and the data you need.
- **Make classes visually distinct** at your flight altitude. If two classes look
  the same in a 40-px aerial blob, merge them.
- **Never reorder** the list after labeling begins — ids are baked into every label.

## 2. How much data (the honest numbers for from-scratch)
From-scratch has no ImageNet prior, so it needs more than transfer learning would:

| Target | Images (min) | Instances per class (min) | Expect |
|---|---|---|---|
| Prove it learns (pipeline test) | ~300 | ~100 | overfits, low mAP — fine as a check |
| Usable prototype | ~2,000 | ~1,000 | rough but real detections |
| Genuinely accurate | ~10,000+ | ~1,500+ each | competitive mAP |

Small objects at altitude need the **higher** end. There is no shortcut here —
this is the price of full ownership.

## 3. Capture guidance (drives real-world accuracy)
Match your training images to how the drone will actually fly:
- **Altitude & angle:** capture at the altitudes/gimbal angles you'll deploy at.
- **Diversity:** lighting (dawn/noon/dusk), weather, seasons, backgrounds, urban/rural.
- **Object scale:** include the smallest objects you need to detect — these are
  the hardest and most under-represented.
- **Negatives:** include images with **no** target objects (background). They cut
  false positives. The pipeline supports empty label files for these.

## 4. Labeling tool — pick for security
| Tool | Where data lives | Verdict for you |
|---|---|---|
| **CVAT (self-hosted)** | **Your network** | ✅ Recommended. Mission imagery never leaves your infrastructure. Exports YOLO format directly. |
| Roboflow (cloud) | Their servers | ⚠️ Convenient but uploads your imagery to a third party — wrong for sensitive/defense data. |
| Label Studio (self-hosted) | Your network | ✅ Fine alternative to CVAT. |

**Export format:** YOLO — one `.txt` per image, lines of `class cx cy w h`
(normalized 0..1). That is exactly what `ingest_dataset.py` expects.

## 5. Split & pin
Don't hand-split. Run:
```bash
./.venv/bin/python scripts/ingest_dataset.py --src /path/to/export --num-classes N --val-frac 0.2
```
This validates every label, refuses on errors, splits train/val deterministically,
and hash-pins the result (`data/manifests/`). Keep a small **untouched holdout**
you never train on for honest final numbers.

## 6. Then train
```bash
./.venv/bin/python -m src.train \
    --images data/train/images --labels data/train/labels \
    --num-classes N --manifest data/manifests/train.json \
    --epochs 100 --imgsz 1280         # NOTE: no --pretrained-backbone = from scratch
```

## Quality checklist before training
- [ ] Classes defined in `configs/dataset.yaml`, visually distinct at altitude
- [ ] Images captured at deployment altitude/angle, diverse conditions
- [ ] Background (no-object) images included
- [ ] Labeled with a **self-hosted** tool (data stayed on your network)
- [ ] `ingest_dataset.py` passed with **0 errors**
- [ ] A holdout set set aside, never trained on
