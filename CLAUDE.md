# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A from-scratch PyTorch aerial object detector (**not** YOLO — for ownership and the
concrete security controls below) for a high-altitude drone. Backbone (ResNet-50)
→ FPN neck → anchor-based detect head. Train on a dev GPU (CUDA preferred; Apple
MPS works but is less numerically stable), deploy to an NVIDIA Jetson via signed
ONNX → TensorRT. VisDrone-DET is the prototyping dataset (10 classes); the pipeline
is dataset-agnostic and own mission data is the end goal (`docs/DATA_PLAN.md`).

## Non-negotiable security rules (see `SECURITY.md`)

These override convenience. Violating them defeats the point of the project.

1. **Never call `torch.load()` / `torch.save()`.** All weight I/O goes through
   `src/secure_io.py`. `.pt`/`.pth` are pickle = RCE. Everything on disk is
   **safetensors**. `secure_io._forbidden` exists to make an accidental
   `torch.load` loud.
2. Weights cross the dev→Jetson boundary **only** as Ed25519-signed safetensors/ONNX.
   `save_model` writes `<file>` + `<file>.sig`; `load_model` verifies signature +
   sha256 and raises `SecurityError` on any mismatch. Never swallow `SecurityError`.
3. The **private signing key** (`keys/model_ed25519`) never leaves the dev machine
   and is never committed. The Jetson gets only `keys/model_ed25519.pub`.
4. External datasets are **untrusted** until hash-pinned by `scripts/ingest_dataset.py`
   (`src/data/integrity.py`). Keep an own-captured holdout separate from third-party data.
5. Keep `src/model/detector.py` **export-friendly**: no exotic ops, and **NMS stays
   in post-process, never in `forward()`**, so the ONNX/TensorRT graph stays clean.

## Architecture

The pipeline is a straight line; each stage has a single entry module.

- **`src/secure_io.py`** — the enforced trust boundary. `generate_keypair`,
  `save_model` (sign), `load_model` (verify+load, Jetson path),
  `load_pretrained_safe` (safetensors-only backbone loader). Nothing else in the
  repo may touch `torch.load/save`.
- **`src/model/`** — the owned model:
  - `detector.py` `CustomDetector`: ResNet-50 stem/layers → `FPNNeck` (fuses
    c3/c4/c5) → `DetectHead` (per-scale cls logits + box regressions). GroupNorm
    throughout (batch-independent, stable at small batch, ONNX-clean) — this fixed
    the from-scratch loss blow-up. `forward()` returns raw `(cls, reg)` per level.
  - `anchors.py` — anchors per FPN level (strides 8/16/32 for c3/c4/c5), IoU,
    encode/decode, and `match` (assigns GT to anchors). If training reports
    `num_pos == 0`, the anchor↔box scale is mismatched — look here.
  - `loss.py` — focal cls + smooth-L1 reg over matched anchors.
  - `postprocess.py` — decode + NMS + overlapping-tile merge for tiny-object inference.
- **`src/data/`** — `dataset.py` (image + `cls cx cy w h` normalized-label loader,
  with train-time SAHI-style tile cropping + hflip + brightness/contrast aug; val
  is whole-frame deterministic), `integrity.py` (hash-pinning), `validate.py`
  (label/class-range checks).
- **CLI entry points** (all run as modules, e.g. `python -m src.train`):
  `train.py`, `evaluate.py` (mAP@0.5 + per-class AP), `predict.py` (`--tile` for
  small objects), `export.py` (sign + ONNX for Jetson).
- **`scripts/`** — one-shot/dev tools: `gen_keys.py`, `fetch_backbone.py`
  (ImageNet ResNet-50 → safetensors), `convert_visdrone.py`, `ingest_dataset.py`
  (the trust gate), plus smoke/unit checks (see Testing).

### Data flow

`data_src/{train,val}` (raw, converted) → `scripts/ingest_dataset.py` validates +
splits + hash-pins → `data/{train,val}/{images,labels}` + `data/manifests` (the
trusted, pinned copy training reads). Wait for **`INGEST OK`**. Class ids/order live
in `configs/dataset.yaml` and are written by ingest — never hand-edit `paths:`, and
never reorder `names:` after labeling (it silently remaps every existing label).

## Common commands

Activate the venv first: `source .venv/bin/activate` (`.venv\Scripts\activate` on Windows).

```bash
# One-time setup
python scripts/gen_keys.py                       # signing keypair
python scripts/fetch_backbone.py                 # ImageNet backbone -> safetensors

# Ingest (the trust gate) — VisDrone is 10 classes, ids 0..9
python scripts/convert_visdrone.py --src VisDrone2019-DET-train --out data_src/train --copy
python scripts/ingest_dataset.py --src data_src/train --num-classes 10 --val-frac 0.0
python scripts/ingest_dataset.py --src data_src/val   --num-classes 10 --val-frac 1.0

# Train (from-scratch = omit --pretrained-backbone; pretrained REQUIRES --allow-pretrained)
python -m src.train --images data/train/images --labels data/train/labels \
    --num-classes 10 --epochs 50 --imgsz 768 --batch 2 --workers 2 --lr 1e-4 \
    --out dist/detector.safetensors
#   --resume dist/detector.safetensors   continue from a checkpoint (chunked training)
#   --tile 1024 (default)                train-time tile crop;  --no-augment disables aug
#   OOM? lower --batch to 1 or --imgsz to 640

# Evaluate — the real accuracy number (low train loss != good detections)
python -m src.evaluate --weights dist/detector.safetensors --num-classes 10 \
    --images data/val/images --labels data/val/labels --imgsz 768

# Detect on an image (tiled for small objects)
python -m src.predict --weights dist/detector.safetensors --num-classes 10 \
    --image aerial.jpg --imgsz 1280 --tile 640 --out result.jpg

# Export for Jetson (signs the ONNX)
python -m src.export --weights dist/detector.safetensors --num-classes 10
```

**Always match `--num-classes` to the data (10 for VisDrone).** A wrong value makes
`validate.py` reject or silently mislabel boxes.

## Containerized pipeline (Colima)

A CPU-only Docker image (`Dockerfile` + `docker-compose.yml`) runs the pipeline
reproducibly on Colima. **No GPU passthrough on macOS** — this is for
deterministic `ingest`/`evaluate`/smoke-tests/`pip-audit` and CPU debugging.
**GPU training stays native** (CUDA host, or Apple MPS); do not train in-container.

- Base is `python:3.9-slim` to match the host `.venv` (3.9.6); torch is the
  **CPU wheel** (`--index-url .../whl/cpu`), rest from pinned `requirements.txt`.
- The repo is bind-mounted at `/app`, so `keys/`, `dist/`, `data/` ride in as
  mounts and are **never baked into an image layer** (`.dockerignore` enforces
  this too). The private signing key stays on the host.

```bash
colima start                                   # 4 vCPU / 8GiB / docker runtime
docker compose build
docker compose run --rm pipeline python scripts/test_secure_io.py
docker compose run --rm pipeline python scripts/smoke_train.py
docker compose run --rm pipeline python -m src.evaluate --weights dist/detector.safetensors \
    --num-classes 10 --images data/val/images --labels data/val/labels --imgsz 768
docker compose run --rm audit                  # pip-audit the pinned deps
```

## Testing

There is no pytest suite; correctness is guarded by runnable scripts:

```bash
python scripts/test_secure_io.py   # sign/verify round-trip + tamper rejection
python scripts/smoke_train.py      # end-to-end fwd/loss/backward; asserts num_pos > 0
python scripts/demo_detect.py      # inference sanity on a sample image
```

Run `smoke_train.py` after any change to `model/`, `anchors.py`, `loss.py`, or
`dataset.py`, and `test_secure_io.py` after any change to `secure_io.py`.

## Gotchas

- **MPS instability:** the regression head can diverge on Apple MPS (fused-clip
  numerical issue). `train.py` uses `clip_grad_value` + `foreach=False`; if loss
  still climbs, drop `--lr` to 2e-5. CUDA at LR 1e-4 is the reliable path.
- **Aug is not reproducible yet:** `dataset.py` uses an unseeded per-worker
  `default_rng()`. Fixed-seed determinism is an open TODO.
- **Training saves every epoch**, not best-only; in-loop val/mAP + save-BEST is a
  known pending item. `dist/` holds several experiment checkpoints (`visdrone_*`).

---

# Project: Drone Object Detection (High-Altitude)

## Stack
- PyTorch + torchvision (RetinaNet/FCOS)
- NO Ultralytics/YOLO packages -- secure, self-owned pipeline
- Target deployment: Jetson (TensorRT)

## Code Change Rules

### Make minimal changes
- Change ONLY what the request requires
- Do NOT refactor, rename, or restructure unrelated code
- Do NOT "improve" code I didn't ask about
- Preserve existing style, naming, and structure

### Check before adding
- ALWAYS search the codebase before writing new code
- If similar/related code already exists, MODIFY it -- don't duplicate
- Never add a function that duplicates existing functionality
- Point out existing code that already does the job

### Adapt to existing code
- Match the current code's conventions, not general best practice
- Follow existing patterns even if you'd write it differently
- Use the project's existing utilities/helpers

### Before large changes
- If a change touches many files or restructures anything, ASK FIRST
- Explain what you'd change and wait for approval

## Response Style
- Be concise -- no long explanations unless asked
- Show only the changed code, not entire files
- Skip preamble and summaries
