# ImgRecognitionModel — Custom Aerial Detector (secure-by-default)

Self-built PyTorch object detector (Backbone → FPN Neck → Head) for a
high-altitude drone. **No YOLO / no third-party detector in the recognition
path** — for ownership *and* with the concrete security controls in
[`SECURITY.md`](./SECURITY.md).

- **Train:** any CUDA GPU (RTX 20xx+, Colab). MPS (Apple) works but is less
  numerically stable — see notes below.
- **Deploy:** NVIDIA Jetson (signed ONNX → TensorRT).
- **Data:** VisDrone (aerial, 10 classes); the pipeline is dataset-agnostic.

## Layout
```
src/model/detector.py   # Backbone(ResNet-50) + FPN neck + detect head (GroupNorm), export-friendly
src/model/loss.py       # focal cls + smooth-L1 reg over multi-scale anchors
src/model/postprocess.py# decode + NMS + tiling (overlap) for tiny-object inference
src/data/dataset.py     # image + label loader (cls cx cy w h, normalized)
src/data/integrity.py   # hash-pin untrusted datasets
src/secure_io.py        # safetensors + Ed25519 sign/verify (never pickle / torch.load)
src/train.py            # training loop (warmup, MPS-safe clipping, --resume, signed ckpts)
src/evaluate.py         # mAP@0.5 + per-class AP + precision/recall on the val set
src/predict.py          # inference on any image, with --tile for small objects
src/export.py           # sign + export ONNX for the Jetson
scripts/gen_keys.py     # one-time signing key   scripts/fetch_backbone.py # ImageNet backbone
scripts/convert_visdrone.py # VisDrone annotations -> label format
scripts/ingest_dataset.py   # validate + split + hash-pin data (the trust gate)
```

## Quickstart — full pipeline
Works on any CUDA machine (Windows RTX 2060, Linux, Colab). On Windows use
`\` paths and `.venv\Scripts\activate`; on Linux/Mac use `/` and `source .venv/bin/activate`.

### 1. Environment
```bash
git clone https://github.com/socialdfindustries-hub/ImgRecognitionModel.git
cd ImgRecognitionModel
python -m venv .venv && source .venv/bin/activate
# CUDA PyTorch (pick the build for your CUDA from pytorch.org):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install safetensors cryptography opencv-python-headless albumentations pillow numpy pyyaml
python -c "import torch; print('cuda:', torch.cuda.is_available())"   # must be True
```

### 2. Keys + pretrained backbone
```bash
python scripts/gen_keys.py          # signing key (private key stays on this machine)
python scripts/fetch_backbone.py    # ImageNet ResNet-50 -> safetensors + hash-pin
```

### 3. Data — download, convert, ingest (VisDrone)
```bash
curl -L -o train.zip https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-train.zip
curl -L -o val.zip   https://github.com/ultralytics/yolov5/releases/download/v1.0/VisDrone2019-DET-val.zip
tar -xf train.zip && tar -xf val.zip
python scripts/convert_visdrone.py --src VisDrone2019-DET-train --out data_src/train --copy
python scripts/convert_visdrone.py --src VisDrone2019-DET-val   --out data_src/val   --copy
python scripts/ingest_dataset.py --src data_src/train --num-classes 10 --val-frac 0.0
python scripts/ingest_dataset.py --src data_src/val   --num-classes 10 --val-frac 1.0
```
Wait for **`INGEST OK`**. (The VisDrone zips are hosted on a GitHub release — a
file download only; no YOLO/Ultralytics code is installed or run.)

### 4. Train
```bash
python -m src.train --images data/train/images --labels data/train/labels \
    --num-classes 10 --epochs 50 --imgsz 768 --batch 2 --workers 2 --lr 1e-4 \
    --pretrained-backbone weights/resnet50_imagenet.safetensors --allow-pretrained \
    --out dist/detector.safetensors
```
- Watch `epoch N mean_loss` trend **down**. On CUDA it's stable at LR 1e-4.
- **Out of memory?** lower `--batch` to 1, or `--imgsz` to 640. (6GB GPU: 768/batch2.)
- **Higher accuracy?** `--imgsz 1024` if VRAM allows (better on tiny objects).
- **Train in chunks** (limited session time): add `--resume dist/detector.safetensors`
  to continue from a saved checkpoint.
- Checkpoints are **signed safetensors**, written every epoch.

### 5. Evaluate — the accuracy number
```bash
python -m src.evaluate --weights dist/detector.safetensors --num-classes 10 \
    --images data/val/images --labels data/val/labels --imgsz 768
```
Prints **mAP@0.5**, per-class AP, precision/recall. A low *training loss* does not
prove good detections — this does. On VisDrone (tiny objects) **mAP ≥ 0.30 is a
solid custom-detector result**; watch class 0 (person) AP.

### 6. Detect on an image (tiled for small objects)
```bash
python -m src.predict --weights dist/detector.safetensors --num-classes 10 \
    --image aerial.jpg --imgsz 1280 --tile 640 --out result.jpg
```
`--tile` slices the frame into overlapping tiles so tiny objects are detectable
(SAHI-style). Smaller tiles = finer detection but slower (lower FPS); no `--tile`
= fastest but coarse. Tune tile size for accuracy ↔ speed.

## Deploy handoff → Jetson
1. `python -m src.export --weights dist/detector.safetensors --num-classes 10`
2. Copy `dist/detector.onnx` + `.sig` **and** `keys/model_ed25519.pub` to the Jetson.
3. On the Jetson: verify signature → build TensorRT engine → **re-validate mAP**
   (FP16/INT8 can shift accuracy on tiny objects).

## Notes on hardware / stability
- **CUDA (recommended):** stable at LR 1e-4. RTX 2060 (6GB) → `--imgsz 768 --batch 2`.
- **Apple MPS:** trains, but the regression head can diverge (a fused-clip numerical
  issue). `train.py` uses `clip_grad_value` + `foreach=False` to mitigate; if it still
  climbs, drop `--lr` to 2e-5. CUDA is the reliable path.
- **Colab free:** ~12h sessions, no background execution — use `--resume` for chunks,
  or Colab Pro+ for long/background runs.

## Non-negotiables (security)
See [`SECURITY.md`](./SECURITY.md). Short version: never `torch.load()`; models cross
machines only as **signed safetensors/ONNX**; the **private key never leaves the
training machine**; external data is untrusted until **hash-pinned** by `ingest_dataset.py`.
