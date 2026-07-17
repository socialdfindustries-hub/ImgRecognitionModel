# ImgRecognitionModel — Custom Aerial Detector (secure-by-default)

Self-built PyTorch object detector (Backbone → FPN Neck → Head) for a
high-altitude drone. **Train on the M4 Pro, deploy to NVIDIA Jetson.** No YOLO /
no third-party detector in the recognition path — for ownership *and* with the
concrete security controls in [`SECURITY.md`](./SECURITY.md).

## Layout
```
src/secure_io.py       # security boundary: safetensors + Ed25519 sign/verify (no pickle)
src/model/detector.py  # Backbone(ResNet) + FPN neck + detect head — export-friendly
src/data/integrity.py  # hash-pin untrusted datasets (VisDrone/Roboflow)
src/export.py          # sign + export ONNX for the Jetson
scripts/gen_keys.py    # one-time: make the M4-only signing key
requirements.txt       # pinned deps (run pip-audit before deploy)
```

## Setup (M4 Pro)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/gen_keys.py          # creates keys/ — private key stays on the Mac
```

## Deploy handoff → Jetson
1. `python -m src.export --weights <trained.safetensors> --num-classes N`
2. Copy `dist/detector.onnx` + `.sig` **and** `keys/model_ed25519.pub` to the Jetson.
3. On the Jetson: verify signature → build TensorRT engine → **re-validate mAP**
   (FP16/INT8 can shift accuracy on tiny objects).

## Non-negotiables
See [`SECURITY.md`](./SECURITY.md). Short version: never `torch.load()`; models
cross machines only as **signed safetensors/ONNX**; the **private key never leaves
the M4**; external data is untrusted until hash-pinned.

> Status: scaffold. Training loop (`src/train.py`) and post-process/tracking are
> the next pieces — see the open TODO in chat.
```
