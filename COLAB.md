# Training on Colab — quickstart

Colab gives you a free GPU (T4), ~10–20× faster than the M4 for full training.
This is the plan for tomorrow.

## What's already fixed (done on the M4)
- ✅ `detector.py`: **GroupNorm** in FPN neck + head (stopped the loss explosion)
- ✅ `train.py`: **LR warmup** (epoch 0) + **grad clip 1.0**
- ✅ Pretrained backbone path verified (`weights/resnet50_imagenet.safetensors`)

## Step 0 — get the CODE onto Colab (you have no git remote yet)
**Option A (recommended): push to a private GitHub repo, then clone.**
```bash
# on the Mac, once:
cd ~/Desktop/ImgRecognitionModel
git add -A && git commit -m "GroupNorm + warmup; ready for Colab training"
gh repo create ImgRecognitionModel --private --source=. --push   # needs gh auth
```
Then in Colab: `!git clone https://github.com/<you>/ImgRecognitionModel.git`

**Option B: zip + upload** (code is small — data/weights/keys are gitignored):
```bash
cd ~/Desktop && zip -r imgrec_code.zip ImgRecognitionModel \
  -x '*/.venv/*' '*/data/*' '*/dist/*' '*/weights/*' '*/.git/*'
```
Upload `imgrec_code.zip` in Colab, then `!unzip imgrec_code.zip`.

## Step 1 — Colab notebook cells (copy-paste, in order)

```python
# 1. GPU check
import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))

# 2. deps (Colab already has torch+CUDA; install the rest)
%cd /content/ImgRecognitionModel
!pip install -q safetensors cryptography opencv-python-headless albumentations pymap3d

# 3. signing key (checkpoints are signed). NOTE: this makes a COLAB key.
#    For a DEPLOYED model, re-sign on the M4 with your real key later.
!python scripts/gen_keys.py

# 4. pretrained backbone (secure path: download -> safetensors -> hash-pin)
!python scripts/fetch_backbone.py
```

## Step 2 — get the DATA (VisDrone, 3.7 GB) onto Colab
**Best: put VisDrone in Google Drive once, then mount it** (no re-upload each session):
```python
from google.colab import drive; drive.mount('/content/drive')
# assuming you uploaded VisDrone to Drive at MyDrive/VisDrone
!ln -s "/content/drive/MyDrive/VisDrone" data_src
```
Or download VisDrone directly in Colab from its official links.

Then ingest through your secure gate (validates + hash-pins):
```python
!python scripts/ingest_dataset.py --src data_src --num-classes 10 --val-frac 0.2
```

## Step 3 — TRAIN (full dataset, GPU)
```python
!python -m src.train \
    --images data/train/images --labels data/train/labels \
    --num-classes 10 --epochs 50 --imgsz 1280 --batch 8 --workers 2 \
    --lr 1e-4 \
    --pretrained-backbone weights/resnet50_imagenet.safetensors --allow-pretrained \
    --out dist/detector.safetensors
```
- On a T4, **start batch=4–8** (drop if OOM at 1280px). `device` auto-selects `cuda`.
- Watch the **epoch mean_loss** — with full data it should now trend **down**.
- If reg loss still climbs: drop `--lr` to `5e-5`, or lower `imgsz` to 1024.

## Step 4 — get the trained model back
```python
from google.colab import files
files.download('dist/detector.safetensors')     # signed checkpoint
```
Copy it (+ the `.pub` key) to the M4 / Jetson. For deployment, **re-sign on the M4**
with your real key (the Colab key is only for the training run).

## If it STILL doesn't converge on full data (debug order)
1. Lower `--lr` (1e-4 → 5e-5 → 2e-5)
2. Check the **cls vs reg** split in the logs — if reg explodes, LR too high; if cls
   climbs, may need more warmup epochs
3. Confirm labels are sane (class ids 0–9, boxes in 0..1) — bad labels = bad training
4. Try `--imgsz 1024` and larger `--batch` for steadier gradients

## Security reminder
- The **private signing key never leaves the M4** for the *deployed* model. The
  Colab key is throwaway — fine for training checkpoints, re-sign for deploy.
- Datasets stay untrusted until `ingest_dataset.py` hash-pins them.
- Model crosses machines only as **signed safetensors** (see SECURITY.md).
