"""
fetch_backbone.py — get ImageNet-pretrained ResNet-50 weights for transfer
learning, and immediately convert them to SAFETENSORS + hash-pin them.

This is the one sanctioned place we touch torchvision's downloaded weights.
torchvision is a trusted source, but we still (a) never keep the .pth around and
(b) re-emit as safetensors so nothing in training ever torch.load()s a pickle.
Transfer learning from ImageNet is the biggest single accuracy win for the model.

  ./.venv/bin/python scripts/fetch_backbone.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
import torchvision
from safetensors.torch import save_file
from src.secure_io import sha256_file

OUT = Path("weights/resnet50_imagenet.safetensors")
OUT.parent.mkdir(exist_ok=True)

print("downloading torchvision ResNet-50 (IMAGENET1K_V2)…")
weights = torchvision.models.ResNet50_Weights.IMAGENET1K_V2
model = torchvision.models.resnet50(weights=weights)
sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
save_file(sd, str(OUT))

digest = sha256_file(OUT)
pin = OUT.with_suffix(".sha256.json")
pin.write_text(json.dumps({"file": OUT.name, "sha256": digest,
                           "source": "torchvision ResNet50_Weights.IMAGENET1K_V2"},
                          indent=2))
print(f"wrote {OUT}")
print(f"sha256 {digest}")
print(f"pinned -> {pin}  (commit this)")
print("\nuse it:  python -m src.train --pretrained-backbone weights/resnet50_imagenet.safetensors …")
