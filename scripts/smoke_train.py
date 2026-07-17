"""Smoke test: one real training step on synthetic data (no dataset/opencv needed).
Proves detector -> loss -> backward -> optimizer -> SIGNED checkpoint all work
on this machine (mps/cpu). Uses a tiny image so it runs fast."""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.model import CustomDetector
from src.model.loss import DetectionLoss
from src.secure_io import generate_keypair, save_model, load_model
from src.train import pick_device

device = pick_device()
print(f"device: {device}")

torch.manual_seed(0)
num_classes = 6
model = CustomDetector(num_classes).to(device)
criterion = DetectionLoss(num_classes)
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

# synthetic batch: 2 images, 256px (small for speed), a couple boxes each
imgs = torch.rand(2, 3, 256, 256, device=device)
targets = [
    {"boxes": torch.tensor([[30, 30, 90, 120], [140, 60, 200, 190]], dtype=torch.float32),
     "labels": torch.tensor([1, 3])},
    {"boxes": torch.tensor([[10, 10, 60, 80]], dtype=torch.float32),
     "labels": torch.tensor([0])},
]

model.train()
l0 = None
for step in range(3):
    out = model(imgs)
    losses = criterion(out, targets)
    opt.zero_grad()
    losses["loss"].backward()
    opt.step()
    print(f"step {step}: loss={float(losses['loss']):.4f} "
          f"cls={float(losses['cls']):.4f} reg={float(losses['reg']):.4f} "
          f"pos={losses['num_pos']}")
    if l0 is None:
        l0 = float(losses["loss"])

assert losses["num_pos"] > 0, "no positive anchors matched — check anchor/box scale"
print("forward+backward+step: OK, positive anchors matched")

# signed checkpoint round-trip (the deploy path)
tmp = Path(tempfile.mkdtemp())
priv, pub = tmp / "k", tmp / "k.pub"
generate_keypair(priv, pub)
w = tmp / "detector.safetensors"
save_model(model.state_dict(), w, priv, metadata={"num_classes": num_classes})
sd = load_model(w, pub, device="cpu")
assert len(sd) == len(model.state_dict())
print("signed checkpoint save + verified load: OK")
print("\nTRAIN LOOP SMOKE TEST PASSED")
