"""
train.py — train the custom detector on the M4 Pro.

Run:
  ./.venv/bin/python -m src.train \
      --images data/train/images --labels data/train/labels \
      --num-classes 10 --epochs 50 --imgsz 1280 \
      --manifest data/manifests/train.json

Checkpoints are written as SIGNED safetensors via secure_io (no pickle, ever) so
the same file can be verified and loaded on the Jetson without a trust gap.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import AerialDetectionDataset, collate
from .model import CustomDetector
from .model.loss import DetectionLoss
from .secure_io import save_model


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="dataset hash manifest to verify before training")
    ap.add_argument("--pretrained-backbone", type=Path, default=None,
                    help="verified safetensors ResNet weights (transfer learning). "
                         "Deviates from the from-scratch decision — requires "
                         "--allow-pretrained. Intended only for baseline benchmarking.")
    ap.add_argument("--allow-pretrained", action="store_true",
                    help="explicit opt-in required to use --pretrained-backbone")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--tile", type=int, default=1024,
                    help="full-res crop window for training (SAHI-style small-object "
                         "sampling); the crop is resized to --imgsz. 0 = whole-frame "
                         "resize (no tiling). Objects keep native pixel scale when "
                         "tile>=imgsz — the main lever for tiny objects at altitude.")
    ap.add_argument("--no-augment", action="store_true",
                    help="disable train-time augmentation (tile crop, flip, "
                         "brightness/contrast). Use for a whole-frame no-aug baseline.")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--signing-key", type=Path, default=Path("keys/model_ed25519"))
    ap.add_argument("--out", type=Path, default=Path("dist/detector.safetensors"))
    ap.add_argument("--resume", type=Path, default=None,
                    help="signed checkpoint to CONTINUE training from (verified via "
                         "the .pub key). Lets you train in chunks across sessions.")
    args = ap.parse_args()

    # --- from-scratch guard -------------------------------------------------
    # Project decision: train fully from scratch (own 100% of weights, air-gap
    # safe). A pretrained backbone can ONLY be used as an explicit, acknowledged
    # baseline — it can never sneak into a deployed model by accident.
    if args.pretrained_backbone is not None and not args.allow_pretrained:
        raise SystemExit(
            "refusing --pretrained-backbone without --allow-pretrained.\n"
            "This project trains FROM SCRATCH by default (see docs/DATA_PLAN.md).\n"
            "Add --allow-pretrained only to train a THROWAWAY transfer-learning\n"
            "baseline for comparison — never for the deployed model.")
    from_scratch = args.pretrained_backbone is None
    if from_scratch:
        print("mode: FROM SCRATCH (no pretrained weights — 100% self-trained)")
    else:
        print("mode: BASELINE with pretrained backbone — NOT for deployment")

    device = pick_device()
    print(f"device: {device}")

    augment = not args.no_augment
    ds = AerialDetectionDataset(args.images, args.labels, args.imgsz, args.manifest,
                                train=augment, tile=(args.tile if augment else 0))
    print(f"data: {len(ds)} images | augment={augment} "
          f"| tile={args.tile if augment else 0} -> imgsz={args.imgsz}")
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, collate_fn=collate,
                    pin_memory=(device == "cuda"))

    model = CustomDetector(args.num_classes, args.pretrained_backbone).to(device)
    if args.resume is not None:
        from .secure_io import load_model
        pub = Path(str(args.signing_key) + ".pub")
        sd = load_model(args.resume, pub, device=device)   # verifies signature first
        model.load_state_dict(sd)
        print(f"resumed from {args.resume} (signature verified) — continuing training")
    criterion = DetectionLoss(args.num_classes)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for step, (imgs, targets) in enumerate(dl):
            # linear LR warmup over the first epoch — from-scratch detection
            # training diverges without it (see scripts/demo_detect.py).
            if epoch == 0:
                warm = min(1.0, (step + 1) / max(len(dl), 1))
                for g in opt.param_groups:
                    g["lr"] = args.lr * warm
            imgs = imgs.to(device)
            out = model(imgs)
            losses = criterion(out, targets)
            opt.zero_grad()
            losses["loss"].backward()
            # Robust gradient clipping. clip_grad_value_ is a hard per-element cap
            # that can't be miscomputed (the fused foreach norm path misbehaves on
            # MPS, letting the reg head diverge). Keep the norm clip too, foreach off.
            torch.nn.utils.clip_grad_value_(model.parameters(), 2.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, foreach=False)
            opt.step()
            running += float(losses["loss"])
            if step % 20 == 0:
                print(f"e{epoch} s{step} loss={float(losses['loss']):.4f} "
                      f"cls={float(losses['cls']):.4f} reg={float(losses['reg']):.4f} "
                      f"pos={losses['num_pos']}")
        sched.step()
        print(f"epoch {epoch} mean_loss={running / max(len(dl),1):.4f}")

        if args.signing_key.exists():
            save_model(model.state_dict(), args.out, args.signing_key,
                       metadata={"epoch": epoch, "num_classes": args.num_classes,
                                 "imgsz": args.imgsz, "from_scratch": from_scratch})
            print(f"  saved signed checkpoint -> {args.out}")
        else:
            print("  WARNING: no signing key; checkpoint NOT saved. Run scripts/gen_keys.py")


if __name__ == "__main__":
    main()
