"""
export.py — package a trained model for the Jetson. Runs on the M4.

Emits safetensors weights + ONNX graph, each with a detached Ed25519 signature.
On the Jetson: verify signatures, THEN build the TensorRT engine from the ONNX.
Do the ONNX->TensorRT conversion on-device (engines are hardware-specific) and
re-validate mAP after conversion — FP16/INT8 can move accuracy on tiny objects.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from .model import CustomDetector
from .secure_io import save_model, _load_private, sha256_file, SecurityError


def sign_blob(path: Path, signing_key_path: Path) -> None:
    """Sign any deploy artifact (e.g. the ONNX file) with the M4-only key."""
    manifest = {"sha256": sha256_file(path), "kind": path.suffix}
    mb = json.dumps(manifest, sort_keys=True).encode()
    sig = _load_private(signing_key_path).sign(mb)
    sig_path = path.with_suffix(path.suffix + ".sig")
    sig_path.write_bytes(len(mb).to_bytes(4, "big") + mb + sig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, required=True,
                    help="trained safetensors weights (already signed)")
    ap.add_argument("--num-classes", type=int, required=True)
    ap.add_argument("--signing-key", type=Path, default=Path("keys/model_ed25519"))
    ap.add_argument("--out", type=Path, default=Path("dist/detector.onnx"))
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    if not args.signing_key.exists():
        raise SecurityError(
            "signing key missing — run scripts/gen_keys.py on the M4 first")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    model = CustomDetector(num_classes=args.num_classes).eval()

    from .secure_io import load_model  # verified load of our own signed weights
    verify_key = args.signing_key.with_suffix(".pub")
    model.load_state_dict(load_model(args.weights, verify_key), strict=False)

    dummy = torch.zeros(1, 3, args.imgsz, args.imgsz)
    torch.onnx.export(
        model, dummy, str(args.out),
        input_names=["images"], opset_version=17,
        dynamic_axes={"images": {0: "batch"}},
    )
    sign_blob(args.out, args.signing_key)
    print(f"exported + signed: {args.out}  sha256={sha256_file(args.out)[:12]}…")


if __name__ == "__main__":
    main()
