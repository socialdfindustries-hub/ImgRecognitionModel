"""
secure_io.py — the security boundary for model artifacts.

Every weight file that enters (pretrained backbone) or leaves (trained model,
deployed to Jetson) the system passes through here. Nothing else in the codebase
is allowed to call torch.load() or torch.save() directly.

Threats this module closes:
  1. Pickle RCE  — .pt/.pth are Python pickle; loading one runs arbitrary code.
                   We forbid pickle entirely and use safetensors (pure tensor data).
  2. Tampered deploy — a model swapped in transit (M4 -> Jetson, or over the
                   WiFi "model deploy" link) is caught by Ed25519 signature check.
                   Private (signing) key lives ONLY on the M4. The Jetson holds
                   the PUBLIC key, so a captured drone cannot forge a valid model.
  3. Silent corruption — payload hash is signed, so a truncated/edited file fails.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.exceptions import InvalidSignature


# --- key management -------------------------------------------------------

def generate_keypair(private_path: Path, public_path: Path) -> None:
    """Run ONCE on the M4. Keep the private key on the Mac only.
    Copy ONLY the public key to the Jetson."""
    priv = Ed25519PrivateKey.generate()
    private_path.write_bytes(
        priv.private_bytes_raw()
    )
    public_path.write_bytes(
        priv.public_key().public_bytes_raw()
    )
    private_path.chmod(0o600)


def _load_private(path: Path) -> Ed25519PrivateKey:
    return Ed25519PrivateKey.from_private_bytes(path.read_bytes())


def _load_public(path: Path) -> Ed25519PublicKey:
    return Ed25519PublicKey.from_public_bytes(path.read_bytes())


# --- hashing --------------------------------------------------------------

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- save (on the M4, after training) ------------------------------------

def save_model(state_dict: dict, out_path: Path, signing_key_path: Path,
               metadata: dict | None = None) -> Path:
    """Save weights as safetensors (no pickle) and emit a detached signature.

    Produces:
      <out_path>            safetensors weights
      <out_path>.sig        Ed25519 signature over sha256(weights)+metadata
    """
    out_path = Path(out_path)
    # safetensors requires contiguous CPU tensors and rejects arbitrary objects.
    clean = {k: v.detach().cpu().contiguous() for k, v in state_dict.items()}
    save_file(clean, str(out_path))

    manifest = {
        "sha256": sha256_file(out_path),
        "metadata": metadata or {},
    }
    manifest_bytes = json.dumps(manifest, sort_keys=True).encode()
    signature = _load_private(signing_key_path).sign(manifest_bytes)

    sig_path = out_path.with_suffix(out_path.suffix + ".sig")
    sig_path.write_bytes(len(manifest_bytes).to_bytes(4, "big")
                         + manifest_bytes + signature)
    return sig_path


# --- load (on the Jetson, before inference) ------------------------------

def load_model(weights_path: Path, verify_key_path: Path,
               device: str = "cpu") -> dict:
    """Verify signature, then load safetensors weights. Raises on any mismatch.

    This is the ONLY sanctioned way to load a model on the Jetson.
    """
    weights_path = Path(weights_path)
    sig_path = weights_path.with_suffix(weights_path.suffix + ".sig")
    if not sig_path.exists():
        raise SecurityError(f"missing signature for {weights_path}")

    raw = sig_path.read_bytes()
    mlen = int.from_bytes(raw[:4], "big")
    manifest_bytes = raw[4:4 + mlen]
    signature = raw[4 + mlen:]

    try:
        _load_public(verify_key_path).verify(signature, manifest_bytes)
    except InvalidSignature:
        raise SecurityError("model signature INVALID — refusing to load")

    manifest = json.loads(manifest_bytes)
    actual = sha256_file(weights_path)
    if actual != manifest["sha256"]:
        raise SecurityError(
            f"weights hash mismatch: signed {manifest['sha256'][:12]}… "
            f"got {actual[:12]}… — file tampered or corrupt")

    return load_file(str(weights_path), device=device)


# --- loading UNTRUSTED pretrained backbones (transfer learning) ----------

def load_pretrained_safe(path: Path, device: str = "cpu") -> dict:
    """Load a third-party pretrained backbone.

    Only accepts safetensors. If you were handed a .pt/.pth, convert it in a
    throwaway sandbox with weights_only=True first (see scripts/convert_pt.py)
    and hash-pin the result — never torch.load() an untrusted .pt in this repo.
    """
    path = Path(path)
    if path.suffix != ".safetensors":
        raise SecurityError(
            f"refusing to load '{path.suffix}' — convert to safetensors first")
    return load_file(str(path), device=device)


class SecurityError(RuntimeError):
    """Raised whenever an integrity/authenticity check fails. Never swallow this."""


# Belt-and-suspenders: make an accidental torch.load() loud, not silent.
def _forbidden(*_a, **_k):
    raise SecurityError(
        "torch.load() is disabled in this project — use secure_io.load_model / "
        "load_pretrained_safe. Pickle deserialization is an RCE vector.")
