"""End-to-end proof of the security spine:
   generate keys -> save+sign real weights -> verify+load (M4 role -> Jetson role)
   -> then tamper and confirm the load is REFUSED.

Run:  ./.venv/bin/python scripts/test_secure_io.py
"""
import sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import torch
from src.secure_io import (
    generate_keypair, save_model, load_model, SecurityError,
)

tmp = Path(tempfile.mkdtemp())
priv, pub = tmp / "k_ed25519", tmp / "k_ed25519.pub"
weights = tmp / "detector.safetensors"

# --- M4 side: make keys, train (fake), save + sign -----------------------
generate_keypair(priv, pub)
real_sd = {"neck.weight": torch.randn(256, 512, 1, 1),
           "head.cls.bias": torch.randn(240)}
save_model(real_sd, weights, priv, metadata={"classes": 80, "epoch": 42})
print("1. saved + signed real weights (safetensors, no pickle)")

# --- Jetson side: verify signature + hash, then load ---------------------
loaded = load_model(weights, pub)
assert torch.equal(loaded["neck.weight"], real_sd["neck.weight"])
print("2. Jetson verified signature + hash, loaded OK, tensors match")

# --- attacker: tamper with the weights after signing --------------------
b = bytearray(weights.read_bytes())
b[-1] ^= 0xFF                      # flip one bit in the payload
weights.write_bytes(b)
try:
    load_model(weights, pub)
    print("FAIL: tampered weights were loaded!")
    sys.exit(1)
except SecurityError as e:
    print(f"3. tamper REJECTED: {e}")

# --- attacker: try their own key (forge a signature) --------------------
fpriv, fpub = tmp / "forge", tmp / "forge.pub"
generate_keypair(fpriv, fpub)
weights.write_bytes(bytes(b[:-1]) + bytes([b[-1] ^ 0xFF]))  # restore payload
save_model(real_sd, weights, fpriv)          # re-sign with WRONG key
try:
    load_model(weights, pub)                 # verify with the REAL pub key
    print("FAIL: forged signature accepted!")
    sys.exit(1)
except SecurityError as e:
    print(f"4. forged signature REJECTED: {e}")

print("\nSECURITY SPINE OK — signed load works, tamper + forgery both refused.")
