"""Run ONCE on the M4 Pro. Generates the model-signing keypair.

  keys/model_ed25519       PRIVATE — stays on the M4. Never commit, never copy
                           to the Jetson. Back it up offline.
  keys/model_ed25519.pub   PUBLIC  — copy THIS to the Jetson. Safe if the drone
                           is captured: it can only verify, not forge.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.secure_io import generate_keypair

priv = Path("keys/model_ed25519")
pub = Path("keys/model_ed25519.pub")
if priv.exists():
    raise SystemExit(f"{priv} already exists — refusing to overwrite a signing key")
generate_keypair(priv, pub)
print(f"wrote {priv} (KEEP SECRET) and {pub} (copy to Jetson)")
