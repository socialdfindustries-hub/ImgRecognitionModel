"""
integrity.py — treat every external dataset (VisDrone, Roboflow exports) as
untrusted until proven otherwise.

Data poisoning is the quiet threat: a poisoned image set can plant a backdoor
(model goes blind to a triggered object) while validation mAP looks fine. We
cannot fully detect that here, but we CAN:
  - pin the exact bytes we trained on (hash manifest), so results are reproducible
    and a swapped-in dataset is caught;
  - keep an OWN, drone-captured holdout set that never mixes with third-party data,
    so a backdoor in the public set does not silently pass validation.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


def build_manifest(data_dir: Path, out: Path) -> int:
    """Record sha256 of every file under data_dir. Commit this manifest."""
    data_dir = Path(data_dir)
    entries = {}
    for p in sorted(data_dir.rglob("*")):
        if p.is_file():
            entries[str(p.relative_to(data_dir))] = _sha256(p)
    out.write_text(json.dumps(entries, indent=2, sort_keys=True))
    return len(entries)


def verify_manifest(data_dir: Path, manifest: Path) -> None:
    """Fail loudly if the dataset differs from what we pinned."""
    data_dir = Path(data_dir)
    expected = json.loads(Path(manifest).read_text())
    mismatched, missing = [], []
    for rel, want in expected.items():
        f = data_dir / rel
        if not f.exists():
            missing.append(rel)
        elif _sha256(f) != want:
            mismatched.append(rel)
    if missing or mismatched:
        raise DatasetIntegrityError(
            f"dataset drift: {len(missing)} missing, {len(mismatched)} altered "
            f"(e.g. {(missing + mismatched)[:3]})")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class DatasetIntegrityError(RuntimeError):
    pass
