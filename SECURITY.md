# Security Analysis & Threat Model — Custom Aerial Recognition Model

Scope: the **model pipeline** — build/train on the M4 Pro, deploy to the NVIDIA
Jetson. This is the analysis you asked for: what in a normal ML pipeline causes a
security issue, and what was removed or replaced to close it.

> **Key point:** building a *custom* detector (instead of YOLO) buys you
> **ownership** of code/weights/data. It does **not** by itself buy you
> **security**. The real attack surface is deserialization, data poisoning,
> model theft on capture, and adversarial evasion. Those are addressed below.

---

## Threats found, and how they are removed in this repo

### 1. Pickle deserialization RCE — **CRITICAL** — *removed*
- **Cause:** PyTorch `.pt`/`.pth` files are Python *pickle*; `torch.load()` on one
  executes arbitrary code. Two entry points: the downloaded pretrained ResNet
  backbone (runs code on the **M4 during training**), and any weights moved to
  the **Jetson**.
- **Removed by:** `src/secure_io.py` — all weights are **safetensors** (pure
  tensor data, cannot execute code). `torch.load()` is banned in-repo;
  `load_pretrained_safe()` rejects any non-`.safetensors` file.

### 2. Tampered / spoofed model on deploy — **CRITICAL** — *removed*
- **Cause:** the design moves a model M4 → Jetson and even mentions "model deploy
  over WiFi". An unauthenticated model can be swapped by anyone on that path.
- **Removed by:** **Ed25519 detached signatures**. The M4 signs (private key,
  `keys/model_ed25519`, never leaves the Mac). The Jetson **verifies** with the
  public key only (`keys/model_ed25519.pub`). `secure_io.load_model()` refuses to
  load on bad signature or hash mismatch. A captured drone holds only the public
  key → **cannot forge** a valid model.

### 3. Data poisoning / backdoor — **HIGH** — *mitigated*
- **Cause:** training on third-party data (VisDrone, Roboflow exports). A poisoned
  set can plant a trigger-activated blind spot while val mAP looks normal.
- **Mitigated by:** `src/data/integrity.py` hash-pins the exact dataset bytes
  (drift is caught). Process rule: keep an **own drone-captured holdout** that
  never mixes with third-party data, so a public-set backdoor can't pass
  validation. *Full backdoor detection is out of scope — treat external data as
  untrusted.*

### 4. Model theft on physical capture — **HIGH** — *documented control*
- **Cause:** a drone can be recovered by an adversary. Weights / ONNX / TensorRT
  engine sit on Jetson storage in the clear → IP theft + tailored evasion attacks.
- **Control:** encrypt artifacts at rest (JetPack disk encryption / secure boot),
  decrypt into memory at runtime. `.gitignore` keeps weights out of the repo.
  Treat the airframe as recoverable by design.

### 5. Adversarial evasion — **MEDIUM/HIGH** — *design control*
- **Cause:** adversarial patches can hide/misclassify objects; small aerial
  targets are especially vulnerable. Custom ≠ robust (often the opposite).
- **Control:** adversarial training augmentation; **do not make the camera a
  single point of decision** — fuse with LiDAR + RTK/GPS (already in the design)
  so a fooled detection is cross-checked.

### 6. Dependency supply chain — **MEDIUM** — *mitigated*
- **Cause:** `pip install` pulls transitive packages that could be malicious.
- **Mitigated by:** pinned `requirements.txt`; run `pip-audit` before deploy;
  build the Jetson image from a pinned manifest, not live `pip install`.

### 7. Insecure export graph (ONNX/TensorRT) — **LOW (security) / HIGH (reliability)**
- **Cause:** exotic ops or in-graph NMS break TensorRT conversion → last-minute
  hacks on-device.
- **Handled by:** `detector.py` keeps NMS in post-process, not `forward()`, so the
  exported graph stays clean and auditable.

---

## Rules for this repo (do not violate)
1. **Never** call `torch.load()` / `torch.save()` — use `secure_io` only.
2. Weights cross the M4→Jetson boundary **only** as signed safetensors/ONNX.
3. The **private signing key never leaves the M4** and is never committed.
4. External datasets are **untrusted** until hash-pinned + validated on an own holdout.
5. `pip-audit` is green before any deploy.

## Out of scope (track separately)
Flight-controller/MAVLink security, RF/telemetry link encryption, ESP32-S3 node
hardening, ground-station dashboard auth. The "two independent brains" split means
a compromised Jetson must **not** be able to command flight — verify that isolation
at the integration layer.
