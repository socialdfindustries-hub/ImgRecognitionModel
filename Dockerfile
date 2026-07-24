# Reproducible CPU pipeline for Colima/Docker on macOS (no GPU passthrough).
# Purpose: deterministic ingest / evaluate / smoke-tests / pip-audit and CPU
# debugging. GPU training stays NATIVE (CUDA host, or Apple MPS) — this image
# is intentionally CPU-only.
#
# Python 3.9 matches the host .venv (3.9.6) so behaviour is identical.
FROM python:3.9-slim-bookworm

# opencv-python-headless + torch runtime libs. No GUI/CUDA libs on purpose.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch first (from the CPU wheel index — avoids the ~1GB CUDA wheel),
# then the rest of the PINNED deps (torch/torchvision already satisfied → skipped).
COPY requirements.txt .
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Non-root. Source is bind-mounted at runtime (see docker-compose.yml); keys,
# data/, dist/ come in as mounts too and are NEVER baked into the image.
RUN useradd -m runner
USER runner

CMD ["python", "-c", "import torch; print('container OK — torch', torch.__version__, 'cpu')"]
