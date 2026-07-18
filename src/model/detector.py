"""
detector.py — the custom detector: Backbone (transfer-learned CNN) -> Neck (FPN)
-> Head (boxes + classes). This is the code you own end-to-end.

Security-relevant notes:
  - The backbone is loaded via secure_io.load_pretrained_safe (safetensors only),
    never a raw torch.load of a downloaded .pt.
  - Keep this module EXPORT-FRIENDLY: no exotic ops that break ONNX/TensorRT on
    the Jetson. NMS is intentionally NOT baked into forward() — it runs in
    post-process so the exported graph stays clean and convertible.
"""
from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torchvision

from ..secure_io import load_pretrained_safe


class FPNNeck(nn.Module):
    """Feature Pyramid Network — fuses multi-scale features. Critical for the
    tiny objects you get at altitude."""

    def __init__(self, in_channels: list[int], out_channels: int = 256):
        super().__init__()
        self.lateral = nn.ModuleList(
            nn.Conv2d(c, out_channels, 1) for c in in_channels)
        # GroupNorm after the smooth conv keeps FPN activations bounded so the
        # from-scratch neck doesn't diverge. GroupNorm is batch-independent
        # (stable at small batch) and exports cleanly to ONNX/TensorRT.
        self.smooth = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(32, out_channels))
            for _ in in_channels)

    def forward(self, feats: list[torch.Tensor]) -> list[torch.Tensor]:
        laterals = [l(f) for l, f in zip(self.lateral, feats)]
        for i in range(len(laterals) - 1, 0, -1):
            up = nn.functional.interpolate(
                laterals[i], size=laterals[i - 1].shape[-2:], mode="nearest")
            laterals[i - 1] = laterals[i - 1] + up
        return [s(l) for s, l in zip(self.smooth, laterals)]


class DetectHead(nn.Module):
    """Per-scale prediction of box regressions + class logits (anchor-based)."""

    def __init__(self, in_channels: int, num_classes: int, num_anchors: int = 3,
                 prior: float = 0.01):
        super().__init__()
        # Conv towers WITH GroupNorm before the final predictors. The norm keeps
        # activations bounded so neck/head training doesn't diverge (this was the
        # root cause of loss blow-up). GroupNorm is batch-independent and
        # ONNX/TensorRT-friendly.
        self.cls_tower = self._tower(in_channels)
        self.reg_tower = self._tower(in_channels)
        self.cls = nn.Conv2d(in_channels, num_anchors * num_classes, 3, padding=1)
        self.reg = nn.Conv2d(in_channels, num_anchors * 4, 3, padding=1)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Focal-loss prior-bias: start the cls head predicting ~`prior` fg prob,
        # so the thousands of background anchors don't dominate the early gradient.
        nn.init.constant_(self.cls.bias, -math.log((1 - prior) / prior))

    @staticmethod
    def _tower(ch: int, n: int = 2):
        layers = []
        for _ in range(n):
            layers += [nn.Conv2d(ch, ch, 3, padding=1),
                       nn.GroupNorm(32, ch), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)

    def forward(self, feats: list[torch.Tensor]):
        return [(self.cls(self.cls_tower(f)), self.reg(self.reg_tower(f)))
                for f in feats]


class CustomDetector(nn.Module):
    def __init__(self, num_classes: int, pretrained_backbone: Path | None = None):
        super().__init__()
        # ResNet-50 as feature extractor. weights=None -> we supply our own,
        # verified safetensors weights rather than letting torchvision download.
        resnet = torchvision.models.resnet50(weights=None)
        if pretrained_backbone is not None:
            sd = load_pretrained_safe(pretrained_backbone)
            missing, unexpected = resnet.load_state_dict(sd, strict=False)
            if unexpected:
                raise RuntimeError(f"unexpected keys in backbone: {unexpected[:5]}")
        self.stem = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4

        self.neck = FPNNeck(in_channels=[512, 1024, 2048])
        self.head = DetectHead(256, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.stem(x)
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        feats = self.neck([c3, c4, c5])
        return self.head(feats)  # NMS happens in post-process, not here
