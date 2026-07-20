"""
postprocess.py — turn raw head outputs into clean detections.
decode (anchors + deltas -> xyxy) -> confidence filter -> per-class NMS.
Also a tiler for SAHI-style small-object detection at altitude.

NMS lives HERE (not in the model's forward) so the exported ONNX graph stays
clean and TensorRT-convertible.
"""
from __future__ import annotations

import torch
from torchvision.ops import batched_nms

from . import anchors as A
from .loss import _flatten_level


def decode(deltas: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Inverse of anchors.encode: (dx,dy,dw,dh)+anchors -> xyxy."""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + aw / 2
    ay = anchors[:, 1] + ah / 2
    cx = deltas[:, 0] * aw + ax
    cy = deltas[:, 1] * ah + ay
    w = torch.exp(deltas[:, 2].clamp(max=7.0)) * aw
    h = torch.exp(deltas[:, 3].clamp(max=7.0)) * ah
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)


@torch.no_grad()
def detections_from_outputs(head_outputs, num_classes, imgsz,
                            score_thr=0.3, iou_thr=0.5, max_det=300):
    """Single-image head outputs -> (boxes[K,4], scores[K], labels[K])."""
    device = head_outputs[0][0].device
    all_anchors, all_cls, all_reg = [], [], []
    for (cls, reg), stride, scale in zip(head_outputs, A.STRIDES, A.SCALES):
        _, _, h, w = cls.shape
        all_anchors.append(A.anchors_for_level(h, w, stride, scale, device))
        c, r = _flatten_level(cls, reg, num_classes)
        all_cls.append(c[0])            # single image
        all_reg.append(r[0])
    anchors = torch.cat(all_anchors, 0)
    scores_all = torch.sigmoid(torch.cat(all_cls, 0))     # (N, C)
    boxes_all = decode(torch.cat(all_reg, 0), anchors)    # (N, 4)

    scores, labels = scores_all.max(dim=1)
    keep = scores >= score_thr
    boxes, scores, labels = boxes_all[keep], scores[keep], labels[keep]
    boxes[:, 0::2] = boxes[:, 0::2].clamp(0, imgsz)
    boxes[:, 1::2] = boxes[:, 1::2].clamp(0, imgsz)

    if boxes.numel() == 0:
        return boxes, scores, labels
    nms_keep = batched_nms(boxes, scores, labels, iou_thr)[:max_det]
    return boxes[nms_keep], scores[nms_keep], labels[nms_keep]


def tile_image(img: torch.Tensor, tile: int, overlap: int | None = None):
    """Yield (tile_tensor, x_off, y_off) covering a C,H,W image, with OVERLAP so an
    object straddling a tile edge is fully contained in at least one tile (else it
    gets cut and missed). Default overlap = 20% of the tile. Small aerial objects
    survive far better tiled than downscaled to one frame."""
    _, H, W = img.shape
    if overlap is None:
        overlap = max(tile // 5, 1)          # 20% overlap
    step = max(tile - overlap, 1)
    ys = list(range(0, max(H - overlap, 1), step))
    xs = list(range(0, max(W - overlap, 1), step))
    for y in ys:
        for x in xs:
            y2, x2 = min(y + tile, H), min(x + tile, W)
            yield img[:, y:y2, x:x2], x, y
