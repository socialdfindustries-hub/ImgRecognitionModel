"""
anchors.py — anchor generation, IoU matching, and box encode/decode.
RetinaNet-style, kept plain (no exotic ops) so training and export stay aligned.
"""
from __future__ import annotations

import torch

# One entry per FPN level (matches detector: c3,c4,c5 -> strides 8,16,32).
STRIDES = (8, 16, 32)
SCALES = (32.0, 64.0, 128.0)          # base anchor size per level (pixels)
ASPECT_RATIOS = (0.5, 1.0, 2.0)       # -> num_anchors = 3


def anchors_for_level(h: int, w: int, stride: int, scale: float,
                      device) -> torch.Tensor:
    """Return (h*w*A, 4) anchors in xyxy, ordered (row, col, aspect)."""
    ys = (torch.arange(h, device=device) + 0.5) * stride
    xs = (torch.arange(w, device=device) + 0.5) * stride
    cy, cx = torch.meshgrid(ys, xs, indexing="ij")
    centers = torch.stack([cx, cy], dim=-1).reshape(-1, 1, 2)  # (h*w,1,2)

    ratios = torch.tensor(ASPECT_RATIOS, device=device)
    ws = scale * torch.sqrt(ratios)
    hs = scale / torch.sqrt(ratios)
    half = torch.stack([ws, hs], dim=-1) / 2.0                # (A,2)

    mins = centers - half                                     # (h*w,A,2)
    maxs = centers + half
    return torch.cat([mins, maxs], dim=-1).reshape(-1, 4)


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU between (N,4) and (M,4) xyxy -> (N,M)."""
    area_a = (a[:, 2] - a[:, 0]).clamp(min=0) * (a[:, 3] - a[:, 1]).clamp(min=0)
    area_b = (b[:, 2] - b[:, 0]).clamp(min=0) * (b[:, 3] - b[:, 1]).clamp(min=0)
    lt = torch.max(a[:, None, :2], b[None, :, :2])
    rb = torch.min(a[:, None, 2:], b[None, :, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def encode(gt: torch.Tensor, anchors: torch.Tensor) -> torch.Tensor:
    """Encode gt xyxy relative to anchors -> (dx,dy,dw,dh)."""
    aw = anchors[:, 2] - anchors[:, 0]
    ah = anchors[:, 3] - anchors[:, 1]
    ax = anchors[:, 0] + aw / 2
    ay = anchors[:, 1] + ah / 2
    gw = gt[:, 2] - gt[:, 0]
    gh = gt[:, 3] - gt[:, 1]
    gx = gt[:, 0] + gw / 2
    gy = gt[:, 1] + gh / 2
    return torch.stack([(gx - ax) / aw, (gy - ay) / ah,
                        torch.log(gw / aw), torch.log(gh / ah)], dim=1)


def match(anchors: torch.Tensor, gt_boxes: torch.Tensor, gt_labels: torch.Tensor,
          pos_thr: float = 0.5, neg_thr: float = 0.4):
    """Assign each anchor. Returns (cls_target[N] with -1=ignore,0=bg,>0 class+1,
    reg_target[N,4], pos_mask[N])."""
    n = anchors.shape[0]
    cls_t = torch.zeros(n, dtype=torch.int64, device=anchors.device)   # 0 = bg
    reg_t = torch.zeros((n, 4), dtype=torch.float32, device=anchors.device)
    if gt_boxes.numel() == 0:
        return cls_t, reg_t, torch.zeros(n, dtype=torch.bool, device=anchors.device)

    iou = box_iou(anchors, gt_boxes)          # (N, M)
    max_iou, arg = iou.max(dim=1)             # best gt per anchor
    pos = max_iou >= pos_thr
    ignore = (max_iou < pos_thr) & (max_iou >= neg_thr)

    # guarantee every gt gets its single best anchor
    best_anchor = iou.argmax(dim=0)           # (M,)
    pos[best_anchor] = True
    ignore[best_anchor] = False

    cls_t[ignore] = -1
    matched_gt = arg[pos]
    cls_t[pos] = gt_labels[matched_gt] + 1    # +1 so 0 stays background
    reg_t[pos] = encode(gt_boxes[matched_gt], anchors[pos])
    return cls_t, reg_t, pos
