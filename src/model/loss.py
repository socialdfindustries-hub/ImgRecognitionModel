"""
loss.py — RetinaNet-style detection loss: sigmoid focal loss for classification
+ smooth-L1 for box regression, over multi-scale anchors.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from . import anchors as A


def _flatten_level(cls, reg, num_classes):
    """(B,Anc*C,H,W),(B,Anc*4,H,W) -> (B,H*W*Anc,C),(B,H*W*Anc,4) in anchor order."""
    b, _, h, w = cls.shape
    na = cls.shape[1] // num_classes
    cls = cls.view(b, na, num_classes, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, num_classes)
    reg = reg.view(b, na, 4, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
    return cls, reg


def sigmoid_focal(logits, targets_onehot, alpha=0.25, gamma=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, targets_onehot, reduction="none")
    p_t = p * targets_onehot + (1 - p) * (1 - targets_onehot)
    loss = ce * ((1 - p_t) ** gamma)
    alpha_t = alpha * targets_onehot + (1 - alpha) * (1 - targets_onehot)
    return (alpha_t * loss).sum()


class DetectionLoss:
    def __init__(self, num_classes: int):
        self.num_classes = num_classes

    def __call__(self, head_outputs, targets):
        device = head_outputs[0][0].device
        # per-level anchors + flattened predictions
        level_anchors, cls_preds, reg_preds = [], [], []
        for (cls, reg), stride, scale in zip(head_outputs, A.STRIDES, A.SCALES):
            _, _, h, w = cls.shape
            level_anchors.append(A.anchors_for_level(h, w, stride, scale, device))
            c, r = _flatten_level(cls, reg, self.num_classes)
            cls_preds.append(c)
            reg_preds.append(r)
        anchors = torch.cat(level_anchors, dim=0)            # (N,4)
        cls_preds = torch.cat(cls_preds, dim=1)              # (B,N,C)
        reg_preds = torch.cat(reg_preds, dim=1)              # (B,N,4)

        cls_loss = reg_preds.new_zeros(())
        reg_loss = reg_preds.new_zeros(())
        total_pos = 0
        for i, t in enumerate(targets):
            cls_t, reg_t, pos = A.match(
                anchors, t["boxes"].to(device), t["labels"].to(device))
            valid = cls_t >= 0                               # drop ignored anchors
            onehot = torch.zeros((anchors.shape[0], self.num_classes), device=device)
            fg = cls_t > 0
            onehot[fg, cls_t[fg] - 1] = 1.0
            cls_loss = cls_loss + sigmoid_focal(cls_preds[i][valid], onehot[valid])
            if pos.any():
                reg_loss = reg_loss + F.smooth_l1_loss(
                    reg_preds[i][pos], reg_t[pos], reduction="sum")
            total_pos += int(pos.sum())

        norm = max(total_pos, 1)
        cls_loss = cls_loss / norm
        reg_loss = reg_loss / norm
        return {"loss": cls_loss + reg_loss, "cls": cls_loss.detach(),
                "reg": reg_loss.detach(), "num_pos": total_pos}
