from __future__ import annotations

import torch
import torch.nn.functional as F


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1.0,
) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = torch.sum(probs * target, dim=dims)
    denominator = torch.sum(probs, dim=dims) + torch.sum(target, dim=dims)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


def bce_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    dice_weight: float = 0.6,
    bce_weight: float = 0.4,
    positive_weight: float | None = None,
) -> torch.Tensor:
    pos_weight = None
    if positive_weight is not None:
        pos_weight = torch.as_tensor(positive_weight, device=logits.device, dtype=logits.dtype)
    bce = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
    dice = soft_dice_loss(logits, target)
    return bce_weight * bce + dice_weight * dice


def confusion_from_logits(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = torch.sigmoid(logits) >= threshold
    truth = target >= 0.5
    true_positive = (pred & truth).sum(dtype=torch.float32)
    false_positive = (pred & ~truth).sum(dtype=torch.float32)
    false_negative = (~pred & truth).sum(dtype=torch.float32)
    true_negative = (~pred & ~truth).sum(dtype=torch.float32)
    return true_positive, false_positive, false_negative, true_negative


def metrics_from_confusion(
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
    true_negative: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> dict[str, float]:
    tp = float(true_positive.detach().cpu())
    fp = float(false_positive.detach().cpu())
    fn = float(false_negative.detach().cpu())
    tn = float(true_negative.detach().cpu())
    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    accuracy = (tp + tn + eps) / (tp + fp + fn + tn + eps)
    return {
        "dice": dice,
        "iou": iou,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


def dice_coefficient(logits: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    tp, fp, fn, _ = confusion_from_logits(logits, target, threshold=threshold)
    return (2.0 * tp + 1e-7) / (2.0 * tp + fp + fn + 1e-7)


def iou_score(logits: torch.Tensor, target: torch.Tensor, *, threshold: float = 0.5) -> torch.Tensor:
    tp, fp, fn, _ = confusion_from_logits(logits, target, threshold=threshold)
    return (tp + 1e-7) / (tp + fp + fn + 1e-7)
