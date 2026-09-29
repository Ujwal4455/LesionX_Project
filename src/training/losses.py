import torch
import torch.nn.functional as F


def weighted_bce_loss(logits, targets, pos_weight=None):
    if pos_weight is None:
        return F.binary_cross_entropy_with_logits(logits, targets)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)


def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    p_t = torch.exp(-bce)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    modulating = (1 - p_t) ** gamma
    return (alpha_factor * modulating * bce).mean()
