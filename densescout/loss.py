import torch


def centernet_focal_loss(logits: torch.Tensor, target: torch.Tensor, alpha: float = 2.0, beta: float = 4.0) -> torch.Tensor:
    """CenterNet-style Gaussian focal loss used by DenseScout.

    The model returns logits. Targets are Gaussian heatmaps in [0, 1] with exact
    positive peaks equal to 1. The no-positive branch is handled explicitly.
    """
    pred = torch.sigmoid(logits).clamp(min=1e-4, max=1.0 - 1e-4)
    pos = target.eq(1).float()
    neg = target.lt(1).float()
    neg_weights = torch.pow(1.0 - target, beta)
    pos_loss = torch.log(pred) * torch.pow(1.0 - pred, alpha) * pos
    neg_loss = torch.log(1.0 - pred) * torch.pow(pred, alpha) * neg_weights * neg
    num_pos = pos.sum()
    if num_pos.item() == 0:
        loss = -neg_loss.sum()
    else:
        loss = -(pos_loss.sum() + neg_loss.sum()) / num_pos
    return loss
