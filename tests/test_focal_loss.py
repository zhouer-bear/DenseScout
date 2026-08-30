import torch
from densescout.loss import centernet_focal_loss


def test_focal_loss_positive_branch_finite():
    logits = torch.zeros(2, 1, 80, 80)
    target = torch.zeros_like(logits)
    target[:, :, 10, 10] = 1.0
    loss = centernet_focal_loss(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_focal_loss_no_positive_branch_finite():
    logits = torch.zeros(1, 1, 80, 80)
    target = torch.zeros_like(logits)
    loss = centernet_focal_loss(logits, target)
    assert torch.isfinite(loss)
    assert loss.item() > 0
