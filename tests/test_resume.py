import importlib.util
from pathlib import Path

import torch

from densescout.model import DenseScout


def _load_train_module():
    path = Path(__file__).resolve().parents[1] / "tools" / "train.py"
    spec = importlib.util.spec_from_file_location("train_tool", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_resume_checkpoint_roundtrip_includes_scheduler_and_best(tmp_path):
    train = _load_train_module()
    model = DenseScout(pretrained=False)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5, eta_min=1e-6)
    opt.zero_grad(set_to_none=True)
    loss = next(model.parameters()).sum() * 0.0
    loss.backward()
    opt.step()
    scheduler.step()
    path = tmp_path / "ckpt.pt"
    train.save_ckpt(path, model, opt, scheduler, epoch=3, best_metric=1.23)

    model2 = DenseScout(pretrained=False)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=5, eta_min=1e-6)
    start_epoch, best = train.load_resume(path, model2, opt2, scheduler2, "cpu")

    assert start_epoch == 4
    assert best == 1.23
    assert scheduler2.state_dict()["last_epoch"] == scheduler.state_dict()["last_epoch"]
    assert opt2.param_groups[0]["lr"] == opt.param_groups[0]["lr"]
