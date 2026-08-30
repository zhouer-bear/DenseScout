#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from densescout.datasets import DenseScoutRecordsDataset
from densescout.loss import centernet_focal_loss
from densescout.model import DenseScout


def save_ckpt(path, model, optimizer, scheduler, epoch, best_metric):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_metric': best_metric,
    }, path)


def load_resume(path, model, optimizer, scheduler, device):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt['state_dict'], strict=True)
    optimizer.load_state_dict(ckpt['optimizer'])
    if 'scheduler' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler'])
    else:
        for _ in range(int(ckpt.get('epoch', -1)) + 1):
            scheduler.step()
    start_epoch = int(ckpt.get('epoch', -1)) + 1
    best = float(ckpt.get('best_metric', float('inf')))
    return start_epoch, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train-records', required=True)
    ap.add_argument('--val-records')
    ap.add_argument('--epochs', type=int, default=60)
    ap.add_argument('--batch-size', type=int, default=16)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--output-dir', required=True)
    ap.add_argument('--resume')
    args = ap.parse_args()

    device = args.device if args.device == 'cpu' or torch.cuda.is_available() else 'cpu'
    model = DenseScout(pretrained=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    start_epoch = 0
    best = float('inf')
    if args.resume:
        start_epoch, best = load_resume(args.resume, model, optimizer, scheduler, device)

    train_ds = DenseScoutRecordsDataset(args.train_records, train=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = None
    if args.val_records:
        val_loader = DataLoader(DenseScoutRecordsDataset(args.val_records, train=False), batch_size=args.batch_size, shuffle=False, num_workers=4)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    history = []
    history_path = out / 'history.json'
    if args.resume and history_path.exists():
        history = json.loads(history_path.read_text(encoding='utf-8'))

    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_loss = 0.0
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = centernet_focal_loss(model(imgs), targets)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item())
        scheduler.step()

        metric = train_loss / max(len(train_loader), 1)
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs, targets = imgs.to(device), targets.to(device)
                    val_loss += float(centernet_focal_loss(model(imgs), targets).item())
            metric = val_loss / max(len(val_loader), 1)

        if metric < best:
            best = metric
            save_ckpt(out / 'best.pt', model, optimizer, scheduler, epoch, best)
        save_ckpt(out / 'last.pt', model, optimizer, scheduler, epoch, best)

        history.append({'epoch': epoch, 'metric_loss': metric, 'best_metric': best, 'lr': scheduler.get_last_lr()[0]})
        history_path.write_text(json.dumps(history, indent=2), encoding='utf-8')
        print(json.dumps(history[-1]))


if __name__ == '__main__':
    main()
