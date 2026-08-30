#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
import cv2
import torch
from densescout.model import DenseScout
from densescout.decoder import decode_heatmap


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--checkpoint')
    ap.add_argument('--topk', type=int, default=9)
    ap.add_argument('--kernel-size', type=int, default=7)
    ap.add_argument('--score-threshold', type=float, default=0.0)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(args.image)
    h, w = img.shape[:2]
    inp = cv2.cvtColor(cv2.resize(img, (640, 640)), cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(inp).permute(2,0,1).float().unsqueeze(0) / 255.0
    model = DenseScout(pretrained=False).to(args.device).eval()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.device)
    with torch.no_grad():
        logits = model(tensor.to(args.device))
    decoded = decode_heatmap(logits[0], args.topk, kernel_size=args.kernel_size, score_threshold=args.score_threshold, image_size=(w, h))
    rows = []
    for c in decoded['centers']:
        rows.append({'rank': c.rank, 'score': c.score, 'lattice_xy': list(c.lattice_xy), 'proxy_xy': list(c.proxy_xy), 'image_xy': list(c.image_xy)})
    out = {'image': args.image, 'image_size': [w, h], 'topk': args.topk, 'K_eff': decoded['K_eff'], 'centers': rows}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps({'output': args.output, 'K_eff': decoded['K_eff']}, indent=2))

if __name__ == '__main__':
    main()
