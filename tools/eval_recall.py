#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import cv2
import torch
import yaml

from densescout.model import DenseScout
from densescout.decoder import decode_heatmap
from densescout.metrics import recall_from_centers


def load_checkpoint(model, path, device):
    ckpt = torch.load(path, map_location=device)
    state = ckpt.get('state_dict', ckpt.get('model', ckpt)) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state, strict=True)


def load_eval_config(path):
    if not path:
        return {}
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f'Config must be a YAML mapping: {path}')
    return data


def parse_float_list(value):
    if value is None or value == '':
        return []
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(x) for x in str(value).split(',') if str(x).strip()]


def parse_int_list(value):
    if value is None or value == '':
        return []
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x) for x in str(value).split(',') if str(x).strip()]


def resolve_eval_settings(args, config):
    decoder_cfg = config.get('decoder', {}) or {}
    ratios = parse_float_list(args.ratios if args.ratios is not None else config.get('ratios', [0.005, 0.01, 0.02, 0.04]))
    topk = parse_int_list(args.topk if args.topk is not None else config.get('topk', []))
    kernel_size = args.kernel_size if args.kernel_size is not None else int(decoder_cfg.get('kernel_size', 7))
    score_threshold = args.score_threshold if args.score_threshold is not None else float(decoder_cfg.get('score_threshold', 0.0))
    hit_half_size = args.hit_half_size if args.hit_half_size is not None else float(config.get('hit_half_size', 32.0))
    return {
        'ratios': ratios,
        'topk': topk,
        'kernel_size': kernel_size,
        'score_threshold': score_threshold,
        'hit_half_size': hit_half_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', help='Optional YAML config. CLI flags override YAML values.')
    ap.add_argument('--records', required=True)
    ap.add_argument('--checkpoint')
    ap.add_argument('--ratios', default=None, help='Comma-separated area ratios, e.g. 0.005,0.01,0.02,0.04')
    ap.add_argument('--topk', default=None, help='Optional comma-separated fixed K list')
    ap.add_argument('--kernel-size', type=int, default=None)
    ap.add_argument('--score-threshold', type=float, default=None)
    ap.add_argument('--hit-half-size', type=float, default=None)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--limit-samples', type=int, default=0)
    ap.add_argument('--output', required=True)
    args = ap.parse_args()

    config = load_eval_config(args.config)
    settings = resolve_eval_settings(args, config)

    records = json.loads(Path(args.records).read_text(encoding='utf-8'))
    if args.limit_samples:
        records = records[:args.limit_samples]

    model = DenseScout(pretrained=False).to(args.device).eval()
    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.device)

    ratio_ks = [max(1, int(round(r * 80 * 80))) for r in settings['ratios']]
    ks = sorted(set(ratio_ks + settings['topk'])) or [1]
    centers_per_k = {k: {} for k in ks}

    for rec in records:
        img = cv2.imread(rec['file_name'])
        if img is None:
            continue
        h, w = img.shape[:2]
        inp = cv2.cvtColor(cv2.resize(img, (640, 640)), cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(inp).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            logits = model(tensor.to(args.device))
        decoded = decode_heatmap(
            logits[0],
            max(ks),
            kernel_size=settings['kernel_size'],
            score_threshold=settings['score_threshold'],
            image_size=(w, h),
        )
        proxy_centers = [c.proxy_xy for c in decoded['centers']]
        img_id = str(rec.get('image_id', rec.get('file_name')))
        for k in ks:
            centers_per_k[k][img_id] = proxy_centers[:k]

    ratio_by_k = {max(1, int(round(r * 80 * 80))): r for r in settings['ratios']}
    out = []
    for k in ks:
        m = recall_from_centers(records, centers_per_k[k], half_size=settings['hit_half_size'])
        out.append({
            'K': k,
            'ratio': ratio_by_k.get(k),
            'recall': m['recall'],
            'recall_percent': m['recall'] * 100.0,
            'num_gt': m['num_gt'],
            'num_hit': m['num_hit'],
            'kernel_size': settings['kernel_size'],
            'score_threshold': settings['score_threshold'],
            'hit_half_size': settings['hit_half_size'],
            'protocol': 'cell64_square_proxy_topk_positive_localmax',
            'note': 'No-checkpoint runs are API sanity only, not benchmark results.' if not args.checkpoint else '',
        })

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding='utf-8')
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
