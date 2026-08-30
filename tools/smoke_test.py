#!/usr/bin/env python3
import argparse
import json
import torch
from densescout.model import DenseScout, count_parameters
from densescout.decoder import decode_heatmap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cpu')
    args = ap.parse_args()
    model = DenseScout(pretrained=False).to(args.device).eval()
    x = torch.zeros(1, 3, 640, 640, device=args.device)
    with torch.no_grad():
        y = model(x)
    decoded = decode_heatmap(y[0], topk=9, kernel_size=7, image_size=(640, 640))
    result = {'params': count_parameters(model), 'output_shape': list(y.shape), 'K_eff': decoded['K_eff']}
    print(json.dumps(result, indent=2))
    assert list(y.shape) == [1, 1, 80, 80]

if __name__ == '__main__':
    main()
