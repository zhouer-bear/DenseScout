#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from PIL import Image

VISDRONE_CLASSES = {
    1: 'pedestrian', 2: 'people', 3: 'bicycle', 4: 'car', 5: 'van',
    6: 'truck', 7: 'tricycle', 8: 'awning-tricycle', 9: 'bus', 10: 'motor'
}


def convert_split(root: Path, split: str, output: Path):
    img_dir = root / f'VisDrone2019-DET-{split}' / 'images'
    ann_dir = root / f'VisDrone2019-DET-{split}' / 'annotations'
    records = []
    for img_path in sorted(img_dir.glob('*.jpg')):
        ann_path = ann_dir / (img_path.stem + '.txt')
        with Image.open(img_path) as im:
            w, h = im.size
        objects = []
        if ann_path.exists():
            for line in ann_path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                vals = line.split(',')
                if len(vals) < 8:
                    continue
                x, y, bw, bh = map(float, vals[:4])
                score = int(float(vals[4]))
                cls = int(float(vals[5]))
                trunc = int(float(vals[6]))
                occ = int(float(vals[7]))
                if cls <= 0 or cls > 10 or score == 0:
                    continue
                objects.append({'bbox_xyxy': [x, y, x + bw, y + bh], 'category_id': cls, 'truncation': trunc, 'occlusion': occ})
        records.append({'image_id': img_path.stem, 'file_name': str(img_path), 'width': w, 'height': h, 'dataset': 'visdrone', 'objects': objects})
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(records, indent=2), encoding='utf-8')
    print(f'wrote {len(records)} records to {output}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='Directory containing VisDrone2019-DET-train/val')
    ap.add_argument('--split', default='val', choices=['train','val'])
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    convert_split(Path(args.root), args.split, Path(args.output))

if __name__ == '__main__':
    main()
