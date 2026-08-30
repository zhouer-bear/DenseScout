#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from pathlib import Path
from PIL import Image

VALID_CLASSES = set(range(1, 11))


def audit_split(root: Path, split: str):
    img_dir = root / f"VisDrone2019-DET-{split}" / "images"
    ann_dir = root / f"VisDrone2019-DET-{split}" / "annotations"
    rows = 0
    ignored_region_rows = 0
    removed_categories = Counter()
    removed_invalid_boxes = 0
    valid_objects = 0
    missing_ann = 0
    images = 0
    examples = []
    for img_path in sorted(img_dir.glob("*.jpg")):
        images += 1
        ann_path = ann_dir / (img_path.stem + ".txt")
        if not ann_path.exists():
            missing_ann += 1
            continue
        with Image.open(img_path) as im:
            width, height = im.size
        for lineno, line in enumerate(ann_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            rows += 1
            parts = line.split(",")
            if len(parts) < 8:
                removed_invalid_boxes += 1
                examples.append({"file": ann_path.name, "line": lineno, "reason": "short_row", "raw": line})
                continue
            x, y, w, h = map(float, parts[:4])
            score = int(float(parts[4]))
            cls = int(float(parts[5]))
            if score == 0 or cls == 0:
                ignored_region_rows += 1
                removed_categories[str(cls)] += 1
                continue
            if cls not in VALID_CLASSES:
                removed_categories[str(cls)] += 1
                continue
            if w <= 0 or h <= 0:
                removed_invalid_boxes += 1
                examples.append({"file": ann_path.name, "line": lineno, "reason": "non_positive_wh", "bbox": [x, y, w, h], "class": cls, "score": score})
                continue
            valid_objects += 1
    return {
        "split": split,
        "images": images,
        "missing_annotation_files": missing_ann,
        "total_annotation_rows": rows,
        "ignored_region_rows": ignored_region_rows,
        "valid_object_rows": valid_objects,
        "removed_invalid_boxes": removed_invalid_boxes,
        "removed_categories": dict(sorted(removed_categories.items(), key=lambda kv: int(kv[0]) if kv[0].lstrip('-').isdigit() else 999)),
        "final_retained_targets": valid_objects,
        "examples": examples[:20],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--split", default="val", choices=["train", "val", "test-dev"])
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = audit_split(Path(args.root), args.split)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
