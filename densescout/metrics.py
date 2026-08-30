from typing import Iterable, List, Mapping, Sequence, Tuple


def center_in_64_cell(pred_xy: Tuple[float, float], gt_xy: Tuple[float, float], half_size: float = 32.0) -> bool:
    """Square 64x64 selector-cell hit criterion, not an L2-radius circle."""
    return abs(pred_xy[0] - gt_xy[0]) <= half_size and abs(pred_xy[1] - gt_xy[1]) <= half_size


def bbox_center_xyxy(bbox: Sequence[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def scale_point(pt: Tuple[float, float], from_size: Tuple[int, int], to_size: Tuple[int, int]) -> Tuple[float, float]:
    x, y = pt
    fw, fh = from_size
    tw, th = to_size
    return x * tw / float(fw), y * th / float(fh)


def recall_from_centers(records: List[Mapping], centers_by_image: Mapping[str, Sequence[Tuple[float, float]]], half_size: float = 32.0) -> dict:
    total = 0
    hit = 0
    for rec in records:
        img_id = str(rec.get("image_id", rec.get("file_name")))
        centers = centers_by_image.get(img_id, [])
        ow, oh = int(rec["width"]), int(rec["height"])
        for obj in rec.get("objects", []):
            if obj.get("ignore", 0):
                continue
            if "bbox_xyxy" not in obj:
                continue
            total += 1
            gt_orig = bbox_center_xyxy(obj["bbox_xyxy"])
            gt_proxy = scale_point(gt_orig, (ow, oh), (640, 640))
            if any(center_in_64_cell(c, gt_proxy, half_size=half_size) for c in centers):
                hit += 1
    return {"recall": hit / total if total else 0.0, "num_gt": total, "num_hit": hit}
