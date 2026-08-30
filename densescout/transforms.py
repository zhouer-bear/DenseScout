from typing import List, MutableMapping, Tuple
import cv2


def flip_record_horizontal(record: MutableMapping) -> MutableMapping:
    w = float(record["width"])
    for obj in record.get("objects", []):
        if "bbox_xyxy" in obj:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            obj["bbox_xyxy"] = [w - x2, y1, w - x1, y2]
    return record


def flip_record_vertical(record: MutableMapping) -> MutableMapping:
    h = float(record["height"])
    for obj in record.get("objects", []):
        if "bbox_xyxy" in obj:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            obj["bbox_xyxy"] = [x1, h - y2, x2, h - y1]
    return record


def flip_image_and_record(image, record: MutableMapping, horizontal: bool = False, vertical: bool = False):
    out = image
    rec = {**record, "objects": [dict(o) for o in record.get("objects", [])]}
    if horizontal:
        out = cv2.flip(out, 1)
        rec = flip_record_horizontal(rec)
    if vertical:
        out = cv2.flip(out, 0)
        rec = flip_record_vertical(rec)
    return out, rec
