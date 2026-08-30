import json
import random
from pathlib import Path
from typing import Mapping
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from .metrics import bbox_center_xyxy, scale_point
from .transforms import flip_image_and_record

INPUT_SIZE = 640
GRID_SIZE = 80


def draw_gaussian(heatmap: np.ndarray, center, radius: int = 2, k: float = 1.0) -> None:
    diameter = 2 * radius + 1
    sigma = radius / 3.0 if radius > 0 else 1.0
    ax = np.arange(diameter, dtype=np.float32) - radius
    gaussian = np.exp(-(ax[:, None] ** 2 + ax[None, :] ** 2) / (2 * sigma ** 2)).astype(np.float32)
    x, y = int(center[0]), int(center[1])
    h, w = heatmap.shape
    if x < 0 or x >= w or y < 0 or y >= h:
        return
    left, right = min(x, radius), min(w - x, radius + 1)
    top, bottom = min(y, radius), min(h - y, radius + 1)
    np.maximum(heatmap[y-top:y+bottom, x-left:x+right], gaussian[radius-top:radius+bottom, radius-left:radius+right] * k, out=heatmap[y-top:y+bottom, x-left:x+right])


def build_heatmap(record: Mapping, radius: int = 2) -> np.ndarray:
    heatmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    ow, oh = int(record["width"]), int(record["height"])
    for obj in record.get("objects", []):
        if obj.get("ignore", 0) or "bbox_xyxy" not in obj:
            continue
        cx, cy = bbox_center_xyxy(obj["bbox_xyxy"])
        px, py = scale_point((cx, cy), (ow, oh), (INPUT_SIZE, INPUT_SIZE))
        draw_gaussian(heatmap, (px / 8.0, py / 8.0), radius=radius)
    return heatmap


class DenseScoutRecordsDataset(Dataset):
    def __init__(self, records_path, train: bool = True, hflip_prob: float = 0.5, vflip_prob: float = 0.0, radius: int = 2):
        self.records_path = Path(records_path)
        self.records = json.loads(self.records_path.read_text(encoding="utf-8"))
        self.train = train
        self.hflip_prob = hflip_prob
        self.vflip_prob = vflip_prob
        self.radius = radius

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = cv2.imread(rec["file_name"])
        if img is None:
            raise FileNotFoundError(rec["file_name"])
        if self.train:
            h = random.random() < self.hflip_prob
            v = random.random() < self.vflip_prob
            img, rec = flip_image_and_record(img, rec, horizontal=h, vertical=v)
        img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        heatmap = build_heatmap(rec, radius=self.radius)
        return torch.from_numpy(img).permute(2,0,1).float()/255.0, torch.from_numpy(heatmap).unsqueeze(0)
