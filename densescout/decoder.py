from dataclasses import dataclass
from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F


@dataclass
class DecodedCenter:
    rank: int
    score: float
    lattice_xy: Tuple[int, int]
    proxy_xy: Tuple[float, float]
    image_xy: Tuple[float, float]


def _as_bchw(score_map: torch.Tensor) -> torch.Tensor:
    if score_map.ndim == 2:
        score_map = score_map[None, None]
    elif score_map.ndim == 3:
        score_map = score_map[None] if score_map.shape[0] != 1 else score_map[:, None]
    if score_map.ndim != 4 or score_map.shape[1] != 1:
        raise ValueError("score_map must have shape [H,W], [1,H,W], or [B,1,H,W]")
    return score_map


def decode_heatmap(
    score_map: torch.Tensor,
    topk: int,
    kernel_size: int = 7,
    score_threshold: float = 0.0,
    image_size: Tuple[int, int] = (640, 640),
    proxy_size: Tuple[int, int] = (640, 640),
    apply_sigmoid: bool = True,
) -> Dict[str, object]:
    """Decode a response map into ranked patch centers without padding.

    Returns local maxima with strictly positive scores and score > score_threshold.
    `K_eff` is the actual number of returned centers and may be smaller than topk.
    """
    if topk < 0:
        raise ValueError("topk must be non-negative")
    sm = _as_bchw(score_map.detach().float().cpu())
    if sm.shape[0] != 1:
        raise ValueError("decode_heatmap currently decodes one image at a time")
    scores = torch.sigmoid(sm) if apply_sigmoid else sm
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be an odd positive integer")
    if kernel_size > 1:
        pooled = F.max_pool2d(scores, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        scores = scores * (scores == pooled)
    h, w = scores.shape[-2:]
    flat = scores.reshape(-1)
    # Paper-default gate: logits -> sigmoid score -> local-max sparse map ->
    # keep sparse scores > 0 and above the user threshold. Negative logits can
    # still be valid local peaks after sigmoid; we do not threshold logits by 0.
    keep = (flat > 0) & (flat > float(score_threshold))
    idx = torch.nonzero(keep, as_tuple=False).flatten()
    if idx.numel() == 0 or topk == 0:
        return {"centers": [], "K_eff": 0}
    vals = flat[idx]
    order = torch.argsort(vals, descending=True)
    idx = idx[order][:topk]
    vals = vals[order][:topk]
    img_w, img_h = image_size
    proxy_w, proxy_h = proxy_size
    stride_x = proxy_w / float(w)
    stride_y = proxy_h / float(h)
    sx = img_w / float(proxy_w)
    sy = img_h / float(proxy_h)
    centers: List[DecodedCenter] = []
    for rank, (linear, score) in enumerate(zip(idx.tolist(), vals.tolist()), start=1):
        y = linear // w
        x = linear % w
        px = (x + 0.5) * stride_x
        py = (y + 0.5) * stride_y
        centers.append(DecodedCenter(rank, float(score), (int(x), int(y)), (px, py), (px * sx, py * sy)))
    return {"centers": centers, "K_eff": len(centers)}
