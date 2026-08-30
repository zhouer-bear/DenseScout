"""DenseScout: budgeted patch-center selection frontend."""
from .model import DenseScout
from .decoder import decode_heatmap
from .loss import centernet_focal_loss

__all__ = ["DenseScout", "decode_heatmap", "centernet_focal_loss"]
