import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseScout(nn.Module):
    """MobileNetV3-Small + Tiny-FPN + single-channel response head.

    Input: [B, 3, 640, 640]
    Output: [B, 1, 80, 80]
    """

    def __init__(self, fpn_channels: int = 64, pretrained: bool = False):
        super().__init__()
        try:
            import timm
        except ImportError as exc:
            raise ImportError("DenseScout requires timm. Install requirements.txt first.") from exc
        self.backbone = timm.create_model(
            "mobilenetv3_small_100",
            pretrained=pretrained,
            features_only=True,
            out_indices=(2, 3, 4),
        )
        channels = self.backbone.feature_info.channels()
        c3_ch, c4_ch, c5_ch = channels
        self.lat_c5 = nn.Conv2d(c5_ch, fpn_channels, kernel_size=1)
        self.lat_c4 = nn.Conv2d(c4_ch, fpn_channels, kernel_size=1)
        self.lat_c3 = nn.Conv2d(c3_ch, fpn_channels, kernel_size=1)
        self.head = nn.Sequential(
            nn.Conv2d(fpn_channels, fpn_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c3, c4, c5 = self.backbone(x)
        p5 = self.lat_c5(c5)
        p4 = self.lat_c4(c4) + F.interpolate(p5, size=c4.shape[-2:], mode="nearest")
        p3 = self.lat_c3(c3) + F.interpolate(p4, size=c3.shape[-2:], mode="nearest")
        return self.head(p3)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
