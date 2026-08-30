from densescout.metrics import scale_point
from densescout.decoder import decode_heatmap
import torch


def test_coordinate_scale_original_to_proxy():
    assert scale_point((960, 540), (1920, 1080), (640, 640)) == (320, 320)


def test_decoder_center_maps_to_image_size():
    sm = torch.full((80, 80), -10.0)
    sm[0, 0] = 10.0
    out = decode_heatmap(sm, topk=1, kernel_size=1, image_size=(1920, 1080))
    c = out['centers'][0]
    assert c.proxy_xy == (4.0, 4.0)
    assert c.image_xy == (12.0, 6.75)
