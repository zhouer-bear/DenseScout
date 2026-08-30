import torch
from densescout.decoder import decode_heatmap


def test_local_max_suppresses_non_peak_neighbor():
    sm = torch.full((80, 80), -10.0)
    sm[10, 10] = 10.0
    sm[10, 11] = 9.0
    out = decode_heatmap(sm, topk=2, kernel_size=3, score_threshold=0.5)
    assert out['K_eff'] == 1
    assert out['centers'][0].lattice_xy == (10, 10)
