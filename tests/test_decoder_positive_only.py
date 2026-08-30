import torch
from densescout.decoder import decode_heatmap


def test_decoder_does_not_pad_when_threshold_removes_all_candidates():
    sm = torch.full((1, 80, 80), -100.0)
    out = decode_heatmap(sm, topk=9, kernel_size=7, score_threshold=1.0)
    assert out['K_eff'] == 0
    assert out['centers'] == []


def test_negative_logit_local_peak_can_be_selected_after_sigmoid():
    sm = torch.full((80, 80), -10.0)
    sm[20, 30] = -1.0
    out = decode_heatmap(sm, topk=3, kernel_size=7, score_threshold=0.0)
    assert out['K_eff'] >= 1
    assert out['centers'][0].lattice_xy == (30, 20)
    assert 0.0 < out['centers'][0].score < 0.5
