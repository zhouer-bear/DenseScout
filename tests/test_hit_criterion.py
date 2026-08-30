from densescout.metrics import center_in_64_cell


def test_hit_criterion_is_square_not_l2():
    assert center_in_64_cell((32, 32), (0, 0))
    assert not center_in_64_cell((33, 0), (0, 0))
